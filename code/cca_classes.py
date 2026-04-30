from scipy.optimize import least_squares, fsolve, brentq
from scipy.special import gammaln
from scipy.stats import norm
import numpy as np

###############################################################################################
# Common functions
###############################################################################################

def compute_lcl_usd(M_bn, Bd_bn, fx_rate, r_d, r_f, T=1.0):
    """
    Local-currency liabilities in USD.
    
    LCL$ = (M·e^{r_d·T} + B_d) · e^{-r_f·T} / X_F
    
    All inputs in billions of local currency. fx_rate = LC per USD.
    r_d, r_f in decimal. Returns USD billions.
    """
    lcl_lc = M_bn * np.exp(r_d * T) + Bd_bn
    return lcl_lc * np.exp(-r_f * T) / fx_rate


def compute_barrier(debt, r_f, T=1.0):
    """
    KMV-style distress barrier: B_f = ST + 0.5·LT + interest.
    All in USD millions.
    """
    interest = (debt) * r_f * T
    return (debt + interest)

###############################################################################################
# M0/M1/M2 Solvers 
###############################################################################################
class BaselineCCAPricer:
    """
    M0 — Baseline sovereign CCA (Gray, Merton & Bodie, 2007).
    """

    def solve_CCA_M0(self, LCL_usd, sigma_lcl, B_f, r_f, T,
                     tol=1e-8, max_iter=100):
        if any(np.isnan(x) or x <= 0 for x in [LCL_usd, sigma_lcl, B_f]):
            return {'implied_V': np.nan, 'implied_sigma_V': np.nan, 'converged': False}

        sqt = np.sqrt(T)

        # Initial guess: σ_V from de-leveraged liability vol.
        sigma_V = max(sigma_lcl * LCL_usd / (LCL_usd + B_f), 1e-4)

        for _ in range(max_iter):
            # Step 1: solve eq1 for V via Brent. Call is monotone increasing in V.
            def eq1(V):
                d1 = (np.log(V / B_f) + (r_f + 0.5 * sigma_V**2) * T) / (sigma_V * sqt)
                d2 = d1 - sigma_V * sqt
                return V * norm.cdf(d1) - B_f * np.exp(-r_f * T) * norm.cdf(d2) - LCL_usd

            try:
                V = brentq(eq1, B_f * 1e-3, (LCL_usd + B_f) * 100, xtol=1e-10)
            except ValueError:
                V = brentq(eq1, 1e-6, (LCL_usd + B_f) * 1e6, xtol=1e-10)

            # Step 2: back σ_V out from eq2 directly.
            d1          = (np.log(V / B_f) + (r_f + 0.5 * sigma_V**2) * T) / (sigma_V * sqt)
            sigma_V_new = LCL_usd * sigma_lcl / (V * norm.cdf(d1))

            if abs(sigma_V_new - sigma_V) < tol:
                sigma_V = sigma_V_new
                break
            sigma_V = sigma_V_new

        return {'implied_V': V, 'implied_sigma_V': sigma_V, 'converged': True}

class ConvenienceYieldCCAPricer:
    def __init__(self, gamma):
        self.gamma = gamma

    def _sigma_total(self, sigma_V, sigma_y):
        return np.sqrt(sigma_V**2 + (self.gamma * sigma_y)**2)

    def solve_CCA_M1(self, LCL_usd, sigma_lcl, B_f, r_f, y, sigma_y, T,
                     tol=1e-8, max_iter=100):
        if any(np.isnan(x) or x <= 0 for x in [LCL_usd, sigma_lcl, B_f]) or np.isnan(y):
            return {'implied_V': np.nan, 'implied_sigma_V': np.nan,
                    'converged': False, 'convenience_yield': y}

        gy, gs = self.gamma * y, self.gamma * sigma_y
        sqt    = np.sqrt(T)
        phi    = np.exp(-gy * T + 0.5 * gs**2 * T)   # V_eff = V * phi

        # Initial guess: σ_V from de-leveraged liability vol.
        sigma_V = max(sigma_lcl * LCL_usd / (LCL_usd + B_f), 1e-4)

        for _ in range(max_iter):
            sigma_total = np.sqrt(sigma_V**2 + gs**2)

            # Step 1: solve eq1 for V via Brent. Call is monotone in V.
            def eq1(V):
                Veff = V * phi
                d1   = (np.log(Veff / B_f) + (r_f + 0.5 * sigma_total**2) * T) / (sigma_total * sqt)
                d2   = d1 - sigma_total * sqt
                return Veff * norm.cdf(d1) - B_f * np.exp(-r_f * T) * norm.cdf(d2) - LCL_usd

            try:
                V = brentq(eq1, B_f * 1e-3, (LCL_usd + B_f) * 100, xtol=1e-10)
            except ValueError:
                V = brentq(eq1, 1e-6, (LCL_usd + B_f) * 1e6, xtol=1e-10)

            # Step 2: back σ_V out from eq2.
            Veff            = V * phi
            d1              = (np.log(Veff / B_f) + (r_f + 0.5 * sigma_total**2) * T) / (sigma_total * sqt)
            sigma_total_new = LCL_usd * sigma_lcl / (Veff * norm.cdf(d1))
            sigma_V_new     = np.sqrt(max(sigma_total_new**2 - gs**2, 1e-8))

            if abs(sigma_V_new - sigma_V) < tol:
                sigma_V = sigma_V_new
                break
            sigma_V = sigma_V_new

        return {'implied_V': V,
                'implied_sigma_V': self._sigma_total(sigma_V, sigma_y),
                'converged': True,
                'convenience_yield': y}


class LogNormalJumpCCAPricer:
    def __init__(self, mu_J, sigma_J):
        self.mu_J    = mu_J
        self.sigma_J = sigma_J
        self.k       = np.exp(mu_J + 0.5 * sigma_J**2) - 1

    def sigma_total(self, sigma_diff, lam):
        return np.sqrt(sigma_diff**2 + lam * (self.mu_J**2 + self.sigma_J**2))

    def _series(self, V, B, r, T, sigma_diff, lam):
        lam_T = lam * (1 + self.k) * T
        sqt   = np.sqrt(T)

        sigma_pois = np.sqrt(lam_T)
        n_low  = max(0, int(lam_T - 6 * sigma_pois))
        n_high = int(lam_T + 6 * sigma_pois) + 1
        ns     = np.arange(n_low, n_high + 1)

        log_w   = -lam_T + ns * np.log(lam_T) - gammaln(ns + 1)
        weights = np.exp(log_w)

        r_ns     = r - lam * self.k + ns * (self.mu_J + 0.5 * self.sigma_J**2) / T
        sigma_ns = np.sqrt(sigma_diff**2 + ns * self.sigma_J**2 / T)

        d1s = (np.log(V / B) + (r_ns + 0.5 * sigma_ns**2) * T) / (sigma_ns * sqt)
        d2s = d1s - sigma_ns * sqt

        calls  = V * norm.cdf(d1s) - B * np.exp(-r_ns * T) * norm.cdf(d2s)
        deltas = norm.cdf(d1s)

        return float(weights @ calls), float(weights @ deltas)

    def solve_CCA_M2(self, LCL_usd, sigma_lcl, B_f, r_f, T, lam,
                     prev_solution=None, tol=1e-8, max_iter=100):
        if any(np.isnan(x) or x <= 0 for x in [LCL_usd, sigma_lcl, B_f]):
            return {'implied_V': np.nan, 'implied_sigma_diff': np.nan,
                    'implied_sigma_V': np.nan, 'converged': False}

        # Warm-start σ_diff from previous observation if available.
        if prev_solution is not None and prev_solution.get('implied_sigma_diff'):
            sigma_diff = prev_solution['implied_sigma_diff']
        else:
            sigma_diff = max(sigma_lcl * LCL_usd / (LCL_usd + B_f), 1e-4)

        for _ in range(max_iter):
            # Step 1: Brent on eq1 (call price = LCL) at fixed σ_diff.
            def eq1(V):
                call, _ = self._series(V, B_f, r_f, T, sigma_diff, lam)
                return call - LCL_usd

            try:
                V = brentq(eq1, B_f * 1e-3, (LCL_usd + B_f) * 100, xtol=1e-10)
            except ValueError:
                V = brentq(eq1, 1e-6, (LCL_usd + B_f) * 1e6, xtol=1e-10)

            # Step 2: Brent on eq2 (delta-vol identity) at the V just found.
            #   σ_diff·V·Δ + λk·V·Δ = LCL·σ_lcl
            def eq2(s):
                _, delta = self._series(V, B_f, r_f, T, s, lam)
                return s * V * delta - LCL_usd * sigma_lcl
            try:
                sigma_diff_new = brentq(eq2, 1e-6, 5.0, xtol=1e-10)
            except ValueError:
                # Eq2 has no root in the interval — happens at structural floor.
                sigma_diff_new = 1e-6

            if abs(sigma_diff_new - sigma_diff) < tol:
                sigma_diff = sigma_diff_new
                break
            sigma_diff = sigma_diff_new

        return {'implied_V': V,
                'implied_sigma_diff': sigma_diff,
                'implied_sigma_V': self.sigma_total(sigma_diff, lam),
                'converged': True}