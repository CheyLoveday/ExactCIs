import math
from functools import lru_cache
from typing import List, Tuple, Dict

# ─── 0. Validation ────────────────────────────────────────────────────────────

def _validate_counts(a: int, b: int, c: int, d: int) -> None:
    if not all(isinstance(x, int) and x >= 0 for x in (a, b, c, d)):
        raise ValueError("All counts must be non‑negative integers")
    if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        raise ValueError("Cannot compute odds ratio with empty margins")

# ─── 1. Support & PMF for noncentral hypergeometric ───────────────────────────

@lru_cache(maxsize=None)
def _support(n1: int, n2: int, m: int) -> Tuple[int, ...]:
    low = max(0, m - n2)
    high = min(m, n1)
    return tuple(range(low, high + 1))

def _pmf_weights(n1: int, n2: int, m: int, theta: float) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    supp = _support(n1, n2, m)
    if theta <= 0:
        w = [1.0 if k == supp[0] else 0.0 for k in supp]
    else:
        logt = math.log(theta)
        logs = [
            math.log(math.comb(n1, k))
            + math.log(math.comb(n2, m - k))
            + k * logt
            for k in supp
        ]
        M = max(logs)
        w = [math.exp(l - M) for l in logs]
    S = sum(w)
    return supp, tuple(wi / S for wi in w)

def _pmf(k: int, n1: int, n2: int, m: int, theta: float) -> float:
    supp, probs = _pmf_weights(n1, n2, m, theta)
    return probs[supp.index(k)]

# ─── 2. Bisection & Plateau‑aware root‑finding ─────────────────────────────────

def _find_root(f, lo: float = 1e-8, hi: float = 1.0,
               tol: float = 1e-8, maxiter: int = 60) -> float:
    f_lo, f_hi = f(lo), f(hi)
    while f_lo * f_hi > 0:
        hi *= 2; f_hi = f(hi)
        if hi > 1e16:
            raise RuntimeError("Failed to bracket root")
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol * max(1, hi):
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)

def _find_smallest_theta(f, alpha: float, **kwargs) -> float:
    two_sided = kwargs.pop("two_sided", True)
    θ0 = _find_root(lambda θ: f(θ) - alpha/2 if two_sided else f(θ) - alpha, **kwargs)
    lo, hi = θ0 * 0.5, θ0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if f(mid) <= (alpha/2 if two_sided else alpha):
            hi = mid
        else:
            lo = mid
        if hi/lo < 1.0001:
            break
    return hi

# ─── 3. Conditional (Fisher) CI ───────────────────────────────────────────────

def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    _validate_counts(a, b, c, d)
    n1, n2, m = a + b, c + d, a + c
    supp = _support(n1, n2, m)
    kmin, kmax = supp[0], supp[-1]

    def cdf_tail(theta: float, upper: bool) -> float:
        return sum(_pmf(k, n1, n2, m, theta)
                   for k in supp if (k >= a if upper else k <= a))

    low = 0.0 if a == kmin else _find_smallest_theta(lambda θ: cdf_tail(θ, True), alpha, lo=1e-8, hi=1.0)
    high = float('inf') if a == kmax else _find_smallest_theta(lambda θ: cdf_tail(θ, False), alpha, lo=1.0)
    return low, high

# ─── 4. Mid‑P adjusted CI ─────────────────────────────────────────────────────

def exact_ci_midp(a: int, b: int, c: int, d: int,
                  alpha: float = 0.05) -> Tuple[float, float]:
    _validate_counts(a, b, c, d)
    n1, n2, m = a + b, c + d, a + c
    supp, _ = _support(n1, n2, m), None

    def midp(theta: float) -> float:
        supp, probs = _pmf_weights(n1, n2, m, theta)
        idx = supp.index(a)
        less = sum(p for i,p in enumerate(probs) if supp[i] < a)
        eq   = probs[idx]
        more = sum(p for i,p in enumerate(probs) if supp[i] > a)
        return 2 * min(less + 0.5*eq, more + 0.5*eq)

    low = 0.0 if a == supp[0] else _find_smallest_theta(midp, alpha, lo=1e-8, hi=1.0, two_sided=False)
    high = float('inf') if a == supp[-1] else _find_smallest_theta(midp, alpha, lo=1.0, two_sided=False)
    return low, high

# ─── 5. Blaker’s exact CI ─────────────────────────────────────────────────────

def exact_ci_blaker(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05) -> Tuple[float, float]:
    _validate_counts(a, b, c, d)
    n1, n2, m = a + b, c + d, a + c
    supp, _ = _support(n1, n2, m), None

    def blaker_p(theta: float) -> float:
        supp, probs = _pmf_weights(n1, n2, m, theta)
        cdf_lo, run = [], 0.0
        for p in probs:
            run += p; cdf_lo.append(run)
        cdf_hi, run = [], 0.0
        for p in reversed(probs):
            run += p; cdf_hi.append(run)
        cdf_hi = list(reversed(cdf_hi))
        fvals = [min(l, h) for l,h in zip(cdf_lo, cdf_hi)]
        f_obs = fvals[supp.index(a)]
        return sum(p for f,p in zip(fvals, probs) if f <= f_obs)

    low = 0.0 if a == supp[0] else _find_smallest_theta(blaker_p, alpha, lo=1e-8, hi=1.0)
    high = float('inf') if a == supp[-1] else _find_smallest_theta(blaker_p, alpha, lo=1.0)
    return low, high

# ─── 6. Barnard’s unconditional exact CI ─────────────────────────────────────

def _pvalue_barnard(a: int, c: int, n1: int, n2: int,
                    theta: float, grid_size: int) -> float:
    eps = 1e-6
    best = 0.0
    try:
        import numpy as np
        ks = np.arange(n1+1)
        ls = np.arange(n2+1)
        K, L = np.meshgrid(ks, ls, indexing='ij')
        for i in range(grid_size+1):
            p1 = eps + i*(1-2*eps)/grid_size
            p2 = (theta * p1) / (1 - p1 + theta * p1)
            Pk = (np.math.comb(n1, K) * p1**K * (1-p1)**(n1-K))
            Pl = (np.math.comb(n2, L) * p2**L * (1-p2)**(n2-L))
            joint = Pk * Pl
            p_obs = joint[a, c]
            best = max(best, float(np.sum(joint[joint <= p_obs])))
    except ImportError:
        for i in range(grid_size+1):
            p1 = eps + i*(1-2*eps)/grid_size
            p2 = (theta * p1) / (1 - p1 + theta * p1)
            p_obs = (math.comb(n1, a) * p1**a * (1-p1)**(n1-a)
                     * math.comb(n2, c) * p2**c * (1-p2)**(n2-c))
            total = 0.0
            for k in range(n1+1):
                for l in range(n2+1):
                    p_kl = (math.comb(n1, k) * p1**k * (1-p1)**(n1-k)
                            * math.comb(n2, l) * p2**l * (1-p2)**(n2-l))
                    if p_kl <= p_obs:
                        total += p_kl
            best = max(best, total)
    return best

def exact_ci_unconditional(a: int, b: int, c: int, d: int,
                           alpha: float = 0.05, grid_size: int = 200
) -> Tuple[float, float]:
    _validate_counts(a, b, c, d)
    n1, n2 = a + b, c + d
    low = 0.0 if a == 0 else _find_smallest_theta(
        lambda θ: _pvalue_barnard(a, c, n1, n2, θ, grid_size), alpha, lo=1e-8, hi=1.0
    )
    high = float('inf') if a == n1 else _find_smallest_theta(
        lambda θ: _pvalue_barnard(a, c, n1, n2, θ, grid_size), alpha, lo=1.0
    )
    return low, high

# ─── 7. Haldane–Anscombe Wald CI w/o SciPy ────────────────────────────────────

def _normal_quantile(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    if p == 0.5:
        return 0.0
    q = p if p < 0.5 else 1-p
    t = math.sqrt(-2 * math.log(q))
    # Abramowitz & Stegun
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1*t + c2*t*t
    den = 1 + d1*t + d2*t*t + d3*t*t*t
    x = t - num/den
    return -x if p < 0.5 else x

def ci_wald_haldane(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05) -> Tuple[float, float]:
    _validate_counts(a, b, c, d)
    a2, b2, c2, d2 = a+0.5, b+0.5, c+0.5, d+0.5
    or_hat = (a2 * d2) / (b2 * c2)
    se = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
    z = _normal_quantile(1 - alpha/2)
    lo = math.exp(math.log(or_hat) - z*se)
    hi = math.exp(math.log(or_hat) + z*se)
    return lo, hi

# ─── Orchestrator ─────────────────────────────────────────────────────────────

def compute_all_cis(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05, grid_size: int = 200
) -> Dict[str, Tuple[float, float]]:
    _validate_counts(a, b, c, d)
    return {
        "conditional":  exact_ci_conditional(a, b, c, d, alpha),
        "midp":         exact_ci_midp(a, b, c, d, alpha),
        "blaker":       exact_ci_blaker(a, b, c, d, alpha),
        "barnard":      exact_ci_unconditional(a, b, c, d, alpha, grid_size),
        "wald_haldane": ci_wald_haldane(a, b, c, d, alpha),
    }

