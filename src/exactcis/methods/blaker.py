"""
Blaker's exact confidence interval for odds ratio.

This module implements Blaker's exact confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple

from exactcis.core import validate_counts, support, pmf_weights, find_smallest_theta


def exact_ci_blaker(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Blaker's exact confidence interval for the odds ratio.

    This method uses the acceptability function f(k)=min[P(K≤k),P(K≥k)] and inverts it
    exactly for monotonic, non-flip intervals. It is appropriate for routine exact
    inference when Fisher is overly conservative, fields standardized on Blaker
    (e.g. genomics, toxicology), and exact coverage with minimal over-coverage.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    validate_counts(a, b, c, d)

    n1, n2, m = a + b, c + d, a + c
    supp = support(n1, n2, m)

    def blaker_p(theta: float) -> float:
        supp, probs = pmf_weights(n1, n2, m, theta)
        cdf_lo, run = [], 0.0
        for p in probs:
            run += p
            cdf_lo.append(run)
        cdf_hi, run = [], 0.0
        for p in reversed(probs):
            run += p
            cdf_hi.append(run)
        cdf_hi = list(reversed(cdf_hi))
        fvals = [min(l, h) for l, h in zip(cdf_lo, cdf_hi)]
        f_obs = fvals[supp.index(a)]
        return sum(p for f, p in zip(fvals, probs) if f <= f_obs)

    low = 0.0 if a == supp[0] else find_smallest_theta(
        blaker_p, alpha, lo=1e-8, hi=1.0
    )
    high = float('inf') if a == supp[-1] else find_smallest_theta(
        blaker_p, alpha, lo=1.0
    )
    return low, high
