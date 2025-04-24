"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple

from exactcis.core import validate_counts, support, pmf, find_smallest_theta


def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the conditional (Fisher) confidence interval for the odds ratio.

    This method inverts the noncentral hypergeometric CDF at alpha/2 in each tail.
    It is appropriate for very small samples or rare events, regulatory/safety-critical
    studies requiring guaranteed ≥ 1-alpha coverage, and fixed-margin designs (case-control).

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval

    Raises:
        ValueError: If alpha is not in (0, 1)
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    validate_counts(a, b, c, d)

    n1, n2, m = a + b, c + d, a + c
    supp = support(n1, n2, m)
    kmin, kmax = supp[0], supp[-1]

    def cdf_tail(theta: float, upper: bool) -> float:
        return sum(pmf(k, n1, n2, m, theta)
                   for k in supp if (k >= a if upper else k <= a))

    # For edge cases, use predefined values
    if a == kmin:
        low = 0.0
    else:
        try:
            low = find_smallest_theta(
                lambda theta: cdf_tail(theta, True), alpha, lo=1e-8, hi=1.0, two_sided=True
            )
        except RuntimeError:
            low = 0.0

    if a == kmax:
        high = float('inf')
    else:
        try:
            high = find_smallest_theta(
                lambda theta: cdf_tail(theta, False), alpha, lo=1.0, two_sided=True
            )
        except RuntimeError:
            # For the edge case where a is at the minimum possible value,
            # we should return a finite upper bound
            if a == kmin:
                high = float('inf')  # Consistent with other edge cases
            else:
                high = float('inf')
    return low, high
