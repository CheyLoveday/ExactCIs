"""
Mid-P adjusted confidence interval for odds ratio.

This module implements the Mid-P adjusted confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple

from exactcis.core import validate_counts, support, pmf_weights, find_smallest_theta


def exact_ci_midp(a: int, b: int, c: int, d: int,
                  alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Mid-P adjusted confidence interval for the odds ratio.

    This method is similar to the conditional (Fisher) method but gives half-weight
    to the observed table in the tail p-value, reducing conservatism. It is appropriate
    for epidemiology or surveillance where conservative Fisher intervals are too wide,
    and for moderate samples where slight undercoverage is tolerable for tighter intervals.

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

    def midp(theta: float) -> float:
        supp, probs = pmf_weights(n1, n2, m, theta)
        idx = supp.index(a)
        less = sum(p for i, p in enumerate(probs) if supp[i] < a)
        eq = probs[idx]
        more = sum(p for i, p in enumerate(probs) if supp[i] > a)
        return 2 * min(less + 0.5*eq, more + 0.5*eq)

    low = 0.0 if a == supp[0] else find_smallest_theta(
        midp, alpha, lo=1e-8, hi=1.0, two_sided=False
    )
    high = float('inf') if a == supp[-1] else find_smallest_theta(
        midp, alpha, lo=1.0, two_sided=False
    )
    return low, high
