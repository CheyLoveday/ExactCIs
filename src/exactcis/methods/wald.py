"""
Haldane-Anscombe Wald confidence interval for odds ratio.

This module implements the Haldane-Anscombe Wald confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import math
from typing import Tuple

from exactcis.core import validate_counts
from exactcis.utils.stats import normal_quantile


def ci_wald_haldane(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Haldane-Anscombe Wald confidence interval for the odds ratio.

    This method adds 0.5 to each cell and applies the standard log-OR ± z·SE formula.
    It includes a pure-Python normal quantile fallback if SciPy is absent.
    It is appropriate for large samples where asymptotic Wald is reasonable,
    quick approximate intervals for routine reporting, and when speed and
    convenience outweigh strict exactness.

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
    
    a2, b2, c2, d2 = a+0.5, b+0.5, c+0.5, d+0.5
    or_hat = (a2 * d2) / (b2 * c2)
    se = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
    z = normal_quantile(1 - alpha/2)
    lo = math.exp(math.log(or_hat) - z*se)
    hi = math.exp(math.log(or_hat) + z*se)
    return lo, hi
