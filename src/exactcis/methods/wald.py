"""
Haldane-Anscombe Wald confidence interval for odds ratio.

This module implements the Haldane-Anscombe Wald confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import math
from typing import Tuple

from exactcis.utils.validation import validate_counts
from exactcis.utils.estimates import compute_log_or_with_se
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
    
    # Use centralized estimate and standard error calculation
    estimation_result = compute_log_or_with_se(a, b, c, d, method="wald_haldane")
    log_or = estimation_result.point_estimate
    se_log_or = estimation_result.standard_error
    
    # Use original normal quantile for exact parity
    z = normal_quantile(1 - alpha/2)
    
    # Calculate confidence interval bounds
    lower_log = log_or - z * se_log_or
    upper_log = log_or + z * se_log_or
    
    return math.exp(lower_log), math.exp(upper_log)
