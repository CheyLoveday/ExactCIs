"""
Centralized statistical estimates for ExactCIs.

This module consolidates point estimates, standard errors, and related 
statistical calculations. It follows a principle where high-level API functions
determine the correction policy, apply it once, and then call raw calculation
functions. This centralizes logic and improves efficiency.
"""

import math
import logging
from typing import Tuple, Union

from exactcis.models import EstimationResult
from exactcis.utils.continuity import get_corrected_counts, detect_zero_cells
from exactcis.utils.mathops import (
    safe_division, safe_log_ratio, safe_log, exp_safe, clip_probability,
    wald_variance_rr
)
from exactcis.utils.validation import validate_counts
from exactcis.constants import (
    MIN_PROB, MAX_PROB, CorrectionMethod, WaldMethod, Distribution
)
# Import the new solver function
from exactcis.utils.solvers import wald_ci_from_log_estimate

logger = logging.getLogger(__name__)

# ##################################################################
# Private Raw Calculation Functions
# These functions perform pure calculations and expect valid inputs.
# They do not contain any correction logic.
# ##################################################################

def _raw_log_or(a: float, b: float, c: float, d: float) -> float:
    """Calculates log odds ratio on raw (potentially corrected) counts.""" 
    return safe_log_ratio(a * d, b * c)

def _raw_se_log_or_wald(a: float, b: float, c: float, d: float) -> float:
    """Calculates Wald SE for log OR on raw (potentially corrected) counts."""
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return float('inf')
    variance = (1/a + 1/b + 1/c + 1/d)
    return math.sqrt(variance)

def _raw_log_rr(a: float, b: float, c: float, d: float) -> float:
    """Calculates log relative risk on raw (potentially corrected) counts.""" 
    n1 = a + b
    n2 = c + d
    # Let safe_log_ratio handle division by zero issues
    risk1 = safe_division(a, n1)
    risk2 = safe_division(c, n2)
    return safe_log_ratio(risk1, risk2)

# ##################################################################
# High-Level API Functions (Estimate + SE)
# The primary entry points for getting an estimate and its SE. 
# They centralize the correction logic.
# ##################################################################

def compute_log_or_with_se(a: Union[int, float], b: Union[int, float],
                          c: Union[int, float], d: Union[int, float],
                          method: WaldMethod = "wald") -> EstimationResult:
    """
    Compute log odds ratio and its SE using a centralized correction policy.

    Returns:
        An EstimationResult object containing the estimate, standard error,
        and a record of any correction applied.
    """
    validate_counts(a, b, c, d, allow_zero_margins=True)
    
    # Get corrected counts from the central policy function
    correction = get_corrected_counts(a, b, c, d, method=method)
    
    # Extract corrected counts from CorrectionResult
    a_corr, b_corr, c_corr, d_corr = (
        correction.a, correction.b, correction.c, correction.d
    )
    
    # Call raw calculation functions with corrected counts
    log_or = _raw_log_or(a_corr, b_corr, c_corr, d_corr)
    se = _raw_se_log_or_wald(a_corr, b_corr, c_corr, d_corr)
    
    return EstimationResult(
        point_estimate=log_or,
        standard_error=se,
        log_estimate=log_or,
        correction_applied=correction.correction_applied
    )

def compute_log_rr_with_se(a: Union[int, float], b: Union[int, float],
                          c: Union[int, float], d: Union[int, float],
                          method: str = "wald") -> EstimationResult:
    """
    Compute log relative risk and its SE using a centralized correction policy.
    TODO: Extend for different RR methods like katz, correlated.

    Returns:
        An EstimationResult object containing the estimate, standard error,
        and a record of any correction applied.
    """
    validate_counts(a, b, c, d, allow_zero_margins=True)
    
    # Get corrected counts from the central policy function
    # We pass 'wald' for now as the RR correction policy isn't defined yet
    correction = get_corrected_counts(a, b, c, d, method="wald")
    
    a_corr, b_corr, c_corr, d_corr = (
        correction.a, correction.b, correction.c, correction.d
    )

    # Calculate estimate and SE with corrected counts
    log_rr = _raw_log_rr(a_corr, b_corr, c_corr, d_corr)
    variance = wald_variance_rr(a_corr, b_corr, c_corr, d_corr, method="standard")
    se = math.sqrt(variance) if variance > 0 and not math.isinf(variance) else float('inf')

    return EstimationResult(
        point_estimate=log_rr,
        standard_error=se,
        log_estimate=log_rr,
        correction_applied=correction.correction_applied
    )

# ##################################################################
# Public-Facing Point Estimate & SE Functions (Legacy Support)
# These functions maintain the original API for backward compatibility.
# ##################################################################

def compute_odds_ratio(a: Union[int, float], b: Union[int, float],
                      c: Union[int, float], d: Union[int, float],
                      correction: CorrectionMethod = None) -> float:
    """
    Compute odds ratio.

    This function is a wrapper around the modern compute_log_or_with_se
    for backward compatibility. The 'correction' parameter is deprecated
    as correction policy is now handled centrally.
    """
    # NOTE: The 'correction' parameter is deprecated and will be ignored.
    # The modern API handles correction policy automatically.
    result = compute_log_or_with_se(a, b, c, d, method="wald")
    return exp_safe(result.point_estimate)


def compute_relative_risk(a: Union[int, float], b: Union[int, float],
                         c: Union[int, float], d: Union[int, float],
                         correction: CorrectionMethod = None) -> float:
    """
    Compute relative risk.

    This function is a wrapper around the modern compute_log_rr_with_se
    for backward compatibility. The 'correction' parameter is deprecated
    as correction policy is now handled centrally.
    """
    # NOTE: The 'correction' parameter is deprecated and will be ignored.
    # The modern API handles correction policy automatically.
    result = compute_log_rr_with_se(a, b, c, d, method="wald")
    return exp_safe(result.point_estimate)

# ##################################################################
# Confidence Interval and Other Utilities
# ##################################################################

def ci_from_log_estimate_se(log_estimate: float, se: float, alpha: float = 0.05,
                           distribution: Distribution = "normal") -> Tuple[float, float]:
    """
    Compute CI by delegating to the appropriate solver.
    
    This function acts as a high-level wrapper, delegating the actual numerical
    computation to the centralized solver functions. This separates the
    statistical modeling from the numerical implementation.
    """
    # Delegate directly to the wald_ci_from_log_estimate solver.
    # This keeps the estimates module clean and focused on statistical policy.
    return wald_ci_from_log_estimate(
        log_estimate=log_estimate, 
        se=se, 
        alpha=alpha, 
        distribution=distribution
    )

def estimate_precision_diagnostics(a: Union[int, float], b: Union[int, float],
                                 c: Union[int, float], d: Union[int, float],
                                 alpha: float = 0.05) -> dict:
    """Provide diagnostic information about estimate precision and reliability."""
    validate_counts(a, b, c, d, allow_zero_margins=True)
    n = a + b + c + d
    n1, n2 = a + b, c + d
    has_zeros, zero_cells, zero_margins = detect_zero_cells(a, b, c, d)
    min_expected = 0
    if n > 0 and (a+c) > 0 and (b+d) > 0 and n1 > 0 and n2 > 0:
        min_expected = min(n1 * (a + c) / n, n1 * (b + d) / n,
                           n2 * (a + c) / n, n2 * (b + d) / n)

    diagnostics = {
        "total_sample_size": n,
        "group_sizes": (n1, n2),
        "has_zero_cells": has_zeros,
        "min_expected_cell": min_expected,
        "asymptotic_adequate": min_expected >= 5,
        "sparse_table": has_zeros or min_expected < 5,
    }
    return diagnostics
