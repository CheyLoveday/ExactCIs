"""
Blaker's exact confidence interval for odds ratio.

This module implements Blaker's exact confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple
import logging
import numpy as np
from scipy.stats import nchypergeom_fisher
from scipy.optimize import brentq

from exactcis.core import (
    validate_counts,
    support, # This now returns SupportData
    log_nchg_pmf, # Keep existing imports as needed
    find_smallest_theta,
    nchg_pdf, 
    estimate_point_or,
    SupportData # Changed from Support to SupportData
)

logger = logging.getLogger(__name__)


def blaker_acceptability(n1: int, n2: int, m1: int, theta: float, support_x: np.ndarray) -> np.ndarray:
    # DEBUG LOGGING FOR SPECIFIC CASE
    if n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6:
        logger.info(f"[DEBUG_ACCEPTABILITY] n1=5,n2=10,m1=7,theta={theta:.2e}, support_x={support_x}")
    
    probs = nchg_pdf(support_x, n1, n2, m1, theta)

    if n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6:
        logger.info(f"[DEBUG_ACCEPTABILITY] Probs from nchg_pdf: {probs}")
    return probs


def blaker_p_value(a: int, n1: int, n2: int, m1: int, theta: float, s: SupportData) -> float:
    """
    Calculates Blaker's p-value for a given theta, for observed count 'a'.
    s: SupportData object containing s.x (support array), s.min_val, s.max_val, and s.offset.
    """
    # 1. Calculate acceptability P(X=k|theta) for all k in the support s.x
    accept_probs_all_k = blaker_acceptability(n1, n2, m1, theta, s.x)
    
    # DEBUG LOGGING FOR SPECIFIC CASE
    is_debug_case = (a == 5 and n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6)
    if is_debug_case:
        logger.info(f"[DEBUG_PVAL] For a=5,n1=5,n2=10,m1=7,theta={theta:.2e}")
        logger.info(f"[DEBUG_PVAL] support s.x={s.x}, s.min_val={s.min_val}, s.offset={s.offset}")
        logger.info(f"[DEBUG_PVAL] accept_probs_all_k = {accept_probs_all_k}")

    # 2. Get the acceptability probability for the observed 'a'
    # The index for 'a' in the support_x array (s.x) is (a - s.min_val) or (s.offset + a).
    # s.offset should map 'a' (absolute count) to its 0-based index within s.x.
    idx_a = s.offset + a
    
    # Basic check for index validity.
    if not (0 <= idx_a < len(accept_probs_all_k)):
        logger.warning(f"Blaker p-value: Calculated index for 'a' ({idx_a}) is out of bounds for acceptability array (len {len(accept_probs_all_k)}). a={a}, s.offset={s.offset}, s.x_len={len(s.x)}. This may indicate an issue with 'a' or support generation.")
        # If 'a' is impossible under 'theta', its probability should be 0.
        # If P(a|theta) is 0, p-value is sum of P(k|theta) for k where P(k|theta) <= 0.
        # This could lead to p_val = 0. For robustness, if idx_a is truly invalid, one might return 1.0 or raise error.
        # However, nchg_pdf returning 0 for k outside actual support range is expected.
        # If current_accept_prob_at_a is problematic due to bad index, we might have issues.
        # For now, relying on upstream validation of 'a' against s.min_val/s.max_val.
        # A robust nchg_pdf should handle cases where k is outside its natural range for given Ns,M,theta (typically returns 0 probability).
        # If idx_a is bad, the next line will raise IndexError. Let's assume it's valid for now.
        pass

    current_accept_prob_at_a = accept_probs_all_k[idx_a]
    
    if is_debug_case:
        logger.info(f"[DEBUG_PVAL] idx_a = {idx_a}, current_accept_prob_at_a = {current_accept_prob_at_a}")

    # 3. Sum probabilities for k where P(k|theta) <= P(a|theta) * (1 + epsilon)
    # Epsilon is a small tolerance factor for floating point comparisons, as per Blaker (2000) and R's exact2x2.
    epsilon = 1e-7 
    p_val_terms = accept_probs_all_k[accept_probs_all_k <= current_accept_prob_at_a * (1 + epsilon)]
    p_val = np.sum(p_val_terms)
    
    if is_debug_case:
        logger.info(f"[DEBUG_PVAL] p_val_terms = {p_val_terms}")
        logger.info(f"[DEBUG_PVAL] Final p_val = {p_val}")

    return p_val


def exact_ci_blaker(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Blaker's exact confidence interval for the odds ratio.
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    Raises:
        ValueError: If inputs are invalid (negative counts, empty margins, invalid alpha)
    """
    validate_counts(a, b, c, d)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    
    # Special handling for zero cells to ensure the true OR is contained in CI
    # When a=0, the true OR is 0.0, so the lower bound should be 0.0
    if a == 0:
        logger.info(f"Blaker CI: Zero in cell a, ensuring lower bound is 0.0")
        # Compute upper bound using Haldane-corrected values but set lower to 0.0
        # Fall through to normal computation but will override lower bound at the end
        
    n1, n2 = a + b, c + d
    m1, _ = a + c, b + d # m2 is not directly needed for nchg_pdf with m1

    # Add point estimate for logging, using Haldane correction
    or_point_est = estimate_point_or(a,b,c,d, correction_type='haldane')
    logger.info(f"Blaker exact_ci_blaker: Input ({a},{b},{c},{d}), alpha={alpha}. Point OR estimate (Haldane corrected): {or_point_est:.4f}")

    s = support(n1, n2, m1) # SupportData object
    kmin, kmax = s.min_val, s.max_val

    # Initialize raw_theta_low and raw_theta_high for cases where search might fail
    raw_theta_low = 0.0 
    raw_theta_high = float('inf')

    # Lower bound calculation - always search
    blaker_p_value_lower = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
    logger.info(f"Blaker exact_ci_blaker: Finding lower bound. Target p-value for search: {alpha/2.0:.4f}")
    
    # Determine search range for lower bound based on OR estimate
    # Search from a small number up to the point estimate (or a large number if OR is small/invalid)
    lo_search_lower = 1e-9
    hi_search_lower = or_point_est if (np.isfinite(or_point_est) and or_point_est > lo_search_lower) else 1e7
    # Ensure hi > lo for the search
    if hi_search_lower <= lo_search_lower: hi_search_lower = 1e7

    if a == kmin: 
        logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmin ({kmin}). Standard search for lower bound.")

    candidate_raw_theta_low = find_smallest_theta(
        func=blaker_p_value_lower, 
        alpha=alpha / 2.0, 
        lo=lo_search_lower, 
        hi=hi_search_lower, 
        increasing=False
    )
    if candidate_raw_theta_low is not None:
        raw_theta_low = candidate_raw_theta_low
    else:
        logger.warning(f"Blaker exact_ci_blaker: Lower bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_low to 0.0.")
        raw_theta_low = 0.0 # Keep default if search fails

    # Upper bound calculation - always search
    blaker_p_value_upper = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
    logger.info(f"Blaker exact_ci_blaker: Finding upper bound. Target p-value for search: {alpha/2.0:.4f}")

    # Determine search range for upper bound based on OR estimate
    # Search from the point estimate (or a small number if OR is small/invalid) up to a large number
    hi_search_upper = 1e7
    lo_search_upper = or_point_est if (np.isfinite(or_point_est) and or_point_est > 1e-9) else 1e-9
    # Ensure lo < hi for the search
    if lo_search_upper >= hi_search_upper: lo_search_upper = 1e-9 

    if a == kmax: 
        logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmax ({kmax}). Standard search for upper bound.")

    candidate_raw_theta_high = find_smallest_theta(
        func=blaker_p_value_upper, 
        alpha=alpha / 2.0, 
        lo=lo_search_upper, 
        hi=hi_search_upper, 
        increasing=True
    )
    if candidate_raw_theta_high is not None:
        raw_theta_high = candidate_raw_theta_high
    else:
        logger.warning(f"Blaker exact_ci_blaker: Upper bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_high to inf.")
        raw_theta_high = float('inf') # Keep default if search fails

    # Convert, check finiteness, and check for crossed bounds
    try:
        logger.info(f"Blaker exact_ci_blaker DEBUG: raw_theta_low IS: {raw_theta_low}, TYPE IS: {type(raw_theta_low)}")
        theta_low = float(raw_theta_low) # Attempt conversion
        logger.info(f"Blaker exact_ci_blaker DEBUG: converted theta_low IS: {theta_low}, TYPE IS: {type(theta_low)}")

        logger.info(f"Blaker exact_ci_blaker DEBUG: raw_theta_high IS: {raw_theta_high}, TYPE IS: {type(raw_theta_high)}")
        theta_high = float(raw_theta_high) # Attempt conversion
        logger.info(f"Blaker exact_ci_blaker DEBUG: converted theta_high IS: {theta_high}, TYPE IS: {type(theta_high)}")
        
        is_finite_low = np.isfinite(theta_low)
        is_finite_high = np.isfinite(theta_high)
        logger.info(f"Blaker exact_ci_blaker DEBUG: is_finite_low IS: {is_finite_low}, TYPE IS: {type(is_finite_low)}")
        logger.info(f"Blaker exact_ci_blaker DEBUG: is_finite_high IS: {is_finite_high}, TYPE IS: {type(is_finite_high)}")

        both_finite = is_finite_low and is_finite_high
        logger.info(f"Blaker exact_ci_blaker DEBUG: combined 'both_finite' IS: {both_finite}, TYPE IS: {type(both_finite)}")

        if not both_finite:
            logger.warning(f"Blaker CI calculation resulted in non-finite bounds for ({a},{b},{c},{d}): low={theta_low}, high={theta_high}. Defaulting to (0, inf) for safety.")
            return 0.0, float('inf')
        
        # Check for crossed bounds AFTER ensuring they are finite numbers
        # Allow for small numerical precision errors
        tolerance = 1e-6
        if theta_low > theta_high + tolerance:
            logger.warning(f"Blaker CI: Lower bound {theta_low:.4f} > Upper bound {theta_high:.4f} for ({a},{b},{c},{d}). This can indicate issues with root finding or p-value function. Returning (0, inf) as a safe default. Original raw values: low={raw_theta_low}, high={raw_theta_high}")
            return 0.0, float('inf')
        elif theta_low > theta_high:
            # Small numerical precision issue - but check for zero cell first
            if a == 0:
                logger.info(f"Blaker CI: Zero cell case with numerical precision issue. Overriding lower bound to 0.0")
                return 0.0, theta_high
            # Normal numerical precision issue - average the bounds
            avg_bound = (theta_low + theta_high) / 2.0
            logger.info(f"Blaker CI: Minor numerical precision issue. Lower {theta_low:.8f} > Upper {theta_high:.8f} by {theta_low - theta_high:.2e}. Using average {avg_bound:.8f} for both bounds.")
            return avg_bound, avg_bound

        # Override lower bound for zero cell cases to ensure true OR is included
        if a == 0:
            logger.info(f"Blaker CI: Overriding lower bound to 0.0 for zero cell case")
            return 0.0, theta_high
        
        return theta_low, theta_high

    except ValueError as e:
        logger.error(f"Blaker exact_ci_blaker: ValueError during float conversion or finiteness check for ({a},{b},{c},{d}): {e}. Details: raw_theta_low='{raw_theta_low}' (type {type(raw_theta_low)}), raw_theta_high='{raw_theta_high}' (type {type(raw_theta_high)})", exc_info=True)
        # Return strings to indicate error type, consistent with previous error handling pattern
        return (f"ValueError: {e}", f"ValueError: {e}") 
    except TypeError as e:
        logger.error(f"Blaker exact_ci_blaker: TypeError during float conversion or finiteness check for ({a},{b},{c},{d}): {e}. Details: raw_theta_low='{raw_theta_low}' (type {type(raw_theta_low)}), raw_theta_high='{raw_theta_high}' (type {type(raw_theta_high)})", exc_info=True)
        return (f"TypeError: {e}", f"TypeError: {e}")
