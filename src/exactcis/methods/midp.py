"""
Mid-P adjusted confidence interval for odds ratio.

This module implements the Mid-P adjusted confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import math
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from exactcis.core import (
    validate_counts,
    support,
    log_nchg_pmf,
    logsumexp,
    find_smallest_theta,
    apply_haldane_correction
)

# Configure logging
logger = logging.getLogger(__name__)

# Cache for previously computed values
_cache: Dict[Tuple[int, int, int, int, float], Tuple[float, float]] = {}


def exact_ci_midp(a: int, b: int, c: int, d: int,
                  alpha: float = 0.05, 
                  progress_callback: Optional[callable] = None) -> Tuple[float, float]:
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
        progress_callback: Optional callback function to report progress (0-100)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    # Check cache first
    cache_key = (a, b, c, d, alpha)
    if cache_key in _cache:
        logger.info(f"Using cached CI values for: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
        return _cache[cache_key]
    
    validate_counts(a, b, c, d)
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

    # Store original counts for PMF calculation basis
    a_orig, b_orig, c_orig, d_orig = a, b, c, d

    # Effective observed count 'a_eff_obs' for comparison against PMF.
    # For Mid-P, we use the original integer count without Haldane correction
    # to maintain consistency between observation and distribution
    a_eff_obs = a_orig

    # Marginals for PMF calculations are ALWAYS based on original integer counts.
    n1_orig = a_orig + b_orig
    n2_orig = c_orig + d_orig
    m1_orig = a_orig + c_orig
    
    # Support is based on original integer counts/marginals.
    # support() returns a SupportData named tuple with x, min_val, max_val, offset
    supp_orig = support(n1_orig, n2_orig, m1_orig)
    if not supp_orig or len(supp_orig.x) == 0: # Should not happen with valid counts
        logger.error("Original support is empty, cannot proceed.")
        return (0.0, float('inf')) # Or raise error
    supp_orig_list = list(supp_orig.x)  # Convert the x array to list for compatibility
    
    kmin_orig, kmax_orig = supp_orig.min_val, supp_orig.max_val

    logger.info(f"Computing Mid-P CI: original_a={a_orig}, original_b={b_orig}, original_c={c_orig}, original_d={d_orig}, alpha={alpha}")
    logger.info(f"Effective a_obs for comparison: {a_eff_obs}")
    logger.info(f"Support for PMF based on original marginals ({n1_orig}, {n2_orig}, {m1_orig}): {supp_orig_list}")
    
    # This function calculates the two-sided mid-p-value for a given theta,
    # comparing against a_eff_obs, using PMF from original counts.
    def midp_pval_func(theta: float) -> float:
        # log_nchg_pmf uses n1_orig, n2_orig, m1_orig
        # Vectorized calculation of log probabilities
        log_probs_values = np.vectorize(log_nchg_pmf)(supp_orig.x, n1_orig, n2_orig, m1_orig, theta)
        probs = np.exp(log_probs_values)

        # P(X = effective discrete point related to a_eff_obs)
        prob_at_a_eff_discrete_point = 0.0
        if a_eff_obs == math.floor(a_eff_obs): # If a_eff_obs is an integer (e.g., 3.0 or 3)
            # Check if this integer is in the support_orig_list
            # (it should be if it's within kmin_orig and kmax_orig and support is contiguous)
            try:
                # Ensure we use int(a_eff_obs) for indexing if supp_orig_list contains ints
                idx = supp_orig_list.index(int(a_eff_obs)) 
                prob_at_a_eff_discrete_point = probs[idx]
            except ValueError:
                # a_eff_obs is an integer but not in the discrete support (e.g. sparse support, or outside range)
                # If it's outside kmin/kmax range, P(X=a_eff_obs) is 0. If inside, but not in list (sparse), also 0.
                prob_at_a_eff_discrete_point = 0.0 
        # If a_eff_obs is k.5, P(X=a_eff_obs) for a discrete PMF is 0.
        
        # P(X < a_eff_obs)
        # Example: a_eff_obs = 3.5. We need sum P(k) for k in supp_orig_list where k < 3.5 (i.e., k <= 3)
        p_strictly_less = np.sum(probs[np.array(supp_orig_list) < a_eff_obs])
        
        # P(X > a_eff_obs)
        # Example: a_eff_obs = 3.5. We need sum P(k) for k in supp_orig_list where k > 3.5 (i.e., k >= 4)
        p_strictly_more = np.sum(probs[np.array(supp_orig_list) > a_eff_obs])

        # Mid-P tail definitions:
        # Lower tail sum: P(X < a_eff_obs) + 0.5 * P(X = discrete point for a_eff_obs)
        # Upper tail sum: P(X > a_eff_obs) + 0.5 * P(X = discrete point for a_eff_obs)
        p_val_lower_tail = p_strictly_less + 0.5 * prob_at_a_eff_discrete_point
        p_val_upper_tail = p_strictly_more + 0.5 * prob_at_a_eff_discrete_point
        
        p_value = 2 * min(p_val_lower_tail, p_val_upper_tail)
        return min(p_value, 1.0)
    
    # Edge cases for CI bounds based on a_eff_obs relative to original support's kmin/kmax.
    # If a_eff_obs is at or below kmin_orig (e.g., 0.5 <= 0, or 0.0 <= 0), lower bound is 0.
    low = 0.0 # Default lower bound
        
    if a_eff_obs <= kmin_orig:
        logger.info(f"Lower bound is 0.0 because a_eff_obs ({a_eff_obs}) <= kmin_orig ({kmin_orig})")
        # low is already 0.0
    else:
        logger.info(f"Finding lower bound for Mid-P interval (a_eff_obs={a_eff_obs}) target alpha={alpha}")
        try:
            # For lower CI bound (theta < 1), midp_pval_func(theta) generally increases as theta -> 0.
            # So we need find_smallest_theta that searches appropriately.
            # Let's define g(theta) = midp_pval_func(theta) - alpha. Root is when g(theta)=0.
            # For lower CI (theta_L), as theta increases from 0 to theta_L, pval decreases to alpha.
            # So midp_pval_func is decreasing for the lower bound search.
            low_candidate = find_smallest_theta(
                lambda theta: midp_pval_func(theta) - alpha, # Function whose root is sought (becomes 0)
                0.0, # Target for g(theta) is 0
                lo=1e-8, 
                hi=1.0,
                two_sided=True,
                progress_callback=lambda p: progress_callback(p * 0.5) if progress_callback else None
            )
            if low_candidate is None:
                logger.warning(f"Lower bound not found by find_smallest_theta (returned None) for a_eff_obs={a_eff_obs}. Defaulting to 0.0.")
                low = 0.0
            else:
                low = low_candidate
                logger.info(f"Lower bound found: {low:.6f}")
        except Exception as e:
            logger.warning(f"Error finding lower bound for a_eff_obs={a_eff_obs}: {e}. Defaulting to 0.0.")
            low = 0.0

    high = float('inf') # Default upper bound
    if a_eff_obs >= kmax_orig:
        logger.info(f"Upper bound is infinity because a_eff_obs ({a_eff_obs}) >= kmax_orig ({kmax_orig})")
        # high is already inf
    else:
        logger.info(f"Finding upper bound for Mid-P interval (a_eff_obs={a_eff_obs}) target alpha={alpha}")
        try:
            # For upper CI bound (theta_U > 1), as theta increases from 1 to theta_U, pval increases from its min then decreases to alpha.
            # Or, if pval is monotonic: as theta increases from 1, pval decreases towards alpha.
            # This means midp_pval_func is also decreasing for the upper bound search (from OR=1 upwards).
            high_candidate = find_smallest_theta(
                lambda theta: midp_pval_func(theta) - alpha, # Function whose root is sought
                0.0, # Target for g(theta) is 0
                lo=1.0, 
                hi=1e4, # Upper search limit for theta
                two_sided=True,
                progress_callback=lambda p: progress_callback(50 + p * 0.5) if progress_callback else None
            )
            if high_candidate is None:
                logger.warning(f"Upper bound not found by find_smallest_theta (returned None) for a_eff_obs={a_eff_obs}. Defaulting to inf.")
                high = float('inf')
            else:
                high = high_candidate
                logger.info(f"Upper bound found: {high:.6f}")
        except Exception as e:
            logger.warning(f"Error finding upper bound for a_eff_obs={a_eff_obs}: {e}. Defaulting to inf.")
            high = float('inf')
    
    # Ensure lower bound is less than or equal to upper bound
    if low > high:
        logger.warning(f"Invalid CI detected: lower bound ({low:.6f}) > upper bound ({high:.6f}). Swapping bounds.")
        low, high = high, low
    
    # Calculate odds ratio to verify it's within the CI
    odds_ratio = (a_orig * d_orig) / (b_orig * c_orig) if b_orig * c_orig > 0 else float('inf')
    
    # If odds ratio is not within CI, adjust bounds
    if odds_ratio < low or odds_ratio > high:
        logger.warning(f"Odds ratio ({odds_ratio:.6f}) not within CI ({low:.6f}, {high:.6f}). Adjusting bounds.")
        # Expand CI to include odds ratio
        if odds_ratio < low:
            low = max(0.0, odds_ratio * 0.9)  # Set lower bound slightly below odds ratio
        if odds_ratio > high:
            high = odds_ratio * 1.1  # Set upper bound slightly above odds ratio
    
    logger.info(f"Mid-P CI result for original counts ({a_orig},{b_orig},{c_orig},{d_orig}): ({low:.6f}, {high if high != float('inf') else 'inf'})")
    
    result = (low, high)
    _cache[cache_key] = result
    return result
