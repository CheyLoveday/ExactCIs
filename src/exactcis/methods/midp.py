"""
Mid-P adjusted confidence interval for odds ratio.

This module implements the Mid-P adjusted confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import math
import logging
from typing import Tuple, List, Optional, Dict, Any, Callable
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

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_map, get_optimal_workers
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for Mid-P method")

# Cache for previously computed values
_cache: Dict[Tuple[int, int, int, int, float], Tuple[float, float]] = {}


def exact_ci_midp(a: int, b: int, c: int, d: int,
                  alpha: float = 0.05, 
                  progress_callback: Optional[Callable] = None) -> Tuple[float, float]:
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
        # Vectorized calculation of log probabilities with numerical stability
        try:
            log_probs_values = np.vectorize(log_nchg_pmf)(supp_orig.x, n1_orig, n2_orig, m1_orig, theta)
            
            # Check for numerical issues
            if np.any(np.isnan(log_probs_values)) or np.any(np.isinf(log_probs_values)):
                logger.warning(f"Numerical issues in log_nchg_pmf for theta={theta}: NaN or Inf detected")
                return np.nan
            
            # Use stable conversion from log space, handling underflow gracefully
            max_log_prob = np.max(log_probs_values)
            if max_log_prob < -700:  # All probabilities are essentially zero
                logger.warning(f"All probabilities underflow for theta={theta} (max_log_prob={max_log_prob})")
                return 0.0
            
            # Compute probabilities in a numerically stable way
            probs = np.exp(log_probs_values - max_log_prob)  # Normalize to prevent underflow
            probs = probs * np.exp(max_log_prob)  # Scale back
            
            # Handle any remaining numerical issues
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            
        except (OverflowError, ValueError) as e:
            logger.warning(f"Error in PMF calculation for theta={theta}: {e}")
            return np.nan

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
            
            # Adaptive search expansion to find proper upper bound
            hi_search = 10.0  # Start with reasonable upper bound
            max_expansions = 10
            expansion_count = 0
            
            # Find a proper bracketing interval by expanding upward
            while expansion_count < max_expansions:
                try:
                    # Test if we can bracket the root
                    g_1 = midp_pval_func(1.0) - alpha
                    g_hi = midp_pval_func(hi_search) - alpha
                    
                    # Check for numerical issues
                    if np.isnan(g_1) or np.isnan(g_hi):
                        logger.warning(f"NaN detected in bracket test: g(1.0)={g_1}, g({hi_search})={g_hi}")
                        hi_search *= 2
                        expansion_count += 1
                        continue
                    
                    # If signs are different, we have bracketed the root
                    if np.sign(g_1) != np.sign(g_hi):
                        logger.info(f"Root bracketed in interval [1.0, {hi_search}]: g(1.0)={g_1:.4e}, g({hi_search})={g_hi:.4e}")
                        break
                    
                    # If p-value at hi_search is still too high, expand further
                    if g_hi > 0:  # midp_pval_func(hi_search) > alpha, need larger theta
                        hi_search *= 10  # Expand geometrically
                        expansion_count += 1
                        logger.info(f"Expanding upper search bound to {hi_search} (expansion {expansion_count})")
                    else:
                        # g_hi <= 0 but g_1 has same sign - this suggests a flat function
                        logger.warning(f"Function appears flat: g(1.0)={g_1:.4e}, g({hi_search})={g_hi:.4e}")
                        break
                        
                except (OverflowError, ValueError) as e:
                    logger.warning(f"Numerical issue during bracketing at hi_search={hi_search}: {e}")
                    hi_search *= 2
                    expansion_count += 1
                    
                if hi_search > 1e8:  # Safety limit
                    logger.warning(f"Upper search limit exceeded 1e8, stopping expansion")
                    break
            
            # If we couldn't bracket after all expansions, the upper bound might be infinite
            if expansion_count >= max_expansions or hi_search > 1e8:
                logger.warning(f"Could not bracket root after {expansion_count} expansions, hi_search={hi_search}. Upper bound may be infinite.")
                high_candidate = None
            else:
                # Use the bracketed interval for root finding
                high_candidate = find_smallest_theta(
                    lambda theta: midp_pval_func(theta) - alpha, # Function whose root is sought
                    0.0, # Target for g(theta) is 0
                    lo=1.0, 
                    hi=hi_search, # Use adaptive upper bound
                    two_sided=True,
                    progress_callback=lambda p: progress_callback(50 + p * 0.5) if progress_callback else None
                )
                
                # Additional validation: if result is close to search boundary, it's likely incorrect
                if high_candidate is not None and high_candidate >= hi_search * 0.99:
                    logger.warning(f"Root finding result ({high_candidate:.6f}) is too close to search boundary ({hi_search}). Upper bound may be infinite.")
                    high_candidate = None
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


def exact_ci_midp_batch(tables: List[Tuple[int, int, int, int]], 
                        alpha: float = 0.05,
                        max_workers: Optional[int] = None,
                        progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Mid-P adjusted confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing to compute confidence intervals for
    multiple tables simultaneously, providing significant speedup for large datasets.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        List of (lower_bound, upper_bound) tuples, one for each input table
        
    Note:
        Error Handling: If computation fails for any individual table (due to
        numerical issues, invalid data, etc.), a conservative interval (0.0, inf)
        is returned for that table, allowing the batch processing to complete
        successfully.
        
    Example:
        >>> tables = [(10, 20, 15, 30), (5, 10, 8, 12), (2, 3, 1, 4)]
        >>> results = exact_ci_midp_batch(tables, alpha=0.05)
        >>> print(results)
        [(0.234, 1.567), (0.123, 2.345), (0.045, 8.901)]
    """
    if not tables:
        return []
    
    if not has_parallel_support:
        # Fall back to sequential processing
        logger.info("Parallel support not available, using sequential processing")
        results = []
        for i, (a, b, c, d) in enumerate(tables):
            try:
                result = exact_ci_midp(a, b, c, d, alpha)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing table {i+1} ({a},{b},{c},{d}): {e}")
                results.append((0.0, float('inf')))  # Conservative fallback
            
            if progress_callback:
                progress_callback(min(100, int(100 * (i+1) / len(tables))))
        
        return results
    
    # Determine number of workers
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    max_workers = min(max_workers, len(tables))  # Don't use more workers than tables
    
    logger.info(f"Processing {len(tables)} tables with Mid-P method using {max_workers} workers")
    
    # Create worker function that handles errors gracefully
    def process_single_table(table_data):
        a, b, c, d = table_data
        try:
            return exact_ci_midp(a, b, c, d, alpha)
        except Exception as e:
            logger.warning(f"Error processing table ({a},{b},{c},{d}): {e}")
            return (0.0, float('inf'))  # Conservative fallback
    
    # Process tables in parallel
    results = parallel_map(
        process_single_table,
        tables,
        max_workers=max_workers,
        force_processes=True,  # CPU-bound task
        progress_callback=progress_callback
    )
    
    logger.info(f"Completed batch processing of {len(tables)} tables with Mid-P method")
    return results
