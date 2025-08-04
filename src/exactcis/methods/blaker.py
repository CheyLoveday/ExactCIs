"""
Blaker's exact confidence interval for odds ratio.

This module implements Blaker's exact confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import logging
import numpy as np
from scipy.stats import nchypergeom_fisher
from scipy.optimize import brentq
from functools import lru_cache

from exactcis.core import (
    validate_counts,
    support, # This now returns SupportData
    log_nchg_pmf, # Keep existing imports as needed
    find_smallest_theta,
    nchg_pdf, 
    estimate_point_or,
    SupportData # Changed from Support to SupportData
)

# Logger setup
logger = logging.getLogger(__name__)

# Import JIT-compiled functions if available
try:
    from exactcis.utils.jit_functions import (
        nchg_pdf_cached,
        sum_probs_less_than_threshold,
        HAS_NUMBA
    )
    logger.info("Using JIT-compiled functions for Blaker method")
except ImportError:
    HAS_NUMBA = False
    logger.warning("JIT-compiled functions not available, using standard Python")

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_compute_ci, get_optimal_workers
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for Blaker method")


# No need for a custom cache class - using lru_cache from functools instead

def blaker_acceptability(n1: int, n2: int, m1: int, theta: float, support_x: np.ndarray) -> np.ndarray:
    """
    Calculate PMF values for Blaker's method with optimized caching to avoid redundant calculations.
    
    This function uses both process-local lru_cache and inter-process shared cache to dramatically 
    improve performance, as the same PMF calculations are repeated many times during root-finding.
    """
    # DEBUG LOGGING FOR SPECIFIC CASE
    is_debug_case = (n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6)
    if is_debug_case and logger.isEnabledFor(logging.INFO):
        logger.info(f"[DEBUG_ACCEPTABILITY] n1=5,n2=10,m1=7,theta={theta:.2e}, support_x={support_x}")
    
    # Round theta for cache stability
    theta_rounded = round(theta, 12)  # Avoid floating point precision issues
    
    # Convert support_x to tuple for caching
    support_tuple = tuple(support_x)
    
    # Try to get PMF values from shared cache first (for batch mode)
    try:
        from ..utils.shared_cache import get_shared_cache
        shared_cache = get_shared_cache()
        cached_probs = shared_cache.get_pmf(n1, n2, m1, theta_rounded, support_tuple)
        if cached_probs is not None:
            if is_debug_case and logger.isEnabledFor(logging.INFO):
                logger.info(f"[DEBUG_ACCEPTABILITY] Using PMF values from shared cache")
            return cached_probs
    except (ImportError, AttributeError):
        # Shared cache not available or not initialized
        pass
    
    # If not in shared cache, use process-local cache or calculate
    if HAS_NUMBA:
        probs = np.array(nchg_pdf_cached(n1, n2, m1, theta_rounded, support_tuple))
    else:
        # Fallback to standard implementation
        probs = nchg_pdf(support_x, n1, n2, m1, theta)
    
    # Store in shared cache for other processes to use (for batch mode)
    try:
        shared_cache.set_pmf(n1, n2, m1, theta_rounded, support_tuple, probs)
        if is_debug_case and logger.isEnabledFor(logging.INFO):
            logger.info(f"[DEBUG_ACCEPTABILITY] Stored PMF values in shared cache")
    except (NameError, AttributeError):
        # Shared cache not available or not initialized
        pass

    if is_debug_case and logger.isEnabledFor(logging.INFO):
        logger.info(f"[DEBUG_ACCEPTABILITY] Probs from nchg_pdf: {probs}")
    
    return probs


def blaker_p_value(a: int, n1: int, n2: int, m1: int, theta: float, s: SupportData, 
                support_tuple: Optional[tuple] = None, skip_validation: bool = False) -> float:
    """
    Calculates Blaker's p-value for a given theta, for observed count 'a'.
    s: SupportData object containing s.x (support array), s.min_val, s.max_val, and s.offset.
    support_tuple: Optional pre-converted tuple of s.x for improved performance.
    
    This optimized version uses JIT compilation for better performance.
    """
    # 1. Calculate acceptability P(X=k|theta) for all k in the support s.x
    if support_tuple is None:
        # Convert support to tuple if not provided
        support_tuple = tuple(s.x)
    
    # Use the support_tuple instead of s.x directly
    accept_probs_all_k = blaker_acceptability(n1, n2, m1, theta, s.x if support_tuple is None else np.array(support_tuple))
    
    # DEBUG LOGGING FOR SPECIFIC CASE
    is_debug_case = (a == 5 and n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6)
    if is_debug_case:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[DEBUG_PVAL] For a=5,n1=5,n2=10,m1=7,theta={theta:.2e}")
            logger.info(f"[DEBUG_PVAL] support s.x={s.x}, s.min_val={s.min_val}, s.offset={s.offset}")
            logger.info(f"[DEBUG_PVAL] accept_probs_all_k = {accept_probs_all_k}")

    # 2. Get the acceptability probability for the observed 'a'
    idx_a = s.offset + a
    
    # This check is a safeguard when validation is not skipped
    if not skip_validation and not (0 <= idx_a < len(accept_probs_all_k)):
        logger.error(f"Blaker p-value: Calculated index for 'a' ({idx_a}) is out of bounds. This should have been caught by earlier validation. a={a}, s.min_val={s.min_val}, s.max_val={s.max_val}")
        # Return a p-value that will not incorrectly satisfy the root-finder
        return 1.0

    current_accept_prob_at_a = accept_probs_all_k[idx_a]
    
    if is_debug_case:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[DEBUG_PVAL] idx_a = {idx_a}, current_accept_prob_at_a = {current_accept_prob_at_a}")

    # 3. Sum probabilities for k where P(k|theta) <= P(a|theta) * (1 + epsilon)
    # Epsilon is a small tolerance factor for floating point comparisons, as per Blaker (2000) and R's exact2x2.
    epsilon = 1e-7 
    
    # Use JIT-compiled function for summation if available
    if HAS_NUMBA:
        p_val = sum_probs_less_than_threshold(accept_probs_all_k, idx_a, epsilon)
    else:
        # Fallback to vectorized NumPy operations
        mask = accept_probs_all_k <= current_accept_prob_at_a * (1 + epsilon)
        p_val = np.sum(accept_probs_all_k[mask])
    
    if is_debug_case:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[DEBUG_PVAL] Final p_val = {p_val}")

    return p_val




def _find_better_bracket(p_value_func, target_alpha: float, lo: float, hi: float, increasing: bool = True) -> Tuple[float, float]:
    """
    Use 3-point evaluation to find better brackets for root finding.
    
    Args:
        p_value_func: Function that returns p-value given theta
        target_alpha: Target p-value we're searching for
        lo: Lower bound of search range
        hi: Upper bound of search range  
        increasing: Whether p-value function is increasing with theta
        
    Returns:
        Tuple of (improved_lo, improved_hi) bounds
    """
    try:
        # Evaluate at 3 strategic points
        if increasing:
            # For increasing functions, check points closer to where we expect the root
            mid = np.sqrt(lo * hi)  # Geometric mean for log-scale spacing
            test_points = [lo + 0.1 * (mid - lo), mid, hi - 0.1 * (hi - mid)]
        else:
            # For decreasing functions, similar but different strategy
            mid = np.sqrt(lo * hi)
            test_points = [lo + 0.1 * (mid - lo), mid, hi - 0.1 * (hi - mid)]
        
        # Evaluate p-values at test points
        p_values = []
        for theta in test_points:
            try:
                p_val = p_value_func(theta)
                p_values.append(p_val)
            except:
                p_values.append(None)
        
        # Find the best bracketing based on which points are above/below target
        valid_points = [(theta, p_val) for theta, p_val in zip(test_points, p_values) if p_val is not None]
        
        if len(valid_points) >= 2:
            # Find points that bracket the target
            if increasing:
                below_target = [(theta, p_val) for theta, p_val in valid_points if p_val < target_alpha]
                above_target = [(theta, p_val) for theta, p_val in valid_points if p_val > target_alpha]
                
                if below_target and above_target:
                    # Use the best bracketing points
                    new_lo = max(theta for theta, _ in below_target)
                    new_hi = min(theta for theta, _ in above_target)
                    return max(lo, new_lo * 0.9), min(hi, new_hi * 1.1)  # Small safety margin
            else:
                above_target = [(theta, p_val) for theta, p_val in valid_points if p_val > target_alpha]
                below_target = [(theta, p_val) for theta, p_val in valid_points if p_val < target_alpha]
                
                if above_target and below_target:
                    # For decreasing function, logic is reversed
                    new_lo = max(theta for theta, _ in below_target)
                    new_hi = min(theta for theta, _ in above_target)
                    return max(lo, new_lo * 0.9), min(hi, new_hi * 1.1)
                    
    except Exception:
        # If pre-bracketing fails, return original bounds
        pass
    
    return lo, hi


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
        ValueError: If inputs are invalid (negative counts, empty margins, invalid alpha, or 'a' outside support).
        RuntimeError: If the root-finding algorithm fails to converge.
    """
    # No need to clear cache - using lru_cache which handles this automatically
    
    validate_counts(a, b, c, d)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    
    n1, n2 = a + b, c + d
    m1, _ = a + c, b + d # m2 is not directly needed for nchg_pdf with m1

    # Add point estimate for logging, using Haldane correction
    or_point_est = estimate_point_or(a,b,c,d, correction_type='haldane')
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Blaker exact_ci_blaker: Input ({a},{b},{c},{d}), alpha={alpha}. Point OR estimate (Haldane corrected): {or_point_est:.4f}")

    s = support(n1, n2, m1) # SupportData object
    kmin, kmax = s.min_val, s.max_val

    # Guard against impossible 'a' for the given marginals
    if not (kmin <= a <= kmax):
        raise ValueError(f"Count 'a' ({a}) is outside the valid support range [{kmin}, {kmax}] for the given marginals.")
    
    # Pre-validate 'a' range for p-value calculations to skip repeated boundary checks
    idx_a = s.offset + a
    if not (0 <= idx_a < len(s.x)):
        raise ValueError(f"Calculated index for 'a' ({idx_a}) is out of support bounds. a={a}, s.min_val={s.min_val}, s.max_val={s.max_val}")

    # Pre-convert support data to tuple to avoid redundant conversions
    support_tuple = tuple(s.x)

    # Initialize raw_theta_low and raw_theta_high for cases where search might fail
    raw_theta_low = 0.0 
    raw_theta_high = float('inf')
    
    # Determine search range for lower bound based on OR estimate
    lo_search_lower = 1e-9
    hi_search_lower = or_point_est if (np.isfinite(or_point_est) and or_point_est > lo_search_lower) else 1e7
    # Ensure hi > lo for the search
    if hi_search_lower <= lo_search_lower: hi_search_lower = 1e7

    # Lower bound calculation (skip validation since we pre-validated above)
    blaker_p_value_lower = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s, support_tuple, skip_validation=True)
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Blaker exact_ci_blaker: Finding lower bound. Target p-value for search: {alpha/2.0:.4f}")
    
    if a > kmin:
        # Use 3-point pre-bracketing to improve root finding efficiency
        improved_lo, improved_hi = _find_better_bracket(
            blaker_p_value_lower, alpha / 2.0, lo_search_lower, hi_search_lower, increasing=False
        )
        
        candidate_raw_theta_low = find_smallest_theta(
            func=blaker_p_value_lower, 
            alpha=alpha / 2.0,  # Divide alpha by 2 as in original implementation
            lo=improved_lo, 
            hi=improved_hi, 
            increasing=False
        )
        if candidate_raw_theta_low is not None:
            raw_theta_low = candidate_raw_theta_low
        else:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Blaker exact_ci_blaker: Lower bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_low to 0.0.")
            raw_theta_low = 0.0
    else: # a == kmin, lower bound is 0
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmin ({kmin}). Lower bound is 0.")
        raw_theta_low = 0.0

    # Upper bound calculation (skip validation since we pre-validated above)
    blaker_p_value_upper = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s, support_tuple, skip_validation=True)
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Blaker exact_ci_blaker: Finding upper bound. Target p-value for search: {alpha/2.0:.4f}")

    # Determine search range for upper bound based on OR estimate
    hi_search_upper = 1e7
    lo_search_upper = or_point_est if (np.isfinite(or_point_est) and or_point_est > 1e-9) else 1e-9
    # Ensure lo < hi for the search
    if lo_search_upper >= hi_search_upper: lo_search_upper = 1e-9

    if a < kmax:
        # Use 3-point pre-bracketing to improve root finding efficiency
        improved_lo, improved_hi = _find_better_bracket(
            blaker_p_value_upper, alpha / 2.0, lo_search_upper, hi_search_upper, increasing=True
        )
        
        candidate_raw_theta_high = find_smallest_theta(
            func=blaker_p_value_upper, 
            alpha=alpha / 2.0,  # Divide alpha by 2 as in original implementation
            lo=improved_lo, 
            hi=improved_hi, 
            increasing=True
        )
        if candidate_raw_theta_high is not None:
            raw_theta_high = candidate_raw_theta_high
        else:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Blaker exact_ci_blaker: Upper bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_high to inf.")
            raw_theta_high = float('inf')
    else: # a == kmax, upper bound is infinity
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmax ({kmax}). Upper bound is infinity.")
        raw_theta_high = float('inf')

    theta_low = float(raw_theta_low)
    theta_high = float(raw_theta_high)

    if not (np.isfinite(theta_low) and np.isfinite(theta_high)):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f"Blaker CI calculation resulted in non-finite bounds for ({a},{b},{c},{d}): low={theta_low}, high={theta_high}. This may be expected if a is at support boundary.")

    # Check for crossed bounds AFTER ensuring they are finite numbers
    tolerance = 1e-6
    if np.isfinite(theta_low) and np.isfinite(theta_high) and theta_low > theta_high + tolerance:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f"Blaker CI: Lower bound {theta_low:.4f} > Upper bound {theta_high:.4f} for ({a},{b},{c},{d}). This can indicate issues with root finding or p-value function. Returning (0, inf) as a safe default. Original raw values: low={raw_theta_low}, high={raw_theta_high}")
        return 0.0, float('inf')
    
    return theta_low, theta_high


def exact_ci_blaker_batch(tables: List[Tuple[int, int, int, int]], 
                          alpha: float = 0.05,
                          max_workers: Optional[int] = None,
                          backend: Optional[str] = None,
                          progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Blaker's exact confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing to compute confidence intervals for
    multiple tables simultaneously, providing significant speedup for large datasets.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        List of (lower_bound, upper_bound) tuples, one for each input table
        
    Note:
        Error Handling: If computation fails for any individual table (due to
        numerical issues, invalid data, etc.), a conservative interval (0.0, inf)
        is returned for that table, allowing the batch processing to complete
        successfully.
        
        Backend Selection: For methods that use Numba-accelerated functions,
        the 'thread' backend may be more efficient. If not specified, the backend
        is auto-detected based on the method.
        
    Example:
        >>> tables = [(10, 20, 15, 30), (5, 10, 8, 12), (2, 3, 1, 4)]
        >>> results = exact_ci_blaker_batch(tables, alpha=0.05)
        >>> print(results)
        [(0.234, 1.567), (0.123, 2.345), (0.045, 8.901)]
    """
    if not tables:
        return []
    
    if not has_parallel_support:
        # Fall back to sequential processing
        if logger.isEnabledFor(logging.INFO):
            logger.info("Parallel support not available, using sequential processing")
        results = []
        for i, (a, b, c, d) in enumerate(tables):
            try:
                result = exact_ci_blaker(a, b, c, d, alpha)
                results.append(result)
            except Exception as e:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Error processing table {i+1} ({a},{b},{c},{d}): {e}")
                results.append((0.0, float('inf')))  # Conservative fallback
            
            if progress_callback:
                progress_callback(min(100, int(100 * (i+1) / len(tables))))
        
        return results
    
    # Determine number of workers
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    max_workers = min(max_workers, len(tables))  # Don't use more workers than tables
    
    # Initialize shared cache for parallel processing  
    from exactcis.utils.shared_cache import init_shared_cache_for_parallel
    cache = init_shared_cache_for_parallel()
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Processing {len(tables)} tables with Blaker method using {max_workers} workers")
    
    # Use the improved parallel processing with shared cache
    results = parallel_compute_ci(
        exact_ci_blaker,
        tables,
        alpha=alpha,
        timeout=None,  # No timeout for batch processing
        backend=backend,
        max_workers=max_workers
    )
    
    # Report final statistics
    try:
        from exactcis.utils.shared_cache import get_shared_cache
        cache = get_shared_cache()
        stats = cache.get_stats()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Cache statistics: {stats['hit_rate_percent']:.1f}% hit rate ({stats['hits']}/{stats['total_lookups']} lookups)")
    except Exception:
        pass  # Don't fail if cache stats unavailable
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Completed batch processing of {len(tables)} tables with Blaker method")
    return results
