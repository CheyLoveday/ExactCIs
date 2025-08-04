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
import hashlib

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

logger = logging.getLogger(__name__)

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_map, get_optimal_workers
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for Blaker method")


class BlakerPMFCache:
    """High-performance cache for Blaker PMF calculations with LRU eviction."""
    
    def __init__(self, max_size: int = 1024):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
        
    def _generate_cache_key(self, n1: int, n2: int, m1: int, 
                          theta: float, support_hash: str) -> str:
        """Generate stable cache key for PMF parameters."""
        theta_rounded = round(theta, 12)  # Avoid floating point precision issues
        return f"{n1}_{n2}_{m1}_{theta_rounded}_{support_hash}"
    
    def get_pmf_values(self, n1: int, n2: int, m1: int, 
                      theta: float, support_x: np.ndarray) -> np.ndarray:
        """Get cached PMF values or calculate if not cached."""
        support_hash = hashlib.md5(support_x.tobytes()).hexdigest()[:8]
        cache_key = self._generate_cache_key(n1, n2, m1, theta, support_hash)
        
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        
        # Calculate PMF values using standard implementation
        pmf_values = nchg_pdf(support_x, n1, n2, m1, theta)
        
        # Store in cache with LRU eviction if needed
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = pmf_values
        self.access_count[cache_key] = 1
        
        return pmf_values
    
    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self.access_count:
            return
            
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.access_count.clear()

# Global cache instance
_blaker_cache = BlakerPMFCache(max_size=2048)  # Increased cache size for better performance

def _clear_blaker_cache():
    """Clear the PMF cache. Should be called at the start of each CI calculation."""
    global _blaker_cache
    _blaker_cache.clear()

def blaker_acceptability(n1: int, n2: int, m1: int, theta: float, support_x: np.ndarray) -> np.ndarray:
    """
    Calculate PMF values for Blaker's method with optimized caching to avoid redundant calculations.
    
    This function uses a high-performance cache with LRU eviction to dramatically improve performance,
    as the same PMF calculations are repeated many times during root-finding.
    """
    # DEBUG LOGGING FOR SPECIFIC CASE
    if n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6:
        logger.info(f"[DEBUG_ACCEPTABILITY] n1=5,n2=10,m1=7,theta={theta:.2e}, support_x={support_x}")
    
    # Get PMF values from cache or calculate
    probs = _blaker_cache.get_pmf_values(n1, n2, m1, theta, support_x)

    if n1 == 5 and n2 == 10 and m1 == 7 and theta > 1e6:
        logger.info(f"[DEBUG_ACCEPTABILITY] Probs from nchg_pdf: {probs}")
    
    return probs


def blaker_p_value(a: int, n1: int, n2: int, m1: int, theta: float, s: SupportData) -> float:
    """
    Calculates Blaker's p-value for a given theta, for observed count 'a'.
    s: SupportData object containing s.x (support array), s.min_val, s.max_val, and s.offset.
    
    This optimized version uses vectorized operations for better performance.
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
    idx_a = s.offset + a
    
    # This check is a safeguard. The primary validation should happen in the calling function.
    if not (0 <= idx_a < len(accept_probs_all_k)):
        logger.error(f"Blaker p-value: Calculated index for 'a' ({idx_a}) is out of bounds. This should have been caught by earlier validation. a={a}, s.min_val={s.min_val}, s.max_val={s.max_val}")
        # Return a p-value that will not incorrectly satisfy the root-finder
        return 1.0

    current_accept_prob_at_a = accept_probs_all_k[idx_a]
    
    if is_debug_case:
        logger.info(f"[DEBUG_PVAL] idx_a = {idx_a}, current_accept_prob_at_a = {current_accept_prob_at_a}")

    # 3. Sum probabilities for k where P(k|theta) <= P(a|theta) * (1 + epsilon)
    # Epsilon is a small tolerance factor for floating point comparisons, as per Blaker (2000) and R's exact2x2.
    epsilon = 1e-7 
    
    # Vectorized comparison - much faster than loop
    mask = accept_probs_all_k <= current_accept_prob_at_a * (1 + epsilon)
    p_val = np.sum(accept_probs_all_k[mask])
    
    if is_debug_case:
        logger.info(f"[DEBUG_PVAL] p_val_terms = {accept_probs_all_k[mask]}")
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
        ValueError: If inputs are invalid (negative counts, empty margins, invalid alpha, or 'a' outside support).
        RuntimeError: If the root-finding algorithm fails to converge.
    """
    # Clear cache at the beginning of each CI calculation
    _clear_blaker_cache()
    
    try:
        validate_counts(a, b, c, d)
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        
        n1, n2 = a + b, c + d
        m1, _ = a + c, b + d # m2 is not directly needed for nchg_pdf with m1

        # Add point estimate for logging, using Haldane correction
        or_point_est = estimate_point_or(a,b,c,d, correction_type='haldane')
        logger.info(f"Blaker exact_ci_blaker: Input ({a},{b},{c},{d}), alpha={alpha}. Point OR estimate (Haldane corrected): {or_point_est:.4f}")

        s = support(n1, n2, m1) # SupportData object
        kmin, kmax = s.min_val, s.max_val

        # Guard against impossible 'a' for the given marginals
        if not (kmin <= a <= kmax):
            raise ValueError(f"Count 'a' ({a}) is outside the valid support range [{kmin}, {kmax}] for the given marginals.")

        # Initialize raw_theta_low and raw_theta_high for cases where search might fail
        raw_theta_low = 0.0 
        raw_theta_high = float('inf')
        
        # Determine search range for lower bound based on OR estimate
        lo_search_lower = 1e-9
        hi_search_lower = or_point_est if (np.isfinite(or_point_est) and or_point_est > lo_search_lower) else 1e7
        # Ensure hi > lo for the search
        if hi_search_lower <= lo_search_lower: hi_search_lower = 1e7

        # Lower bound calculation
        blaker_p_value_lower = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
        logger.info(f"Blaker exact_ci_blaker: Finding lower bound. Target p-value for search: {alpha/2.0:.4f}")
        
        if a > kmin:
            candidate_raw_theta_low = find_smallest_theta(
                func=blaker_p_value_lower, 
                alpha=alpha / 2.0,  # Divide alpha by 2 as in original implementation
                lo=lo_search_lower, 
                hi=hi_search_lower, 
                increasing=False
            )
            if candidate_raw_theta_low is not None:
                raw_theta_low = candidate_raw_theta_low
            else:
                logger.warning(f"Blaker exact_ci_blaker: Lower bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_low to 0.0.")
                raw_theta_low = 0.0
        else: # a == kmin, lower bound is 0
            logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmin ({kmin}). Lower bound is 0.")
            raw_theta_low = 0.0

        # Upper bound calculation
        blaker_p_value_upper = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
        logger.info(f"Blaker exact_ci_blaker: Finding upper bound. Target p-value for search: {alpha/2.0:.4f}")

        # Determine search range for upper bound based on OR estimate
        hi_search_upper = 1e7
        lo_search_upper = or_point_est if (np.isfinite(or_point_est) and or_point_est > 1e-9) else 1e-9
        # Ensure lo < hi for the search
        if lo_search_upper >= hi_search_upper: lo_search_upper = 1e-9

        if a < kmax:
            candidate_raw_theta_high = find_smallest_theta(
                func=blaker_p_value_upper, 
                alpha=alpha / 2.0,  # Divide alpha by 2 as in original implementation
                lo=lo_search_upper, 
                hi=hi_search_upper, 
                increasing=True
            )
            if candidate_raw_theta_high is not None:
                raw_theta_high = candidate_raw_theta_high
            else:
                logger.warning(f"Blaker exact_ci_blaker: Upper bound search failed for ({a},{b},{c},{d}). Defaulting raw_theta_high to inf.")
                raw_theta_high = float('inf')
        else: # a == kmax, upper bound is infinity
            logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmax ({kmax}). Upper bound is infinity.")
            raw_theta_high = float('inf')

        theta_low = float(raw_theta_low)
        theta_high = float(raw_theta_high)

        if not (np.isfinite(theta_low) and np.isfinite(theta_high)):
            logger.warning(f"Blaker CI calculation resulted in non-finite bounds for ({a},{b},{c},{d}): low={theta_low}, high={theta_high}. This may be expected if a is at support boundary.")

        # Check for crossed bounds AFTER ensuring they are finite numbers
        tolerance = 1e-6
        if np.isfinite(theta_low) and np.isfinite(theta_high) and theta_low > theta_high + tolerance:
            logger.warning(f"Blaker CI: Lower bound {theta_low:.4f} > Upper bound {theta_high:.4f} for ({a},{b},{c},{d}). This can indicate issues with root finding or p-value function. Returning (0, inf) as a safe default. Original raw values: low={raw_theta_low}, high={raw_theta_high}")
            return 0.0, float('inf')
        
        return theta_low, theta_high
    finally:
        # Clear cache at the end of each CI calculation to free memory
        _clear_blaker_cache()


def exact_ci_blaker_batch(tables: List[Tuple[int, int, int, int]], 
                          alpha: float = 0.05,
                          max_workers: Optional[int] = None,
                          progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Blaker's exact confidence intervals for multiple 2x2 tables in parallel.
    
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
        >>> results = exact_ci_blaker_batch(tables, alpha=0.05)
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
                result = exact_ci_blaker(a, b, c, d, alpha)
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
    
    logger.info(f"Processing {len(tables)} tables with Blaker method using {max_workers} workers")
    
    # Create worker function that handles errors gracefully
    def process_single_table(table_data):
        a, b, c, d = table_data
        try:
            return exact_ci_blaker(a, b, c, d, alpha)
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
    
    logger.info(f"Completed batch processing of {len(tables)} tables with Blaker method")
    return results
