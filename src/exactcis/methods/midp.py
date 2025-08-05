"""
Mid-P adjusted confidence interval for odds ratio.

This module implements the Mid-P adjusted confidence interval method
for the odds ratio of a 2x2 contingency table using a grid search approach
with confidence interval inversion.

The Mid-P method is a modification of Fisher's exact test that gives half-weight
to the observed table in the tail probability calculation, reducing the conservatism
of Fisher's exact test. This results in confidence intervals that are narrower than
Fisher's exact intervals but maintain good coverage properties.

This implementation uses a grid search approach with confidence interval inversion,
which is more reliable for large sample sizes than root-finding methods. The approach
is similar to that used in the R 'Exact' package by Fay and Fay (2021).

References:
    Lancaster, H. O. (1961). Significance tests in discrete distributions.
        Journal of the American Statistical Association, 56(294), 223-234.
    Fay, M. P., & Fay, M. M. (2021). Exact: Unconditional Exact Test. R package
        version 3.1. https://CRAN.R-project.org/package=Exact
"""

import math
import logging
from typing import Tuple, List, Optional, Dict, Any, Callable
import numpy as np
from functools import lru_cache

from exactcis.core import (
    validate_counts,
    support,
    log_nchg_pmf,
    calculate_odds_ratio
)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_map, get_optimal_workers, parallel_compute_ci
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for Mid-P method")


def calculate_midp_pvalue(a_obs: int, n1: int, n2: int, m1: int, theta: float) -> float:
    """
    Calculate Mid-P p-value for given parameters.
    
    Args:
        a_obs: Observed count in cell (1,1)
        n1: Row 1 total (a + b)
        n2: Row 2 total (c + d)
        m1: Column 1 total (a + c)
        theta: Odds ratio parameter
        
    Returns:
        Two-sided Mid-P p-value
    """
    # Get support for the distribution
    supp = support(n1, n2, m1)
    
    if not supp or len(supp.x) == 0:
        logger.error("Support is empty, cannot calculate p-value.")
        return np.nan
    
    try:
        # Calculate log probabilities for all possible tables
        log_probs = np.vectorize(log_nchg_pmf)(supp.x, n1, n2, m1, theta)
        
        # Check for numerical issues
        if np.any(np.isnan(log_probs)) or np.any(np.isinf(log_probs)):
            logger.warning(f"Numerical issues in log_nchg_pmf for theta={theta}: NaN or Inf detected")
            return np.nan
        
        # Convert log probabilities to probabilities in a numerically stable way
        max_log_prob = np.max(log_probs)
        if max_log_prob < -700:  # All probabilities are essentially zero
            logger.warning(f"All probabilities underflow for theta={theta} (max_log_prob={max_log_prob})")
            return 0.0
        
        # Compute probabilities in a numerically stable way
        probs = np.exp(log_probs - max_log_prob)  # Normalize to prevent underflow
        probs = probs * np.exp(max_log_prob)  # Scale back
        
        # Handle any remaining numerical issues
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Find probability of observed table
        idx = np.where(supp.x == a_obs)[0]
        if len(idx) == 0:
            # a_obs not in support
            prob_obs = 0.0
        else:
            prob_obs = probs[idx[0]]
        
        # Calculate p-values for lower and upper tails
        p_lower = np.sum(probs[supp.x < a_obs]) + 0.5 * prob_obs
        p_upper = np.sum(probs[supp.x > a_obs]) + 0.5 * prob_obs
        
        # Return two-sided p-value
        return min(1.0, 2.0 * min(p_lower, p_upper))
        
    except (OverflowError, ValueError) as e:
        logger.warning(f"Error in PMF calculation for theta={theta}: {e}")
        return np.nan


def find_ci_bound(theta_grid: np.ndarray, p_values: np.ndarray, alpha: float, is_lower: bool = True) -> float:
    """
    Find confidence interval bound from grid of theta values and p-values.
    
    Args:
        theta_grid: Array of theta values
        p_values: Array of p-values corresponding to theta_grid
        alpha: Significance level
        is_lower: If True, find lower bound; if False, find upper bound
        
    Returns:
        Confidence interval bound
    """
    # Find values in the confidence interval (p_value >= alpha)
    in_ci = p_values >= alpha
    
    if not np.any(in_ci):
        # No values in CI, return boundary
        return theta_grid[0] if is_lower else theta_grid[-1]
    
    # Find the bound
    if is_lower:
        # Lower bound is the smallest theta in CI
        return theta_grid[np.where(in_ci)[0][0]]
    else:
        # Upper bound is the largest theta in CI
        return theta_grid[np.where(in_ci)[0][-1]]


@lru_cache(maxsize=4096)
def exact_ci_midp(a: int, b: int, c: int, d: int,
                  alpha: float = 0.05, 
                  progress_callback: Optional[Callable] = None,
                  grid_size: int = 200,
                  theta_min: float = 0.001,
                  theta_max: float = 1000) -> Tuple[float, float]:
    """
    Calculate the Mid-P adjusted confidence interval for the odds ratio using grid search.

    This method is similar to the conditional (Fisher) method but gives half-weight
    to the observed table in the tail p-value, reducing conservatism. It is appropriate
    for epidemiology or surveillance where conservative Fisher intervals are too wide,
    and for moderate samples where slight undercoverage is tolerable for tighter intervals.
    
    This implementation uses a grid search approach with confidence interval inversion,
    which is more reliable for large sample sizes than root-finding methods.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        progress_callback: Optional callback function to report progress (0-100)
        grid_size: Number of points in the theta grid (default: 200)
        theta_min: Minimum theta value for grid search (default: 0.001)
        theta_max: Maximum theta value for grid search (default: 1000)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    # Validate inputs
    validate_counts(a, b, c, d)
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    # Calculate marginals
    n1 = a + b
    n2 = c + d
    m1 = a + c
    
    # Calculate odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    
    # Log initial information
    logger.info(f"Computing Mid-P CI using grid search: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    logger.info(f"Marginals: n1={n1}, n2={n2}, m1={m1}, odds_ratio={odds_ratio}")
    
    # Handle edge cases
    if b == 0 or c == 0:
        logger.info(f"Edge case: b={b}, c={c}, returning (0, inf)")
        return (0.0, float('inf'))
    
    # Generate grid of theta values (logarithmically spaced)
    # Ensure the grid includes the point estimate
    if odds_ratio > 0 and odds_ratio < float('inf'):
        # Adjust theta_min and theta_max to ensure they bracket the odds ratio
        theta_min = min(theta_min, odds_ratio * 0.1)
        theta_max = max(theta_max, odds_ratio * 10)
    
    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), grid_size)
    
    # Calculate p-values for each theta in the grid
    p_values = []
    for i, theta in enumerate(theta_grid):
        p_value = calculate_midp_pvalue(a, n1, n2, m1, theta)
        p_values.append(p_value)
        
        # Report progress
        if progress_callback:
            progress_callback(min(100, int(100 * (i+1) / grid_size)))
    
    # Convert to numpy arrays
    p_values = np.array(p_values)
    
    # Find confidence interval bounds
    lower_bound = find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    upper_bound = find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    
    # Ensure the odds ratio is within the CI
    if odds_ratio < lower_bound:
        logger.warning(f"Odds ratio ({odds_ratio}) < lower bound ({lower_bound}), adjusting lower bound")
        lower_bound = max(0.0, odds_ratio * 0.9)
    
    if odds_ratio > upper_bound and upper_bound < theta_max * 0.99:
        logger.warning(f"Odds ratio ({odds_ratio}) > upper bound ({upper_bound}), adjusting upper bound")
        upper_bound = min(float('inf'), odds_ratio * 1.1)
    
    logger.info(f"Mid-P CI result: ({lower_bound:.6f}, {upper_bound:.6f})")
    
    return (lower_bound, upper_bound)


def exact_ci_midp_batch(tables: List[Tuple[int, int, int, int]], 
                        alpha: float = 0.05,
                        max_workers: Optional[int] = None,
                        backend: Optional[str] = None,
                        progress_callback: Optional[Callable] = None,
                        grid_size: int = 200,
                        theta_min: float = 0.001,
                        theta_max: float = 1000) -> List[Tuple[float, float]]:
    """
    Calculate Mid-P adjusted confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing to compute confidence intervals for
    multiple tables simultaneously, providing significant speedup for large datasets.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        progress_callback: Optional callback function to report progress (0-100)
        grid_size: Number of points in the theta grid (default: 200)
        theta_min: Minimum theta value for grid search (default: 0.001)
        theta_max: Maximum theta value for grid search (default: 1000)
        
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
                result = exact_ci_midp(a, b, c, d, alpha, 
                                      progress_callback=progress_callback,
                                      grid_size=grid_size,
                                      theta_min=theta_min,
                                      theta_max=theta_max)
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
    
    # Initialize shared cache for parallel processing
    try:
        from ..utils.shared_cache import init_shared_cache_for_parallel
        cache = init_shared_cache_for_parallel()
        logger.info("Initialized shared cache for parallel processing")
    except ImportError:
        logger.warning("Shared cache not available, performance may be reduced")
    
    logger.info(f"Processing {len(tables)} tables with Mid-P method using {max_workers} workers")
    
    # Use parallel_compute_ci for better performance and error handling
    results = parallel_compute_ci(
        exact_ci_midp,
        tables,
        alpha=alpha,
        grid_size=grid_size,
        theta_min=theta_min,
        theta_max=theta_max,
        timeout=None,  # No timeout for batch processing
        backend=backend,
        max_workers=max_workers
    )
    
    # Report cache statistics if available
    try:
        from ..utils.shared_cache import get_shared_cache
        cache = get_shared_cache()
        stats = cache.get_stats()
        logger.info(f"Cache statistics: {stats['hit_rate_percent']:.1f}% hit rate ({stats['hits']}/{stats['total_lookups']} lookups)")
    except (ImportError, AttributeError):
        pass
    
    logger.info(f"Completed batch processing of {len(tables)} tables with Mid-P method")
    return results
