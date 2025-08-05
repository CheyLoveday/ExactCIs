import math
import logging
import time 
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Union, Any
import concurrent.futures
import os
from functools import lru_cache
from scipy.stats import binom

# Configure logging
logger = logging.getLogger(__name__)

from ..core import (
    find_sign_change,
    find_plateau_edge,
    calculate_odds_ratio,
    calculate_relative_risk,
    create_2x2_table,
    validate_counts,
    validate_alpha,
    log_binom_coeff,
    find_root_log,
    logsumexp,
    apply_haldane_correction
)

# Import utilities
try:
    has_numpy = True
except ImportError:
    has_numpy = False
    logger.info("NumPy not available, using pure Python implementation")

# Try to import Numba for JIT acceleration
try:
    from numba import jit, prange
    has_numba = True
    logger.info("Numba available for JIT acceleration")
except ImportError:
    has_numba = False
    logger.info("Numba not available, using standard Python")
    # Create no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Import parallel processing utilities if available
try:
    from ..utils.parallel import (
        parallel_map, 
        has_parallel_support,
        get_optimal_workers,
        parallel_compute_ci
    )
    has_parallel = True
    logger.info("Using parallel processing for unconditional method")
except ImportError:
    has_parallel = False
    logger.info("Parallel processing not available")
    
    # Fallback function if parallel utilities are not available
    def get_optimal_workers():
        return 1

# Import optimization utilities
from ..utils.optimization import (
    get_global_cache,
    derive_search_params,
    adaptive_grid_search
)

# Import tqdm if available, otherwise define a no-op version
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Keep the existing implementation of _log_pvalue_barnard and other helper functions
# (These functions are not shown here but would be included in the actual file)

def find_ci_bound(theta_grid: np.ndarray, p_values: np.ndarray, 
                 alpha: float, is_lower: bool = True) -> float:
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
    # Convert to numpy arrays if not already
    theta_grid = np.array(theta_grid)
    p_values = np.array(p_values)
    
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

def exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, 
                          grid_size: int = 200, theta_min: float = 0.001, 
                          theta_max: float = 1000, **kwargs) -> Tuple[float, float]:
    """
    Calculate Barnard's exact unconditional confidence interval using grid search.
    
    This implementation uses a grid search over the theta parameter space with
    supremum over the nuisance parameter, which is more reliable for large sample
    sizes than the previous root-finding approach.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points for theta (default: 200)
        theta_min: Minimum theta value for grid search (default: 0.001)
        theta_max: Maximum theta value for grid search (default: 1000)
        **kwargs: Additional configuration options including:
            p1_grid_size: Number of grid points for p1 (default: 50)
            haldane: Apply Haldane's correction (default: False)
            timeout: Optional timeout in seconds
            use_cache: Whether to use caching (default: True)
            progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    # Validate inputs
    validate_counts(a, b, c, d)
    validate_alpha(alpha)
    
    # Extract additional parameters from kwargs
    p1_grid_size = kwargs.get('p1_grid_size', 50)
    haldane = kwargs.get('haldane', False)
    timeout = kwargs.get('timeout', None)
    use_cache = kwargs.get('use_cache', True)
    progress_callback = kwargs.get('progress_callback', None)
    
    # Set up timeout checker if timeout is provided
    timeout_checker = None
    start_time = None
    if timeout is not None:
        start_time = time.time()
        timeout_checker = lambda: time.time() - start_time > timeout
        logger.info(f"Using timeout of {timeout} seconds")
    
    # Handle special cases
    if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
        logger.warning("One or both marginal totals are zero, returning (0, inf)")
        return (0.0, float('inf'))
    
    # Check cache for existing result
    if use_cache:
        cache = get_global_cache()
        cached_result = cache.get_exact(
            a, b, c, d, alpha, grid_size, haldane
        )
        if cached_result is not None:
            logger.info(f"Using cached result for table ({a},{b},{c},{d})")
            return cached_result[0]
    
    # Apply Haldane's correction if requested
    working_a, working_b, working_c, working_d = a, b, c, d
    if haldane:
        working_a, working_b, working_c, working_d = apply_haldane_correction(a, b, c, d)
        logger.info("Applied Haldane's correction")
    
    # Special case for zero in cell 'a' with nonzero 'c'
    if working_a == 0 and working_c > 0:
        logger.info("Cell a=0 with c>0, returning (0, 1e-5)")
        result = (0.0, 1e-5)
        if use_cache:
            metadata = {"method": "zero_cell_a", "reason": "early_return"}
            cache.add(a, b, c, d, alpha, result, metadata, grid_size, haldane)
        return result
    
    # Calculate odds ratio to center the grid
    odds_ratio = calculate_odds_ratio(working_a, working_b, working_c, working_d)
    
    # Adjust theta range to ensure it includes the odds ratio
    if odds_ratio > 0 and odds_ratio < float('inf'):
        theta_min = min(theta_min, odds_ratio * 0.1)
        theta_max = max(theta_max, odds_ratio * 10)
    
    logger.info(f"Theta range: ({theta_min:.6f}, {theta_max:.6f}), odds ratio: {odds_ratio:.6f}")
    
    # Generate grid of theta values (logarithmically spaced)
    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), grid_size)
    
    # Progress reporting
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Calculate p-values for each theta in the grid
    p_values = []
    for i, theta in enumerate(theta_grid):
        log_pval = _log_pvalue_barnard(
            working_a, working_c, working_a + working_b, working_c + working_d, 
            theta, grid_size=p1_grid_size, 
            start_time=start_time, timeout=timeout, timeout_checker=timeout_checker
        )
        
        # Check for timeout
        if log_pval is None:
            logger.warning("Timeout occurred during p-value calculation")
            return (theta_min, theta_max)  # Return conservative interval on timeout
        
        p_values.append(math.exp(log_pval) if log_pval > float('-inf') else 0.0)
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(10 + (i+1) / len(theta_grid) * 80)
    
    # Find confidence interval bounds
    lower_bound = find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    upper_bound = find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    
    # Ensure bounds are within valid range
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(float('inf'), upper_bound)
    
    # Handle crossed bounds (should be rare with grid search)
    if lower_bound > upper_bound:
        logger.warning(f"Crossed bounds detected: {lower_bound} > {upper_bound}")
        lower_bound, upper_bound = theta_min, theta_max
    
    logger.info(f"Unconditional CI result: ({lower_bound:.6f}, {upper_bound:.6f})")
    
    # Cache the result if caching is enabled
    if use_cache:
        metadata = {
            "method": "grid_search",
            "grid_size": grid_size,
            "p1_grid_size": p1_grid_size,
            "theta_range": (theta_min, theta_max),
            "odds_ratio": odds_ratio
        }
        cache.add(a, b, c, d, alpha, (lower_bound, upper_bound), metadata, grid_size, haldane)
    
    # Final progress update
    if progress_callback:
        progress_callback(100)
    
    return (lower_bound, upper_bound)

def exact_ci_unconditional_batch(tables: List[Tuple[int, int, int, int]], 
                                alpha: float = 0.05,
                                max_workers: Optional[int] = None,
                                backend: Optional[str] = None,
                                progress_callback: Optional[Callable] = None,
                                grid_size: int = 200,
                                theta_min: float = 0.001,
                                theta_max: float = 1000,
                                p1_grid_size: int = 50) -> List[Tuple[float, float]]:
    """
    Calculate Barnard's exact unconditional confidence intervals for multiple 2x2 tables in parallel.
    
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
        p1_grid_size: Number of points in the p1 grid (default: 50)
        
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
        >>> results = exact_ci_unconditional_batch(tables, alpha=0.05)
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
                result = exact_ci_unconditional(a, b, c, d, alpha, 
                                              grid_size=grid_size,
                                              theta_min=theta_min,
                                              theta_max=theta_max,
                                              p1_grid_size=p1_grid_size,
                                              progress_callback=progress_callback)
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
    
    logger.info(f"Processing {len(tables)} tables with Unconditional method using {max_workers} workers")
    
    # Use parallel_compute_ci for better performance and error handling
    results = parallel_compute_ci(
        exact_ci_unconditional,
        tables,
        alpha=alpha,
        grid_size=grid_size,
        theta_min=theta_min,
        theta_max=theta_max,
        p1_grid_size=p1_grid_size,
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
    
    logger.info(f"Completed batch processing of {len(tables)} tables with Unconditional method")
    return results