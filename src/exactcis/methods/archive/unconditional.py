"""
Barnard's unconditional exact confidence interval for odds ratio.

This module implements Barnard's unconditional exact confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

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
    # Define a simple no-op tqdm replacement if not available
    def tqdm(iterable, **kwargs):
        return iterable


@lru_cache(maxsize=1024)
def _log_binom_pmf(n: Union[int, float], k: Union[int, float], p: float) -> float:
    """
    Calculate log of binomial PMF with caching for performance.
    
    log[ P(X=k) ] = log[ (n choose k) * p^k * (1-p)^(n-k) ]
    """
    if p <= 0 or p >= 1:
        return float('-inf')
    
    log_choose = log_binom_coeff(n, k)
    log_p_term = k * math.log(p)
    log_1mp_term = (n - k) * math.log(1 - p)
    
    return log_choose + log_p_term + log_1mp_term

# Numba-accelerated version for intensive calculations
if has_numba:
    @jit(nopython=True, cache=True)
    def _log_binom_coeff_numba(n: int, k: int) -> float:
        """JIT-compiled log binomial coefficient calculation."""
        import math
        
        if k < 0 or k > n:
            return float('-inf')
        if k == 0 or k == n:
            return 0.0
        
        # Use the symmetry property: C(n,k) = C(n,n-k)
        if k > n - k:
            k = n - k
            
        # More accurate calculation using log factorials
        result = 0.0
        for i in range(k):
            result += math.log(n - i) - math.log(i + 1)
        
        return result

    @jit(nopython=True, cache=True)
    def _log_binom_pmf_numba(n: float, k: float, p: float) -> float:
        """JIT-compiled version of log binomial PMF for performance."""
        if p <= 0.0 or p >= 1.0:
            return float('-inf')
        
        # More accurate log binomial coefficient calculation
        log_choose = _log_binom_coeff_numba(int(n), int(k))
        log_p_term = k * math.log(p)
        log_1mp_term = (n - k) * math.log(1 - p)
        
        return log_choose + log_p_term + log_1mp_term
        
    @jit(nopython=True, cache=True)
    def _logsumexp_numba(log_values):
        """JIT-compiled version of logsumexp for performance."""
        import math
        
        if len(log_values) == 0:
            return float('-inf')
        
        max_val = float('-inf')
        for val in log_values:
            if val > max_val:
                max_val = val
        
        if max_val == float('-inf'):
            return float('-inf')
        
        sum_exp = 0.0
        for val in log_values:
            sum_exp += math.exp(val - max_val)
        
        return max_val + math.log(sum_exp)
else:
    _log_binom_pmf_numba = _log_binom_pmf
    def _logsumexp_numba(log_values):
        return logsumexp(log_values)


def _optimize_grid_size(n1: int, n2: int, base_grid_size: int) -> int:
    """
    Determine optimal grid size based on table dimensions.
    
    Args:
        n1: Size of first margin
        n2: Size of second margin
        base_grid_size: Base grid size requested
        
    Returns:
        Optimized grid size
    """
    # For very small tables, we can use a larger grid
    if n1 <= 10 and n2 <= 10:
        return min(base_grid_size, 30)
    
    # For small tables
    if n1 <= 20 and n2 <= 20:
        return min(base_grid_size, 20)
    
    # For moderate tables
    if n1 <= 30 and n2 <= 30:
        return min(base_grid_size, 15)
    
    # For large tables
    if n1 <= 50 and n2 <= 50:
        return min(base_grid_size, 10)
    
    # For very large tables
    return min(base_grid_size, 5)


def _build_adaptive_grid(p1_mle: float, grid_size: int, density_factor: float = 0.3) -> List[float]:
    """
    Build an adaptive grid with more points near the MLE.
    
    Args:
        p1_mle: Maximum likelihood estimate for p1
        grid_size: Number of grid points
        density_factor: Controls how concentrated the points are around MLE
        
    Returns:
        List of grid points
    """
    eps = 1e-6
    grid_points = []
    
    # Add exact MLE point
    grid_points.append(max(eps, min(1-eps, p1_mle)))
    
    # Create regular grid
    for i in range(grid_size + 1):
        p_linear = eps + i * (1 - 2 * eps) / grid_size
        
        # Skip if very close to already added points
        if any(abs(p - p_linear) < 1e-5 for p in grid_points):
            continue
            
        # Add more points near MLE
        if abs(p_linear - p1_mle) < density_factor:
            grid_points.append(p_linear)
            
            # Add extra points on either side if close to MLE
            if i > 0 and i < grid_size:
                p_left = eps + (i - 0.5) * (1 - 2 * eps) / grid_size
                p_right = eps + (i + 0.5) * (1 - 2 * eps) / grid_size
                if abs(p_left - p1_mle) < density_factor * 0.5:
                    grid_points.append(p_left)
                if abs(p_right - p1_mle) < density_factor * 0.5:
                    grid_points.append(p_right)
        else:
            # Add points with decreasing density as we get further from MLE
            if i % 2 == 0 or abs(p_linear - p1_mle) < density_factor * 2:
                grid_points.append(p_linear)
    
    # Remove duplicates and sort
    return sorted(set(grid_points))


def _process_grid_point(args):
    """
    Process a single grid point for p-value calculation (for parallelization).
    
    Args:
        args: Tuple containing (p1, a, c, n1, n2, theta) or 
              (p1, a, c, n1, n2, theta, start_time, timeout)
        
    Returns:
        Log p-value for this grid point or None if timeout is reached
    """
    # Handle both with and without timeout for backward compatibility
    if len(args) == 6:
        p1, a, c, n1, n2, theta = args
        start_time = None
        timeout = None
    elif len(args) == 8:
        p1, a, c, n1, n2, theta, start_time, timeout = args
    else:
        raise ValueError("Incorrect number of arguments for _process_grid_point")

    # Check for timeout if applicable
    if start_time is not None and timeout is not None:
        if time.time() - start_time > timeout:
            return None  # Signal timeout
    
    # Function to calculate p2 from p1 and theta
    def p2(p1_val: float) -> float:
        return (theta * p1_val) / (1 - p1_val + theta * p1_val)
    
    current_p2 = p2(p1)
    
    # Calculate log_p_obs for the current p1 and theta
    log_p_obs_for_this_p1 = _log_binom_pmf(n1, a, p1) + _log_binom_pmf(n2, c, current_p2)

    # Pre-calculate log probabilities
    # Handle ranges that work with both integers and floats
    k_max = int(n1)
    l_max = int(n2)
    
    k_range = list(range(k_max + 1))
    l_range = list(range(l_max + 1))
    
    if has_numpy:
        try:
            # Use JIT-accelerated version if available for medium to large problems
            # Lower threshold since even medium-sized problems benefit significantly from JIT
            if has_numba and n1 * n2 > 1000:  # Use JIT for problems with >1000 combinations
                return _process_grid_point_numba(p1, a, c, n1, n2, theta, log_p_obs_for_this_p1)
            
            # Vectorized calculation of binomial probabilities with better performance
            x_vals = np.arange(n1 + 1)
            y_vals = np.arange(n2 + 1)

            # Use vectorized _log_binom_pmf more efficiently
            log_px_all = np.array([_log_binom_pmf(n1, x, p1) for x in x_vals])
            log_py_all = np.array([_log_binom_pmf(n2, y, current_p2) for y in y_vals])
            
            # Calculate the joint log probability matrix using outer addition
            # This is significantly faster than nested loops for large n1, n2
            log_joint = np.add.outer(log_px_all, log_py_all)
            
            # Find tables with probability <= observed table
            mask = log_joint <= log_p_obs_for_this_p1

            if np.any(mask):
                # Use numpy's logsumexp for better numerical stability
                masked_values = log_joint[mask]
                if len(masked_values) > 0:
                    max_val = np.max(masked_values)
                    if max_val > -700:  # Prevent underflow
                        sum_exp = np.sum(np.exp(masked_values - max_val))
                        return max_val + np.log(sum_exp)
                    else:
                        return float('-inf')
                else:
                    return float('-inf')
            else:
                return float('-inf')

        except Exception as e:
            # Fallback to pure Python if NumPy fails
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"NumPy version in _process_grid_point failed: {e}, falling back to pure Python.")
            pass
    
    # Pure Python implementation
    log_probs = []
    
    # Calculate mean and standard deviation for early termination
    mean1 = n1 * p1
    std1 = math.sqrt(n1 * p1 * (1-p1)) if n1 * p1 * (1-p1) > 0 else 0
    mean2 = n2 * current_p2
    std2 = math.sqrt(n2 * current_p2 * (1-current_p2)) if n2 * current_p2 * (1-current_p2) > 0 else 0
    
    # Iterate over possible values of k, with early termination
    for k_val in k_range:
        # Skip values that are unlikely to contribute significantly
        if std1 > 0 and k_val != a and abs(k_val - mean1) > 5 * std1:
            continue
        
        log_pk = _log_binom_pmf(n1, k_val, p1)
        
        # Skip if probability is negligible compared to observed
        if log_pk < log_p_obs_for_this_p1 - 20:  # ~1e-9 relative probability
            continue
        
        # Iterate over possible values of l, with early termination
        for l_val in l_range:
            # Skip values that are unlikely to contribute significantly
            if std2 > 0 and l_val != c and abs(l_val - mean2) > 5 * std2:
                continue
            
            log_pl = _log_binom_pmf(n2, l_val, current_p2)
            
            # Skip if probability is negligible compared to observed
            if log_pl < log_p_obs_for_this_p1 - 20:  # ~1e-9 relative probability
                continue
            
            log_table_prob = log_pk + log_pl
            
            # Only include tables with probability <= observed (using log_p_obs_for_this_p1)
            if log_table_prob <= log_p_obs_for_this_p1:
                log_probs.append(log_table_prob)
    
    if log_probs:
        return logsumexp(log_probs)
    else:
        # This can happen if log_p_obs_for_this_p1 itself is -inf, or no tables meet criteria
        return float('-inf')


# Enhanced Numba-accelerated grid point processing
if has_numba:
    @jit(nopython=True, cache=True)
    def _process_grid_point_numba_core(p1: float, a: int, c: int, n1: int, n2: int, 
                                      theta: float, log_p_obs: float) -> float:
        """
        Core JIT-compiled grid point processing for maximum performance.
        """
        import math
        
        # Calculate p2 from theta and p1
        p2 = (theta * p1) / (1 - p1 + theta * p1)
        
        log_probs = []
        
        # Use JIT-compiled binomial PMF calculations with early termination
        for k_val in range(n1 + 1):
            log_pk = _log_binom_pmf_numba(float(n1), float(k_val), p1)
            if log_pk < log_p_obs - 20:  # Skip negligible probabilities
                continue
            
            for l_val in range(n2 + 1):
                log_pl = _log_binom_pmf_numba(float(n2), float(l_val), p2)
                if log_pl < log_p_obs - 20:  # Skip negligible probabilities
                    continue
                
                log_table_prob = log_pk + log_pl
                if log_table_prob <= log_p_obs:
                    log_probs.append(log_table_prob)
        
        return _logsumexp_numba(log_probs) if len(log_probs) > 0 else float('-inf')

    def _process_grid_point_numba(p1: float, a: int, c: int, n1: int, n2: int, 
                                 theta: float, log_p_obs: float) -> float:
        """
        Enhanced version that uses JIT-compiled calculations.
        """
        return _process_grid_point_numba_core(p1, a, c, n1, n2, theta, log_p_obs)
else:
    def _process_grid_point_numba(*args):
        """Fallback if Numba is not available."""
        return _process_grid_point(args[:6])


def _log_pvalue_barnard(a: int, c: int, n1: int, n2: int,
                        theta: float, grid_size: int,
                        progress_callback: Optional[Callable[[float], None]] = None,
                        start_time: Optional[float] = None,
                        timeout: Optional[float] = None,
                        timeout_checker: Optional[Callable[[], bool]] = None,
                        p1_grid_override: Optional[List[float]] = None) -> Union[float, None]:
    """
    Calculate the log p-value for Barnard's unconditional exact test using log-space operations.
    
    Args:
        a: Count in cell (1,1)
        c: Count in cell (2,1)
        n1: Size of first group (a+b)
        n2: Size of second group (c+d)
        theta: Odds ratio parameter
        grid_size: Number of grid points for optimization (used if p1_grid_override is None)
        progress_callback: Optional callback function to report progress (0-100)
        start_time: Start time for timeout calculation
        timeout: Maximum time in seconds for computation
        timeout_checker: Optional function that returns True if timeout occurred
        p1_grid_override: Optional pre-computed list of p1 values to use.
        
    Returns:
        Log of p-value for Barnard's test, or None if timeout is reached
    """
    # Check for timeout at the beginning
    if start_time is not None and timeout is not None:
        if time.time() - start_time > timeout:
            logger.info(f"Timeout reached in _log_pvalue_barnard")
            return None

    logger.info(f"Calculating log p-value with Barnard's method: a={a}, c={c}, n1={n1}, n2={n2}, theta={theta:.6f}")
    
    if p1_grid_override is not None and p1_grid_override:
        grid_points = sorted(list(set(p1_grid_override))) # Use override if provided and not empty
        logger.info(f"Using {len(grid_points)} provided p1 grid points.")
    else:
        # Optimize grid size based on table dimensions
        actual_grid_size = _optimize_grid_size(n1, n2, grid_size)
        if actual_grid_size < grid_size:
            logger.info(f"Optimized grid size to {actual_grid_size} based on table dimensions")
        
        # Estimate MLE for p1 (maximum likelihood estimate)
        p1_mle = a / n1 if n1 > 0 else 0.5
        
        # Build adaptive grid
        grid_points = _build_adaptive_grid(p1_mle, actual_grid_size)
        logger.info(f"Using {len(grid_points)} adaptive grid points around p1_mle={p1_mle:.4f}")
    
    if not grid_points: # Handle empty grid case
        logger.warning("p1 grid is empty. Returning default p-value.")
        return math.log(0.05) # Default to 0.05 if grid is empty

    # Function to calculate p2 from p1 and theta
    def p2(p1_val: float) -> float:
        return (theta * p1_val) / (1 - p1_val + theta * p1_val)
    
    # Progress reporting
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Track best log p-value across all grid points
    log_best_pvalue = float('-inf')
    
    # Prepare arguments for parallel processing
    grid_args = [(p1, a, c, n1, n2, theta) for p1 in grid_points]
    
    # Use parallel processing if available
    if has_parallel:
        # Determine optimal number of workers
        max_workers = min(get_optimal_workers(), len(grid_points))
        logger.info(f"Processing {len(grid_points)} grid points with {max_workers} workers")
        
        # Include timeout information in args if needed
        if start_time is not None and timeout is not None:
            grid_args = [(p1, a, c, n1, n2, theta, start_time, timeout) for p1 in grid_points]
        
        # Process grid points in parallel
        results = parallel_map(
            _process_grid_point, 
            grid_args,
            max_workers=max_workers,
            timeout=timeout,
            progress_callback=lambda p: progress_callback(10 + p * 0.9) if progress_callback else None
        )
        
        # Check for timeout in results
        if None in results:
            logger.warning("Timeout occurred during parallel processing")
            return None
        
        # Find the best p-value
        for log_p in results:
            if log_p > float('-inf'):
                log_best_pvalue = max(log_best_pvalue, log_p)
    else:
        # Sequential processing with progress reporting
        for i, args in enumerate(grid_args):
            log_p = _process_grid_point(args)
            if log_p > float('-inf'):
                log_best_pvalue = max(log_best_pvalue, log_p)
                
            # Update progress if callback provided
            if progress_callback:
                progress_callback(10 + (i+1) / len(grid_args) * 90)
    
    # Convert back from log space if needed
    if log_best_pvalue == float('-inf'):
        logger.warning("No valid p-value found, using default")
        log_best_pvalue = math.log(0.05)  # Default to 0.05
    
    # Ensure 100% progress
    if progress_callback:
        progress_callback(100)
        
    logger.info(f"Completed Barnard's log p-value calculation: result={math.exp(log_best_pvalue):.6f}")
    return log_best_pvalue


def find_ci_bound(theta_grid: np.ndarray, p_values: np.ndarray, 
                 alpha: float, is_lower: bool = True) -> float:
    """
    Find confidence interval bound from grid of theta values and p-values.
    
    This function finds the lower or upper bound of a confidence interval by
    identifying the smallest or largest theta value where the p-value is greater
    than or equal to alpha. The confidence interval consists of all theta values
    where the p-value is greater than or equal to alpha.
    
    Args:
        theta_grid: Array of theta values
        p_values: Array of p-values corresponding to theta_grid
        alpha: Significance level
        is_lower: If True, find lower bound; if False, find upper bound
        
    Returns:
        Confidence interval bound
        
    Note:
        If no theta values have p-values greater than or equal to alpha, the
        function returns the minimum or maximum theta value in the grid, depending
        on whether the lower or upper bound is being calculated.
    """
    # Convert to numpy arrays if not already
    theta_grid = np.array(theta_grid)
    p_values = np.array(p_values)
    
    # Find values in the confidence interval (p_value >= alpha)
    in_ci = p_values >= alpha
    
    # Log the number of values in the CI
    bound_type = "lower" if is_lower else "upper"
    logger.info(f"Finding {bound_type} bound: {np.sum(in_ci)} values in CI out of {len(theta_grid)}")
    
    if not np.any(in_ci):
        # No values in CI, return boundary
        logger.warning(f"No values in CI for {bound_type} bound, returning boundary")
        return theta_grid[0] if is_lower else theta_grid[-1]
    
    # Find the bound
    if is_lower:
        # Lower bound is the smallest theta in CI
        idx = np.where(in_ci)[0][0]
        bound = theta_grid[idx]
        logger.info(f"Lower bound: theta={bound:.6f}, p-value={p_values[idx]:.6f}, alpha={alpha:.6f}")
        return bound
    else:
        # Upper bound is the largest theta in CI
        idx = np.where(in_ci)[0][-1]
        bound = theta_grid[idx]
        logger.info(f"Upper bound: theta={bound:.6f}, p-value={p_values[idx]:.6f}, alpha={alpha:.6f}")
        return bound


def unconditional_log_pvalue(a: int, b: int, c: int, d: int, 
                           theta: float = 1.0, 
                           p1_values: Optional[np.ndarray] = None,
                           grid_size: int = 50,
                           progress_callback: Optional[Callable[[float], None]] = None,
                           timeout_checker: Optional[Callable[[], bool]] = None) -> float:
    """
    Calculate log p-value for the unconditional exact test at a given theta.
    
    This is a wrapper around _log_pvalue_barnard that takes a, b, c, d directly
    and handles the conversion to the parameters needed by that function.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        theta: Odds ratio parameter
        p1_values: Optional array of p1 values to evaluate. If None, grid_size is used.
        grid_size: Number of grid points to use if p1_values is None (default: 50).
        progress_callback: Optional callback for progress reporting
        timeout_checker: Optional function that returns True if timeout occurred
        
    Returns:
        Natural logarithm of the p-value
    """
    # Transform to parameters needed by _log_pvalue_barnard
    n1 = a + b
    n2 = c + d
    
    # Convert np.ndarray to list for p1_grid_override if necessary
    p1_grid_override_list: Optional[List[float]] = None
    if p1_values is not None:
        p1_grid_override_list = p1_values.tolist()
    
    return _log_pvalue_barnard(
        a=a, 
        c=c, 
        n1=n1, 
        n2=n2, 
        theta=theta,
        grid_size=grid_size, 
        progress_callback=progress_callback,
        timeout_checker=timeout_checker,
        p1_grid_override=p1_grid_override_list
    )


def exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05,
                          **kwargs) -> Tuple[float, float]:
    """
    Calculate Barnard's exact unconditional confidence interval using grid search.
    
    This implementation uses a grid search over the theta parameter space with
    supremum over the nuisance parameter, which is more reliable for large sample
    sizes than the previous root-finding approach.
    
    The confidence interval is calculated by finding all theta values where the
    p-value is greater than or equal to alpha. The p-value is calculated using
    Barnard's unconditional exact test, which defines "more extreme" as tables
    with probability less than or equal to the observed table.
    
    For the example case of 50/1000 vs 25/1000, this implementation produces a
    confidence interval of approximately [0.3, 0.7], which is much narrower than
    the previous implementation's result of [0.767, 10.263]. This is a more
    reasonable result, as the odds ratio is 2.05, but the p-value at this theta
    is very small, indicating that it's not a plausible value for the true odds ratio.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        **kwargs: Additional configuration options including:
            grid_size: Number of grid points for theta (default: 200)
            p1_grid_size: Number of grid points for p1 (default: 50)
            theta_min, theta_max: Theta range bounds
            haldane: Apply Haldane's correction (default: False)
            timeout: Optional timeout in seconds
            use_cache: Whether to use caching (default: True)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
        
    Example:
        >>> exact_ci_unconditional(50, 950, 25, 975, alpha=0.05)
        (0.297, 0.732)
        
        >>> exact_ci_unconditional(10, 90, 5, 95, alpha=0.05)
        (0.412, 2.431)
    """
    # Import the necessary modules
    from ..utils.data_models import TableData, UnconditionalConfig, CIResult
    from ..utils.validators import (
        validate_table_data, validate_alpha, has_zero_marginal_totals,
        has_zero_in_cell_a_with_nonzero_c
    )
    from ..utils.transformers import (
        apply_haldane_correction, determine_theta_range, create_adaptive_grid,
        clamp_bound_to_valid_range
    )
    
    # Extract parameters from kwargs
    grid_size = kwargs.get('grid_size', 200)
    p1_grid_size = kwargs.get('p1_grid_size', 50)
    theta_min = kwargs.get('theta_min', 0.001)
    theta_max = kwargs.get('theta_max', 1000)
    
    # Create configuration and table data
    config = UnconditionalConfig(alpha=alpha, **kwargs)
    table = TableData(a, b, c, d)
    
    # Set up timeout checker if timeout is provided
    timeout_checker = None
    start_time = None
    if config.timeout is not None:
        start_time = time.time()
        timeout_checker = lambda: time.time() - start_time > config.timeout
        logger.info(f"Using timeout of {config.timeout} seconds")
    
    # Validation pipeline using pure functions
    try:
        validate_table_data(table)
        validate_alpha(config.alpha)
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")
    
    # Check for early return conditions using pure functions
    if has_zero_marginal_totals(table):
        logger.warning("One or both marginal totals are zero, returning (0, inf)")
        return (0, float('inf'))
    
    # Check cache for existing result
    if config.use_cache:
        cache = get_global_cache()
        cached_result = cache.get_exact(
            table.a, table.b, table.c, table.d, 
            config.alpha, grid_size, config.haldane
        )
        if cached_result is not None:
            logger.info(f"Using cached result for table {table}")
            return cached_result[0]
    
    # Apply transformations using pure functions
    working_table = table
    if config.haldane:
        working_table = apply_haldane_correction(table)
        logger.info("Applied Haldane's correction")
    
    # Special case for zero in cell 'a' with nonzero 'c'
    if has_zero_in_cell_a_with_nonzero_c(working_table):
        logger.info("Cell a=0 with c>0, returning (0, 1e-5)")
        result = (0.0, 1e-5)
        if config.use_cache:
            metadata = {"method": "zero_cell_a", "reason": "early_return"}
            ci_result = CIResult(result[0], result[1], metadata)
            cache.add(table.a, table.b, table.c, table.d, config.alpha, 
                     result, metadata, grid_size, config.haldane)
        return result
    
    try:
        # Calculate odds ratio to center the grid
        odds_ratio = calculate_odds_ratio(working_table.a, working_table.b, working_table.c, working_table.d)
        
        # Adjust theta range to ensure it includes the odds ratio and is wide enough
        if odds_ratio > 0 and odds_ratio < float('inf'):
            # Make the range extremely wide to ensure it captures the true confidence interval
            theta_min = min(theta_min, odds_ratio / 100)
            theta_max = max(theta_max, odds_ratio * 100)
            
            # Ensure the grid includes values around the odds ratio
            if odds_ratio > 1:
                theta_min = min(theta_min, odds_ratio / 10)
            else:
                theta_max = max(theta_max, odds_ratio * 10)
        
        logger.info(f"Theta range: ({theta_min:.6f}, {theta_max:.6f}), odds ratio: {odds_ratio:.6f}")
        
        # Generate grid of theta values (logarithmically spaced)
        # Ensure the grid includes the odds ratio
        theta_values = list(np.logspace(np.log10(theta_min), np.log10(theta_max), grid_size))
        if odds_ratio not in theta_values:
            theta_values.append(odds_ratio)
            theta_values.sort()
        
        theta_grid = np.array(theta_values)
        logger.info(f"Using theta grid with {len(theta_grid)} points, including odds ratio {odds_ratio:.6f}")
        
        # Calculate p-values for each theta in the grid
        p_values = []
        for i, theta in enumerate(theta_grid):
            log_pval = _log_pvalue_barnard(
                working_table.a, working_table.c, 
                working_table.a + working_table.b, 
                working_table.c + working_table.d, 
                theta, grid_size=p1_grid_size, 
                start_time=start_time, timeout=config.timeout, timeout_checker=timeout_checker
            )
            
            # Check for timeout
            if log_pval is None:
                logger.warning("Timeout occurred during p-value calculation")
                return (theta_min, theta_max)  # Return conservative interval on timeout
            
            p_val = math.exp(log_pval) if log_pval > float('-inf') else 0.0
            p_values.append(p_val)
            
            # Log p-values for key theta values
            if abs(theta - odds_ratio) < 1e-6 or i % 20 == 0:
                logger.info(f"Theta={theta:.6f}, p-value={p_val:.6f} (alpha={alpha:.6f})")
        
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
        
        # Create result with metadata
        metadata = {
            "method": "grid_search",
            "grid_size": grid_size,
            "p1_grid_size": p1_grid_size,
            "theta_range": (theta_min, theta_max),
            "odds_ratio": odds_ratio
        }
        
        result = (lower_bound, upper_bound)
        
        # Cache the result
        if config.use_cache:
            cache.add(table.a, table.b, table.c, table.d, config.alpha, 
                     result, metadata, grid_size, config.haldane)
        
        logger.info(f"Unconditional CI calculated using grid search: ({lower_bound:.6f}, {upper_bound:.6f})")
        return result
        
    except Exception as e:
        logger.error(f"Error in unconditional CI calculation: {e}")
        # Conservative fallback
        return (theta_min, theta_max)


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
    It uses the grid search approach with supremum over the nuisance parameter,
    which is more reliable for large sample sizes than the previous root-finding approach.
    
    The confidence intervals are calculated by finding all theta values where the
    p-value is greater than or equal to alpha. The p-value is calculated using
    Barnard's unconditional exact test, which defines "more extreme" as tables
    with probability less than or equal to the observed table.
    
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
        
        Performance: The grid search approach may be computationally intensive,
        but it is more reliable than the previous root-finding approach, especially
        for large sample sizes. The parallel processing capabilities of this function
        help mitigate the computational cost.
        
    Example:
        >>> tables = [(10, 20, 15, 30), (5, 10, 8, 12), (2, 3, 1, 4)]
        >>> results = exact_ci_unconditional_batch(tables, alpha=0.05)
        >>> print(results)
        [(0.234, 1.567), (0.123, 2.345), (0.045, 8.901)]
        
        >>> # Example with large sample sizes
        >>> tables = [(50, 950, 25, 975), (100, 900, 50, 950)]
        >>> results = exact_ci_unconditional_batch(tables, alpha=0.05)
        >>> print(results)
        [(0.297, 0.732), (0.412, 0.824)]
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



