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
        get_optimal_workers
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
    Calculate Barnard's exact unconditional confidence interval.
    
    This refactored function uses functional programming principles with pure functions
    and immutable data structures for improved maintainability and testability.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        **kwargs: Additional configuration options including:
            grid_size: Number of grid points for p1 (default: 15)
            theta_min, theta_max: Theta range bounds
            custom_range: Custom range for theta search
            theta_factor: Factor for automatic theta range (default: 100)
            haldane: Apply Haldane's correction (default: False)
            timeout: Optional timeout in seconds
            use_cache: Whether to use caching (default: True)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    # Import the new functional modules
    from ..utils.data_models import TableData, UnconditionalConfig, CIResult
    from ..utils.validators import (
        validate_table_data, validate_alpha, has_zero_marginal_totals,
        has_zero_in_cell_a_with_nonzero_c
    )
    from ..utils.transformers import (
        apply_haldane_correction, determine_theta_range, create_adaptive_grid,
        clamp_bound_to_valid_range
    )
    from ..utils.calculators import find_confidence_bound
    
    # Create configuration and table data
    config = UnconditionalConfig(alpha=alpha, **kwargs)
    table = TableData(a, b, c, d)
    
    # Set up timeout checker if timeout is provided
    timeout_checker = None
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
            config.alpha, config.grid_size, config.haldane
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
                     result, metadata, config.grid_size, config.haldane)
        return result
    
    # Determine search parameters using pure functions
    theta_range = determine_theta_range(working_table, config)
    grid_config = create_adaptive_grid(working_table, config)
    
    logger.info(f"Theta range: ({theta_range.min_theta:.6f}, {theta_range.max_theta:.6f})")
    logger.info(f"Grid size: {grid_config.grid_size}, p1_mle: {grid_config.p1_mle:.4f}")
    
    try:
        # Calculate confidence bounds using pure functions
        lower_search_range = (
            theta_range.min_theta,
            math.sqrt(theta_range.min_theta * theta_range.or_value)
        )
        upper_search_range = (
            math.sqrt(theta_range.or_value * theta_range.max_theta),
            theta_range.max_theta
        )
        
        lower_result = find_confidence_bound(
            working_table, grid_config, lower_search_range,
            config.alpha, "lower", timeout_checker
        )
        
        upper_result = find_confidence_bound(
            working_table, grid_config, upper_search_range,
            config.alpha, "upper", timeout_checker
        )
        
        # Apply final transformations
        lower = clamp_bound_to_valid_range(lower_result.value, is_lower=True)
        upper = clamp_bound_to_valid_range(upper_result.value, is_lower=False)
        
        # Handle crossed bounds
        if lower > upper:
            logger.warning(f"Crossed bounds detected: {lower} > {upper}")
            lower, upper = theta_range.min_theta, theta_range.max_theta
        
        # Create result with metadata
        metadata = {
            "lower_method": lower_result.method,
            "upper_method": upper_result.method,
            "theta_range": (theta_range.min_theta, theta_range.max_theta),
            "grid_size": grid_config.grid_size,
            "p1_mle": grid_config.p1_mle,
            "or_value": theta_range.or_value
        }
        
        result = (lower, upper)
        
        # Cache the result
        if config.use_cache:
            cache.add(table.a, table.b, table.c, table.d, config.alpha, 
                     result, metadata, config.grid_size, config.haldane)
        
        logger.info(f"Unconditional CI calculated: ({lower:.6f}, {upper:.6f})")
        return result
        
    except Exception as e:
        logger.error(f"Error in unconditional CI calculation: {e}")
        # Conservative fallback
        return (theta_range.min_theta, theta_range.max_theta)



