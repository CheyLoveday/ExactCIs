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
            # Vectorized calculation of binomial probabilities
            x_vals = np.arange(n1 + 1)
            y_vals = np.arange(n2 + 1)

            log_px_all = _log_binom_pmf(n1, x_vals, p1)
            log_py_all = _log_binom_pmf(n2, y_vals, current_p2)
            
            # Calculate the joint log probability matrix using outer addition
            # This is significantly faster than nested loops for large n1, n2
            log_joint = np.add.outer(log_px_all, log_py_all)
            
            # Find tables with probability <= observed table
            mask = log_joint <= log_p_obs_for_this_p1

            if np.any(mask):
                return logsumexp(log_joint[mask].flatten().tolist())
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
        
        # Process grid points in parallel
        results = parallel_map(
            _process_grid_point, 
            grid_args,
            max_workers=max_workers,
            progress_callback=lambda p: progress_callback(10 + p * 0.9) if progress_callback else None
        )
        
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
                          grid_size: int = 15,
                          theta_min: Optional[float] = None,
                          theta_max: Optional[float] = None,
                          custom_range: Optional[Tuple[float, float]] = None,
                          theta_factor: float = 100.0,
                          haldane: bool = False, 
                          timeout: Optional[float] = None,
                          optimization_params: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[Callable[[float, str], None]] = None,
                          adaptive_grid: bool = True,
                          use_cache: bool = True,
                          optimization_strategy: str = "auto") -> Tuple[float, float]:
    """
    Calculate Barnard's exact unconditional confidence interval.
    
    This unified function uses a grid search over the nuisance parameter p1 and then
    finds the confidence interval for the odds ratio. It combines the best features
    of the previous implementations with configurable optimization strategies.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points for p1 (default: 50)
        theta_min: Minimum value of theta to consider (default: None, auto-determined)
        theta_max: Maximum value of theta to consider (default: None, auto-determined)
        custom_range: Custom range for theta search (min, max)
        theta_factor: Factor for determining automatic theta range (default: 100)
        haldane: Apply Haldane's correction (default: False)
        timeout: Optional timeout in seconds
        optimization_params: Additional optimization parameters
        progress_callback: Optional callback for progress reporting
        adaptive_grid: Whether to use adaptive grid refinement (default: True)
        use_cache: Whether to use caching for speedup (default: True)
        optimization_strategy: Strategy for optimization ("auto", "conservative", "aggressive")
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    # Initialize parameters
    optimization_params = optimization_params or {}
    
    # Set up timeout checker if timeout is provided
    timeout_checker = None
    if timeout is not None:
        start_time = time.time()
        def timeout_checker():
            return time.time() - start_time > timeout
        logger.info(f"Using timeout of {timeout} seconds")
    
    # Check cache for exact match first
    cache = get_global_cache()
    cached_result = cache.get_exact(a, b, c, d, alpha, grid_size, haldane)
    if cached_result is not None:
        logger.info(f"Using cached CI for ({a},{b},{c},{d}) at alpha={alpha}, grid_size={grid_size}, haldane={haldane}")
        return cached_result[0]
    
    # Look for similar tables in cache to guide search
    search_params = {}
    similar_entries = cache.get_similar(a, b, c, d, alpha)
    if similar_entries:
        search_params = derive_search_params(a, b, c, d, similar_entries)
        logger.info(f"Using search parameters derived from similar tables: {search_params}")
        
        # Override grid_size if suggested by similar tables
        if "grid_size" in search_params and grid_size == 50:  # Only override default
            grid_size = search_params["grid_size"]
        
        # Use predicted theta range if available and not explicitly set
        if "predicted_theta_range" in search_params and theta_min is None and theta_max is None and custom_range is None:
            theta_min, theta_max = search_params["predicted_theta_range"]
            logger.info(f"Using predicted theta range from similar tables: ({theta_min}, {theta_max})")
    
    # Validation
    n1 = a + b
    n2 = c + d
    
    if n1 == 0 or n2 == 0:
        logger.warning("One or both marginal totals are zero, returning (0, inf)")
        result = (0, float('inf'))
        cache.add(a, b, c, d, alpha, result, {"method": "early_return", "reason": "zero_marginal"}, grid_size, haldane)
        return result
    
    if haldane:
        a_h, b_h, c_h, d_h = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        a, b, c, d = a_h, b_h, c_h, d_h
        logger.info("Applied Haldane's correction")
    
    # Determine search range for theta
    if custom_range is not None:
        min_theta, max_theta = custom_range
        logger.info(f"Using custom theta range: ({min_theta}, {max_theta})")
    elif theta_min is not None and theta_max is not None:
        min_theta, max_theta = theta_min, theta_max
        logger.info(f"Using provided theta range: ({min_theta}, {max_theta})")
    else:
        # Calculate odds ratio as center point
        or_value = (a * d) / (b * c) if b * c > 0 else (a * d) if a * d > 0 else 1.0
        
        # Special case for zero in a cell - the odds ratio should be 0 when a=0 and c>0
        if a == 0 and c > 0:
            logger.info("Cell a=0, adjusting confidence interval to include 0")
            result = (0.0, 1e-5)  
            cache.add(a, b, c, d, alpha, result, {"method": "early_return", "reason": "zero_in_a"}, grid_size, haldane)
            return result
            
        # Handle edge cases
        if or_value == 0:
            or_value = 1e-6
        elif or_value == float('inf'):
            or_value = 1e6
        
        # Set range centered on OR with appropriate width
        min_theta = or_value / theta_factor
        max_theta = or_value * theta_factor
        
        # Ensure reasonable bounds
        min_theta = max(min_theta, 1e-6)
        max_theta = min(max_theta, 1e6)
        
        logger.info(f"Auto-determined theta range: ({min_theta}, {max_theta}) based on OR={or_value}")
    
    # Start timing
    start_time = time.time()
    
    # Create result tracking variables
    calculation_metadata = {
        "grid_size": grid_size,
        "theta_range": (min_theta, max_theta),
        "total_iterations": 0
    }
    
    # Optimize grid size based on table dimensions
    n = a + b + c + d
    if n < 40 and grid_size > 20:
        grid_size = 20
        logger.info(f"Optimized grid size to {grid_size} based on table dimensions")
    
    # Initialize low and high
    low, high = float('inf'), float('-inf')
    
    try:
        # Calculate MLE p1 for adaptive grid
        p1_mle = a / (a + b) if a + b > 0 else 0.5
        
        # Use adaptive grid around MLE
        p1_values = np.concatenate([
            np.linspace(max(0.001, p1_mle - 0.4), p1_mle - 0.05, grid_size // 3),
            np.linspace(p1_mle - 0.05, p1_mle + 0.05, grid_size // 3),
            np.linspace(p1_mle + 0.05, min(0.999, p1_mle + 0.4), grid_size // 3)
        ])
        p1_values = np.unique(p1_values)
        logger.info(f"Using {len(p1_values)} adaptive grid points around p1_mle={p1_mle:.4f}")
        
        # Function to cache log p-values
        cached_pvalues = {}
        
        def cached_log_pvalue(theta):
            if theta in cached_pvalues:
                return cached_pvalues[theta]
            result = unconditional_log_pvalue(a, b, c, d, theta, p1_values=p1_values, grid_size=grid_size, progress_callback=progress_callback, timeout_checker=timeout_checker)
            cached_pvalues[theta] = result
            calculation_metadata["total_iterations"] += 1
            return result
        
        # Calculate lower bound
        try:
            # Use log(alpha) as our target
            log_alpha = math.log(alpha)
            
            # Initialize bounds for lower CI limit
            if min_theta < 1:
                theta_lower = min_theta  # Use provided lower bound
                theta_mid = math.sqrt(min_theta)  # Geometric mean as midpoint
            else:
                theta_lower = min_theta / 10
                theta_mid = min_theta
            
            # Search for sign change in log p-value - log(alpha)
            sign_change = find_sign_change(
                lambda theta: cached_log_pvalue(theta) - log_alpha,
                theta_lower,
                theta_mid
            )
            
            if sign_change:
                low, _ = sign_change
                logger.info(f"Found sign change for lower bound: {low:.6f}")
            else:
                # Fallback: try adaptive grid search
                crossings = adaptive_grid_search(
                    lambda theta: math.exp(cached_log_pvalue(theta)),
                    (theta_lower, theta_mid),
                    target_value=alpha,
                    initial_points=15,
                    refinement_rounds=2
                )
                
                if crossings:
                    low = min(crossings)
                    logger.info(f"Found lower bound using adaptive search: {low:.6f}")
                else:
                    # Conservative fallback
                    low = min_theta
                    logger.warning(f"Using fallback lower bound: {low}")
            
            # Refine lower bound
            plateau_result = find_plateau_edge(
                lambda theta: math.exp(cached_log_pvalue(theta)),
                lo=max(1e-10, low * 0.9),
                hi=low * 1.1,
                target=alpha,
                timeout_checker=timeout_checker
            )
            
            # Handle tuple return value correctly
            if plateau_result is not None:
                low = plateau_result[0]  # First element is the result value
                logger.info(f"Lower bound calculated: {low:.6f}")
            else:
                logger.warning("Plateau edge detection failed for lower bound")
            
        except Exception as e:
            logger.error(f"Error calculating lower bound: {e}")
            # Fallback to a conservative estimate
            low = min_theta
            logger.warning(f"Using fallback lower bound: {low}")
        
        # Calculate upper bound
        try:
            # Initialize bounds for upper CI limit
            if max_theta > 1:
                theta_upper = max_theta  # Use provided upper bound
                theta_mid = math.sqrt(max_theta)  # Geometric mean as midpoint
            else:
                theta_upper = max_theta * 10
                theta_mid = max_theta
            
            # Search for sign change in log p-value - log(alpha)
            sign_change = find_sign_change(
                lambda theta: cached_log_pvalue(theta) - log_alpha,
                theta_mid,
                theta_upper
            )
            
            if sign_change:
                high, _ = sign_change
                logger.info(f"Found sign change for upper bound: {high:.6f}")
            else:
                # Fallback: try adaptive grid search
                crossings = adaptive_grid_search(
                    lambda theta: math.exp(cached_log_pvalue(theta)),
                    (theta_mid, theta_upper),
                    target_value=alpha,
                    initial_points=15,
                    refinement_rounds=2
                )
                
                if crossings:
                    high = max(crossings)
                    logger.info(f"Found upper bound using adaptive search: {high:.6f}")
                else:
                    # Conservative fallback
                    high = max_theta
                    logger.warning(f"Using fallback upper bound: {high}")
            
            # Refine upper bound
            plateau_result = find_plateau_edge(
                lambda theta: math.exp(cached_log_pvalue(theta)),
                lo=high * 0.9,
                hi=min(high * 1.1, 1e16),
                target=alpha,
                timeout_checker=timeout_checker
            )
            
            # Handle tuple return value correctly
            if plateau_result is not None:
                high = plateau_result[0]  # First element is the result value
                logger.info(f"Upper bound calculated: {high:.6f}")
            else:
                logger.warning("Plateau edge detection failed for upper bound")
            
        except Exception as e:
            logger.error(f"Error calculating upper bound: {e}")
            # Fallback to a conservative estimate
            high = max_theta
            logger.warning(f"Using fallback upper bound: {high}")
        
        # Perform additional refinement if needed and if we have non-infinite bounds
        if low != float('inf') and high != float('inf'):
            try:
                # Get even more precise bounds using minimal grid with parallel processing
                refined_p1_values = np.linspace(0.001, 0.999, 20)
                
                # Function to compute refined p-values
                cached_refined_pvalues = {}
                
                def refined_log_pvalue(theta):
                    if theta in cached_refined_pvalues:
                        return cached_refined_pvalues[theta]
                    result = unconditional_log_pvalue(a, b, c, d, theta, p1_values=refined_p1_values, grid_size=20, progress_callback=progress_callback, timeout_checker=timeout_checker)
                    cached_refined_pvalues[theta] = result
                    calculation_metadata["total_iterations"] += 1
                    return result
                
                # Refine the lower bound using plateau edge detection
                plateau_result = find_plateau_edge(
                    lambda theta: math.exp(refined_log_pvalue(theta)),
                    lo=max(1e-10, low * 0.98),
                    hi=low * 1.02,
                    target=alpha,
                    timeout_checker=timeout_checker
                )
                
                # Handle tuple return value correctly
                if plateau_result is not None:
                    low = plateau_result[0]  # First element is the result value
                else:
                    logger.warning("Plateau edge detection failed for lower bound")
                
                if high < float('inf'):
                    # Use narrow ranges for refinement
                    if has_parallel:
                        # For values near upper bound
                        upper_thetas = np.linspace(high * 0.98, high * 1.02, 11)
                        upper_pvalues = parallel_map(
                            refined_log_pvalue,
                            upper_thetas
                        )
                    else:
                        # Sequential computation
                        upper_thetas = np.linspace(high * 0.98, high * 1.02, 7)
                        upper_pvalues = [refined_log_pvalue(theta) for theta in upper_thetas]
                    
                    # Refine the upper bound using plateau edge detection
                    plateau_result = find_plateau_edge(
                        lambda theta: math.exp(refined_log_pvalue(theta)),
                        lo=high * 0.98,
                        hi=min(high * 1.02, 1e16),
                        target=alpha,
                        timeout_checker=timeout_checker
                    )
                    
                    # Handle tuple return value correctly
                    if plateau_result is not None:
                        high = plateau_result[0]  # First element is the result value
                    else:
                        logger.warning("Plateau edge detection failed for upper bound")
                
                logger.info(f"Refined bounds: ({low:.6f}, {high if high != float('inf') else 'inf'})")
            except Exception as e:
                logger.error(f"Error in refinement: {e}")
                # Continue with the unrefined bounds
                pass
    except Exception as e:
        logger.error(f"Error in unconditional CI calculation: {e}")
        # Return conservative bounds on error
        low, high = min_theta, max_theta
    
    # Finalize results
    elapsed_time = time.time() - start_time
    calculation_metadata["elapsed_time"] = elapsed_time
    
    # Handle special cases and sanity checks
    if low == float('inf') or high == float('-inf'):
        logger.warning("Bounds calculation failed, using fallback")
        low, high = min_theta, max_theta
    
    # Ensure bounds are within valid range
    low = max(low, 0)
    
    # Log the final result
    logger.info(f"Unconditional CI calculated: ({low}, {high if high != float('inf') else 'inf'})")
    
    # Add to cache
    result = (low, high)
    cache.add(a, b, c, d, alpha, result, calculation_metadata, grid_size, haldane)
    
    return result



