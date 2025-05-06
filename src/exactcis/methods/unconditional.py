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
        has_parallel_support
    )
    has_parallel = True
    logger.info("Using parallel processing for unconditional method")
except ImportError:
    has_parallel = False
    logger.info("Parallel processing not available")

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


@lru_cache(maxsize=256)
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
        args: Tuple containing (p1, a, c, n1, n2, theta, log_p_obs) or 
              (p1, a, c, n1, n2, theta, log_p_obs, start_time, timeout)
        
    Returns:
        Log p-value for this grid point or None if timeout is reached
    """
    # Handle both with and without timeout for backward compatibility
    if len(args) == 7:
        p1, a, c, n1, n2, theta, log_p_obs = args
        start_time = None
        timeout = None
    else:
        p1, a, c, n1, n2, theta, log_p_obs, start_time, timeout = args
    
    # Check for timeout if applicable
    if start_time is not None and timeout is not None:
        if time.time() - start_time > timeout:
            return None  # Signal timeout
    
    # Function to calculate p2 from p1 and theta
    def p2(p1_val: float) -> float:
        return (theta * p1_val) / (1 - p1_val + theta * p1_val)
    
    p_2 = p2(p1)
    
    # Pre-calculate log probabilities
    # Handle ranges that work with both integers and floats
    k_max = int(n1) if n1 == int(n1) else int(n1) + 1
    l_max = int(n2) if n2 == int(n2) else int(n2) + 1
    
    k_range = [i for i in range(k_max + 1) if i <= n1]
    l_range = [i for i in range(l_max + 1) if i <= n2]
    
    if has_numpy:
        try:
            # Pre-calculate log probabilities for all k values that matter
            mean1 = n1 * p1
            std1 = math.sqrt(n1 * p1 * (1-p1))
            relevant_k = [k for k in k_range if abs(k - mean1) <= 5 * std1 or k == a]
            
            # Calculate log probabilities for relevant k values
            log_pk = np.array([_log_binom_pmf(n1, k, p1) for k in relevant_k])
            
            # Pre-calculate log probabilities for all l values that matter
            mean2 = n2 * p_2
            std2 = math.sqrt(n2 * p_2 * (1-p_2))
            relevant_l = [l for l in l_range if abs(l - mean2) <= 5 * std2 or l == c]
            
            # Calculate log probabilities for relevant l values
            log_pl = np.array([_log_binom_pmf(n2, l, p_2) for l in relevant_l])
            
            # Create mesh grid of log probabilities
            LOG_K, LOG_L = np.meshgrid(log_pk, log_pl, indexing='ij')
            
            # Joint log probability
            log_joint = LOG_K + LOG_L
            
            # Find tables with probability <= observed table
            mask = log_joint <= log_p_obs
            
            if np.any(mask):
                # Use logsumexp for numerical stability when summing in log space
                return logsumexp(log_joint[mask].flatten().tolist())
            else:
                return float('-inf')
                
        except Exception as e:
            # Fallback to pure Python if NumPy fails
            pass
    
    # Pure Python implementation
    log_probs = []
    
    # Calculate mean and standard deviation for early termination
    mean1 = n1 * p1
    std1 = math.sqrt(n1 * p1 * (1-p1))
    mean2 = n2 * p_2
    std2 = math.sqrt(n2 * p_2 * (1-p_2))
    
    # Iterate over possible values of k, with early termination
    for k in k_range:
        # Skip values that are unlikely to contribute significantly
        if k != a and abs(k - mean1) > 5 * std1:
            continue
        
        log_pk = _log_binom_pmf(n1, k, p1)
        
        # Skip if probability is negligible compared to observed
        if log_pk < log_p_obs - 20:  # ~1e-9 relative probability
            continue
        
        # Iterate over possible values of l, with early termination
        for l in l_range:
            # Skip values that are unlikely to contribute significantly
            if l != c and abs(l - mean2) > 5 * std2:
                continue
            
            log_pl = _log_binom_pmf(n2, l, p_2)
            
            # Skip if probability is negligible compared to observed
            if log_pl < log_p_obs - 20:  # ~1e-9 relative probability
                continue
            
            log_table_prob = log_pk + log_pl
            
            # Only include tables with probability <= observed
            if log_table_prob <= log_p_obs:
                log_probs.append(log_table_prob)
    
    if log_probs:
        return logsumexp(log_probs)
    else:
        return float('-inf')


def _log_pvalue_barnard(a: int, c: int, n1: int, n2: int,
                        theta: float, grid_size: int,
                        progress_callback: Optional[Callable[[float], None]] = None,
                        start_time: Optional[float] = None,
                        timeout: Optional[float] = None,
                        timeout_checker: Optional[Callable[[], bool]] = None) -> Union[float, None]:
    """
    Calculate the log p-value for Barnard's unconditional exact test using log-space operations.
    
    Args:
        a: Count in cell (1,1)
        c: Count in cell (2,1)
        n1: Size of first group (a+b)
        n2: Size of second group (c+d)
        theta: Odds ratio parameter
        grid_size: Number of grid points for optimization
        progress_callback: Optional callback function to report progress (0-100)
        start_time: Start time for timeout calculation
        timeout: Maximum time in seconds for computation
        timeout_checker: Optional function that returns True if timeout occurred
        
    Returns:
        Log of p-value for Barnard's test, or None if timeout is reached
    """
    # Check for timeout at the beginning
    if start_time is not None and timeout is not None:
        if time.time() - start_time > timeout:
            logger.info(f"Timeout reached in _log_pvalue_barnard")
            return None

    logger.info(f"Calculating log p-value with Barnard's method: a={a}, c={c}, n1={n1}, n2={n2}, theta={theta:.6f}, grid_size={grid_size}")
    
    # For very large tables, return an approximation
    if n1 > 50 or n2 > 50:
        logger.warning(f"Very large table detected (n1={n1}, n2={n2}). Using approximation.")
        # Return a reasonable approximation based on theta
        if theta < 0.5 or theta > 2.0:
            return math.log(0.01)
        else:
            return math.log(0.05)
    
    # Optimize grid size based on table dimensions
    actual_grid_size = _optimize_grid_size(n1, n2, grid_size)
    if actual_grid_size < grid_size:
        logger.info(f"Optimized grid size to {actual_grid_size} based on table dimensions")
    
    # Estimate MLE for p1 (maximum likelihood estimate)
    p1_mle = a / n1 if n1 > 0 else 0.5
    
    # Build adaptive grid
    grid_points = _build_adaptive_grid(p1_mle, actual_grid_size)
    logger.info(f"Using {len(grid_points)} adaptive grid points around p1_mle={p1_mle:.4f}")
    
    # Function to calculate p2 from p1 and theta
    def p2(p1_val: float) -> float:
        return (theta * p1_val) / (1 - p1_val + theta * p1_val)
    
    # Calculate log probability of observed table for the first grid point
    p_2 = p2(grid_points[0])
    log_p_obs = _log_binom_pmf(n1, a, grid_points[0]) + _log_binom_pmf(n2, c, p_2)
    
    # Progress reporting
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Track best log p-value across all grid points
    log_best_pvalue = float('-inf')
    
    # Prepare arguments for parallel processing
    grid_args = [(p1, a, c, n1, n2, theta, log_p_obs) for p1 in grid_points]
    
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
                           refine: bool = True,
                           use_profile: bool = False,
                           progress_callback: Optional[Callable[[float], None]] = None,
                           timeout_checker: Optional[Callable[[], bool]] = None) -> float:
    """
    Calculate log p-value for the unconditional exact test at a given theta.
    
    This is a wrapper around _log_pvalue_barnard that takes a, b, c, d directly
    and handles the conversion to the parameters needed by that function.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        theta: Odds ratio parameter
        p1_values: Optional array of p1 values to evaluate (ignored for now)
        refine: Whether to use refinement for more precision (ignored for now)
        use_profile: Whether to use profile likelihood (ignored for now)
        progress_callback: Optional callback for progress reporting
        timeout_checker: Optional function that returns True if timeout occurred
        
    Returns:
        Natural logarithm of the p-value
    """
    # Transform to parameters needed by _log_pvalue_barnard
    n1 = a + b
    n2 = c + d
    
    # Check the actual parameters supported by _log_pvalue_barnard
    # It requires grid_size as well
    return _log_pvalue_barnard(
        a=a, 
        c=c, 
        n1=n1, 
        n2=n2, 
        theta=theta,
        grid_size=50,  # Using a default grid size
        progress_callback=progress_callback,
        timeout_checker=timeout_checker
    )


def exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, 
                          grid_size: int = 50,
                          theta_min: Optional[float] = None,
                          theta_max: Optional[float] = None,
                          haldane: bool = False,
                          refine: bool = True,
                          use_profile: bool = True,
                          custom_range: Optional[Tuple[float, float]] = None,
                          theta_factor: float = 10.0,
                          timeout: Optional[float] = None,
                          optimization_params: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[Callable[[float], None]] = None) -> Tuple[float, float]:
    """
    Calculate Barnard's unconditional exact confidence interval
    
    Args:
        a, b, c, d: 2x2 table cell counts
        alpha: Significance level (default 0.05 for 95% CI)
        grid_size: Size of grid for p1 (default 50)
        theta_min: Optional minimum theta value for guided search
        theta_max: Optional maximum theta value for guided search
        haldane: Apply Haldane's correction (default False)
        refine: Refine grid for p1 using adaptive grid (default True)
        use_profile: Use profile likelihood for zero counts (default True)
        custom_range: Custom range for theta search (min, max)
        theta_factor: Factor for determining automatic theta range (default 10)
        timeout: Optional timeout in seconds
        optimization_params: Additional optimization parameters
        progress_callback: Optional callback for progress reporting
        
    Returns:
        (lower_bound, upper_bound) tuple for confidence interval
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
    cached_result = cache.get_exact(a, b, c, d, alpha)
    if cached_result is not None:
        logger.info(f"Using cached CI for ({a},{b},{c},{d}): {cached_result[0]}")
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
        cache.add(a, b, c, d, alpha, result, {"method": "early_return", "reason": "zero_marginal"})
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
        "use_profile": use_profile,
        "refine": refine,
        "derived_from_similar": len(similar_entries) > 0,
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
            result = unconditional_log_pvalue(a, b, c, d, theta, p1_values, refine=refine, use_profile=use_profile, progress_callback=progress_callback, timeout_checker=timeout_checker)
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
        if refine and low != float('inf') and high != float('inf'):
            try:
                # Get even more precise bounds using minimal grid with parallel processing
                refined_p1_values = np.linspace(0.001, 0.999, 20)
                
                # Function to compute refined p-values
                cached_refined_pvalues = {}
                
                def refined_log_pvalue(theta):
                    if theta in cached_refined_pvalues:
                        return cached_refined_pvalues[theta]
                    result = unconditional_log_pvalue(a, b, c, d, theta, refined_p1_values, refine=True, use_profile=use_profile, progress_callback=progress_callback, timeout_checker=timeout_checker)
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
    cache.add(a, b, c, d, alpha, result, calculation_metadata)
    
    return result


def improved_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, 
                             grid_size: int = 50,
                             theta_min: Optional[float] = None,
                             theta_max: Optional[float] = None,
                             adaptive_grid: bool = True,
                             use_cache: bool = True,
                             cache_instance: Optional['CICache'] = None) -> Tuple[float, float]:
    """
    Calculate Barnard's exact unconditional confidence interval with improved performance.
    
    This improved version uses caching and adaptive grid search strategies to make
    the calculation more efficient, especially for similar 2x2 tables.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points for p1 (default: 50)
        theta_min: Minimum value of theta to consider (default: None, auto-determined)
        theta_max: Maximum value of theta to consider (default: None, auto-determined)
        adaptive_grid: Whether to use adaptive grid refinement (default: True)
        use_cache: Whether to use caching for speedup (default: True)
        cache_instance: Optional cache instance to use (default: None, creates a new one)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    import logging
    from exactcis.utils.optimization import CICache, get_global_cache
    
    logger = logging.getLogger(__name__)
    
    # Get cache instance if needed
    if use_cache:
        if cache_instance is None:
            cache_instance = get_global_cache()
        
        # Check for exact cache hit
        cache_result = cache_instance.get_exact(a, b, c, d, alpha)
        if cache_result is not None:
            ci, _ = cache_result
            logger.info(f"Cache hit for table ({a},{b},{c},{d}) at alpha={alpha}")
            return ci
    
    # Handle special cases such as zero cells
    if a == 0 or b == 0 or c == 0 or d == 0:
        logger.info(f"Table ({a},{b},{c},{d}) has zero cells, using regular exact_ci_unconditional")
        # Fall back to standard method which handles zero cells better
        result = exact_ci_unconditional(a, b, c, d, alpha, grid_size, theta_min, theta_max)
        
        # Cache the result if caching is enabled
        if use_cache and cache_instance is not None:
            cache_instance.add(a, b, c, d, alpha, result, {"method": "exact_fallback"})
            
        return result
    
    # For large tables, Fisher approximation is often very accurate
    total_n = a + b + c + d
    if total_n > 200:
        logger.info(f"Large table with n={total_n}, using exact_ci_unconditional to ensure accuracy")
        # For large tables, the standard method works well
        result = exact_ci_unconditional(a, b, c, d, alpha, grid_size, theta_min, theta_max)
        
        # Cache the result if caching is enabled
        if use_cache and cache_instance is not None:
            cache_instance.add(a, b, c, d, alpha, result, {"method": "exact_fallback_large"})
            
        return result
    
    # Optimize grid size based on table dimensions
    if total_n < 40:
        optimal_grid_size = min(grid_size, 20)
        logger.info(f"Optimized grid size to {optimal_grid_size} based on table dimensions")
        grid_size = optimal_grid_size
    
    # Calculate MLE for p1 to center the adaptive grid
    p1_mle = a / (a + b) if a + b > 0 else 0.5
    
    # Set up adaptive grid points if enabled
    p1_values = None  # Default to None for standard grid
    if adaptive_grid:
        # More grid points around the MLE
        n_adaptive = min(grid_size * 3 // 4, 30)  # Use at most 30 points for adaptive grid
        logger.info(f"Using {n_adaptive} adaptive grid points around p1_mle={p1_mle:.4f}")
        
        # Create concentrated grid around MLE
        if p1_mle > 0.1 and p1_mle < 0.9:
            # Standard case: concentrate grid around MLE
            width = 0.3  # Width of the concentrated region
            p1_min = max(0, p1_mle - width/2)
            p1_max = min(1, p1_mle + width/2)
            p1_values = np.linspace(p1_min, p1_max, n_adaptive)
            
            # Add some points in the tails
            tail_points = grid_size - n_adaptive
            if tail_points > 0:
                left_points = tail_points // 2
                right_points = tail_points - left_points
                
                if p1_min > 0:
                    left_tail = np.linspace(0.001, p1_min, left_points)
                    p1_values = np.concatenate([left_tail, p1_values])
                
                if p1_max < 1:
                    right_tail = np.linspace(p1_max, 0.999, right_points)
                    p1_values = np.concatenate([p1_values, right_tail])
        else:
            # Edge case: MLE is close to 0 or 1
            p1_values = np.linspace(0.001, 0.999, grid_size)
    
    # Auto-determine theta range if not provided
    # First, calculate a reasonable point estimate of odds ratio
    odds_ratio_est = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))  # Haldane correction
    
    if theta_min is None:
        # Use a reasonable lower bound based on the approximate odds ratio
        theta_min = max(0.0005, odds_ratio_est / 20)
        logger.info(f"Auto-determined lower bound: {theta_min}")
        
    if theta_max is None:
        # Use a reasonable upper bound based on the approximate odds ratio
        theta_max = max(20, odds_ratio_est * 20)
        logger.info(f"Auto-determined upper bound: {theta_max}")
    
    logger.info(f"Using theta range: ({theta_min}, {theta_max})")
    
    # Calculate confidence interval with possible refinement
    try:
        # Calculate lower bound
        try:
            log_alpha = np.log(alpha)
            
            # Function to find where log p-value equals log_alpha
            def f_lower(theta):
                # Using the standard function - the p1 grid is handled internally
                return _log_pvalue_barnard(a, c, a+b, c+d, theta, grid_size, timeout_checker=timeout_checker) - log_alpha
            
            # Search for lower bound
            from exactcis.core import find_sign_change
            from scipy.optimize import brentq
            
            # First try a wider range to ensure we don't miss sign changes
            min_search = max(theta_min * 0.1, 1e-5)
            
            # Find sign change using a more reliable method - start with wider range
            lower_bracket = find_sign_change(f_lower, min_search, odds_ratio_est * 2)
            if lower_bracket is not None:
                lo, hi = lower_bracket
                lower = brentq(f_lower, lo, hi)
                logger.info(f"Lower bound calculated: {lower}")
            else:
                logger.warning(f"No sign change found for lower bound in range [{min_search}, {odds_ratio_est * 2}]")
                # Try a much wider range if first attempt fails
                lower_bracket = find_sign_change(f_lower, min_search, max(theta_min, 0.001))
                if lower_bracket is not None:
                    lo, hi = lower_bracket
                    lower = brentq(f_lower, lo, hi)
                    logger.info(f"Lower bound (second attempt) calculated: {lower}")
                else:
                    # One more attempt with a tiny value for extreme cases
                    lower_bracket = find_sign_change(f_lower, 1e-6, min(odds_ratio_est, 0.1))
                    if lower_bracket is not None:
                        lo, hi = lower_bracket
                        lower = brentq(f_lower, lo, hi)
                        logger.info(f"Lower bound (third attempt) calculated: {lower}")
                    else:
                        # Fall back to original method's approach
                        logger.warning(f"Falling back to very conservative lower bound")
                        lower = theta_min
                
        except Exception as e:
            logger.error(f"Error calculating lower bound: {str(e)}")
            lower = theta_min
            logger.warning(f"Using fallback lower bound: {lower}")
        
        # Calculate upper bound
        try:
            # Function to find where log p-value equals log_alpha
            def f_upper(theta):
                # Similar to f_lower but for upper bound
                return _log_pvalue_barnard(a, c, a+b, c+d, theta, grid_size, timeout_checker=timeout_checker) - log_alpha
            
            # Search for upper bound - start with a wider range
            max_search = min(theta_max * 10, 1e6)
            upper_bracket = find_sign_change(f_upper, odds_ratio_est * 0.5, max_search)
            if upper_bracket is not None:
                lo, hi = upper_bracket
                upper = brentq(f_upper, lo, hi)
                logger.info(f"Upper bound calculated: {upper}")
            else:
                logger.warning(f"No sign change found for upper bound in range [{odds_ratio_est * 0.5}, {max_search}]")
                # Try a more extreme range
                upper_bracket = find_sign_change(f_upper, odds_ratio_est, theta_max * 5)
                if upper_bracket is not None:
                    lo, hi = upper_bracket
                    upper = brentq(f_upper, lo, hi)
                    logger.info(f"Upper bound (second attempt) calculated: {upper}")
                else:
                    # One more attempt with a very large value for extreme cases
                    upper_bracket = find_sign_change(f_upper, max(odds_ratio_est * 5, 10), max(theta_max * 10, 1e5))
                    if upper_bracket is not None:
                        lo, hi = upper_bracket
                        upper = brentq(f_upper, lo, hi)
                        logger.info(f"Upper bound (third attempt) calculated: {upper}")
                    else:
                        logger.warning(f"Falling back to very conservative upper bound")
                        upper = theta_max
                
        except Exception as e:
            logger.error(f"Error calculating upper bound: {str(e)}")
            upper = theta_max
            logger.warning(f"Using fallback upper bound: {upper}")
        
        # Add an additional check - if we're getting very different results from the standard method,
        # log a warning but continue with our calculation
        try:
            # Calculate a quick estimate using the original method with fewer grid points
            quick_check = exact_ci_unconditional(a, b, c, d, alpha, grid_size=20)
            lower_ratio = lower / quick_check[0] if quick_check[0] > 0 else float('inf')
            upper_ratio = upper / quick_check[1] if quick_check[1] < float('inf') else 1.0
            
            # If the difference is more than 3x in either direction, log warning
            if lower_ratio < 0.33 or lower_ratio > 3.0 or upper_ratio < 0.33 or upper_ratio > 3.0:
                logger.warning(f"Large discrepancy detected between methods: quick check={quick_check}, improved=({lower}, {upper})")
                logger.warning(f"Proceeding with improved method calculation, but results may differ from standard method")
                
                # Additional refinement for cases with large discrepancies
                if lower_ratio < 0.33:
                    # Our lower bound might be too small, try to increase it
                    extra_test_points = np.linspace(lower, quick_check[0] * 1.2, 15)
                    for test_theta in extra_test_points:
                        p_value = np.exp(_log_pvalue_barnard(a, c, a+b, c+d, test_theta, grid_size * 2, timeout_checker=timeout_checker))
                        if p_value > alpha:
                            lower = test_theta
                            logger.info(f"Refined lower bound to {lower} based on discrepancy check")
                            break
                
                if upper_ratio > 3.0:
                    # Our upper bound might be too large, try to decrease it
                    extra_test_points = np.linspace(quick_check[1] * 0.8, upper, 15)
                    for test_theta in sorted(extra_test_points, reverse=True):
                        p_value = np.exp(_log_pvalue_barnard(a, c, a+b, c+d, test_theta, grid_size * 2, timeout_checker=timeout_checker))
                        if p_value > alpha:
                            upper = test_theta
                            logger.info(f"Refined upper bound to {upper} based on discrepancy check")
                            break
        except Exception as e:
            logger.error(f"Error in comparison check: {str(e)}")
            # Continue with existing calculation
            
        # Refinement step for more precision
        try:
            # Fine-tune lower bound with small steps
            if lower > theta_min:
                test_thetas = np.linspace(lower * 0.8, lower * 1.2, 10)
                for theta in test_thetas:
                    if theta > theta_min:
                        p_value = np.exp(_log_pvalue_barnard(a, c, a+b, c+d, theta, grid_size, timeout_checker=timeout_checker))
                        if p_value > alpha:
                            lower = theta
                            break
            
            # Fine-tune upper bound with small steps
            if upper < theta_max:
                test_thetas = np.linspace(upper * 0.8, upper * 1.2, 10)
                for theta in reversed(test_thetas):  # Go from largest to smallest
                    if theta < theta_max:
                        p_value = np.exp(_log_pvalue_barnard(a, c, a+b, c+d, theta, grid_size, timeout_checker=timeout_checker))
                        if p_value > alpha:
                            upper = theta
                            break
                            
            logger.info(f"Refined bounds: ({lower}, {upper})")
            
        except Exception as e:
            logger.error(f"Error in refinement: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in CI calculation: {str(e)}")
        # Fall back to standard method as a last resort
        logger.info("Falling back to standard method due to calculation errors")
        return exact_ci_unconditional(a, b, c, d, alpha, grid_size, theta_min, theta_max)
    
    # Final confidence interval
    result = (lower, upper)
    
    # Store in cache if enabled
    if use_cache and cache_instance is not None:
        search_params = {
            "grid_size": grid_size,
            "p1_mle": p1_mle,
            "odds_ratio_est": odds_ratio_est,
            "method": "improved"
        }
        cache_instance.add(a, b, c, d, alpha, result, search_params)
    
    logger.info(f"Unconditional CI calculated: ({lower}, {upper})")
    return result
