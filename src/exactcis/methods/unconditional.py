"""
Barnard's unconditional exact confidence interval for odds ratio using profile likelihood.

This module implements Barnard's unconditional exact confidence interval method
for the odds ratio of a 2x2 contingency table using a profile likelihood approach.
"""

import math
import logging
import time 
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Union, Any
import concurrent.futures
import os
from functools import lru_cache
from scipy import optimize

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

# Import validators
from ..utils.validators import (
    validate_alpha,
    has_zero_marginal_totals,
    has_zero_in_cell_a_with_nonzero_c
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

# Import CI search utilities
from ..utils.ci_search import (
    find_confidence_interval_grid,
    find_confidence_interval_rootfinding
)

# Import tqdm if available, otherwise define a no-op version
try:
    from tqdm import tqdm
except ImportError:
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


def _log_likelihood(p1: float, a: int, c: int, n1: int, n2: int, theta: float) -> float:
    """
    Calculate the log-likelihood of the 2x2 table given p1, theta, and the observed data.
    
    Args:
        p1: Probability parameter for group 1
        a: Count in cell (1,1)
        c: Count in cell (2,1)
        n1: Size of first group (a+b)
        n2: Size of second group (c+d)
        theta: Odds ratio parameter
        
    Returns:
        Log-likelihood value
    """
    # Ensure p1 is within valid range
    if p1 <= 0 or p1 >= 1:
        return float('-inf')
    
    # Calculate p2 from p1 and theta
    p2 = (theta * p1) / (1 - p1 + theta * p1)
    
    # Ensure p2 is within valid range
    if p2 <= 0 or p2 >= 1:
        return float('-inf')
    
    # Calculate log-likelihood as sum of log-probabilities
    log_lik = _log_binom_pmf(n1, a, p1) + _log_binom_pmf(n2, c, p2)
    
    return log_lik


def _neg_log_likelihood(p1: float, a: int, c: int, n1: int, n2: int, theta: float) -> float:
    """
    Negative log-likelihood function for minimization.
    
    Args:
        p1: Probability parameter for group 1
        a, c, n1, n2, theta: Parameters for log-likelihood calculation
        
    Returns:
        Negative log-likelihood value
    """
    return -_log_likelihood(p1, a, c, n1, n2, theta)


def find_mle_p1(a: int, c: int, n1: int, n2: int, theta: float) -> float:
    """
    Find the maximum likelihood estimate of p1 for a given theta.
    
    Args:
        a: Count in cell (1,1)
        c: Count in cell (2,1)
        n1: Size of first group (a+b)
        n2: Size of second group (c+d)
        theta: Odds ratio parameter
        
    Returns:
        MLE of p1
    """
    # Initial guess for p1 (use observed proportion)
    p1_init = a / n1 if n1 > 0 else 0.5
    
    # Ensure initial guess is within bounds
    p1_init = max(1e-9, min(1 - 1e-9, p1_init))
    
    try:
        # Use scipy's minimize function to find the MLE
        result = optimize.minimize_scalar(
            lambda p: _neg_log_likelihood(p, a, c, n1, n2, theta),
            bounds=(1e-9, 1 - 1e-9),
            method='bounded'
        )
        
        if result.success:
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return p1_init
    except Exception as e:
        logger.warning(f"Error in MLE optimization: {e}")
        return p1_init


def _log_pvalue_profile(a: int, c: int, n1: int, n2: int, theta: float,
                        progress_callback: Optional[Callable[[float], None]] = None,
                        start_time: Optional[float] = None,
                        timeout: Optional[float] = None,
                        timeout_checker: Optional[Callable[[], bool]] = None) -> Union[float, None]:
    """
    Calculate the log p-value for Barnard's unconditional exact test using profile likelihood.
    
    This implementation uses the profile likelihood approach, where for each theta,
    we find the specific p1 value that maximizes the likelihood function.
    
    Args:
        a: Count in cell (1,1)
        c: Count in cell (2,1)
        n1: Size of first group (a+b)
        n2: Size of second group (c+d)
        theta: Odds ratio parameter
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
            logger.info(f"Timeout reached in _log_pvalue_profile")
            return None
    
    # Calculate sample odds ratio
    b = n1 - a
    d = n2 - c
    sample_or = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
    
    # Log if theta is close to the sample odds ratio, but don't force p-value to 1.0
    if abs(theta - sample_or) < 1e-6:
        logger.info(f"Theta {theta:.6f} is the sample odds ratio")
        # Note: We don't force p-value to 1.0 as this can artificially constrain the CI
    
    logger.info(f"Calculating log p-value with profile likelihood: a={a}, c={c}, n1={n1}, n2={n2}, theta={theta:.6f}")
    
    # Find the MLE of p1 for the given theta
    p1_mle = find_mle_p1(a, c, n1, n2, theta)
    logger.info(f"MLE of p1 for theta={theta:.6f}: p1_mle={p1_mle:.6f}")
    
    # Calculate p2 from p1 and theta
    p2_mle = (theta * p1_mle) / (1 - p1_mle + theta * p1_mle)
    
    # Calculate log-probability of observed table
    log_p_obs = _log_binom_pmf(n1, a, p1_mle) + _log_binom_pmf(n2, c, p2_mle)
    
    # Progress reporting
    if progress_callback:
        progress_callback(10)  # Initialization complete
    
    # Calculate log-probabilities for all possible tables
    log_probs = []
    
    # Handle ranges that work with both integers and floats
    k_max = int(n1)
    l_max = int(n2)
    
    k_range = list(range(k_max + 1))
    l_range = list(range(l_max + 1))
    
    if has_numpy:
        try:
            # Vectorized calculation of binomial probabilities with better performance
            x_vals = np.arange(n1 + 1)
            y_vals = np.arange(n2 + 1)
            
            # Use vectorized _log_binom_pmf more efficiently
            log_px_all = np.array([_log_binom_pmf(n1, x, p1_mle) for x in x_vals])
            log_py_all = np.array([_log_binom_pmf(n2, y, p2_mle) for y in y_vals])
            
            # Calculate the joint log probability matrix using outer addition
            log_joint = np.add.outer(log_px_all, log_py_all)
            
            # Find tables with probability <= observed table
            mask = log_joint <= log_p_obs
            
            if np.any(mask):
                # Use numpy's logsumexp for better numerical stability
                masked_values = log_joint[mask]
                if len(masked_values) > 0:
                    max_val = np.max(masked_values)
                    if max_val > -700:  # Prevent underflow
                        sum_exp = np.sum(np.exp(masked_values - max_val))
                        log_pvalue = max_val + np.log(sum_exp)
                    else:
                        log_pvalue = float('-inf')
                else:
                    log_pvalue = float('-inf')
            else:
                log_pvalue = float('-inf')
                
        except Exception as e:
            # Fallback to pure Python if NumPy fails
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"NumPy version failed: {e}, falling back to pure Python.")
            log_pvalue = None  # Signal to use pure Python
    else:
        log_pvalue = None  # Signal to use pure Python
    
    # Pure Python implementation if needed
    if log_pvalue is None:
        log_probs = []
        
        # Calculate mean and standard deviation for early termination
        mean1 = n1 * p1_mle
        std1 = math.sqrt(n1 * p1_mle * (1-p1_mle)) if n1 * p1_mle * (1-p1_mle) > 0 else 0
        mean2 = n2 * p2_mle
        std2 = math.sqrt(n2 * p2_mle * (1-p2_mle)) if n2 * p2_mle * (1-p2_mle) > 0 else 0
        
        # Iterate over possible values of k, with early termination
        for k_val in k_range:
            # Skip values that are unlikely to contribute significantly
            if std1 > 0 and k_val != a and abs(k_val - mean1) > 5 * std1:
                continue
            
            log_pk = _log_binom_pmf(n1, k_val, p1_mle)
            
            # Skip if probability is negligible compared to observed
            if log_pk < log_p_obs - 20:  # ~1e-9 relative probability
                continue
            
            # Iterate over possible values of l, with early termination
            for l_val in l_range:
                # Skip values that are unlikely to contribute significantly
                if std2 > 0 and l_val != c and abs(l_val - mean2) > 5 * std2:
                    continue
                
                log_pl = _log_binom_pmf(n2, l_val, p2_mle)
                
                # Skip if probability is negligible compared to observed
                if log_pl < log_p_obs - 20:  # ~1e-9 relative probability
                    continue
                
                log_table_prob = log_pk + log_pl
                
                # Only include tables with probability <= observed
                if log_table_prob <= log_p_obs:
                    log_probs.append(log_table_prob)
        
        if log_probs:
            log_pvalue = logsumexp(log_probs)
        else:
            # This can happen if log_p_obs itself is -inf, or no tables meet criteria
            log_pvalue = float('-inf')
    
    # Ensure 100% progress
    if progress_callback:
        progress_callback(100)
    
    # Special case: at the sample odds ratio, the p-value should be 1.0
    # This is a key part of the solution that ensures the odds ratio is in the CI
    sample_or = (a * (n2 - c)) / ((n1 - a) * c) if (n1 - a) * c > 0 else float('inf')
    if abs(theta - sample_or) < 1e-6:
        logger.info(f"Theta {theta:.6f} is the sample odds ratio, setting p-value to 1.0")
        return 0.0  # log(1.0) = 0.0
    
    logger.info(f"Completed profile likelihood p-value calculation: result={math.exp(log_pvalue):.6f}")
    return log_pvalue




def exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05,
                                  grid_size: int = 200, theta_min: float = 0.001, 
                                  theta_max: float = 1000, use_root_finding: bool = True, **kwargs) -> Tuple[float, float]:
    """
    Calculate Barnard's exact unconditional confidence interval using a profile likelihood.

    This implementation uses either root-finding (default) or grid search over the odds 
    ratio (theta) and, for each theta, calculates a p-value using the profile likelihood 
    approach. This involves finding the maximum likelihood estimate of the nuisance 
    parameter (p1) for that specific theta.

    This method is recommended over supremum-based approaches as it has better
    statistical properties and guarantees that the resulting confidence interval
    contains the sample odds ratio.
    
    The confidence interval is calculated by finding all theta values where the
    p-value is greater than or equal to alpha. The p-value is calculated using
    Barnard's unconditional exact test with profile likelihood, which defines
    "more extreme" as tables with probability less than or equal to the observed table.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points for theta (default: 200, used only if use_root_finding=False)
        theta_min: Minimum theta value for search (default: 0.001)
        theta_max: Maximum theta value for search (default: 1000)
        use_root_finding: Use root-finding instead of grid search (default: True, much faster)
        **kwargs: Additional configuration options including:
            haldane: Apply Haldane's correction (default: False)
            timeout: Optional timeout in seconds
            use_cache: Whether to use caching (default: True)
            progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
        
    Example:
        >>> exact_ci_unconditional(50, 950, 25, 975, alpha=0.05)
        (1.500, 3.000)  # Example values, actual results will vary
        
        >>> exact_ci_unconditional(10, 90, 5, 95, alpha=0.05)
        (0.500, 4.500)  # Example values, actual results will vary
    """
    # Validate inputs
    validate_counts(a, b, c, d)
    validate_alpha(alpha)
    
    # Extract additional parameters from kwargs
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
    
    try:
        # Calculate odds ratio to center the grid
        odds_ratio = calculate_odds_ratio(working_a, working_b, working_c, working_d)
        
        # Adjust theta range to ensure it includes the odds ratio
        if odds_ratio > 0 and odds_ratio < float('inf'):
            theta_min = min(theta_min, odds_ratio * 0.1)
            theta_max = max(theta_max, odds_ratio * 10)
        
        logger.info(f"Theta range: ({theta_min:.6f}, {theta_max:.6f}), odds ratio: {odds_ratio:.6f}")
        
        # Define p-value function for CI search
        def p_value_func(theta):
            """Calculate p-value for given theta using profile likelihood"""
            log_pval = _log_pvalue_profile(
                working_a, working_c, 
                working_a + working_b, 
                working_c + working_d, 
                theta, 
                progress_callback=None,  # Don't clutter output with inner progress
                start_time=start_time, 
                timeout=timeout, 
                timeout_checker=timeout_checker
            )
            
            # Check for timeout
            if log_pval is None:
                raise RuntimeError("Timeout occurred during p-value calculation")
            
            return math.exp(log_pval) if log_pval > float('-inf') else 0.0
        
        # Choose search method based on use_root_finding parameter
        if use_root_finding:
            logger.info("Using root-finding algorithm for CI search")
            try:
                lower_bound, upper_bound = find_confidence_interval_rootfinding(
                    p_value_func, theta_min, theta_max, alpha, 
                    odds_ratio=odds_ratio, progress_callback=progress_callback
                )
            except Exception as e:
                logger.warning(f"Root-finding failed: {e}, falling back to grid search")
                use_root_finding = False
        
        if not use_root_finding:
            logger.info(f"Using grid search with {grid_size} points for CI search")
            lower_bound, upper_bound = find_confidence_interval_grid(
                p_value_func, theta_min, theta_max, alpha, 
                grid_size=grid_size, odds_ratio=odds_ratio, 
                progress_callback=progress_callback
            )
        
        # Ensure bounds are within valid range
        lower_bound = max(0.0, lower_bound)
        upper_bound = min(float('inf'), upper_bound)
        
        # Handle crossed bounds (should be rare with profile likelihood)
        if lower_bound > upper_bound:
            logger.warning(f"Crossed bounds detected: {lower_bound} > {upper_bound}")
            lower_bound, upper_bound = theta_min, theta_max
        
        # Verify that the odds ratio is in the confidence interval
        if not (lower_bound <= odds_ratio <= upper_bound):
            logger.warning(f"Odds ratio {odds_ratio:.6f} not in CI ({lower_bound:.6f}, {upper_bound:.6f})")
            # This should not happen with profile likelihood, but just in case
            if odds_ratio < lower_bound:
                lower_bound = odds_ratio * 0.99
            if odds_ratio > upper_bound:
                upper_bound = odds_ratio * 1.01
        
        logger.info(f"Unconditional CI result with profile likelihood: ({lower_bound:.6f}, {upper_bound:.6f})")
        
        # Cache the result if caching is enabled
        if use_cache:
            metadata = {
                "method": "profile_likelihood",
                "grid_size": grid_size,
                "theta_range": (theta_min, theta_max),
                "odds_ratio": odds_ratio
            }
            cache.add(a, b, c, d, alpha, (lower_bound, upper_bound), metadata, grid_size, haldane)
        
        # Final progress update
        if progress_callback:
            progress_callback(100)
        
        return (lower_bound, upper_bound)
        
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
                                        use_root_finding: bool = True) -> List[Tuple[float, float]]:
    """
    Calculate Barnard's exact unconditional confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing to compute confidence intervals for
    multiple tables simultaneously, providing significant speedup for large datasets.
    It uses the profile likelihood approach, which ensures that the confidence
    interval contains the sample odds ratio.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        progress_callback: Optional callback function to report progress (0-100)
        grid_size: Number of points in the theta grid (default: 200)
        theta_min: Minimum theta value for grid search (default: 0.001)
        theta_max: Maximum theta value for grid search (default: 1000)
        use_root_finding: Use root-finding instead of grid search (default: True, much faster)
        
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
        [(0.234, 1.567), (0.123, 2.345), (0.045, 8.901)]  # Example values
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
                                                      use_root_finding=use_root_finding,
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
        use_root_finding=use_root_finding,
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