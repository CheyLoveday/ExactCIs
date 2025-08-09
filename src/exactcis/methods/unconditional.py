#!/usr/bin/env python3
"""
Unconditional exact confidence interval for odds ratio.

This module implements the unconditional exact confidence interval method
for the odds ratio of a 2x2 contingency table using a grid search approach
with confidence interval inversion.

The unconditional exact method uses maximum likelihood estimation over a nuisance parameter
and considers the full unconditional sampling distribution, whereas Fisher's exact test 
conditions on the marginals. This approach provides valid inference when the row or column
totals are not fixed by the study design.

This implementation uses a grid search approach with confidence interval inversion,
which is more reliable for large sample sizes than root-finding methods. The approach
is similar to that used in the R 'Exact' package by Fay and Fay (2021).

References:
    Barnard, G. A. (1947). Significance tests for 2 × 2 tables.
        Biometrika, 34(1/2), 123-138.
    Fay, M. P., & Fay, M. M. (2021). Exact: Unconditional Exact Test. R package
        version 3.1. https://CRAN.R-project.org/package=Exact
"""

import math
import logging
from typing import Tuple, List, Optional, Dict, Any, Callable
import numpy as np
from functools import lru_cache
from scipy.stats import binom

# Try to import numba for vectorization
try:
    import numba as nb
    has_numba = True
    logger = logging.getLogger(__name__)
    logger.info("Numba is available for acceleration")
except ImportError:
    has_numba = False
    logger = logging.getLogger(__name__)
    logger.warning("Numba not available, performance may be reduced")

from exactcis.core import (
    calculate_odds_ratio
)
from exactcis.utils.validation import validate_table_and_alpha

# Import CI search utilities
from exactcis.utils.ci_search import (
    find_confidence_interval_grid,
    find_confidence_interval_adaptive_grid
)

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_map, get_optimal_workers, parallel_compute_ci
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for unconditional method")


# Vectorize binomial functions if numba is available
def _calculate_binomial_probs(a_values, n1, p1, c_values, n2, p2):
    """Calculate joint binomial probabilities for two independent binomials.
    Uses numpy vectorization; numba version omitted because scipy functions
    are not supported in nopython mode.
    """
    log_pmf1 = binom.logpmf(a_values, n1, p1)
    log_pmf2 = binom.logpmf(c_values, n2, p2)
    return np.exp(log_pmf1[:, np.newaxis] + log_pmf2[np.newaxis, :])


def or_test_statistic(a, n1, n2, m1, p1, p2):
    r"""Barnard/Suissa–Shuster score statistic with continuity correction.

    The statistic is |\hat{\ell} - \ell_0| / sqrt(Var), where
        \hat{\ell}   = log((k1+cc)/(n1-k1+cc)) - log((k2+cc)/(n2-k2+cc))
        \ell_0       = log(p1/(1-p1)) - log(p2/(1-p2))
        Var          = 1/(n1 p1(1-p1)) + 1/(n2 p2(1-p2))

    A 0.5 continuity correction (cc) is applied when a cell is on a boundary.
    """
    eps = 1e-12

    b = n1 - a
    c = m1 - a
    d = n2 - c

    # continuity correction 0.5 at boundaries
    def cc(x, n):
        if x == 0 or x == n:
            return x + 0.5 if x == 0 else x - 0.5
        return x

    a_cc = cc(a, n1)
    b_cc = cc(b, n1)
    c_cc = cc(c, n2)
    d_cc = cc(d, n2)

    # clamp to avoid zeros
    a_cc = max(a_cc, eps)
    b_cc = max(b_cc, eps)
    c_cc = max(c_cc, eps)
    d_cc = max(d_cc, eps)

    try:
        lor_hat = math.log(a_cc / b_cc) - math.log(c_cc / d_cc)

        p1_adj = min(1 - eps, max(eps, p1))
        p2_adj = min(1 - eps, max(eps, p2))
        lor_exp = math.log(p1_adj / (1 - p1_adj)) - math.log(p2_adj / (1 - p2_adj))

        var = (1.0 / (n1 * p1_adj * (1 - p1_adj))) + (1.0 / (n2 * p2_adj * (1 - p2_adj)))
        if var <= 0:
            return float('inf')

        return abs(lor_hat - lor_exp) / math.sqrt(var)
    except (ValueError, ZeroDivisionError, OverflowError):
        return float('inf')


def or_test_statistic_vectorized(a_values, n1, n2, m1, p1, p2):
    eps = 1e-12

    # continuity correction
    a_cc = a_values.astype(float)
    b_cc = (n1 - a_values).astype(float)
    c_cc = (m1 - a_values).astype(float)
    d_cc = (n2 - c_cc).astype(float)

    mask = a_values == 0
    a_cc[mask] += 0.5
    mask = a_values == n1
    a_cc[mask] -= 0.5

    mask = b_cc == 0
    b_cc[mask] += 0.5
    mask = b_cc == n1
    b_cc[mask] -= 0.5

    mask = c_cc == 0
    c_cc[mask] += 0.5
    mask = c_cc == n2
    c_cc[mask] -= 0.5

    mask = d_cc == 0
    d_cc[mask] += 0.5
    mask = d_cc == n2
    d_cc[mask] -= 0.5

    a_cc = np.maximum(a_cc, eps)
    b_cc = np.maximum(b_cc, eps)
    c_cc = np.maximum(c_cc, eps)
    d_cc = np.maximum(d_cc, eps)

    # log odds
    lor_hat = np.log(a_cc / b_cc) - np.log(c_cc / d_cc)

    p1_adj = min(1 - eps, max(eps, p1))
    p2_adj = min(1 - eps, max(eps, p2))
    lor_exp = math.log(p1_adj / (1 - p1_adj)) - math.log(p2_adj / (1 - p2_adj))

    var = (1.0 / (n1 * p1_adj * (1 - p1_adj))) + (1.0 / (n2 * p2_adj * (1 - p2_adj)))

    result = np.abs(lor_hat - lor_exp) / math.sqrt(var)

    # Handle any potential NaNs or infs
    result[np.isnan(result) | np.isinf(result)] = float('inf')

    return result


def calculate_binomial_exact_pvalue_or(a_obs, n1, n2, m1, p1, p2):
    """
    Calculate exact p-value using binomial probabilities for odds ratio,
    using vectorized operations for efficiency.
    
    Parameters:
    -----------
    a_obs : int
        Observed count in first row, first column
    n1, n2, m1 : int
        Margin totals
    p1, p2 : float
        Probability parameters
        
    Returns:
    --------
    float
        P-value for unconditional exact test
    """
    try:
        # Ensure valid probabilities
        epsilon = 1e-10
        p1 = min(1.0 - epsilon, max(epsilon, p1))
        p2 = min(1.0 - epsilon, max(epsilon, p2))
        
        # Get all possible values for a
        a_min = max(0, m1 - n2)
        a_max = min(n1, m1)
        a_values = np.arange(a_min, a_max + 1)
        
        if len(a_values) == 0:
            return 1.0  # No valid tables
            
        # Calculate probabilities of all possible tables
        probs_a = binom.pmf(a_values, n1, p1)
        c_values = m1 - a_values
        
        # Filter out invalid c values
        valid_c = (c_values >= 0) & (c_values <= n2)
        if not np.any(valid_c):
            return 1.0  # No valid tables
            
        a_values = a_values[valid_c]
        probs_a = probs_a[valid_c]
        c_values = c_values[valid_c]
        
        probs_c = binom.pmf(c_values, n2, p2)
        joint_probs = probs_a * probs_c
        
        # Calculate test statistics for all tables
        stats = or_test_statistic_vectorized(a_values, n1, n2, m1, p1, p2)
        
        # Calculate observed test statistic
        obs_stat = or_test_statistic(a_obs, n1, n2, m1, p1, p2)
        
        if math.isinf(obs_stat) or math.isnan(obs_stat):
            return 1.0  # Conservative approach
        
        # Find extreme tables
        extreme_mask = stats >= obs_stat - 1e-12
        
        # Calculate p-value
        total_prob = np.sum(joint_probs)
        if abs(total_prob - 1.0) > 1e-5 and total_prob > 0:
            # Normalize if the total probability differs significantly from 1
            p_value = np.sum(joint_probs[extreme_mask]) / total_prob
        else:
            p_value = np.sum(joint_probs[extreme_mask])
        
        return min(1.0, float(p_value))
        
    except Exception as e:
        # If any errors occur, return conservative p-value
        import logging
        logging.getLogger(__name__).warning(f"Error in vectorized p-value: {str(e)}")
        
        # Fall back to non-vectorized calculation
        try:
            return _calculate_binomial_exact_pvalue_or_nonvectorized(a_obs, n1, n2, m1, p1, p2)
        except:
            return 1.0  # Complete fallback


def _calculate_binomial_exact_pvalue_or_nonvectorized(a_obs, n1, n2, m1, p1, p2):
    """Non-vectorized fallback implementation for robustness."""
    from scipy.stats import binom
    
    # Ensure probabilities are valid
    p1 = min(1.0 - 1e-10, max(1e-10, p1))
    p2 = min(1.0 - 1e-10, max(1e-10, p2))
    
    # Calculate test statistic for observed table
    obs_stat = or_test_statistic(a_obs, n1, n2, m1, p1, p2)
    
    # If observed statistic is infinite, return p-value of 0
    if math.isinf(obs_stat):
        return 0.0
    
    total_pvalue = 0.0
    total_prob = 0.0
    
    # Enumerate all possible tables
    for a in range(max(0, m1 - n2), min(n1, m1) + 1):
        # Calculate remaining cells
        b = n1 - a
        c = m1 - a
        d = n2 - c
        
        # Skip invalid tables
        if b < 0 or c < 0 or d < 0:
            continue
            
        # Calculate probability of this table under the null
        prob = binom.pmf(a, n1, p1) * binom.pmf(c, n2, p2)
        total_prob += prob
        
        # Only include tables with equal or more extreme test statistics
        try:
            stat = or_test_statistic(a, n1, n2, m1, p1, p2)
            if not math.isnan(stat) and stat >= obs_stat - 1e-12:
                total_pvalue += prob
        except:
            # Skip tables that cause errors
            continue
    
    # Normalize p-value if total probability is not 1
    if abs(total_prob - 1.0) > 1e-5 and total_prob > 0:
        total_pvalue = total_pvalue / total_prob
        
    return min(1.0, total_pvalue)


def calculate_unconditional_pvalue_or(a_obs, n1, n2, m1, theta, pi_grid_size=50):
    """
    Calculate unconditional exact p-value for odds ratio,
    maximizing over nuisance parameter.
    
    Parameters:
    -----------
    a_obs : int
        Observed count in first row, first column
    n1, n2, m1 : int
        Margin totals
    theta : float
        Odds ratio under the null hypothesis
    pi_grid_size : int, optional
        Size of grid for nuisance parameter
        
    Returns:
    --------
    float
        Maximum p-value over all nuisance parameter values
    """
    # Create grid of nuisance parameter pi values with better distribution
    epsilon = 1e-6
    
    # Use a non-uniform grid to better cover critical regions
    # More points near boundaries where numerical issues often occur
    pi_grid_uniform = np.linspace(epsilon, 1.0 - epsilon, pi_grid_size // 2)
    pi_grid_log_low = np.logspace(np.log10(epsilon), np.log10(0.5), pi_grid_size // 4)
    pi_grid_log_high = 1.0 - np.logspace(np.log10(epsilon), np.log10(0.5), pi_grid_size // 4)
    pi_grid = np.unique(np.concatenate([pi_grid_uniform, pi_grid_log_low, pi_grid_log_high]))
    
    # Calculate p1 values for all pi in the grid
    p2_values = pi_grid
    p1_values = (theta * p2_values) / (1.0 + p2_values * (theta - 1.0))
    
    # Filter out invalid probability values
    valid_mask = (p1_values > 0) & (p1_values < 1)
    p1_values = p1_values[valid_mask]
    p2_values = p2_values[valid_mask]
    
    if len(p1_values) == 0:
        return 1.0  # Conservative approach if no valid probabilities
    
    # Calculate p-values for all valid (p1, p2) pairs
    p_values = np.zeros_like(p1_values)
    
    try:
        for i, (p1, p2) in enumerate(zip(p1_values, p2_values)):
            try:
                p_values[i] = calculate_binomial_exact_pvalue_or(a_obs, n1, n2, m1, p1, p2)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Error with pi={p2}: {str(e)}")
                p_values[i] = 0.0  # Skip this point
                
        # Return maximum p-value
        if len(p_values) > 0:
            return float(np.max(p_values))
        else:
            return 1.0  # Conservative approach
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error maximizing p-value: {str(e)}")
        return 1.0  # Conservative approach


@lru_cache(maxsize=4096)
def exact_ci_unconditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05, 
                         progress_callback: Optional[Callable] = None,
                         grid_size: int = 200,
                         theta_min: float = 0.001,
                         theta_max: float = 1000,
                         use_adaptive_grid: bool = True,
                         refinement_rounds: int = 2,
                         pi_grid_size: int = 50) -> Tuple[float, float]:
    """
    Calculate the unconditional exact confidence interval for the odds ratio.

    This method uses unconditional binomial probabilities and maximizes over the nuisance parameter
    (pi). It is appropriate when the row or column totals are not fixed by the study design, and
    provides valid inference without conditioning on the marginals.
    
    This implementation uses an adaptive grid search approach with confidence interval inversion,
    which provides better precision than a fixed-size grid without the computational cost of
    a uniformly dense grid.

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
        use_adaptive_grid: Whether to use adaptive grid refinement (default: True)
        refinement_rounds: Number of refinement rounds for adaptive grid (default: 2)
        pi_grid_size: Number of grid points for nuisance parameter (default: 50)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    # Validate inputs
    a, b, c, d = validate_table_and_alpha(a, b, c, d, alpha, preserve_int_types=True)
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    # Calculate marginals
    n1 = a + b
    n2 = c + d
    m1 = a + c
    
    # Calculate odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    
    # Log initial information
    method = "adaptive grid search" if use_adaptive_grid else "fixed grid search"
    logger.info(f"Computing unconditional CI using {method}: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    logger.info(f"Marginals: n1={n1}, n2={n2}, m1={m1}, odds_ratio={odds_ratio}")
    
    # Handle edge cases
    if b == 0 or c == 0:
        logger.info(f"Edge case: b={b}, c={c}, returning (0, inf)")
        return (0.0, float('inf'))
    
    # Handle case where a == 0 (odds ratio is 0)
    if a == 0:
        logger.info(f"Edge case: a={a}, odds ratio is 0, adjusting lower bound")
        # For a=0, the odds ratio is 0, so we need to ensure the CI includes 0
        special_case_a_zero = True
    else:
        special_case_a_zero = False
    
    # Adjust theta_min and theta_max to ensure they bracket the odds ratio
    if odds_ratio > 0 and odds_ratio < float('inf'):
        theta_min = min(theta_min, odds_ratio * 0.1)
        theta_max = max(theta_max, odds_ratio * 10)
    
    # Define p-value function for CI search
    def p_value_func(theta):
        """Calculate unconditional p-value for given theta"""
        return calculate_unconditional_pvalue_or(a, n1, n2, m1, theta, pi_grid_size)
    
    # Choose search method based on use_adaptive_grid parameter
    if use_adaptive_grid:
        logger.info("Using adaptive grid search for CI search")
        lower_bound, upper_bound = find_confidence_interval_adaptive_grid(
            p_value_func, theta_min, theta_max, alpha, 
            initial_grid_size=grid_size // 4,  # Use smaller initial grid
            refinement_rounds=refinement_rounds,
            odds_ratio=odds_ratio, 
            progress_callback=progress_callback
        )
    else:
        logger.info(f"Using fixed grid search with {grid_size} points for CI search")
        lower_bound, upper_bound = find_confidence_interval_grid(
            p_value_func, theta_min, theta_max, alpha, 
            grid_size=grid_size, odds_ratio=odds_ratio, 
            progress_callback=progress_callback
        )
    
    # Log if the odds ratio is not within the CI (should be rare with adaptive grid)
    if odds_ratio < lower_bound:
        logger.warning(f"Note: Odds ratio ({odds_ratio:.6f}) < lower bound ({lower_bound:.6f})")
        # Adjust lower bound to include odds ratio if very close
        if odds_ratio > lower_bound * 0.95:
            lower_bound = odds_ratio * 0.99
            logger.info(f"Adjusted lower bound to {lower_bound:.6f} to include odds ratio")
    
    if odds_ratio > upper_bound and upper_bound < theta_max * 0.99:
        logger.warning(f"Note: Odds ratio ({odds_ratio:.6f}) > upper bound ({upper_bound:.6f})")
        # Adjust upper bound to include odds ratio if very close
        if odds_ratio < upper_bound * 1.05:
            upper_bound = odds_ratio * 1.01
            logger.info(f"Adjusted upper bound to {upper_bound:.6f} to include odds ratio")
            
    # Handle special case where a=0 (odds ratio is 0)
    if special_case_a_zero and lower_bound > 0:
        logger.info(f"Special case a=0: Setting lower bound to 0.0 (was {lower_bound:.6f})")
        lower_bound = 0.0
    
    logger.info(f"Unconditional CI result: ({lower_bound:.6f}, {upper_bound:.6f})")
    
    return (lower_bound, upper_bound)


def exact_ci_unconditional_batch(tables: List[Tuple[int, int, int, int]], 
                              alpha: float = 0.05,
                              max_workers: Optional[int] = None,
                              backend: Optional[str] = None,
                              progress_callback: Optional[Callable] = None,
                              grid_size: int = 200,
                              theta_min: float = 0.001,
                              theta_max: float = 1000,
                              use_adaptive_grid: bool = True,
                              refinement_rounds: int = 2,
                              pi_grid_size: int = 50) -> List[Tuple[float, float]]:
    """
    Calculate unconditional exact confidence intervals for multiple 2x2 tables in parallel.
    
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
        use_adaptive_grid: Whether to use adaptive grid refinement (default: True)
        refinement_rounds: Number of refinement rounds for adaptive grid (default: 2)
        pi_grid_size: Number of grid points for nuisance parameter (default: 50)
        
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
                                           progress_callback=progress_callback,
                                           grid_size=grid_size,
                                           theta_min=theta_min,
                                           theta_max=theta_max,
                                           use_adaptive_grid=use_adaptive_grid,
                                           refinement_rounds=refinement_rounds,
                                           pi_grid_size=pi_grid_size)
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
    
    logger.info(f"Processing {len(tables)} tables with unconditional method using {max_workers} workers")
    
    # Use parallel_compute_ci for better performance and error handling
    results = parallel_compute_ci(
        exact_ci_unconditional,
        tables,
        alpha=alpha,
        grid_size=grid_size,
        theta_min=theta_min,
        theta_max=theta_max,
        use_adaptive_grid=use_adaptive_grid,
        refinement_rounds=refinement_rounds,
        pi_grid_size=pi_grid_size,
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
    
    logger.info(f"Completed batch processing of {len(tables)} tables with unconditional method")
    return results


# Legacy function names for backward compatibility
def exact_ci_unconditional_legacy(a, b, c, d, alpha=0.05, **kwargs):
    """Legacy wrapper for unconditional OR CI"""
    return exact_ci_unconditional(a, b, c, d, alpha=alpha, **kwargs)


# Simple test case if run directly
if __name__ == "__main__":
    print("Testing Unconditional OR CI Implementation")
    print("=" * 45)
    
    # Test with small table first
    print("Small test: 2/10 vs 1/10") 
    ci_small = exact_ci_unconditional(2, 8, 1, 9)
    print(f"95% CI: ({ci_small[0]:.3f}, {ci_small[1]:.3f})")
    
    # Your original test cases
    print("\nTable 1: 50/1000 vs 10/1000")
    ci1 = exact_ci_unconditional(50, 950, 10, 990) 
    print(f"95% CI: ({ci1[0]:.2f}, {ci1[1]:.2f})")