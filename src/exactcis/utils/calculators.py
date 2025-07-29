"""
Pure calculation functions for ExactCIs.
"""

import math
import logging
from functools import lru_cache
from typing import Optional, Callable, List
from .data_models import TableData, GridConfig, BoundResult
from .transformers import calculate_p2_from_theta
from ..core import log_binom_coeff, logsumexp, find_sign_change, find_plateau_edge

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def log_binom_pmf_cached(n: int, k: int, p: float) -> float:
    """
    Calculate log of binomial PMF with caching.
    
    Args:
        n: Number of trials
        k: Number of successes
        p: Probability of success
        
    Returns:
        Log probability mass
    """
    if p <= 0 or p >= 1:
        return float('-inf')
    
    log_choose = log_binom_coeff(n, k)
    log_p_term = k * math.log(p)
    log_1mp_term = (n - k) * math.log(1 - p)
    
    return log_choose + log_p_term + log_1mp_term


def calculate_log_probability_for_table(
    table: TableData, 
    p1: float, 
    p2: float
) -> float:
    """
    Calculate log probability for a specific table configuration.
    
    Args:
        table: Table data
        p1: First probability parameter
        p2: Second probability parameter
        
    Returns:
        Log probability
    """
    return (log_binom_pmf_cached(table.n1, table.a, p1) + 
            log_binom_pmf_cached(table.n2, table.c, p2))


def enumerate_all_possible_tables(n1: int, n2: int) -> List[TableData]:
    """
    Enumerate all possible 2x2 tables with given marginals.
    
    Args:
        n1: First margin total
        n2: Second margin total
        
    Returns:
        List of all possible tables
    """
    tables = []
    for a in range(n1 + 1):
        for c in range(n2 + 1):
            b = n1 - a
            d = n2 - c
            tables.append(TableData(a, b, c, d))
    return tables


def calculate_unconditional_log_pvalue_for_grid_point(
    observed_table: TableData,
    theta: float,
    p1: float
) -> float:
    """
    Calculate log p-value for a single grid point.
    
    Args:
        observed_table: Observed table data
        theta: Odds ratio parameter
        p1: First probability parameter
        
    Returns:
        Log p-value for this grid point
    """
    p2 = calculate_p2_from_theta(p1, theta)
    
    # Calculate log probability for observed table
    log_p_obs = calculate_log_probability_for_table(observed_table, p1, p2)
    
    # Calculate probabilities for all possible tables
    log_probs = []
    for possible_table in enumerate_all_possible_tables(observed_table.n1, observed_table.n2):
        log_p_table = calculate_log_probability_for_table(possible_table, p1, p2)
        
        if log_p_table <= log_p_obs:
            log_probs.append(log_p_table)
    
    return logsumexp(log_probs) if log_probs else float('-inf')


def calculate_unconditional_log_pvalue(
    table: TableData, 
    theta: float, 
    grid: GridConfig,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Optional[float]:
    """
    Calculate log p-value for unconditional test across all grid points.
    
    Args:
        table: Table data
        theta: Odds ratio parameter
        grid: Grid configuration
        timeout_checker: Optional timeout checking function
        
    Returns:
        Maximum log p-value across grid points, or None if timeout
    """
    if timeout_checker and timeout_checker():
        return None
    
    results = []
    for p1 in grid.p1_values:
        if timeout_checker and timeout_checker():
            return None
            
        result = calculate_unconditional_log_pvalue_for_grid_point(table, theta, p1)
        if result > float('-inf'):
            results.append(result)
    
    return max(results) if results else float('-inf')


def create_pvalue_function(
    table: TableData,
    grid: GridConfig,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Callable[[float], float]:
    """
    Create a p-value function for a given table and grid.
    
    Args:
        table: Table data
        grid: Grid configuration
        timeout_checker: Optional timeout checking function
        
    Returns:
        Function that calculates p-value for given theta
    """
    def pvalue_function(theta: float) -> float:
        log_pval = calculate_unconditional_log_pvalue(table, theta, grid, timeout_checker)
        if log_pval is None:
            return float('nan')  # Signal timeout
        return math.exp(log_pval)
    
    return pvalue_function


def find_confidence_bound_via_sign_change(
    pvalue_function: Callable[[float], float],
    search_range: tuple[float, float],
    alpha: float,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Optional[float]:
    """
    Find confidence bound using sign change detection.
    
    Args:
        pvalue_function: Function that calculates p-values
        search_range: (lo, hi) search range
        alpha: Significance level
        timeout_checker: Optional timeout checking function
        
    Returns:
        Confidence bound or None if not found
    """
    def search_function(theta: float) -> float:
        return pvalue_function(theta) - alpha
    
    search_lo, search_hi = search_range
    
    try:
        return find_sign_change(
            search_function,
            search_lo,
            search_hi,
            timeout_checker=timeout_checker
        )
    except Exception as e:
        logger.warning(f"Sign change detection failed: {e}")
        return None


def refine_bound_with_plateau_detection(
    pvalue_function: Callable[[float], float],
    initial_bound: float,
    alpha: float,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Optional[float]:
    """
    Refine bound using plateau edge detection.
    
    Args:
        pvalue_function: Function that calculates p-values
        initial_bound: Initial bound estimate
        alpha: Significance level
        timeout_checker: Optional timeout checking function
        
    Returns:
        Refined bound or None if refinement failed
    """
    try:
        plateau_result = find_plateau_edge(
            pvalue_function,
            lo=max(1e-10, initial_bound * 0.98),
            hi=initial_bound * 1.02,
            target=alpha,
            timeout_checker=timeout_checker
        )
        
        if plateau_result is not None:
            return plateau_result[0]  # First element is the result value
        
        return None
        
    except Exception as e:
        logger.warning(f"Plateau detection failed: {e}")
        return None


def find_confidence_bound(
    table: TableData,
    grid: GridConfig,
    search_range: tuple[float, float],
    alpha: float,
    bound_type: str,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> BoundResult:
    """
    Find confidence interval bound using multiple strategies.
    
    Args:
        table: Table data
        grid: Grid configuration
        search_range: (lo, hi) search range
        alpha: Significance level
        bound_type: "lower" or "upper"
        timeout_checker: Optional timeout checking function
        
    Returns:
        Bound result with value, iterations, and method
    """
    pvalue_function = create_pvalue_function(table, grid, timeout_checker)
    
    # Try sign change detection first
    bound_value = find_confidence_bound_via_sign_change(
        pvalue_function, search_range, alpha, timeout_checker
    )
    
    if bound_value is not None:
        # Try to refine with plateau detection
        refined_bound = refine_bound_with_plateau_detection(
            pvalue_function, bound_value, alpha, timeout_checker
        )
        
        if refined_bound is not None:
            return BoundResult(refined_bound, 1, "plateau_refined")
        
        return BoundResult(bound_value, 1, "sign_change")
    
    # Fallback to conservative estimate
    search_lo, search_hi = search_range
    fallback_value = search_lo if bound_type == "lower" else search_hi
    
    logger.warning(f"Using fallback {bound_type} bound: {fallback_value}")
    return BoundResult(fallback_value, 0, "fallback")