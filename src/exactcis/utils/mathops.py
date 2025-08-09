"""
Safe mathematical operations for ExactCIs.

This module provides numerically stable mathematical operations used across
confidence interval methods, with proper handling of edge cases, overflow,
underflow, and division by zero.
"""

import math
import logging
from typing import Union, Tuple, Optional, List
from functools import lru_cache

from exactcis.constants import (
    SMALL_POS, LARGE_POS, LOG_SMALL, LOG_LARGE, 
    MIN_PROB, MAX_PROB, EPS, TOL
)

logger = logging.getLogger(__name__)


# Basic safe operations
def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with fallback for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Value to return if denominator is zero
        
    Returns:
        numerator/denominator if safe, default otherwise
    """
    if abs(denominator) < EPS:
        return default
    return numerator / denominator


def safe_log(value: float, default: float = LOG_SMALL) -> float:
    """
    Safely compute logarithm with fallback for non-positive values.
    
    Args:
        value: Value to take logarithm of
        default: Value to return for non-positive input
        
    Returns:
        log(value) if value > 0, default otherwise
    """
    if value <= 0:
        return default
    return math.log(value)


def safe_log_ratio(numerator: float, denominator: float, 
                  default_log: float = LOG_SMALL) -> float:
    """
    Safely compute log(numerator/denominator) with proper edge case handling.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default_log: Log value to return for invalid ratios
        
    Returns:
        log(numerator/denominator) if both positive, default_log otherwise
    """
    if numerator <= 0 or denominator <= 0:
        return default_log
    
    # Use log subtraction for numerical stability
    try:
        return math.log(numerator) - math.log(denominator)
    except (ValueError, OverflowError):
        return default_log


def exp_safe(log_value: float, max_exp: float = LOG_LARGE) -> float:
    """
    Safely compute exponential with overflow protection.
    
    Args:
        log_value: Value in log space
        max_exp: Maximum log value to exponentiate
        
    Returns:
        exp(log_value) if safe, appropriate boundary value otherwise
    """
    if log_value <= LOG_SMALL:
        return 0.0
    elif log_value >= max_exp:
        return float('inf') if log_value > LOG_LARGE else LARGE_POS
    else:
        return math.exp(log_value)


def clip_probability(p: float, min_p: float = MIN_PROB, max_p: float = MAX_PROB) -> float:
    """
    Clip probability to valid range avoiding numerical issues at boundaries.
    
    Args:
        p: Probability value to clip
        min_p: Minimum allowed probability
        max_p: Maximum allowed probability
        
    Returns:
        Clipped probability in [min_p, max_p]
    """
    return max(min_p, min(max_p, p))


# Log-space arithmetic for numerical stability
def log_sum_exp(log_values: List[float]) -> float:
    """
    Compute log(sum(exp(log_values))) in a numerically stable way.
    
    This is the standard log-sum-exp trick for avoiding overflow.
    
    Args:
        log_values: List of values in log space
        
    Returns:
        log(sum(exp(x) for x in log_values))
    """
    if not log_values:
        return LOG_SMALL
        
    max_log = max(log_values)
    if max_log <= LOG_SMALL:
        return LOG_SMALL
    
    # Subtract max to prevent overflow
    sum_exp = sum(math.exp(log_val - max_log) for log_val in log_values)
    return max_log + math.log(sum_exp)


def log_diff_exp(log_a: float, log_b: float) -> float:
    """
    Compute log(exp(log_a) - exp(log_b)) in a numerically stable way.
    
    Assumes exp(log_a) >= exp(log_b), i.e., log_a >= log_b.
    
    Args:
        log_a: First value in log space (should be >= log_b)
        log_b: Second value in log space
        
    Returns:
        log(exp(log_a) - exp(log_b))
    """
    if log_a <= log_b:
        return LOG_SMALL
    
    if log_b <= LOG_SMALL:
        return log_a
    
    # Use the identity: log(exp(a) - exp(b)) = a + log(1 - exp(b-a))
    diff = log_b - log_a
    if diff < -10:  # exp(diff) is negligible
        return log_a
    
    return log_a + math.log(1 - math.exp(diff))


def log_mean_exp(log_values: List[float]) -> float:
    """
    Compute log(mean(exp(log_values))) in a numerically stable way.
    
    Args:
        log_values: List of values in log space
        
    Returns:
        log(mean(exp(x) for x in log_values))
    """
    if not log_values:
        return LOG_SMALL
    
    return log_sum_exp(log_values) - math.log(len(log_values))


# Statistical variance calculations
def wald_variance_or(a: Union[int, float], b: Union[int, float], 
                    c: Union[int, float], d: Union[int, float],
                    haldane_corrected: bool = False) -> float:
    """
    Calculate Wald variance for log odds ratio.
    
    This is the standard formula: Var(log(OR)) = 1/a + 1/b + 1/c + 1/d
    
    Args:
        a, b, c, d: Cell counts
        haldane_corrected: Whether counts are already Haldane-corrected
        
    Returns:
        Variance of log odds ratio
    """
    if haldane_corrected:
        # Counts already have correction applied
        cells = [a, b, c, d]
    else:
        # Apply Haldane correction
        cells = [a + 0.5, b + 0.5, c + 0.5, d + 0.5]
    
    # Check for non-positive values
    if any(cell <= 0 for cell in cells):
        return float('inf')
    
    return sum(1.0 / cell for cell in cells)


def wald_variance_rr(a: Union[int, float], b: Union[int, float],
                    c: Union[int, float], d: Union[int, float], 
                    method: str = "standard") -> float:
    """
    Calculate Wald variance for log relative risk.
    
    Different methods handle zero cells differently:
    - "standard": Var(log(RR)) = b/(a*(a+b)) + d/(c*(c+d))
    - "katz": Uses Katz adjustment for zero cells
    - "correlated": For matched pairs/correlated data
    
    Args:
        a, b, c, d: Cell counts  
        method: Variance calculation method
        
    Returns:
        Variance of log relative risk
    """
    if method == "standard":
        n1, n2 = a + b, c + d
        if n1 <= 0 or n2 <= 0 or a <= 0 or c <= 0:
            return float('inf')
        return (b / (a * n1)) + (d / (c * n2))
    
    elif method == "katz":
        # Katz method: add 0.5 to cells with zeros
        a_adj = a + 0.5 if a == 0 else a
        c_adj = c + 0.5 if c == 0 else c
        n1, n2 = a_adj + b, c_adj + d
        
        if n1 <= 0 or n2 <= 0:
            return float('inf')
        return (b / (a_adj * n1)) + (d / (c_adj * n2))
    
    elif method == "correlated":
        # For matched pairs - different variance formula
        # This is a simplified version; actual implementation depends on study design
        if a + c <= 0:
            return float('inf')
        return (b + d) / ((a + c) ** 2)
    
    else:
        raise ValueError(f"Unknown variance method: {method}")


def pooled_variance_estimate(n1: float, n2: float, p_pooled: float) -> float:
    """
    Calculate pooled variance estimate for two proportions.
    
    Used in some statistical tests and confidence intervals.
    
    Args:
        n1: Sample size for group 1
        n2: Sample size for group 2  
        p_pooled: Pooled proportion estimate
        
    Returns:
        Pooled variance estimate
    """
    if n1 <= 0 or n2 <= 0:
        return float('inf')
    
    p_pooled = clip_probability(p_pooled)
    q_pooled = 1 - p_pooled
    
    return p_pooled * q_pooled * (1/n1 + 1/n2)


# Numerical stability helpers
@lru_cache(maxsize=1000)
def cached_log_factorial(n: int) -> float:
    """
    Cached logarithm of factorial for efficiency.
    
    Args:
        n: Non-negative integer
        
    Returns:
        log(n!)
    """
    if n <= 0:
        return 0.0
    return sum(math.log(i) for i in range(1, n + 1))


def stirling_log_factorial(n: float) -> float:
    """
    Stirling's approximation for log(n!) for large n.
    
    More accurate than direct calculation for large values.
    
    Args:
        n: Value to approximate factorial of
        
    Returns:
        Stirling approximation of log(n!)
    """
    if n <= 0:
        return 0.0
    if n < 10:
        # Use exact calculation for small n
        return cached_log_factorial(int(n))
    
    # Stirling: log(n!) ≈ n*log(n) - n + 0.5*log(2πn)
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


def relative_error(expected: float, actual: float) -> float:
    """
    Calculate relative error between expected and actual values.
    
    Args:
        expected: Expected value
        actual: Actual value
        
    Returns:
        Relative error |expected - actual| / |expected|
    """
    if abs(expected) < EPS:
        return abs(actual)  # Absolute error when expected is near zero
    return abs(expected - actual) / abs(expected)


def is_numerically_close(a: float, b: float, rel_tol: float = TOL, 
                        abs_tol: float = EPS) -> bool:
    """
    Check if two values are numerically close within tolerances.
    
    Args:
        a, b: Values to compare
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        
    Returns:
        True if values are close within tolerances
    """
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))