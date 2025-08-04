"""
JIT-compiled functions for performance-critical calculations in ExactCIs.

This module contains Numba-accelerated versions of core mathematical functions
used in confidence interval calculations. These functions provide significant
performance improvements over their pure Python counterparts.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Union
from functools import lru_cache

# Set up logging
logger = logging.getLogger(__name__)

# Try to import Numba
try:
    from numba import njit, float64, int64
    HAS_NUMBA = True
    logger.info("Numba is available for JIT compilation")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba is not available, using slower Python implementations")
    # Create dummy decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


# JIT-compiled version of nchg_pdf
@njit
def _nchg_pdf_jit(k_values: np.ndarray, n1: int, n2: int, m1: int, theta: float) -> np.ndarray:
    """
    JIT-compiled implementation of noncentral hypergeometric PDF calculation.
    
    This function calculates the probability mass function for multiple k values
    using direct computation rather than calling the pmf function for each k.
    
    Args:
        k_values: Array of integers at which to evaluate the PMF
        n1: Size of first group (row 1 total)
        n2: Size of second group (row 2 total)
        m1: Number of successes in the first column (column 1 total)
        theta: Odds ratio parameter
        
    Returns:
        Array of probabilities corresponding to each k in k_values
    """
    # Calculate support bounds
    k_min = max(0, m1 - n2)
    k_max = min(m1, n1)
    
    # Initialize result array
    result = np.zeros(len(k_values))
    
    # Calculate log terms for all k values in support
    log_terms = np.zeros(k_max - k_min + 1)
    for i, k in enumerate(range(k_min, k_max + 1)):
        # Calculate binomial coefficients
        log_binom1 = _log_binom_coeff_jit(n1, k)
        log_binom2 = _log_binom_coeff_jit(n2, m1 - k)
        
        # Calculate log probability
        log_prob = log_binom1 + log_binom2 + k * np.log(theta)
        log_terms[i] = log_prob
    
    # Normalize to get probabilities (shift to avoid overflow)
    max_log_term = np.max(log_terms)
    terms = np.exp(log_terms - max_log_term)
    sum_terms = np.sum(terms)
    probs = terms / sum_terms
    
    # Map k_values to indices in the support
    for i, k in enumerate(k_values):
        if k_min <= k <= k_max:
            idx = int(k - k_min)
            result[i] = probs[idx]
    
    return result


@njit
def _log_binom_coeff_jit(n: int, k: int) -> float:
    """
    JIT-compiled implementation of log binomial coefficient calculation.
    
    Args:
        n: Number of items
        k: Number of items to choose
        
    Returns:
        Natural logarithm of the binomial coefficient
    """
    if k < 0 or k > n:
        return float('-inf')  # Invalid input
    
    if k == 0 or k == n:
        return 0.0  # log(1) = 0
    
    # Use symmetry to reduce computation
    if k > n - k:
        k = n - k
    
    # Calculate log factorial differences
    log_binom = 0.0
    for i in range(1, k + 1):
        log_binom += np.log(n - k + i) - np.log(i)
    
    return log_binom


# Wrapper function that uses JIT implementation if available
def nchg_pdf_jit(k_values: np.ndarray, n1: int, n2: int, m1: int, theta: float) -> np.ndarray:
    """
    Calculate noncentral hypergeometric PDF using JIT compilation if available.
    
    This function is a drop-in replacement for the original nchg_pdf function
    that uses Numba JIT compilation for better performance.
    
    Args:
        k_values: Array of integers at which to evaluate the PMF
        n1: Size of first group (row 1 total)
        n2: Size of second group (row 2 total)
        m1: Number of successes in the first column (column 1 total)
        theta: Odds ratio parameter
        
    Returns:
        Array of probabilities corresponding to each k in k_values
    """
    # Convert inputs to appropriate types for Numba
    k_values_array = np.array(k_values, dtype=np.int64)
    n1_int = int(n1)
    n2_int = int(n2)
    m1_int = int(m1)
    theta_float = float(theta)
    
    return _nchg_pdf_jit(k_values_array, n1_int, n2_int, m1_int, theta_float)


# Cached version of the JIT-compiled function
@lru_cache(maxsize=2048)
def nchg_pdf_cached(n1: int, n2: int, m1: int, theta_rounded: float, support_tuple: tuple) -> tuple:
    """
    Cached version of the JIT-compiled nchg_pdf function.
    
    This function caches the results of nchg_pdf_jit for better performance
    when the same parameters are used multiple times.
    
    Args:
        n1: Size of first group (row 1 total)
        n2: Size of second group (row 2 total)
        m1: Number of successes in the first column (column 1 total)
        theta_rounded: Rounded odds ratio parameter for cache stability
        support_tuple: Tuple of support values
        
    Returns:
        Tuple of probabilities corresponding to each k in support_tuple
    """
    support_array = np.array(support_tuple, dtype=np.int64)
    probs = nchg_pdf_jit(support_array, n1, n2, m1, theta_rounded)
    return tuple(probs.tolist())


# JIT-compiled function for p-value summation logic in blaker_p_value
@njit
def _sum_probs_less_than_threshold_jit(probs: np.ndarray, idx_a: int, epsilon: float = 1e-7) -> float:
    """
    JIT-compiled function to sum probabilities less than or equal to a threshold.
    
    This function implements the core summation logic from blaker_p_value:
    Sum probabilities for k where P(k|theta) <= P(a|theta) * (1 + epsilon)
    
    Args:
        probs: Array of probabilities for all k in support
        idx_a: Index of the observed value 'a' in the support array
        epsilon: Small tolerance factor for floating point comparisons
        
    Returns:
        Sum of probabilities that meet the condition
    """
    # Get the probability at the observed value 'a'
    if 0 <= idx_a < len(probs):
        prob_at_a = probs[idx_a]
    else:
        # Invalid index, return 1.0 (same behavior as in blaker_p_value)
        return 1.0
    
    # Calculate threshold
    threshold = prob_at_a * (1.0 + epsilon)
    
    # Sum probabilities that meet the condition
    p_val = 0.0
    for i in range(len(probs)):
        if probs[i] <= threshold:
            p_val += probs[i]
    
    return p_val


def sum_probs_less_than_threshold(probs: np.ndarray, idx_a: int, epsilon: float = 1e-7) -> float:
    """
    Sum probabilities less than or equal to a threshold using JIT compilation if available.
    
    This function is a drop-in replacement for the summation logic in blaker_p_value
    that uses Numba JIT compilation for better performance.
    
    Args:
        probs: Array of probabilities for all k in support
        idx_a: Index of the observed value 'a' in the support array
        epsilon: Small tolerance factor for floating point comparisons
        
    Returns:
        Sum of probabilities that meet the condition
    """
    # Convert inputs to appropriate types for Numba
    probs_array = np.array(probs, dtype=np.float64)
    idx_a_int = int(idx_a)
    epsilon_float = float(epsilon)
    
    return _sum_probs_less_than_threshold_jit(probs_array, idx_a_int, epsilon_float)