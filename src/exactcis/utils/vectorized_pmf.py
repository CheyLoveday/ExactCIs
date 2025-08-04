"""
Vectorized PMF calculations for improved performance.

This module provides vectorized implementations of PMF calculations
using NumPy for better performance with large tables.
"""

import numpy as np
from typing import Union, Tuple, Optional
import logging

from ..core import log_binom_coeff, SupportData

logger = logging.getLogger(__name__)

def vectorized_log_binom_coeff(n: Union[int, float], k: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of log binomial coefficients.
    
    Args:
        n: Size parameter
        k: Array of values for which to calculate binomial coefficients
        
    Returns:
        Array of log binomial coefficients
    """
    result = np.zeros_like(k, dtype=float)
    
    # Use NumPy's vectorized operations
    valid_mask = (k >= 0) & (k <= n)
    invalid_mask = ~valid_mask
    
    if np.any(invalid_mask):
        result[invalid_mask] = float('-inf')
    
    if np.any(valid_mask):
        k_valid = k[valid_mask]
        
        # Use log-gamma for numerical stability
        log_n_factorial = np.sum(np.log(np.arange(1, n + 1)))
        log_k_factorial = np.array([np.sum(np.log(np.arange(1, ki + 1))) if ki > 0 else 0 for ki in k_valid])
        log_n_minus_k_factorial = np.array([np.sum(np.log(np.arange(1, n - ki + 1))) if n - ki > 0 else 0 for ki in k_valid])
        
        result[valid_mask] = log_n_factorial - log_k_factorial - log_n_minus_k_factorial
    
    return result

def vectorized_log_nchg_pmf(k: np.ndarray, n1: int, n2: int, m1: int, theta: float) -> np.ndarray:
    """
    Vectorized calculation of log noncentral hypergeometric PMF.
    
    Args:
        k: Array of values at which to evaluate the PMF
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes
        theta: Odds ratio parameter
        
    Returns:
        Array of log PMF values
    """
    # Calculate log binomial terms
    log_comb_n1_k = vectorized_log_binom_coeff(n1, k)
    log_comb_n2_m1_k = vectorized_log_binom_coeff(n2, m1 - k)
    
    # Calculate log PMF
    log_theta = np.log(theta) if theta > 0 else float('-inf')
    log_pmf = log_comb_n1_k + log_comb_n2_m1_k + k * log_theta
    
    # Normalize to avoid numerical issues
    max_log_pmf = np.max(log_pmf[np.isfinite(log_pmf)])
    if not np.isfinite(max_log_pmf):
        return np.zeros_like(k, dtype=float)
    
    # Exponentiate and normalize
    pmf = np.exp(log_pmf - max_log_pmf)
    total = np.sum(pmf)
    
    if total > 0:
        pmf = pmf / total
    else:
        pmf = np.zeros_like(k, dtype=float)
    
    return pmf

def batch_pmf_calculation(support_array: np.ndarray, n1: int, n2: int, 
                         m1: int, theta: float) -> np.ndarray:
    """
    Calculate PMF for entire support array at once.
    
    Args:
        support_array: Array of values at which to evaluate the PMF
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes
        theta: Odds ratio parameter
        
    Returns:
        Array of PMF values
    """
    # Handle special cases
    if theta <= 0:
        # All mass at minimum value
        min_val = np.min(support_array)
        return np.array([1.0 if k == min_val else 0.0 for k in support_array])
    
    if theta >= 1e6 or np.isinf(theta):
        # All mass at maximum value
        max_val = np.max(support_array)
        return np.array([1.0 if k == max_val else 0.0 for k in support_array])
    
    # Use vectorized calculation
    return vectorized_log_nchg_pmf(support_array, n1, n2, m1, theta)

def optimized_nchg_pdf(support_array: np.ndarray, n1: int, n2: int, 
                      m1: int, theta: float) -> np.ndarray:
    """
    Optimized version of nchg_pdf using vectorized calculations.
    
    This function is a drop-in replacement for nchg_pdf in core.py.
    
    Args:
        support_array: Array of values at which to evaluate the PMF
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes
        theta: Odds ratio parameter
        
    Returns:
        Array of PMF values
    """
    return batch_pmf_calculation(support_array, n1, n2, m1, theta)