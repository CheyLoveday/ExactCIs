#!/usr/bin/env python3
"""
Utility functions for probability calculations.

This module provides functions for calculating various probability distributions,
especially for binomial probabilities used in exact confidence interval methods.
"""

import math
import numpy as np
from scipy.stats import binom
from typing import Union, Tuple, Optional, List

# Check if numba is available
try:
    import numba as nb
    has_numba = True
except ImportError:
    has_numba = False


def joint_binom_pmf(a_values: np.ndarray, n1: int, p1: float,
                    c_values: np.ndarray, n2: int, p2: float) -> np.ndarray:
    """
    Calculate joint binomial probability mass function for two independent binomials.
    
    Uses numpy vectorization for efficient computation of P(A=a) * P(C=c) for all
    combinations of a and c values.
    
    Parameters
    ----------
    a_values : numpy.ndarray
        Array of values for variable A
    n1 : int
        Size parameter for first binomial
    p1 : float
        Probability parameter for first binomial
    c_values : numpy.ndarray
        Array of values for variable C
    n2 : int
        Size parameter for second binomial
    p2 : float
        Probability parameter for second binomial
        
    Returns
    -------
    numpy.ndarray
        2D array of joint probabilities with shape (len(a_values), len(c_values))
    """
    log_pmf1 = binom.logpmf(a_values, n1, p1)
    log_pmf2 = binom.logpmf(c_values, n2, p2)
    
    # Use broadcasting to create the joint probability matrix
    return np.exp(log_pmf1[:, np.newaxis] + log_pmf2[np.newaxis, :])


def calculate_joint_probs_filtered(a_values: np.ndarray, n1: int, p1: float,
                                 m1: int, n2: int, p2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate joint probabilities, filtering out invalid tables.
    
    This computes P(A=a) * P(C=c) for all valid combinations where:
    - a is in a_values
    - c = m1 - a
    - c is between 0 and n2
    
    Parameters
    ----------
    a_values : numpy.ndarray
        Array of values for variable A
    n1 : int
        Size parameter for first binomial
    p1 : float
        Probability parameter for first binomial
    m1 : int
        Column 1 total (constrains c = m1 - a)
    n2 : int
        Size parameter for second binomial
    p2 : float
        Probability parameter for second binomial
        
    Returns
    -------
    tuple
        (filtered_a, filtered_c, joint_probs)
        where filtered_a and filtered_c are the valid values, and
        joint_probs contains their joint probabilities
    """
    # Calculate c values from a values
    c_values = m1 - a_values
    
    # Filter out invalid c values
    valid_c = (c_values >= 0) & (c_values <= n2)
    
    if not np.any(valid_c):
        return np.array([]), np.array([]), np.array([])
    
    a_filtered = a_values[valid_c]
    c_filtered = c_values[valid_c]
    
    # Calculate probabilities for valid tables
    probs_a = binom.pmf(a_filtered, n1, p1)
    probs_c = binom.pmf(c_filtered, n2, p2)
    joint_probs = probs_a * probs_c
    
    return a_filtered, c_filtered, joint_probs


# Numba-accelerated version of joint binomial calculations
if has_numba:
    @nb.njit(cache=True, fastmath=True)
    def _log_binom_coeff(n: int, k: int) -> float:
        """
        Calculate log binomial coefficient log(n choose k) using log-gamma.
        
        This is a pure-Python implementation that works with numba.
        
        Parameters
        ----------
        n : int
            Total number of items
        k : int
            Number of items to choose
            
        Returns
        -------
        float
            Log of binomial coefficient
        """
        if k < 0 or k > n:
            return -1e308  # -inf substitute
        if k == 0 or k == n:
            return 0.0
        return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)

    @nb.njit(cache=True, fastmath=True)
    def _log_binom_pmf(k: int, n: int, p: float) -> float:
        """
        Calculate log of binomial PMF without using scipy.
        
        Parameters
        ----------
        k : int
            Number of successes
        n : int
            Number of trials
        p : float
            Success probability
            
        Returns
        -------
        float
            Log of binomial probability
        """
        if k < 0 or k > n:
            return -1e308
        if p == 0.0:
            return 0.0 if k == 0 else -1e308
        if p == 1.0:
            return 0.0 if k == n else -1e308
        return _log_binom_coeff(n, k) + k*math.log(p) + (n-k)*math.log1p(-p)

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def joint_binom_pmf_numba(a_values: np.ndarray, n1: int, p1: float,
                              c_values: np.ndarray, n2: int, p2: float) -> np.ndarray:
        """
        Numba-accelerated joint binomial PMF calculation.
        
        Parameters
        ----------
        a_values : numpy.ndarray
            Array of values for variable A
        n1 : int
            Size parameter for first binomial
        p1 : float
            Probability parameter for first binomial
        c_values : numpy.ndarray
            Array of values for variable C
        n2 : int
            Size parameter for second binomial
        p2 : float
            Probability parameter for second binomial
            
        Returns
        -------
        numpy.ndarray
            2D array of joint probabilities
        """
        # Pre-compute 1-D log pmfs
        logp1 = np.empty(len(a_values), dtype=np.float64)
        logp2 = np.empty(len(c_values), dtype=np.float64)
        
        for k in nb.prange(len(a_values)):
            logp1[k] = _log_binom_pmf(a_values[k], n1, p1)
            
        for k in nb.prange(len(c_values)):
            logp2[k] = _log_binom_pmf(c_values[k], n2, p2)
            
        # Convert to probabilities in a numerically stable way
        max1 = logp1.max()
        max2 = logp2.max()
        
        prob1 = np.exp(logp1 - max1)
        prob2 = np.exp(logp2 - max2)
        
        # Scale by the maxima
        prob1 *= math.exp(max1)
        prob2 *= math.exp(max2)
        
        # Calculate joint probabilities using outer product
        joint = np.zeros((len(a_values), len(c_values)), dtype=np.float64)
        for i in nb.prange(len(a_values)):
            for j in range(len(c_values)):
                joint[i, j] = prob1[i] * prob2[j]
                
        return joint

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def unconditional_pvalue_numba(a_obs: int, c_obs: int, n1: int, n2: int, p1: float, p2: float,
                                  score_function: callable) -> float:
        """
        Two-sided exact p-value for unconditional test with fixed (p1,p2).
        
        Parameters
        ----------
        a_obs : int
            Observed count in cell (1,1)
        c_obs : int
            Observed count in cell (2,1)
        n1 : int
            Row 1 total
        n2 : int
            Row 2 total
        p1 : float
            Probability parameter for first group
        p2 : float
            Probability parameter for second group
        score_function : callable
            Function to compute the score statistic given k1,k2,n1,n2,p1,p2
            
        Returns
        -------
        float
            Unconditional exact p-value
        """
        obs_stat = score_function(a_obs, c_obs, n1, n2, p1, p2)
        
        # Pre-compute 1-D log pmfs for all possible tables
        logp1 = np.empty(n1+1, dtype=np.float64)
        logp2 = np.empty(n2+1, dtype=np.float64)
        
        for k in nb.prange(n1+1):
            logp1[k] = _log_binom_pmf(k, n1, p1)
            
        for k in nb.prange(n2+1):
            logp2[k] = _log_binom_pmf(k, n2, p2)
            
        # Convert to probabilities in a numerically stable way
        max1 = logp1.max()
        max2 = logp2.max()
        
        prob1 = np.exp(logp1 - max1)
        prob2 = np.exp(logp2 - max2)
        
        # Scale by the maxima
        prob1 *= math.exp(max1)
        prob2 *= math.exp(max2)
        
        # Calculate p-value by summing over all possible tables
        total_prob = 0.0
        for a in range(n1 + 1):
            for c in range(n2 + 1):
                if score_function(a, c, n1, n2, p1, p2) + 1e-12 >= obs_stat:
                    total_prob += prob1[a] * prob2[c]
                    
        return min(1.0, total_prob)
