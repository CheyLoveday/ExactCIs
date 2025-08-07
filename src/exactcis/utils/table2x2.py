#!/usr/bin/env python3
"""
Utility functions for 2x2 contingency table operations.

This module provides common operations for working with 2x2 contingency tables,
including calculating cell counts, odds ratios, log odds ratios,
and related statistical quantities.
"""

import math
import numpy as np
from typing import Union, Tuple, Optional

# Small epsilon value to prevent numerical issues
EPS = 1e-12


def cell_counts(a: Union[int, float, np.ndarray], n1: int, n2: int, m1: int) -> Union[
        Tuple[Union[int, float], Union[int, float], Union[int, float]],
        Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate remaining cell counts in a 2x2 table from a, n1, n2, m1.
    
    Parameters
    ----------
    a : int, float, or numpy.ndarray
        Count in cell (1,1) or array of counts
    n1 : int
        Row 1 total
    n2 : int
        Row 2 total
    m1 : int
        Column 1 total
        
    Returns
    -------
    tuple
        (b, c, d) - remaining cell counts
    """
    b = n1 - a
    c = m1 - a
    d = n2 - c
    return b, c, d


def log_odds_ratio(a: Union[float, np.ndarray], b: Union[float, np.ndarray], 
                   c: Union[float, np.ndarray], d: Union[float, np.ndarray],
                   cc: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate log odds ratio for a 2x2 table.
    
    Parameters
    ----------
    a, b, c, d : float or numpy.ndarray
        Cell counts (can be corrected counts)
    cc : float, default=0.0
        Additional continuity correction (often not needed if a,b,c,d
        are already corrected)
        
    Returns
    -------
    float or numpy.ndarray
        Log odds ratio: log((a*d)/(b*c))
    """
    if isinstance(a, np.ndarray):
        # Ensure all values are positive to avoid log(0)
        a_adj = np.maximum(a + cc, EPS)
        b_adj = np.maximum(b + cc, EPS)
        c_adj = np.maximum(c + cc, EPS)
        d_adj = np.maximum(d + cc, EPS)
        
        return np.log(a_adj / b_adj) - np.log(c_adj / d_adj)
    else:
        # Scalar case
        a_adj = max(a + cc, EPS)
        b_adj = max(b + cc, EPS)
        c_adj = max(c + cc, EPS)
        d_adj = max(d + cc, EPS)
        
        return math.log(a_adj / b_adj) - math.log(c_adj / d_adj)


def lor_null(p1: float, p2: float) -> float:
    """
    Calculate expected log odds ratio under null hypothesis.
    
    Parameters
    ----------
    p1 : float
        Probability parameter for first group
    p2 : float
        Probability parameter for second group
        
    Returns
    -------
    float
        Expected log odds ratio under null
    """
    # Ensure probabilities are valid
    p1_adj = min(1.0 - EPS, max(EPS, p1))
    p2_adj = min(1.0 - EPS, max(EPS, p2))
    
    return math.log(p1_adj / (1 - p1_adj)) - math.log(p2_adj / (1 - p2_adj))


def lor_var(p1: float, p2: float, n1: int, n2: int) -> float:
    """
    Calculate variance of log odds ratio under null hypothesis.
    
    Parameters
    ----------
    p1 : float
        Probability parameter for first group
    p2 : float
        Probability parameter for second group
    n1 : int
        Size of first group
    n2 : int
        Size of second group
        
    Returns
    -------
    float
        Variance of log odds ratio under null
    """
    # Ensure probabilities are valid
    p1_adj = min(1.0 - EPS, max(EPS, p1))
    p2_adj = min(1.0 - EPS, max(EPS, p2))
    
    return (1.0 / (n1 * p1_adj * (1 - p1_adj))) + (1.0 / (n2 * p2_adj * (1 - p2_adj)))


def score_statistic(a: Union[float, np.ndarray], b: Union[float, np.ndarray],
                    c: Union[float, np.ndarray], d: Union[float, np.ndarray],
                    p1: float, p2: float, n1: int, n2: int) -> Union[float, np.ndarray]:
    """
    Calculate the Barnard/Suissa-Shuster score statistic.
    
    This is the standardized difference between observed and expected log odds ratio:
    |log OR - expected log OR| / sqrt(Var)
    
    Parameters
    ----------
    a, b, c, d : float or numpy.ndarray
        Cell counts (assumed already continuity-corrected if needed)
    p1 : float
        Probability parameter for first group
    p2 : float
        Probability parameter for second group
    n1 : int
        Size of first group
    n2 : int
        Size of second group
        
    Returns
    -------
    float or numpy.ndarray
        Score statistic value(s)
    """
    observed_lor = log_odds_ratio(a, b, c, d)
    expected_lor = lor_null(p1, p2)
    variance = lor_var(p1, p2, n1, n2)
    
    if isinstance(a, np.ndarray):
        result = np.abs(observed_lor - expected_lor) / np.sqrt(variance)
        # Handle numerical issues
        result = np.where(np.isnan(result) | np.isinf(result), float('inf'), result)
        return result
    else:
        try:
            return abs(observed_lor - expected_lor) / math.sqrt(variance)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float('inf')
