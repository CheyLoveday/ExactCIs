"""
Statistical utility functions for ExactCIs package.

This module provides statistical utility functions used by the various
confidence interval methods.
"""

import math
from typing import Union, Tuple


def normal_quantile(p: float) -> float:
    """
    Calculate the quantile of the standard normal distribution.
    
    This is a pure Python implementation that doesn't require SciPy.
    It uses the Abramowitz & Stegun approximation.
    
    Args:
        p: Probability (0 < p < 1)
        
    Returns:
        Quantile of the standard normal distribution
        
    Raises:
        ValueError: If p is not in (0,1)
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    if p == 0.5:
        return 0.0
    q = p if p < 0.5 else 1-p
    t = math.sqrt(-2 * math.log(q))
    # Abramowitz & Stegun
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1*t + c2*t*t
    den = 1 + d1*t + d2*t*t + d3*t*t*t
    x = t - num/den
    return -x if p < 0.5 else x


def add_haldane_correction(a: Union[int, float], b: Union[int, float], 
                           c: Union[int, float], d: Union[int, float]) -> Tuple[float, float, float, float]:
    """
    Add Haldane's continuity correction (0.5 to each cell).
    
    This helps with zero counts and generally improves the accuracy of interval estimates.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        
    Returns:
        Tuple (a, b, c, d) with Haldane's correction applied
    """
    return a + 0.5, b + 0.5, c + 0.5, d + 0.5


def calculate_odds_ratio_with_correction(a: Union[int, float], b: Union[int, float], 
                                        c: Union[int, float], d: Union[int, float],
                                        add_correction: bool = True) -> float:
    """
    Calculate odds ratio with optional Haldane's continuity correction.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        add_correction: Whether to add Haldane's continuity correction (default: True)
        
    Returns:
        Odds ratio (a*d)/(b*c) with or without correction
    """
    from exactcis.core import calculate_odds_ratio
    
    if add_correction:
        a, b, c, d = add_haldane_correction(a, b, c, d)
    
    return calculate_odds_ratio(a, b, c, d)


def calculate_standard_error(a: Union[int, float], b: Union[int, float], 
                            c: Union[int, float], d: Union[int, float],
                            add_correction: bool = True) -> float:
    """
    Calculate standard error of the log odds ratio.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        add_correction: Whether to add Haldane's continuity correction (default: True)
        
    Returns:
        Standard error of the log odds ratio
    """
    if add_correction:
        a, b, c, d = add_haldane_correction(a, b, c, d)
    
    # Avoid division by zero for zero cells
    a_val = max(0.5, a) if a == 0 else a
    b_val = max(0.5, b) if b == 0 else b
    c_val = max(0.5, c) if c == 0 else c
    d_val = max(0.5, d) if d == 0 else d
    
    return math.sqrt(1/a_val + 1/b_val + 1/c_val + 1/d_val)