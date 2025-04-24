"""
Statistical utility functions for ExactCIs package.

This module provides statistical utility functions used by the various
confidence interval methods.
"""

import math


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