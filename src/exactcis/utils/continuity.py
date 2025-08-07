#!/usr/bin/env python3
"""
Utility functions for continuity corrections in statistical methods.

This module provides functions for applying continuity corrections to counts
in contingency tables, especially for boundary cases (zeros or full counts).
"""

import numpy as np
from typing import Union, Tuple


def correct_cell(x: Union[int, float, np.ndarray], n: Union[int, float],
                 delta: float = 0.5) -> Union[float, np.ndarray]:
    """
    Apply continuity correction to a cell count, especially at boundaries.
    
    For boundary cases (0 or n), applies a continuity correction of +delta or -delta
    to avoid zeros and improve statistical properties.
    
    Parameters
    ----------
    x : int, float, or numpy.ndarray
        Cell count or array of cell counts
    n : int or float
        Total count (row or column total)
    delta : float, default=0.5
        Magnitude of continuity correction
        
    Returns
    -------
    float or numpy.ndarray
        Corrected cell count(s)
    """
    if isinstance(x, np.ndarray):
        result = x.astype(float)
        
        # Add delta to zeros
        zeros_mask = x == 0
        if np.any(zeros_mask):
            result[zeros_mask] += delta
            
        # Subtract delta from full counts
        fulls_mask = x == n
        if np.any(fulls_mask):
            result[fulls_mask] -= delta
            
        return result
    else:
        # Scalar case
        if x == 0:
            return float(x) + delta
        elif x == n:
            return float(x) - delta
        else:
            return float(x)


def correct_2x2(a: Union[int, float], n1: int, n2: int, m1: int, 
               delta: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Apply continuity corrections to all cells in a 2x2 table.
    
    Parameters
    ----------
    a : int or float
        Count in cell (1,1)
    n1 : int
        Row 1 total
    n2 : int
        Row 2 total
    m1 : int
        Column 1 total
    delta : float, default=0.5
        Magnitude of continuity correction
        
    Returns
    -------
    tuple
        (a_corrected, b_corrected, c_corrected, d_corrected)
    """
    b = n1 - a
    c = m1 - a
    d = n2 - c
    
    a_cc = correct_cell(a, n1, delta)
    b_cc = correct_cell(b, n1, delta)
    c_cc = correct_cell(c, n2, delta)
    d_cc = correct_cell(d, n2, delta)
    
    return a_cc, b_cc, c_cc, d_cc
