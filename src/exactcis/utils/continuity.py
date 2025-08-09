#!/usr/bin/env python3
"""
Utility functions for continuity corrections in statistical methods.

This module provides the single source of truth for all continuity correction
policies and helper functions.
"""

import numpy as np
from typing import Union, Tuple, List

from exactcis.constants import WaldMethod
from exactcis.models import CorrectionResult

def detect_zero_cells(a: Union[int, float], b: Union[int, float],
                     c: Union[int, float], d: Union[int, float]
) -> Tuple[bool, int, List[str]]:
    """
    Detect zero cells in a 2x2 table.
    
    Args:
        a, b, c, d: Cell counts
        
    Returns:
        Tuple of (has_zeros, zero_count, zero_positions)
        where zero_positions is a list like ['a', 'c'] indicating which cells are zero
    """
    counts = {'a': float(a), 'b': float(b), 'c': float(c), 'd': float(d)}
    zero_positions = [cell for cell, count in counts.items() if count == 0]
    
    return len(zero_positions) > 0, len(zero_positions), zero_positions


def get_corrected_counts(a: float, b: float, c: float, d: float, method: WaldMethod) -> CorrectionResult:
    """
    Central policy function for applying continuity corrections.

    This function determines if a correction is needed based on the data and
    the chosen statistical method, and returns the corrected counts along
    with a record of which correction was applied.

    Args:
        a, b, c, d: The four cell counts of the 2x2 table.
        method: The statistical method being used (e.g., "wald").

    Returns:
        A CorrectionResult object containing the corrected counts and the
        name of the correction that was applied, if any.
    """
    has_zeros, _, _ = detect_zero_cells(a, b, c, d)

    # Apply method-specific correction policies
    # 1) Haldane-Anscombe Wald: always add 0.5 to all cells (parity with historical behavior)
    if method == "wald_haldane":
        return _apply_haldane_correction(a, b, c, d)

    # 2) Standard Wald: add Haldane correction only when zeros are present
    if method == "wald" and has_zeros:
        return _apply_haldane_correction(a, b, c, d)

    # Default: no correction
    return CorrectionResult(a=a, b=b, c=c, d=d, correction_applied=False, correction_amount=0.0)


def _apply_haldane_correction(a: float, b: float, c: float, d: float) -> CorrectionResult:
    """
    Applies the Haldane (or Haldane-Anscombe) continuity correction.

    Adds 0.5 to all cells in the 2x2 table.

    Returns:
        A CorrectionResult object with the corrected counts.
    """
    delta = 0.5
    return CorrectionResult(
        a=a + delta,
        b=b + delta,
        c=c + delta,
        d=d + delta,
        correction_applied=True,
        correction_amount=delta,
        original_counts=(a, b, c, d),
        method="haldane"
    )


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
    ------
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
    ------
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
