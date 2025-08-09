"""
Centralized correction policies for ExactCIs.

This module provides the single source of truth for Haldane and continuity corrections,
ensuring consistent behavior across all methods.
"""

from typing import Tuple, Union


def add_haldane_correction(a: Union[int, float], b: Union[int, float], 
                          c: Union[int, float], d: Union[int, float],
                          correction: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Apply Haldane correction by adding correction value to all cells.
    
    This is the standard Haldane-Anscombe correction that adds a small constant
    (typically 0.5) to all cells to handle zero counts and improve finite-sample
    performance of asymptotic methods.
    
    Args:
        a, b, c, d: Cell counts
        correction: Value to add to each cell (default: 0.5)
        
    Returns:
        Tuple of corrected counts (a_corr, b_corr, c_corr, d_corr)
    """
    return (
        float(a) + correction,
        float(b) + correction,
        float(c) + correction,
        float(d) + correction
    )


def add_continuity_correction(a: Union[int, float], b: Union[int, float],
                             c: Union[int, float], d: Union[int, float],
                             correction: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Apply continuity correction only if there are zero cells.
    
    This selective correction adds the correction value to all cells
    only when one or more cells contain zero. This is often used
    for relative risk calculations.
    
    Args:
        a, b, c, d: Cell counts  
        correction: Value to add to each cell if zeros present (default: 0.5)
        
    Returns:
        Tuple of potentially corrected counts
    """
    # Convert to float for consistency
    a_f, b_f, c_f, d_f = float(a), float(b), float(c), float(d)
    
    # Apply correction only if there are zeros
    if any(x == 0 for x in (a_f, b_f, c_f, d_f)):
        return (
            a_f + correction,
            b_f + correction, 
            c_f + correction,
            d_f + correction
        )
    else:
        return (a_f, b_f, c_f, d_f)


def apply_correction_policy(a: Union[int, float], b: Union[int, float],
                           c: Union[int, float], d: Union[int, float],
                           policy: str = "none", correction: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Apply correction according to the specified policy.
    
    This unified interface allows consistent correction application
    across different methods.
    
    Args:
        a, b, c, d: Cell counts
        policy: Correction policy ("none", "haldane", "continuity")
        correction: Correction value
        
    Returns:
        Tuple of corrected counts
        
    Raises:
        ValueError: If policy is not recognized
    """
    if policy == "none":
        return float(a), float(b), float(c), float(d)
    elif policy == "haldane":
        return add_haldane_correction(a, b, c, d, correction)
    elif policy == "continuity":
        return add_continuity_correction(a, b, c, d, correction)
    else:
        raise ValueError(f"Unknown correction policy: {policy}. "
                        f"Must be one of: 'none', 'haldane', 'continuity'")


# TODO: REVIEW FOR REMOVAL - Duplicate zero cell detection
# This function is duplicated in utils/continuity.py with identical functionality
# Remove after consolidating under single module (prefer continuity.py)
def detect_zero_cells(a: Union[int, float], b: Union[int, float],
                     c: Union[int, float], d: Union[int, float]
) -> Tuple[bool, int, list[str]]:
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


def should_apply_correction(a: Union[int, float], b: Union[int, float],
                           c: Union[int, float], d: Union[int, float],
                           method: str) -> bool:
    """
    Determine whether correction should be applied based on method and data.
    
    This encapsulates the logic for when different methods need corrections.
    
    Args:
        a, b, c, d: Cell counts
        method: Method name (e.g., "wald", "score", "exact")
        
    Returns:
        True if correction should be applied
    """
    has_zeros, _, _ = detect_zero_cells(a, b, c, d)
    
    # Method-specific correction policies
    if method in ("wald", "wald_haldane"):
        return True  # Wald methods typically use Haldane correction
    elif method in ("wald_katz", "wald_correlated"):
        return has_zeros  # Apply only if there are zeros
    elif method in ("score", "score_cc"):
        return has_zeros  # Score methods handle zeros differently
    else:
        return False  # Exact methods typically don't need correction