"""
Centralized validation functions for ExactCIs.

This module provides the single source of truth for input validation,
ensuring consistent behavior across all methods.
"""

from typing import Union


def validate_counts(a: Union[int, float], b: Union[int, float], 
                   c: Union[int, float], d: Union[int, float],
                   allow_zero_margins: bool = False) -> None:
    """
    Validate the counts in a 2x2 contingency table.
    
    This is the centralized validation function used by all methods.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2) 
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        allow_zero_margins: If True, allow zero row or column margins.
                          Some methods (like RR) can handle zero margins.
                          
    Raises:
        ValueError: If any count is negative, or if margins are zero when not allowed
    """
    # Check for negative counts
    if not all(isinstance(x, (int, float)) and x >= 0 for x in (a, b, c, d)):
        raise ValueError("All counts must be non-negative numbers")
    
    # Check for zero margins if not allowed
    if not allow_zero_margins:
        # Check row margins
        if (a + b) == 0:
            raise ValueError("Row 1 margin (a + b) cannot be zero")
        if (c + d) == 0:
            raise ValueError("Row 2 margin (c + d) cannot be zero")
        
        # Check column margins  
        if (a + c) == 0:
            raise ValueError("Column 1 margin (a + c) cannot be zero")
        if (b + d) == 0:
            raise ValueError("Column 2 margin (b + d) cannot be zero")


def normalize_inputs(a: Union[int, float], b: Union[int, float],
                    c: Union[int, float], d: Union[int, float]
) -> tuple[float, float, float, float]:
    """
    Normalize inputs to float type for consistent computation.
    
    Args:
        a, b, c, d: Cell counts
        
    Returns:
        Tuple of normalized float values
    """
    return float(a), float(b), float(c), float(d)


def validate_alpha(alpha: float) -> None:
    """
    Validate the significance level.
    
    Args:
        alpha: Significance level
        
    Raises:
        ValueError: If alpha is not in (0, 1)
    """
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")


def validate_table_and_alpha(a: Union[int, float], b: Union[int, float],
                            c: Union[int, float], d: Union[int, float],
                            alpha: float = 0.05, allow_zero_margins: bool = False,
                            preserve_int_types: bool = False
) -> tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]:
    """
    Validate both table counts and alpha, returning normalized inputs.
    
    This is a convenience function that performs common validation
    and normalization in one step.
    
    Args:
        a, b, c, d: Cell counts
        alpha: Significance level
        allow_zero_margins: Whether to allow zero margins
        preserve_int_types: If True, return integers if inputs were integers
        
    Returns:
        Tuple of normalized counts (a, b, c, d) - floats by default, 
        integers if preserve_int_types=True and all inputs were integers
        
    Raises:
        ValueError: If validation fails
    """
    validate_counts(a, b, c, d, allow_zero_margins=allow_zero_margins)
    validate_alpha(alpha)
    
    if preserve_int_types and all(isinstance(x, int) for x in (a, b, c, d)):
        return int(a), int(b), int(c), int(d)
    else:
        return normalize_inputs(a, b, c, d)