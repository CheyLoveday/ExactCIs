"""
Pure validation functions for ExactCIs.
"""

from typing import Union
from .data_models import TableData, UnconditionalConfig
from .validation import validate_counts


def validate_table_data(table: TableData) -> bool:
    """
    Validate 2x2 table data.
    
    Args:
        table: Table data to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If table data is invalid
    """
    validate_counts(table.a, table.b, table.c, table.d)
    return True


# TODO: REVIEW FOR REMOVAL - Deprecated alpha validation function
# This function is deprecated and replaced by exactcis.utils.validation.validate_alpha
# Remove after confirming no remaining dependencies
def validate_alpha(alpha: float) -> bool:
    """
    Validate alpha parameter.
    
    DEPRECATED: Use exactcis.utils.validation.validate_alpha instead.
    """
    from .validation import validate_alpha as _validate_alpha
    _validate_alpha(alpha)
    return True


def has_zero_marginal_totals(table: TableData) -> bool:
    """
    Check if table has zero marginal totals.
    
    Args:
        table: Table data to check
        
    Returns:
        True if any marginal total is zero
    """
    return table.n1 == 0 or table.n2 == 0


def has_zero_in_cell_a_with_nonzero_c(table: TableData) -> bool:
    """
    Check for special case: a=0 and c>0.
    
    Args:
        table: Table data to check
        
    Returns:
        True if a=0 and c>0
    """
    return table.a == 0 and table.c > 0


def is_valid_theta_range(theta_min: float, theta_max: float) -> bool:
    """
    Validate theta range parameters.
    
    Args:
        theta_min: Minimum theta value
        theta_max: Maximum theta value
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If range is invalid
    """
    if theta_min <= 0:
        raise ValueError("theta_min must be positive")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")
    return True


def is_finite_positive(value: float) -> bool:
    """
    Check if value is finite and positive.
    
    Args:
        value: Value to check
        
    Returns:
        True if finite and positive
    """
    import math
    return math.isfinite(value) and value > 0