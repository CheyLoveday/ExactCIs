"""
Pure transformation functions for ExactCIs.
"""

import math
import numpy as np
from typing import Tuple
from .data_models import TableData, UnconditionalConfig, ThetaRange, GridConfig


def apply_haldane_correction(table: TableData) -> TableData:
    """
    Apply Haldane correction to table data.
    
    Args:
        table: Original table data
        
    Returns:
        Table with Haldane correction applied
    """
    return TableData(
        a=table.a + 0.5,
        b=table.b + 0.5,
        c=table.c + 0.5,
        d=table.d + 0.5
    )


def calculate_odds_ratio_safe(table: TableData) -> float:
    """
    Calculate odds ratio with safe division.
    
    Args:
        table: Table data
        
    Returns:
        Odds ratio value
    """
    if table.b * table.c == 0:
        return (table.a * table.d) if table.a * table.d > 0 else 1.0
    return (table.a * table.d) / (table.b * table.c)


def normalize_odds_ratio(or_value: float) -> float:
    """
    Normalize odds ratio to reasonable range.
    
    Args:
        or_value: Raw odds ratio value
        
    Returns:
        Normalized odds ratio
    """
    if or_value == 0:
        return 1e-6
    elif or_value == float('inf'):
        return 1e6
    else:
        return or_value


def determine_theta_range(table: TableData, config: UnconditionalConfig) -> ThetaRange:
    """
    Determine search range for theta parameter.
    
    Args:
        table: Table data
        config: Configuration
        
    Returns:
        Theta range with computed bounds
    """
    or_value = calculate_odds_ratio_safe(table)
    
    if config.custom_range is not None:
        min_theta, max_theta = config.custom_range
        return ThetaRange(min_theta, max_theta, or_value)
    
    if config.theta_min is not None and config.theta_max is not None:
        return ThetaRange(config.theta_min, config.theta_max, or_value)
    
    # Auto-determine range
    normalized_or = normalize_odds_ratio(or_value)
    
    min_theta = max(normalized_or / config.theta_factor, 1e-6)
    max_theta = min(normalized_or * config.theta_factor, 1e6)
    
    return ThetaRange(min_theta, max_theta, normalized_or)


def optimize_grid_size_for_table(table: TableData, base_grid_size: int) -> int:
    """
    Optimize grid size based on table dimensions.
    
    Args:
        table: Table data
        base_grid_size: Base grid size
        
    Returns:
        Optimized grid size
    """
    if table.total < 40 and base_grid_size > 20:
        return 20
    return base_grid_size


def create_adaptive_grid(table: TableData, config: UnconditionalConfig) -> GridConfig:
    """
    Create adaptive grid for p1 values.
    
    Args:
        table: Table data
        config: Configuration
        
    Returns:
        Grid configuration with p1 values
    """
    p1_mle = table.a / table.n1 if table.n1 > 0 else 0.5
    optimized_size = optimize_grid_size_for_table(table, config.grid_size)
    
    # Create adaptive grid around MLE
    p1_values = np.concatenate([
        np.linspace(max(0.001, p1_mle - 0.4), p1_mle - 0.05, optimized_size // 3),
        np.linspace(p1_mle - 0.05, p1_mle + 0.05, optimized_size // 3),
        np.linspace(p1_mle + 0.05, min(0.999, p1_mle + 0.4), optimized_size // 3)
    ])
    p1_values = np.unique(p1_values)
    
    return GridConfig(p1_values, optimized_size, p1_mle)


def calculate_p2_from_theta(p1: float, theta: float) -> float:
    """
    Calculate p2 from p1 and theta.
    
    Args:
        p1: First probability parameter
        theta: Odds ratio parameter
        
    Returns:
        Second probability parameter
    """
    return (theta * p1) / (1 - p1 + theta * p1)


def clamp_bound_to_valid_range(bound: float, is_lower: bool) -> float:
    """
    Clamp confidence bound to valid range.
    
    Args:
        bound: Bound value to clamp
        is_lower: Whether this is a lower bound
        
    Returns:
        Clamped bound value
    """
    if is_lower:
        return max(bound, 0.0)
    else:
        return bound  # Upper bounds can be infinite