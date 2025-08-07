"""
Shared utilities for confidence interval search algorithms.

This module provides common functions for searching confidence intervals
using various p-value calculation methods. It abstracts the grid search
and root-finding logic that is shared between different CI methods.
"""

import math
import logging
import numpy as np
from typing import Tuple, Callable, Optional, Union, List
from scipy import optimize

# Import adaptive grid search from optimization module
from .optimization import adaptive_grid_search

logger = logging.getLogger(__name__)


def find_confidence_interval_grid(
    p_value_func: Callable[[float], float],
    theta_min: float,
    theta_max: float,
    alpha: float,
    grid_size: int = 200,
    odds_ratio: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[float, float]:
    """
    Find confidence interval using grid search.
    
    Args:
        p_value_func: Function that takes theta and returns p-value
        theta_min: Minimum theta value for search
        theta_max: Maximum theta value for search
        alpha: Significance level
        grid_size: Number of grid points
        odds_ratio: Sample odds ratio (will be included in grid if provided)
        progress_callback: Optional progress callback function
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Generate grid of theta values (logarithmically spaced)
    theta_values = list(np.logspace(np.log10(theta_min), np.log10(theta_max), grid_size))
    
    # Include odds ratio in grid if provided
    if odds_ratio is not None and odds_ratio > 0 and odds_ratio < float('inf'):
        if odds_ratio not in theta_values:
            theta_values.append(odds_ratio)
            theta_values.sort()
    
    theta_grid = np.array(theta_values)
    logger.info(f"Using theta grid with {len(theta_grid)} points")
    
    # Calculate p-values for each theta in the grid
    p_values = []
    for i, theta in enumerate(theta_grid):
        try:
            p_value = p_value_func(theta)
            p_values.append(p_value)
        except Exception as e:
            logger.warning(f"Error calculating p-value for theta={theta}: {e}")
            p_values.append(0.0)  # Conservative fallback
        
        # Report progress
        if progress_callback:
            progress_callback(min(100, int(100 * (i+1) / len(theta_grid))))
    
    # Convert to numpy arrays
    p_values = np.array(p_values)
    
    # Find confidence interval bounds
    lower_bound = _find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    upper_bound = _find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    
    return (lower_bound, upper_bound)


def find_confidence_interval_rootfinding(
    p_value_func: Callable[[float], float],
    theta_min: float,
    theta_max: float,
    alpha: float,
    odds_ratio: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[float, float]:
    """
    Find confidence interval using root-finding algorithm.
    
    This method is more precise than grid search and typically faster
    for well-behaved p-value functions.
    
    Args:
        p_value_func: Function that takes theta and returns p-value
        theta_min: Minimum theta value for search
        theta_max: Maximum theta value for search
        alpha: Significance level
        odds_ratio: Sample odds ratio (used for initialization)
        progress_callback: Optional progress callback function
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    def root_func(theta):
        """Function whose roots we want to find: p_value(theta) - alpha = 0"""
        return p_value_func(theta) - alpha
    
    # Find initial bounds where the root function changes sign
    # We need to find where p_value crosses alpha
    
    if progress_callback:
        progress_callback(10)
    
    try:
        # Find lower bound
        # Look for theta where p_value goes from < alpha to >= alpha
        lower_bound = _find_root_with_bracketing(
            root_func, theta_min, odds_ratio or 1.0, 
            "lower", progress_callback
        )
        
        if progress_callback:
            progress_callback(50)
        
        # Find upper bound  
        # Look for theta where p_value goes from >= alpha to < alpha
        upper_bound = _find_root_with_bracketing(
            root_func, odds_ratio or 1.0, theta_max,
            "upper", progress_callback
        )
        
        if progress_callback:
            progress_callback(100)
        
        return (lower_bound, upper_bound)
        
    except Exception as e:
        logger.warning(f"Root-finding failed: {e}, falling back to grid search")
        # Fallback to grid search
        return find_confidence_interval_grid(
            p_value_func, theta_min, theta_max, alpha, 
            grid_size=100, odds_ratio=odds_ratio, 
            progress_callback=progress_callback
        )


def _find_root_with_bracketing(
    func: Callable[[float], float], 
    a: float, 
    b: float, 
    bound_type: str,
    progress_callback: Optional[Callable[[float], None]] = None
) -> float:
    """
    Find root using bracketing and Brent's method.
    
    Args:
        func: Function whose root we want to find
        a: Left bracket
        b: Right bracket  
        bound_type: "lower" or "upper" for logging
        progress_callback: Optional progress callback
        
    Returns:
        Root of the function
    """
    # First, ensure we have a proper bracket
    fa = func(a)
    fb = func(b)
    
    # If we don't have a sign change, try to extend the bracket
    if fa * fb > 0:
        logger.warning(f"No sign change in initial bracket for {bound_type} bound")
        # Try to find a bracket by expanding the search
        if bound_type == "lower":
            # For lower bound, we expect p-value to increase as theta increases
            # So expand leftward if needed
            while fa * fb > 0 and a > 1e-6:
                a = a * 0.1
                fa = func(a)
        else:
            # For upper bound, expand rightward if needed
            while fa * fb > 0 and b < 1e6:
                b = b * 10
                fb = func(b)
        
        # If we still don't have a bracket, return the boundary
        if fa * fb > 0:
            logger.warning(f"Could not bracket root for {bound_type} bound")
            return a if bound_type == "lower" else b
    
    try:
        # Use Brent's method to find the root
        result = optimize.brentq(func, a, b, xtol=1e-6, rtol=1e-6)
        logger.info(f"Found {bound_type} bound using root-finding: {result:.6f}")
        return result
        
    except Exception as e:
        logger.warning(f"Brent's method failed for {bound_type} bound: {e}")
        # Return the boundary that's closest to having the right p-value
        if abs(fa - 0) < abs(fb - 0):
            return a
        else:
            return b


def _find_ci_bound(theta_grid: np.ndarray, p_values: np.ndarray, 
                  alpha: float, is_lower: bool = True) -> float:
    """
    Find confidence interval bound from grid of theta values and p-values.
    
    Args:
        theta_grid: Array of theta values
        p_values: Array of p-values corresponding to theta_grid
        alpha: Significance level
        is_lower: If True, find lower bound; if False, find upper bound
        
    Returns:
        Confidence interval bound
    """
    # Convert to numpy arrays if not already
    theta_grid = np.array(theta_grid)
    p_values = np.array(p_values)
    
    # Find values in the confidence interval (p_value >= alpha)
    in_ci = p_values >= alpha
    
    # Log the number of values in the CI
    bound_type = "lower" if is_lower else "upper"
    logger.info(f"Finding {bound_type} bound: {np.sum(in_ci)} values in CI out of {len(theta_grid)}")
    
    if not np.any(in_ci):
        # No values in CI, return boundary
        logger.warning(f"No values in CI for {bound_type} bound, returning boundary")
        return theta_grid[0] if is_lower else theta_grid[-1]
    
    # Find the bound
    if is_lower:
        # Lower bound is the smallest theta in CI
        idx = np.where(in_ci)[0][0]
        bound = theta_grid[idx]
        logger.info(f"Lower bound: theta={bound:.6f}, p-value={p_values[idx]:.6f}, alpha={alpha:.6f}")
        return bound
    else:
        # Upper bound is the largest theta in CI
        idx = np.where(in_ci)[0][-1]
        bound = theta_grid[idx]
        logger.info(f"Upper bound: theta={bound:.6f}, p-value={p_values[idx]:.6f}, alpha={alpha:.6f}")
        return bound


def find_confidence_interval_adaptive_grid(
    p_value_func: Callable[[float], float],
    theta_min: float,
    theta_max: float,
    alpha: float,
    initial_grid_size: int = 50,
    refinement_rounds: int = 2,
    odds_ratio: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[float, float]:
    """
    Find confidence interval using adaptive grid search with refinement.
    
    This method first uses a coarse grid to find approximate bounds, then
    refines the grid around those bounds for more precision. This approach
    provides better precision than a fixed-size grid without the computational
    cost of a uniformly dense grid.
    
    Args:
        p_value_func: Function that takes theta and returns p-value
        theta_min: Minimum theta value for search
        theta_max: Maximum theta value for search
        alpha: Significance level
        initial_grid_size: Number of points in the initial grid
        refinement_rounds: Number of refinement rounds
        odds_ratio: Sample odds ratio (will be included in grid if provided)
        progress_callback: Optional progress callback function
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Generate initial grid of theta values (logarithmically spaced)
    theta_values = list(np.logspace(np.log10(theta_min), np.log10(theta_max), initial_grid_size))
    
    # Include odds ratio in grid if provided
    if odds_ratio is not None and odds_ratio > 0 and odds_ratio < float('inf'):
        if odds_ratio not in theta_values:
            theta_values.append(odds_ratio)
            theta_values.sort()
    
    theta_grid = np.array(theta_values)
    logger.info(f"Using initial theta grid with {len(theta_grid)} points")
    
    # Calculate p-values for each theta in the initial grid
    p_values = []
    for i, theta in enumerate(theta_grid):
        try:
            p_value = p_value_func(theta)
            p_values.append(p_value)
        except Exception as e:
            logger.warning(f"Error calculating p-value for theta={theta}: {e}")
            p_values.append(0.0)  # Conservative fallback
        
        # Report progress (50% for initial grid)
        if progress_callback:
            progress_callback(min(50, int(50 * (i+1) / len(theta_grid))))
    
    # Convert to numpy arrays
    p_values = np.array(p_values)
    
    # Find preliminary confidence interval bounds
    prelim_lower = _find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    prelim_upper = _find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    
    logger.info(f"Preliminary CI bounds: ({prelim_lower:.6f}, {prelim_upper:.6f})")
    
    # Refine the lower bound
    lower_bound_func = lambda theta: p_value_func(theta) - alpha
    lower_bound_results = adaptive_grid_search(
        lower_bound_func,
        bounds=(max(theta_min, prelim_lower * 0.5), min(prelim_lower * 1.5, prelim_upper * 0.9)),
        target_value=0.0,
        initial_points=initial_grid_size // 2,
        refinement_rounds=refinement_rounds
    )
    
    # If adaptive_grid_search returns multiple crossings, take the smallest one
    lower_bound = min(lower_bound_results) if lower_bound_results else prelim_lower
    
    # Report progress (75% after lower bound refinement)
    if progress_callback:
        progress_callback(75)
    
    # Refine the upper bound with more conservative range to avoid inflated bounds
    upper_bound_func = lambda theta: p_value_func(theta) - alpha
    
    # Use more conservative upper search bound - if prelim_upper seems inflated, cap it
    max_reasonable_upper = prelim_upper
    if odds_ratio is not None and prelim_upper > odds_ratio * 2.5:
        # If preliminary upper bound is >2.5x the odds ratio, it's likely inflated
        max_reasonable_upper = min(prelim_upper, odds_ratio * 2.5)
        logger.info(f"Capping inflated prelim_upper from {prelim_upper:.3f} to {max_reasonable_upper:.3f}")
    
    upper_bound_results = adaptive_grid_search(
        upper_bound_func,
        bounds=(max(prelim_lower * 1.1, prelim_upper * 0.5), min(theta_max, max_reasonable_upper * 1.5)),
        target_value=0.0,
        initial_points=initial_grid_size // 2,
        refinement_rounds=refinement_rounds
    )
    
    # If adaptive_grid_search returns multiple crossings, take the largest one
    upper_bound = max(upper_bound_results) if upper_bound_results else prelim_upper
    
    # Report progress (100% after upper bound refinement)
    if progress_callback:
        progress_callback(100)
    
    logger.info(f"Refined CI bounds: ({lower_bound:.6f}, {upper_bound:.6f})")
    
    return (lower_bound, upper_bound)