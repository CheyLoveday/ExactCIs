#!/usr/bin/env python3
"""
Utility functions for confidence interval grid construction.

This module provides functions for creating standard grids for odds ratio (theta)
and nuisance parameter (pi) values, used in confidence interval calculations.
"""

import numpy as np
import math
from typing import List, Union, Tuple, Optional


def default_theta_grid(theta_min: float = 1e-3, theta_max: float = 1e3,
                      size: int = 51) -> np.ndarray:
    """
    Create a default log-spaced grid of odds ratio values.
    
    This is the standard grid used by most exact CI methods.
    
    Parameters
    ----------
    theta_min : float, default=1e-3
        Minimum theta value
    theta_max : float, default=1e3
        Maximum theta value
    size : int, default=51
        Number of grid points
        
    Returns
    -------
    numpy.ndarray
        Log-spaced grid of theta values
    """
    return np.logspace(np.log10(theta_min), np.log10(theta_max), size)


def default_pi_grid(n: Union[int, float] = None, base_size: int = 50,
                   eps: float = 1e-6) -> np.ndarray:
    """
    Create a default grid for the nuisance parameter pi.
    
    For small n, a uniform grid is sufficient. For larger n,
    a combination of uniform and log-spaced points can better
    explore the parameter space.
    
    Parameters
    ----------
    n : int or float, optional
        Sample size, used to determine if a more complex grid is needed
    base_size : int, default=50
        Base number of grid points
    eps : float, default=1e-6
        Minimum value to ensure grid doesn't include exact 0 or 1
        
    Returns
    -------
    numpy.ndarray
        Grid of pi values between eps and 1-eps
    """
    if n is not None and n > 100:
        # For large n, use a combination of uniform and log-spaced points
        # to better explore regions near 0 and 1
        uniform_size = base_size // 2
        log_size = base_size - uniform_size
        
        # Uniform grid in the middle
        uniform = np.linspace(0.1, 0.9, uniform_size)
        
        # Log-spaced points near boundaries
        log_low = np.logspace(np.log10(eps), np.log10(0.1), log_size // 2)
        log_high = 1 - np.logspace(np.log10(eps), np.log10(0.1), log_size // 2)
        
        # Combine and sort
        grid = np.concatenate([log_low, uniform, log_high])
        grid.sort()
        return grid
    else:
        # Simple uniform grid for smaller n
        return np.linspace(eps, 1-eps, base_size)


def adaptive_pi_grid_refinement(pi_values: np.ndarray, p_values: np.ndarray,
                              n: int = 50, max_iter: int = 3) -> np.ndarray:
    """
    Refine a pi grid based on p-value evaluations.
    
    This adds more grid points in regions where the p-value
    function changes rapidly, which is important for accurate
    maximization over the nuisance parameter.
    
    Parameters
    ----------
    pi_values : numpy.ndarray
        Current grid of pi values
    p_values : numpy.ndarray
        P-values evaluated at pi_values
    n : int, default=50
        Number of new points to add
    max_iter : int, default=3
        Maximum number of refinement iterations
        
    Returns
    -------
    numpy.ndarray
        Refined grid of pi values
    """
    grid = pi_values.copy()
    
    for _ in range(max_iter):
        # Calculate absolute differences between adjacent p-values
        diffs = np.abs(np.diff(p_values))
        
        if np.max(diffs) < 1e-6:
            # Grid is already fine enough
            break
        
        # Add points where p-value changes rapidly
        indices = np.argsort(-diffs)[:n]
        
        new_points = []
        for idx in indices:
            # Midpoint between adjacent points
            new_pi = (grid[idx] + grid[idx+1]) / 2
            new_points.append(new_pi)
            
        # Add new points to grid and sort
        if new_points:
            grid = np.concatenate([grid, new_points])
            grid.sort()
            
    return grid


def odds_ratio_from_p1p2(p1: float, p2: float) -> float:
    """
    Calculate the odds ratio corresponding to probabilities p1 and p2.
    
    Parameters
    ----------
    p1 : float
        Probability in first group
    p2 : float
        Probability in second group
        
    Returns
    -------
    float
        Odds ratio: (p1/(1-p1)) / (p2/(1-p2))
    """
    eps = 1e-12
    p1_safe = min(1.0 - eps, max(eps, p1))
    p2_safe = min(1.0 - eps, max(eps, p2))
    
    return (p1_safe / (1.0 - p1_safe)) / (p2_safe / (1.0 - p2_safe))


def p2_from_p1_and_or(p1: float, odds_ratio: float) -> float:
    """
    Calculate p2 from p1 and odds ratio.
    
    For a given p1 and odds ratio theta, find the p2 such that:
    (p1/(1-p1)) / (p2/(1-p2)) = theta
    
    Parameters
    ----------
    p1 : float
        Probability in first group
    odds_ratio : float
        Target odds ratio
        
    Returns
    -------
    float
        p2 value that yields the target odds ratio with p1
    """
    eps = 1e-12
    p1_safe = min(1.0 - eps, max(eps, p1))
    odds1 = p1_safe / (1.0 - p1_safe)
    odds2 = odds1 / odds_ratio
    
    return odds2 / (1.0 + odds2)
