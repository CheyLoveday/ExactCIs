#!/usr/bin/env python3
"""
Utility functions for standardized logging across methods.

This module provides helper functions for consistent logging across different
confidence interval methods, making output more uniform and easier to parse.
"""

import logging
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)


def method_banner(method_name: str, a: int, b: int, c: int, d: int, 
                 alpha: float = 0.05) -> None:
    """
    Log a standardized banner at the start of a method calculation.
    
    Parameters
    ----------
    method_name : str
        Name of the method being used
    a, b, c, d : int
        Cell counts in the 2x2 table
    alpha : float, default=0.05
        Significance level
    """
    logger.info(f"Computing {method_name} CI using adaptive grid search: "
               f"a={a}, b={b}, c={c}, d={d}, alpha={alpha:.5f}")
    
    n1 = a + b
    n2 = c + d
    m1 = a + c
    
    odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
    
    logger.info(f"Marginals: n1={n1}, n2={n2}, m1={m1}, odds_ratio={odds_ratio}")


def search_info(theta_min: float, theta_max: float, grid_size: int, 
               search_type: str = "adaptive grid search") -> None:
    """
    Log information about the search strategy.
    
    Parameters
    ----------
    theta_min : float
        Minimum theta value in the grid
    theta_max : float
        Maximum theta value in the grid
    grid_size : int
        Number of grid points
    search_type : str, default="adaptive grid search"
        Type of search being used
    """
    logger.info(f"Using {search_type} for CI search")
    logger.info(f"Using initial theta grid with {grid_size} points "
               f"from {theta_min:.6f} to {theta_max:.6f}")


def ci_result(method_name: str, lower: float, upper: float) -> None:
    """
    Log the final confidence interval result.
    
    Parameters
    ----------
    method_name : str
        Name of the method
    lower : float
        Lower bound of CI
    upper : float
        Upper bound of CI
    """
    logger.info(f"{method_name} CI result: ({lower:.6f}, {upper:.6f})")


def p_value_result(theta: float, p_value: float, alpha: float) -> None:
    """
    Log information about a specific p-value calculation.
    
    Parameters
    ----------
    theta : float
        Theta value tested
    p_value : float
        P-value obtained
    alpha : float
        Significance level
    """
    logger.info(f"theta={theta:.6f}, p-value={p_value:.6f}, alpha={alpha:.6f}")


def debug_info(message: str, **kwargs) -> None:
    """
    Log detailed debug information.
    
    Parameters
    ----------
    message : str
        Message to log
    **kwargs
        Additional information to include in the log message
    """
    if kwargs:
        formatted_kwargs = ', '.join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} ({formatted_kwargs})")
    else:
        logger.debug(message)


def batch_processing_banner(method_name: str, num_tables: int, alpha: float = 0.05,
                           max_workers: Optional[int] = None, backend: Optional[str] = None) -> None:
    """
    Log a standardized banner at the start of batch processing.
    
    Parameters
    ----------
    method_name : str
        Name of the method being used
    num_tables : int
        Number of tables to process
    alpha : float, default=0.05
        Significance level
    max_workers : int, optional
        Number of workers for parallel processing
    backend : str, optional
        Backend used for parallel processing
    """
    worker_info = f" using {max_workers} workers" if max_workers else ""
    backend_info = f" with {backend} backend" if backend else ""
    logger.info(f"Starting batch processing of {num_tables} tables with {method_name} method"
               f"{worker_info}{backend_info}, alpha={alpha:.5f}")


def batch_processing_complete(method_name: str, num_tables: int) -> None:
    """
    Log completion of batch processing.
    
    Parameters
    ----------
    method_name : str
        Name of the method used
    num_tables : int
        Number of tables processed
    """
    logger.info(f"Completed batch processing of {num_tables} tables with {method_name} method")
