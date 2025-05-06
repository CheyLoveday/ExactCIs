"""
Parallel processing utilities for ExactCIs.

This module provides parallelization helpers to speed up computation-intensive
operations in the ExactCIs package.
"""

import logging
import multiprocessing
import numpy as np
from functools import partial
from typing import Callable, List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)


def get_optimal_workers():
    """
    Determine the optimal number of worker processes.
    
    Returns 75% of available cores, but at least 2 and at most 8.
    """
    cores = multiprocessing.cpu_count()
    return max(2, min(8, int(cores * 0.75)))


def parallel_map(func: Callable, items: List[Any], 
                 use_threads: bool = False, 
                 max_workers: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 timeout: Optional[float] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> List[Any]:
    """
    Execute a function over a list of items in parallel.
    
    Args:
        func: Function to execute
        items: List of items to process
        use_threads: Use threads instead of processes (better for I/O bound tasks)
        max_workers: Maximum number of workers (default: auto-determined)
        chunk_size: Size of chunks for processing (default: auto-determined)
        timeout: Maximum time to wait for completion in seconds
        progress_callback: Optional callback to report progress (0-100)
        
    Returns:
        List of results in the same order as input items
    """
    if not items:
        return []
        
    n_items = len(items)
    if n_items == 1:
        # No need for parallelization with just one item
        result = [func(items[0])]
        if progress_callback:
            progress_callback(100)
        return result
        
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    # Adjust workers if we have fewer items than workers
    max_workers = min(max_workers, n_items)
    
    # Auto-determine chunk size if not specified
    if chunk_size is None:
        chunk_size = max(1, n_items // (max_workers * 4))
    
    logger.info(f"Running parallel map with {max_workers} workers, chunk_size={chunk_size}")
    
    # For process pools, we need a simpler approach - progress tracking with shared
    # objects like multiprocessing.Value causes issues with serialization
    if use_threads:
        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor
        # Disable progress tracking for process pools to avoid serialization issues
        progress_callback = None
    
    try:
        with executor_class(max_workers=max_workers) as executor:
            # Use map with chunksize for better performance
            results = list(executor.map(
                func, 
                items, 
                chunksize=chunk_size,
                timeout=timeout
            ))
            
        logger.info(f"Parallel map completed successfully with {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in parallel map: {e}")
        # Fall back to sequential processing on error
        logger.warning("Falling back to sequential processing")
        results = []
        for i, item in enumerate(items):
            results.append(func(item))
            if progress_callback:
                progress_callback(min(100, int(100 * (i+1) / n_items)))
        return results


def parallel_compute_ci(method_func: Callable, 
                        tables: List[Tuple[int, int, int, int]],
                        alpha: float = 0.05,
                        **kwargs) -> List[Tuple[float, float]]:
    """
    Compute confidence intervals for multiple tables in parallel.
    
    Args:
        method_func: CI method function to use
        tables: List of 2x2 tables as (a,b,c,d) tuples
        alpha: Significance level
        **kwargs: Additional arguments to pass to the method function
        
    Returns:
        List of (lower, upper) confidence interval tuples
    """
    def process_table(table):
        a, b, c, d = table
        try:
            return method_func(a, b, c, d, alpha=alpha, **kwargs)
        except Exception as e:
            logger.warning(f"Error computing CI for table {table}: {e}")
            return (0.0, float('inf'))  # Return a conservative interval on error
    
    return parallel_map(process_table, tables)


def chunk_parameter_space(theta_range: Tuple[float, float], 
                          n_chunks: int) -> List[Tuple[float, float]]:
    """
    Split a parameter space into chunks for parallel processing.
    
    Args:
        theta_range: (min, max) range of theta values
        n_chunks: Number of chunks to create
        
    Returns:
        List of (min, max) ranges for each chunk
    """
    min_theta, max_theta = theta_range
    if min_theta <= 0:
        min_theta = 1e-8  # Avoid zero or negative values
        
    # Use logarithmic spacing for better distribution
    log_min = np.log(min_theta)
    log_max = np.log(max_theta)
    
    chunk_points = np.exp(np.linspace(log_min, log_max, n_chunks + 1))
    return [(chunk_points[i], chunk_points[i+1]) for i in range(n_chunks)]


def parallel_find_root(func: Callable[[float], float], 
                       target_value: float,
                       theta_range: Tuple[float, float],
                       max_workers: Optional[int] = None,
                       progress_callback: Optional[Callable[[float], None]] = None) -> float:
    """
    Find a root of func(theta) - target_value = 0 using parallel processing.
    
    This function divides the search space and searches for sign changes in parallel.
    
    Args:
        func: Function to find root for
        target_value: Target value for the function
        theta_range: (min, max) range to search within
        max_workers: Maximum number of workers
        progress_callback: Optional callback for progress reporting
        
    Returns:
        Value of theta where func(theta) ≈ target_value
    """
    min_theta, max_theta = theta_range
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    # Create chunks based on available workers
    n_chunks = max_workers * 2  # Create more chunks than workers for better load balancing
    chunks = chunk_parameter_space((min_theta, max_theta), n_chunks)
    
    # Create adjusted function that returns difference from target
    def adjusted_func(theta):
        return func(theta) - target_value
    
    # Evaluate function at chunk boundaries
    logger.info(f"Evaluating {len(chunks)+1} points to locate sign change")
    
    # First stage: evaluate at all chunk boundaries to locate sign changes
    evaluation_points = [min_theta] + [chunk[1] for chunk in chunks]
    
    if progress_callback:
        progress_callback(10)  # Starting evaluation
        
    # Evaluate function at all points
    values = []
    for i, theta in enumerate(evaluation_points):
        values.append(adjusted_func(theta))
        if progress_callback:
            progress_callback(10 + (i / len(evaluation_points)) * 30)
    
    # Find chunks where sign changes
    sign_change_chunks = []
    for i in range(len(chunks)):
        if values[i] * values[i+1] <= 0:  # Sign change detected
            sign_change_chunks.append(chunks[i])
    
    if not sign_change_chunks:
        logger.warning("No sign changes found in the given range")
        # Return value closest to target
        closest_idx = min(range(len(values)), key=lambda i: abs(values[i]))
        if progress_callback:
            progress_callback(100)
        return evaluation_points[closest_idx]
    
    logger.info(f"Found {len(sign_change_chunks)} chunks with sign changes")
    
    # Second stage: perform binary search in each chunk with sign change
    def binary_search(chunk):
        lo, hi = chunk
        for _ in range(20):  # Maximum iterations
            mid = (lo + hi) / 2
            f_mid = adjusted_func(mid)
            if abs(f_mid) < 1e-6:
                return mid
            elif f_mid * adjusted_func(lo) <= 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    
    # Process each sign change chunk
    if len(sign_change_chunks) == 1:
        # Only one sign change, do binary search directly
        result = binary_search(sign_change_chunks[0])
        if progress_callback:
            progress_callback(100)
        return result
    else:
        # Multiple sign changes, search in parallel
        results = parallel_map(
            binary_search, 
            sign_change_chunks,
            progress_callback=lambda p: progress_callback(40 + p * 0.6) if progress_callback else None
        )
        
        # Get result closest to target value
        closest = min(results, key=lambda x: abs(adjusted_func(x)))
        if progress_callback:
            progress_callback(100)
        return closest
