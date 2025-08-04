"""
Parallel processing utilities for ExactCIs.

This module provides parallelization helpers to speed up computation-intensive
operations in the ExactCIs package.
"""

import logging
import multiprocessing
import numpy as np
from functools import partial
from typing import Callable, List, Dict, Any, Tuple, Optional, Union, Literal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)


def has_numba_support() -> bool:
    """
    Check if Numba is available and can be used for acceleration.
    
    Returns:
        bool: True if Numba is available, False otherwise
    """
    try:
        # Try to import Numba
        import numba
        return True
    except ImportError:
        return False
    
    
def get_optimal_backend(force_processes: bool = False) -> str:
    """
    Determine the optimal backend based on the task and environment.
    
    Args:
        force_processes: Force use of processes even if Numba is available
        
    Returns:
        str: 'thread' if Numba is available and not forcing processes, 'process' otherwise
    """
    if not force_processes and has_numba_support():
        return 'thread'
    return 'process'


def get_optimal_workers():
    """
    Determine the optimal number of worker processes.
    
    Returns 75% of available cores, but at least 2 and at most 8.
    """
    cores = multiprocessing.cpu_count()
    return max(2, min(8, int(cores * 0.75)))


def parallel_map(func: Callable, items: List[Any], 
                 backend: Optional[str] = None,
                 use_threads: bool = False,  # Deprecated, use backend='thread' instead
                 max_workers: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 min_batch_size: int = 4,  # Minimum items per worker for efficient batching
                 timeout: Optional[float] = None,
                 progress_callback: Optional[Callable[[float], None]] = None,
                 force_processes: bool = False) -> List[Any]:
    """
    Execute a function over a list of items in parallel.
    
    Args:
        func: Function to execute
        items: List of items to process
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        use_threads: [Deprecated] Use threads instead of processes (better for I/O bound tasks)
        max_workers: Maximum number of workers (default: auto-determined)
        chunk_size: Size of chunks for processing (default: auto-determined)
        min_batch_size: Minimum number of items per worker for efficient batching (default: 4)
        timeout: Maximum time to wait for completion in seconds
        progress_callback: Optional callback to report progress (0-100)
        force_processes: Force use of processes for CPU-bound tasks (default: False)
        
    Returns:
        List of results in the same order as input items
        
    Note:
        Error Handling: If parallel execution fails (timeout, worker errors, etc.),
        the function automatically falls back to sequential processing to ensure
        all items are processed. This fallback behavior ensures robustness but
        may result in longer execution times.
        
        Backend Selection: The 'thread' backend is more efficient for Numba-accelerated
        functions that release the GIL. The 'process' backend is better for CPU-bound
        tasks that don't release the GIL. If not specified, the backend is auto-detected
        based on the presence of Numba.
        
        Batch Size Autotuning: The function automatically adjusts the number of workers
        and chunk size to ensure each worker gets at least min_batch_size items.
        This reduces overhead for small batches and improves performance.
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
    
    # Batch size autotuning: ensure each worker gets at least min_batch_size items
    if n_items < max_workers * min_batch_size:
        # Reduce number of workers to ensure each gets at least min_batch_size items
        adjusted_workers = max(1, n_items // min_batch_size)
        logger.info(f"Reducing workers from {max_workers} to {adjusted_workers} to ensure each worker gets at least {min_batch_size} items")
        max_workers = adjusted_workers
    
    # Adjust workers if we have fewer items than workers (safety check)
    max_workers = min(max_workers, n_items)
    
    # Auto-determine chunk size if not specified
    if chunk_size is None:
        # Ensure chunk_size is at least min_batch_size
        chunk_size = max(min_batch_size, n_items // max_workers)
        logger.info(f"Auto-determined chunk_size={chunk_size} to ensure efficient batching")
    elif chunk_size < min_batch_size:
        # If specified chunk_size is too small, adjust it
        logger.info(f"Increasing chunk_size from {chunk_size} to {min_batch_size} for efficient batching")
        chunk_size = min_batch_size
    
    logger.info(f"Running parallel map with {max_workers} workers, chunk_size={chunk_size}")
    
    # Determine which backend to use
    if backend is None:
        # Handle deprecated use_threads parameter for backward compatibility
        if use_threads and not force_processes:
            selected_backend = 'thread'
        else:
            selected_backend = get_optimal_backend(force_processes)
    else:
        selected_backend = backend.lower()
    
    # Select executor class based on backend
    if selected_backend == 'thread':
        executor_class = ThreadPoolExecutor
        logger.info(f"Using ThreadPoolExecutor for parallelization")
    else:  # 'process' or any other value
        executor_class = ProcessPoolExecutor
        # Disable progress tracking for process pools to avoid serialization issues
        if progress_callback is not None:
            logger.info(f"Progress tracking disabled for ProcessPoolExecutor to avoid serialization issues")
            progress_callback = None
        logger.info(f"Using ProcessPoolExecutor for parallelization")
    
    try:
        with executor_class(max_workers=max_workers) as executor:
            # Create a dictionary to track futures
            future_to_index = {}
            
            # Submit all tasks
            for i, item in enumerate(items):
                future = executor.submit(func, item)
                future_to_index[future] = i
            
            # Process results as they complete
            results = [None] * len(items)  # Pre-allocate result list
            completed = 0
            
            for future in as_completed(future_to_index.keys(), timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.warning(f"Error in task {index}: {e}")
                    # Handle individual task failure without failing the entire batch
                    try:
                        # Fall back to sequential for this item
                        results[index] = func(items[index])
                        logger.info(f"Successfully recovered task {index} with sequential processing")
                    except Exception as e2:
                        logger.error(f"Failed to recover task {index}: {e2}")
                        # Return a default value or raise an exception
                        results[index] = None  # Or some other default value
                
                completed += 1
                if progress_callback:
                    progress_callback(min(100, int(100 * completed / len(items))))
            
            # Check if any tasks are still None (timed out or failed)
            for i, result in enumerate(results):
                if result is None:
                    logger.warning(f"Task {i} did not complete (timeout or error). Processing sequentially.")
                    try:
                        results[i] = func(items[i])
                    except Exception as e:
                        logger.error(f"Sequential processing for task {i} also failed: {e}")
                        results[i] = None  # Or some other default value
            
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
                        timeout: Optional[float] = None,
                        backend: Optional[str] = None,
                        max_workers: Optional[int] = None,
                        **kwargs) -> List[Tuple[float, float]]:
    """
    Compute confidence intervals for multiple tables in parallel.
    
    Args:
        method_func: CI method function to use
        tables: List of 2x2 tables as (a,b,c,d) tuples
        alpha: Significance level
        timeout: Maximum time to wait for completion in seconds
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        max_workers: Maximum number of workers (default: auto-determined)
        **kwargs: Additional arguments to pass to the method function
        
    Returns:
        List of (lower, upper) confidence interval tuples
        
    Note:
        Error Handling: If computation fails for any individual table, a
        conservative interval (0.0, inf) is returned for that table to
        ensure the function completes successfully.
        
        Backend Selection: For methods that use Numba-accelerated functions
        (like Blaker's method), the 'thread' backend may be more efficient.
        If not specified, the backend is auto-detected based on the method.
    """
    # Check if the method uses Numba
    method_has_numba = False
    if method_func.__name__ == 'exact_ci_blaker':
        # Blaker's method uses Numba if available
        try:
            from exactcis.utils.jit_functions import HAS_NUMBA
            method_has_numba = HAS_NUMBA
        except ImportError:
            pass
    
    # Auto-select backend if not specified
    if backend is None and method_has_numba:
        backend = 'thread'
        logger.info(f"Auto-selected 'thread' backend for Numba-accelerated method: {method_func.__name__}")
    
    # Use a module-level function to avoid pickle issues
    task_data = [(table, method_func.__name__, alpha, kwargs) for table in tables]
    return parallel_map(
        _process_ci_task, 
        task_data, 
        backend=backend,
        timeout=timeout,
        max_workers=max_workers
    )


def _process_ci_task(task_data):
    """
    Process a single CI computation task. 
    
    This is a module-level function to avoid pickle issues with local functions.
    """
    table, method_name, alpha, kwargs = task_data
    a, b, c, d = table
    
    try:
        # Import the method function by name
        if method_name == 'exact_ci_conditional':
            from exactcis.methods.conditional import exact_ci_conditional
            return exact_ci_conditional(a, b, c, d, alpha=alpha, **kwargs)
        elif method_name == 'exact_ci_blaker':
            from exactcis.methods.blaker import exact_ci_blaker
            return exact_ci_blaker(a, b, c, d, alpha=alpha, **kwargs)
        elif method_name == 'exact_ci_midp':
            from exactcis.methods.midp import exact_ci_midp
            return exact_ci_midp(a, b, c, d, alpha=alpha, **kwargs)
        else:
            logger.error(f"Unknown method: {method_name}")
            return (0.0, float('inf'))
            
    except Exception as e:
        logger.warning(f"Error computing CI for table {table} with {method_name}: {e}")
        return (0.0, float('inf'))


def chunk_parameter_space(theta_range: Tuple[float, float], 
                          n_chunks: int) -> List[Tuple[float, float]]:
    """
    Split a parameter space into chunks for parallel processing.
    
    Uses logarithmic spacing which is appropriate for odds ratio parameters
    that typically span several orders of magnitude (e.g., 0.01 to 100).
    
    Args:
        theta_range: (min, max) range of theta values
        n_chunks: Number of chunks to create
        
    Returns:
        List of (min, max) ranges for each chunk
        
    Note:
        Logarithmic Spacing: This function uses logarithmic rather than linear
        spacing because odds ratios are naturally distributed on a log scale.
        This ensures more balanced computational load when searching across
        the parameter space, as equal log-space intervals represent equal
        multiplicative factors in odds ratio space.
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
        
    Note:
        Edge Case Handling: If no sign changes are found in the given range,
        the function returns the theta value that produces the function value
        closest to the target value. This ensures the function always returns
        a result even when a true root doesn't exist within the search range.
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
