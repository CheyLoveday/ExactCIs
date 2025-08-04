#!/usr/bin/env python
"""
Baseline performance profiling script for ExactCIs.
This script measures the performance of key functions targeted for optimization.
"""

import sys
import os
import time
import cProfile
import pstats
import numpy as np
import json
import argparse
from functools import wraps
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from exactcis.core import nchg_pdf, pmf, pmf_weights
from exactcis.methods.blaker import (
    blaker_p_value, 
    blaker_acceptability,
    exact_ci_blaker,
    exact_ci_blaker_batch
)
from exactcis.core import support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store timing results
timing_results = {}

def time_function(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store the timing result
        backend = kwargs.get('backend', 'default')
        if func.__name__ not in timing_results:
            timing_results[func.__name__] = {}
        timing_results[func.__name__][backend] = execution_time
        
        logger.info(f"{func.__name__} took {execution_time:.6f} seconds (backend: {backend})")
        return result
    return wrapper

def profile_function(func, *args, **kwargs):
    """Profile a function and print stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)  # Print top 20 functions by cumulative time
    
    return result

@time_function
def test_nchg_pdf():
    """Test the performance of nchg_pdf function."""
    n1, n2, m1 = 10, 20, 15
    k_values = np.array(range(max(0, m1 - n2), min(m1, n1) + 1))
    
    # Run multiple times to get a better average
    for _ in range(100):
        nchg_pdf(k_values, n1, n2, m1, 1.5)

@time_function
def test_blaker_p_value():
    """Test the performance of blaker_p_value function."""
    a, n1, n2, m1 = 5, 10, 20, 15
    theta = 1.5
    s = support(n1, n2, m1)
    
    # Run multiple times to get a better average
    for _ in range(100):
        blaker_p_value(a, n1, n2, m1, theta, s)

@time_function
def test_exact_ci_blaker():
    """Test the performance of exact_ci_blaker function."""
    a, b, c, d = 5, 5, 10, 10
    
    # Run a few times to get a better average
    for _ in range(5):
        exact_ci_blaker(a, b, c, d)

@time_function
def test_exact_ci_blaker_batch(backend=None):
    """Test the performance of exact_ci_blaker_batch function with specified backend."""
    tables = [
        (5, 5, 10, 10),
        (10, 10, 15, 15),
        (2, 3, 4, 5),
        (7, 8, 9, 10),
        (12, 13, 14, 15)
    ]
    
    # Use the specified backend
    exact_ci_blaker_batch(tables, backend=backend)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Baseline performance profiling for ExactCIs")
    
    parser.add_argument(
        "--backend", 
        choices=["thread", "process", "auto"], 
        default="auto",
        help="Backend to use for parallel processing (thread, process, or auto)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="timing_results.json",
        help="Output file for timing results in JSON format"
    )
    
    parser.add_argument(
        "--skip-core", 
        action="store_true",
        help="Skip core function tests (nchg_pdf, blaker_p_value)"
    )
    
    return parser.parse_args()

def main():
    """Run all performance tests."""
    args = parse_args()
    
    # Convert 'auto' to None for the backend parameter
    backend = None if args.backend == "auto" else args.backend
    
    logger.info(f"Starting baseline performance profiling with backend={args.backend}")
    
    if not args.skip_core:
        logger.info("\n--- Testing nchg_pdf ---")
        test_nchg_pdf()
        
        logger.info("\n--- Testing blaker_p_value ---")
        test_blaker_p_value()
    
    logger.info("\n--- Testing exact_ci_blaker ---")
    test_exact_ci_blaker()
    
    logger.info("\n--- Testing exact_ci_blaker_batch ---")
    test_exact_ci_blaker_batch(backend=backend)
    
    # Test with other backends for comparison if auto was selected
    if args.backend == "auto":
        logger.info("\n--- Testing exact_ci_blaker_batch with thread backend ---")
        test_exact_ci_blaker_batch(backend="thread")
        
        logger.info("\n--- Testing exact_ci_blaker_batch with process backend ---")
        test_exact_ci_blaker_batch(backend="process")
    
    logger.info("\n--- Detailed profiling of exact_ci_blaker ---")
    a, b, c, d = 5, 5, 10, 10
    profile_function(exact_ci_blaker, a, b, c, d)
    
    # Save timing results to JSON file
    with open(args.output, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    logger.info(f"Baseline performance profiling completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()