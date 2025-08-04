"""
Benchmark script for parallel processing improvements in ExactCIs.

This script benchmarks the performance of the parallel processing improvements
for various workloads and documents the results.
"""

import time
import logging
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.conditional import exact_ci_conditional
from exactcis.utils.parallel import (
    parallel_compute_ci,
    has_numba_support
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tables(n_tables: int, seed: int = 42) -> List[Tuple[int, int, int, int]]:
    """
    Generate random 2x2 tables for benchmarking.
    
    Args:
        n_tables: Number of tables to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (a, b, c, d) tuples representing 2x2 tables
    """
    np.random.seed(seed)
    
    # Generate random counts between 1 and 100
    tables = []
    for _ in range(n_tables):
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        c = np.random.randint(1, 100)
        d = np.random.randint(1, 100)
        tables.append((a, b, c, d))
    
    return tables


def benchmark_method(
    method_func: Any,
    tables: List[Tuple[int, int, int, int]],
    backend: Optional[str] = None,
    min_batch_size: int = 4,
    n_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark a confidence interval method with parallel processing.
    
    Args:
        method_func: CI method function to benchmark
        tables: List of tables to process
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        min_batch_size: Minimum batch size for parallel processing
        n_runs: Number of runs to average over
        
    Returns:
        Dictionary with benchmark results
    """
    method_name = method_func.__name__
    n_tables = len(tables)
    
    logger.info(f"Benchmarking {method_name} with {n_tables} tables, backend={backend}, min_batch_size={min_batch_size}")
    
    # Run the benchmark multiple times and take the average
    times = []
    for i in range(n_runs):
        start_time = time.time()
        
        # Don't pass min_batch_size to parallel_compute_ci as it would be passed to the CI method
        results = parallel_compute_ci(
            method_func,
            tables,
            backend=backend
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        logger.info(f"Run {i+1}/{n_runs}: {elapsed_time:.2f} seconds")
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    logger.info(f"Average time: {avg_time:.2f} seconds (min={min_time:.2f}, max={max_time:.2f}, std={std_time:.2f})")
    
    return {
        "method": method_name,
        "n_tables": n_tables,
        "backend": backend,
        "min_batch_size": min_batch_size,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    # Check if Numba is available
    has_numba = has_numba_support()
    logger.info(f"Numba support detected: {has_numba}")
    
    # Define workloads
    workloads = [
        {"n_tables": 10, "name": "small"},
        {"n_tables": 100, "name": "medium"},
        {"n_tables": 1000, "name": "large"}
    ]
    
    # Define methods to benchmark
    methods = [
        exact_ci_midp,
        exact_ci_blaker,
        exact_ci_conditional
    ]
    
    # Define backends to test
    backends = [None, "thread", "process"]
    
    # Define batch sizes to test
    batch_sizes = [4, 8, 16]
    
    # Store results
    all_results = []
    
    # Run benchmarks for each combination
    for workload in workloads:
        n_tables = workload["n_tables"]
        name = workload["name"]
        
        logger.info(f"Generating {n_tables} tables for {name} workload")
        tables = generate_tables(n_tables)
        
        for method_func in methods:
            for backend in backends:
                for batch_size in batch_sizes:
                    # Skip large workloads for some combinations to save time
                    if n_tables == 1000 and batch_size > 4:
                        continue
                    
                    result = benchmark_method(
                        method_func,
                        tables,
                        backend=backend,
                        min_batch_size=batch_size
                    )
                    
                    # Add workload info to result
                    result["workload"] = name
                    
                    all_results.append(result)
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("=================")
    
    # Group by workload and method
    for workload in workloads:
        name = workload["name"]
        logger.info(f"\n{name.upper()} WORKLOAD ({workload['n_tables']} tables)")
        
        for method_func in methods:
            method_name = method_func.__name__
            logger.info(f"\n  {method_name}:")
            
            # Filter results for this workload and method
            filtered_results = [r for r in all_results if r["workload"] == name and r["method"] == method_name]
            
            # Sort by average time
            filtered_results.sort(key=lambda r: r["avg_time"])
            
            # Print top 3 configurations
            for i, result in enumerate(filtered_results[:3]):
                logger.info(f"    #{i+1}: backend={result['backend']}, batch_size={result['min_batch_size']}, time={result['avg_time']:.2f}s")
    
    # Print best overall configuration for each method
    logger.info("\nBest Overall Configurations:")
    logger.info("==========================")
    
    for method_func in methods:
        method_name = method_func.__name__
        
        # Filter results for this method
        filtered_results = [r for r in all_results if r["method"] == method_name]
        
        # Group by backend and batch size
        configs = {}
        for result in filtered_results:
            key = (result["backend"], result["min_batch_size"])
            if key not in configs:
                configs[key] = []
            configs[key].append(result["avg_time"])
        
        # Calculate average time across all workloads for each configuration
        avg_configs = {k: np.mean(v) for k, v in configs.items()}
        
        # Sort by average time
        sorted_configs = sorted(avg_configs.items(), key=lambda x: x[1])
        
        # Print best configuration
        best_config, best_time = sorted_configs[0]
        logger.info(f"  {method_name}: backend={best_config[0]}, batch_size={best_config[1]}, avg_time={best_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark parallel processing improvements")
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark with fewer combinations")
    args = parser.parse_args()
    
    if args.quick:
        # Override the run_benchmarks function with a quicker version
        def run_benchmarks():
            # Check if Numba is available
            has_numba = has_numba_support()
            logger.info(f"Numba support detected: {has_numba}")
            
            # Generate a small set of tables
            tables = generate_tables(20)
            
            # Benchmark each method with auto-detected backend
            for method_func in [exact_ci_midp, exact_ci_blaker, exact_ci_conditional]:
                benchmark_method(method_func, tables, backend=None, n_runs=2)
    
    # Run the benchmarks
    run_benchmarks()