#!/usr/bin/env python3
"""
Benchmark script to measure the performance of bracket expansion in conditional method.
This establishes a baseline before implementing vectorized expansion.
"""

import time
import statistics
import logging
from exactcis.methods.conditional import exact_ci_conditional
from exactcis.utils.shared_cache import reset_shared_cache

# Set logging level to reduce output
logging.basicConfig(level=logging.WARNING)

def benchmark_table(a, b, c, d, name, runs=5):
    """Benchmark a specific table configuration."""
    print(f"\nBenchmarking {name} table: ({a}, {b}, {c}, {d})")
    
    # Warm-up run to initialize cache
    _ = exact_ci_conditional(a, b, c, d)
    
    # Timed runs
    times = []
    results = []
    
    for i in range(runs):
        start = time.perf_counter()
        result = exact_ci_conditional(a, b, c, d)
        end = time.perf_counter()
        
        execution_time = (end - start) * 1000  # Convert to ms
        times.append(execution_time)
        results.append(result)
        
        print(f"  Run {i+1}: {execution_time:.2f}ms, Result: {result}")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Std deviation: {std_dev:.2f}ms")
    
    return {
        'table': (a, b, c, d),
        'name': name,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_dev': std_dev,
        'times': times,
        'results': results
    }

def main():
    """Run benchmark with different table sizes and characteristics."""
    print("Bracket Expansion Benchmark - Baseline")
    print("=====================================")
    
    # Define test cases that require bracket expansion
    # These cases are chosen to trigger different bracket expansion scenarios
    test_cases = [
        # Standard cases
        (10, 15, 12, 18, "medium_balanced"),
        (50, 75, 60, 90, "large_balanced"),
        (100, 200, 150, 300, "very_large"),
        
        # Cases likely to need extensive bracket expansion
        (1, 20, 2, 40, "extreme_imbalanced"),
        (30, 5, 40, 2, "reverse_imbalanced"),
        (1, 1, 20, 20, "sparse_case")
    ]
    
    results = []
    
    # Run benchmarks
    for a, b, c, d, name in test_cases:
        # Reset cache before each test case
        reset_shared_cache()
        result = benchmark_table(a, b, c, d, name)
        results.append(result)
    
    # Summary
    print("\nPerformance Summary:")
    print("===================")
    print(f"{'Table Type':<20} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<20} {result['avg_time']:<15.2f} {result['min_time']:<15.2f} {result['max_time']:<15.2f}")
    
    print("\nBaseline benchmark complete. This data will be used to compare with vectorized implementation.")

if __name__ == "__main__":
    main()