#!/usr/bin/env python3
"""
Scaling benchmark for conditional method with different table sizes.
"""

from exactcis.methods.conditional import exact_ci_conditional
import time
import logging
import statistics

# Set logging level to reduce output
logging.basicConfig(level=logging.WARNING)

def benchmark_table(a, b, c, d, name, runs=3):
    """Benchmark a specific table configuration."""
    print(f"\nBenchmarking {name} table: ({a}, {b}, {c}, {d})")
    
    # Warm-up run
    _ = exact_ci_conditional(a, b, c, d)
    
    # Timed runs
    times = []
    for i in range(runs):
        start = time.perf_counter()
        result = exact_ci_conditional(a, b, c, d)
        end = time.perf_counter()
        execution_time = (end - start) * 1000  # Convert to ms
        times.append(execution_time)
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    
    return {
        'table': (a, b, c, d),
        'name': name,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'times': times
    }

def main():
    """Run scaling benchmark with different table sizes."""
    print("Conditional Method Scaling Benchmark")
    print("===================================")
    
    # Reset cache between benchmarks
    from exactcis.utils.shared_cache import reset_shared_cache
    
    # Define test cases from small to very large
    test_cases = [
        (2, 3, 4, 5, "very_small"),
        (10, 15, 12, 18, "medium_balanced"),
        (50, 75, 60, 90, "large_balanced"),
        (100, 200, 150, 300, "very_large")
    ]
    
    results = []
    
    # Run benchmarks
    for a, b, c, d, name in test_cases:
        # Reset cache before each test case
        reset_shared_cache()
        result = benchmark_table(a, b, c, d, name)
        results.append(result)
    
    # Summary
    print("\nScaling Summary:")
    print("===============")
    print(f"{'Table Size':<15} {'Total N':<10} {'Avg Time (ms)':<15}")
    print("-" * 40)
    
    for result in results:
        a, b, c, d = result['table']
        total_n = a + b + c + d
        print(f"{result['name']:<15} {total_n:<10} {result['avg_time']:<15.2f}")

if __name__ == "__main__":
    main()