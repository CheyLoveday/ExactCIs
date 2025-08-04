#!/usr/bin/env python3
"""
Benchmark script to measure the performance improvement from vectorized bracket expansion.
This script compares the performance with the previous benchmark results.
"""

import time
import statistics
import logging
import json
import os
from pathlib import Path
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

def load_baseline_results():
    """Load baseline results from previous benchmark if available."""
    baseline_file = Path("baseline_benchmark_results.json")
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            return json.load(f)
    return None

def save_results(results, filename="vectorized_benchmark_results.json"):
    """Save benchmark results to a file."""
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        serializable_result['times'] = [float(t) for t in result['times']]
        serializable_result['results'] = [list(r) for r in result['results']]
        serializable_result['table'] = list(result['table'])
        serializable_results.append(serializable_result)
    
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def main():
    """Run benchmark with different table sizes and characteristics."""
    print("Vectorized Bracket Expansion Benchmark")
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
    
    # Save results
    save_results(results)
    
    # Load baseline results if available
    baseline_results = load_baseline_results()
    
    # Summary
    print("\nPerformance Summary:")
    print("===================")
    
    if baseline_results:
        print(f"{'Table Type':<20} {'Baseline (ms)':<15} {'Vectorized (ms)':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Create a lookup for baseline results
        baseline_lookup = {r['name']: r for r in baseline_results}
        
        total_baseline = 0
        total_vectorized = 0
        
        for result in results:
            name = result['name']
            if name in baseline_lookup:
                baseline_time = baseline_lookup[name]['avg_time']
                vectorized_time = result['avg_time']
                
                improvement = (baseline_time - vectorized_time) / baseline_time * 100
                
                print(f"{name:<20} {baseline_time:<15.2f} {vectorized_time:<15.2f} {improvement:+.2f}%")
                
                total_baseline += baseline_time
                total_vectorized += vectorized_time
            else:
                print(f"{name:<20} {'N/A':<15} {result['avg_time']:<15.2f} {'N/A':<15}")
        
        # Overall improvement
        if total_baseline > 0:
            overall_improvement = (total_baseline - total_vectorized) / total_baseline * 100
            print("-" * 70)
            print(f"{'OVERALL':<20} {total_baseline/len(results):<15.2f} {total_vectorized/len(results):<15.2f} {overall_improvement:+.2f}%")
    else:
        print(f"{'Table Type':<20} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<20} {result['avg_time']:<15.2f} {result['min_time']:<15.2f} {result['max_time']:<15.2f}")
        
        print("\nNo baseline results found. Run 'python benchmark_bracket_expansion.py' first and rename the output file to 'baseline_benchmark_results.json'.")
    
    print("\nVectorized benchmark complete.")

if __name__ == "__main__":
    main()