#!/usr/bin/env python
"""
CI Method Comparator for ExactCIs.

This script:
1. Compares performance across different confidence interval methods
2. Validates consistency of results across methods
3. Profiles functions to identify performance bottlenecks
4. Suggests targeted optimization strategies
"""

import time
import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from functools import wraps
import argparse
import os
from pathlib import Path
import json

# Import all methods from exactcis
from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Create directories for results
os.makedirs("profiling_results", exist_ok=True)
os.makedirs("benchmark_results", exist_ok=True)


def time_function(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    return wrapper


@time_function
def run_blaker(a, b, c, d, alpha):
    """Run Blaker's method and time it."""
    return exact_ci_blaker(a, b, c, d, alpha)


@time_function
def run_conditional(a, b, c, d, alpha):
    """Run conditional method and time it."""
    return exact_ci_conditional(a, b, c, d, alpha)


@time_function
def run_midp(a, b, c, d, alpha):
    """Run mid-p method and time it."""
    return exact_ci_midp(a, b, c, d, alpha)


@time_function
def run_unconditional(a, b, c, d, alpha, grid_size=20):
    """Run unconditional method and time it."""
    return exact_ci_unconditional(a, b, c, d, alpha, grid_size=grid_size)


@time_function
def run_wald(a, b, c, d, alpha):
    """Run Wald's method and time it."""
    return ci_wald_haldane(a, b, c, d, alpha)


def profile_method(method_func, a, b, c, d, alpha, output_file):
    """Profile a specific method and save results."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the method
    result = method_func(a, b, c, d, alpha)
    
    profiler.disable()
    
    # Save profile results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    
    with open(output_file, 'w') as f:
        f.write(s.getvalue())
    
    return result


def compare_methods(test_cases):
    """Compare all methods against the test cases."""
    results = []
    
    print("\nRunning method comparison...")
    
    for i, case in enumerate(test_cases):
        a, b, c, d, alpha = case
        
        print(f"\nCase {i+1}: ({a}, {b}, {c}, {d}) with alpha={alpha}")
        
        # Run all methods with timing
        blaker_result, blaker_time = run_blaker(a, b, c, d, alpha)
        conditional_result, conditional_time = run_conditional(a, b, c, d, alpha)
        midp_result, midp_time = run_midp(a, b, c, d, alpha)
        unconditional_result, unconditional_time = run_unconditional(a, b, c, d, alpha)
        wald_result, wald_time = run_wald(a, b, c, d, alpha)
        
        # Store results
        case_results = {
            "case": i+1,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "alpha": alpha,
            "blaker_lower": blaker_result[0],
            "blaker_upper": blaker_result[1],
            "blaker_time": blaker_time,
            "conditional_lower": conditional_result[0],
            "conditional_upper": conditional_result[1],
            "conditional_time": conditional_time,
            "midp_lower": midp_result[0],
            "midp_upper": midp_result[1],
            "midp_time": midp_time,
            "unconditional_lower": unconditional_result[0],
            "unconditional_upper": unconditional_result[1],
            "unconditional_time": unconditional_time,
            "wald_lower": wald_result[0],
            "wald_upper": wald_result[1],
            "wald_time": wald_time
        }
        
        results.append(case_results)
        
        print(f"  Blaker:        ({blaker_result[0]:.4f}, {blaker_result[1]:.4f}) - {blaker_time:.4f}s")
        print(f"  Conditional:   ({conditional_result[0]:.4f}, {conditional_result[1]:.4f}) - {conditional_time:.4f}s")
        print(f"  Mid-P:         ({midp_result[0]:.4f}, {midp_result[1]:.4f}) - {midp_time:.4f}s")
        print(f"  Unconditional: ({unconditional_result[0]:.4f}, {unconditional_result[1]:.4f}) - {unconditional_time:.4f}s")
        print(f"  Wald:          ({wald_result[0]:.4f}, {wald_result[1]:.4f}) - {wald_time:.4f}s")
    
    return results


def profile_slow_methods(test_cases):
    """Profile the methods identified as slow."""
    print("\nProfiling slow methods...")
    
    for i, case in enumerate(test_cases):
        a, b, c, d, alpha = case
        
        # Profile the slow methods
        profile_method(exact_ci_blaker, a, b, c, d, alpha, 
                      f"profiling_results/blaker_case{i+1}.prof")
        profile_method(exact_ci_unconditional, a, b, c, d, alpha, 
                      f"profiling_results/unconditional_case{i+1}.prof")
        
        print(f"Profiled case {i+1} - results in profiling_results/")


def generate_summary(results):
    """Generate a summary of the method comparison results."""
    df = pd.DataFrame(results)
    
    # Calculate average execution times
    avg_times = {
        "Blaker": df["blaker_time"].mean(),
        "Conditional": df["conditional_time"].mean(),
        "Mid-P": df["midp_time"].mean(),
        "Unconditional": df["unconditional_time"].mean(),
        "Wald": df["wald_time"].mean()
    }
    
    # Calculate interval width (upper - lower)
    df["blaker_width"] = df["blaker_upper"] - df["blaker_lower"]
    df["conditional_width"] = df["conditional_upper"] - df["conditional_lower"]
    df["midp_width"] = df["midp_upper"] - df["midp_lower"]
    df["unconditional_width"] = df["unconditional_upper"] - df["unconditional_lower"]
    df["wald_width"] = df["wald_upper"] - df["wald_lower"]
    
    # Calculate average interval width
    avg_widths = {
        "Blaker": df["blaker_width"].mean(),
        "Conditional": df["conditional_width"].mean(),
        "Mid-P": df["midp_width"].mean(),
        "Unconditional": df["unconditional_width"].mean(),
        "Wald": df["wald_width"].mean()
    }
    
    # Create summary table
    summary = []
    for method in ["Blaker", "Conditional", "Mid-P", "Unconditional", "Wald"]:
        summary.append({
            "Method": method,
            "Avg Time (s)": avg_times[method],
            "Avg Interval Width": avg_widths[method],
            "Time Ratio (vs fastest)": avg_times[method] / min(avg_times.values())
        })
    
    # Create performance factor table
    time_table = tabulate(
        summary,
        headers="keys",
        tablefmt="grid",
        floatfmt=".4f"
    )
    
    print("\n===== METHOD PERFORMANCE SUMMARY =====")
    print(time_table)
    
    # Create a plot of the times
    methods = list(avg_times.keys())
    times = list(avg_times.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, times)
    plt.title('Average Execution Time by Method')
    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig('benchmark_results/method_comparison.png')
    
    # Save results to file
    df.to_csv("benchmark_results/full_results.csv", index=False)
    
    with open("benchmark_results/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    return summary


def identify_bottlenecks():
    """Analyze profiling results to identify bottlenecks."""
    print("\n===== PERFORMANCE BOTTLENECKS =====")
    
    prof_files = [f for f in os.listdir("profiling_results") if f.endswith(".prof")]
    
    for prof_file in prof_files:
        print(f"\nAnalyzing {prof_file}...")
        
        # Read the profiling data
        p = pstats.Stats(os.path.join("profiling_results", prof_file))
        
        # Get top 5 time-consuming functions
        s = io.StringIO()
        ps = pstats.Stats(p, stream=s).sort_stats('cumulative')
        ps.print_stats(5)
        
        # Extract and print the important lines
        lines = s.getvalue().strip().split('\n')
        for i, line in enumerate(lines):
            if i >= 5 and i < 10:  # Skip header lines, show top 5 functions
                print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Compare CI methods in ExactCIs")
    parser.add_argument("--profile", action="store_true", help="Profile slow methods")
    parser.add_argument("--cases", type=int, default=5, help="Number of test cases to run")
    args = parser.parse_args()
    
    # Define test cases (a, b, c, d, alpha)
    test_cases = [
        (12, 5, 8, 10, 0.05),  # Common example from README
        (5, 5, 5, 5, 0.05),    # Balanced case
        (2, 20, 4, 10, 0.05),  # Unbalanced case
        (0, 10, 1, 20, 0.05),  # Zero in one cell
        (50, 10, 30, 20, 0.05) # Larger counts
    ]
    
    # Select a subset of test cases if requested
    if args.cases < len(test_cases):
        test_cases = test_cases[:args.cases]
    
    # Run comparison
    print("=== ExactCIs Method Comparison ===")
    results = compare_methods(test_cases)
    
    # Generate summary
    summary = generate_summary(results)
    
    # Profile slow methods if requested
    if args.profile:
        profile_slow_methods(test_cases)
        identify_bottlenecks()
    
    # Print recommendations
    print("\n===== OPTIMIZATION RECOMMENDATIONS =====")
    
    # Sort methods by execution time
    methods_by_time = sorted(summary, key=lambda x: x["Avg Time (s)"])
    slowest_method = methods_by_time[-1]["Method"]
    slowest_time = methods_by_time[-1]["Avg Time (s)"]
    second_slowest = methods_by_time[-2]["Method"]
    second_slowest_time = methods_by_time[-2]["Avg Time (s)"]
    
    print(f"1. Focus on optimizing the {slowest_method} method ({slowest_time:.4f}s)")
    print(f"2. Then consider optimizing {second_slowest} method ({second_slowest_time:.4f}s)")
    print("3. Consider these optimization strategies:")
    print("   - Add caching with @lru_cache to expensive functions")
    print("   - Use vectorized operations with NumPy where possible")
    print("   - Optimize grid search parameters for speed/accuracy trade-offs")
    print("   - Implement early stopping for convergence-based algorithms")
    print("")
    print("See detailed profiling results in the profiling_results/ directory")
    print("and benchmark results in the benchmark_results/ directory")


if __name__ == "__main__":
    main()
