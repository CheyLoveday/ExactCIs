#!/usr/bin/env python

"""
Test profiler for ExactCIs.
"""

import subprocess
import time
import sys
from collections import defaultdict
import statistics
import os

def run_test(test_path, test_name, runs=3):
    """Run a specific test multiple times and measure performance."""
    results = []
    for i in range(runs):
        cmd = f"uv run pytest {test_path}::{test_name} -v"
        print(f"Run {i+1}/{runs}: {cmd}")
        start_time = time.time()
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        duration = time.time() - start_time
        if process.returncode != 0:
            print(f"Test failed with exit code {process.returncode}")
            print("STDOUT:")
            print(process.stdout)
            print("STDERR:")
            print(process.stderr)
            return None
        results.append(duration)
        print(f"Duration: {duration:.3f} seconds")

    stats = {
        "min": min(results),
        "max": max(results),
        "mean": statistics.mean(results),
        "median": statistics.median(results),
        "stdev": statistics.stdev(results) if len(results) > 1 else 0
    }
    return stats

def profile_tests(tests):
    """Profile a list of tests."""
    results = defaultdict(dict)
    for test_file, test_name in tests:
        print(f"\n=== Profiling {test_file}::{test_name} ===")
        stats = run_test(test_file, test_name)
        if stats:
            results[test_file][test_name] = stats
    return results

def print_results(results):
    """Print performance results in order."""
    print("\n=== Test Performance Results (ordered by median time) ===\n")

    # Flatten results for sorting
    flat_results = []
    for test_file, tests in results.items():
        for test_name, stats in tests.items():
            flat_results.append((test_file, test_name, stats))

    # Sort by median time
    flat_results.sort(key=lambda x: x[2]["median"])

    # Print tabular results
    print(f"{'Test':<60} {'Median (s)':<10} {'Mean (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'StdDev (s)':<10}")
    print("-" * 110)

    for test_file, test_name, stats in flat_results:
        print(f"{test_file}::{test_name:<40} {stats['median']:<10.3f} {stats['mean']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f} {stats['stdev']:<10.3f}")

    # Save results to file
    with open("profile_results.txt", "w") as f:
        f.write(f"{'Test':<60} {'Median (s)':<10} {'Mean (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'StdDev (s)':<10}\n")
        f.write("-" * 110 + "\n")
        for test_file, test_name, stats in flat_results:
            f.write(f"{test_file}::{test_name:<40} {stats['median']:<10.3f} {stats['mean']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f} {stats['stdev']:<10.3f}\n")

def main():
    # Define tests to profile based on the issue description
    tests = [
        # Very Fast Tests (milliseconds)
        ("tests/test_utils/test_stats.py", "test_normal_quantile_basic"),
        ("tests/test_core.py", "test_validate_counts_valid"),
        ("tests/test_core.py", "test_support"),
        ("tests/test_methods/test_wald.py", "test_ci_wald_haldane_basic"),

        # Moderately Fast Tests (milliseconds to seconds)
        ("tests/test_core.py", "test_find_root"),
        ("tests/test_core.py", "test_pmf_weights"),
        ("tests/test_core.py", "test_find_smallest_theta"),
        ("tests/test_methods/test_conditional.py", "test_exact_ci_conditional_basic"),

        # Medium Tests (seconds)
        ("tests/test_methods/test_midp.py", "test_exact_ci_midp_basic"),
        ("tests/test_methods/test_blaker.py", "test_exact_ci_blaker_basic"),
        ("tests/test_exactcis.py", "test_compute_all_cis"),

        # Slow Tests (seconds to minutes)
        ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_basic"),
        ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_small_counts"),
        ("tests/test_integration.py", "test_readme_example"),

        # Very Slow Tests (minutes+) - commented out to avoid long run times
        # ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_grid_size"),
        # ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_numpy_fallback"),
        # ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_moderate_imbalance"),
        # ("tests/test_integration.py", "test_large_imbalance"),
    ]

    # Allow running a subset of tests via environment variable
    test_subset = os.environ.get("PROFILE_TESTS", "all")
    if test_subset != "all":
        categories = {
            "fast": tests[:4],
            "moderate": tests[4:8],
            "medium": tests[8:11],
            "slow": tests[11:14],
            # "very_slow": tests[14:],
        }
        if test_subset in categories:
            tests = categories[test_subset]
            print(f"Running only {test_subset} tests")
        else:
            print(f"Unknown test subset: {test_subset}. Using all tests.")

    # Run profiling
    results = profile_tests(tests)

    # Print and save results
    print_results(results)

if __name__ == "__main__":
    main()
