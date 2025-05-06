#!/usr/bin/env python
"""
Advanced profiler for ExactCIs to identify slow functions in the test suite.

This script:
1. Runs selected tests with Python's cProfile
2. Analyzes and sorts the results by cumulative time
3. Identifies the specific functions causing slowdowns
4. Generates a detailed report with improvement recommendations
"""

import cProfile
import pstats
import io
import subprocess
import sys
import os
from pathlib import Path
import time
import argparse
from pstats import SortKey


def get_slow_tests():
    """Find tests marked with @pytest.mark.slow."""
    slow_tests = []
    
    # Run pytest to list all tests with markers
    cmd = "uv run pytest --collect-only -v"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Process output to find slow-marked tests
    for line in result.stdout.split('\n'):
        if 'slow' in line and 'test_' in line:
            # Extract test file and test name
            parts = line.strip().split('::')
            if len(parts) >= 2:
                test_file = parts[0]
                test_name = parts[1]
                slow_tests.append((test_file, test_name))
    
    return slow_tests


def profile_test(test_file, test_name, output_dir='profiling_results'):
    """Profile a single test and save results."""
    print(f"\n[+] Profiling {test_file}::{test_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate profile filename
    test_file_short = Path(test_file).stem
    profile_filename = f"{output_dir}/{test_file_short}_{test_name}.prof"
    stats_filename = f"{output_dir}/{test_file_short}_{test_name}_stats.txt"
    
    # Run the test with cProfile
    cmd = f"uv run python -m cProfile -o {profile_filename} -m pytest {test_file}::{test_name} -v"
    print(f"Running: {cmd}")
    
    start_time = time.time()
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if process.returncode != 0:
        print(f"Error running test: {process.stderr}")
        return None
    
    print(f"Test completed in {duration:.3f} seconds")
    
    # Create readable stats from the profile
    p = pstats.Stats(profile_filename)
    
    # Open a text file for writing the stats
    with open(stats_filename, 'w') as stats_file:
        # Redirect stdout to the file
        sys.stdout = stats_file
        
        # Print standard stats sorted by cumulative time
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
        
        # Print stats sorted by total time
        p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
        
        # Print caller stats for the slowest functions
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_callers(10)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
    
    print(f"Profiling results saved to {profile_filename}")
    print(f"Analysis saved to {stats_filename}")
    
    # Return a summary of the top time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profile_filename, stream=s).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(10)
    
    return {
        'test': f"{test_file}::{test_name}",
        'duration': duration,
        'profile_file': profile_filename,
        'stats_file': stats_filename,
        'top_stats': s.getvalue()
    }


def profile_module(module_path, output_dir='profiling_results'):
    """Profile an entire module to identify slow functions."""
    print(f"\n[+] Profiling module {module_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate profile filename
    module_name = Path(module_path).stem
    profile_filename = f"{output_dir}/{module_name}_module.prof"
    stats_filename = f"{output_dir}/{module_name}_module_stats.txt"
    
    # Import the module and profile it directly
    cmd = f"uv run python -c \"import cProfile; import {module_name.replace('/', '.')}; cProfile.run('import {module_name.replace('/', '.')}', '{profile_filename}')\""
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        
        # Create readable stats from the profile
        p = pstats.Stats(profile_filename)
        
        # Open a text file for writing the stats
        with open(stats_filename, 'w') as stats_file:
            sys.stdout = stats_file
            p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
            p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
            sys.stdout = sys.__stdout__
        
        print(f"Module profiling results saved to {profile_filename}")
        print(f"Analysis saved to {stats_filename}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error profiling module: {e}")


def print_summary(results):
    """Print a summary of profiling results with recommendations."""
    if not results:
        print("\nNo profiling results to display.")
        return
    
    print("\n===== PROFILING SUMMARY =====")
    print("-" * 80)
    print(f"{'Test':<50} {'Duration (s)':<15}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['duration'], reverse=True):
        print(f"{result['test']:<50} {result['duration']:<15.3f}")
    
    print("\n===== SLOW FUNCTIONS IDENTIFIED =====")
    print("Below are the functions consuming the most time across tests:\n")
    
    # Collect and count slow functions across all tests
    slow_functions = {}
    for result in results:
        lines = result['top_stats'].split('\n')
        for line in lines[5:15]:  # Skip header lines, focus on top 10 functions
            if line.strip() and 'function calls' not in line:
                parts = line.strip().split()
                if len(parts) >= 6:
                    func_name = parts[5]
                    if func_name in slow_functions:
                        slow_functions[func_name] += 1
                    else:
                        slow_functions[func_name] = 1
    
    # Display the most commonly occurring slow functions
    for func, count in sorted(slow_functions.items(), key=lambda x: x[1], reverse=True)[:10]:
        if count > 1:  # Show functions that appear in multiple test profiles
            print(f"- {func} (appears in {count} profiles)")
    
    print("\n===== RECOMMENDATIONS =====")
    print("Based on the profiling results, consider these optimizations:")
    print("1. Review the implementation of frequently appearing slow functions")
    print("2. Consider caching results for expensive calculations")
    print("3. Check for unnecessary loops or inefficient algorithms")
    print("4. Look for opportunities to use vectorized operations with NumPy")
    print("5. Run more specific profiling on the identified slow functions")
    print("\nDetailed profiles are available in the profiling_results/ directory")


def main():
    parser = argparse.ArgumentParser(description="Profile ExactCIs tests to identify slow functions")
    parser.add_argument("--test", help="Specific test to profile (format: test_file.py::test_name)")
    parser.add_argument("--module", help="Specific module to profile")
    parser.add_argument("--all-slow", action="store_true", help="Profile all tests marked as slow")
    parser.add_argument("--output-dir", default="profiling_results", help="Directory to store profiling results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    if args.test:
        # Profile a specific test
        if "::" in args.test:
            test_file, test_name = args.test.split("::")
            result = profile_test(test_file, test_name, args.output_dir)
            if result:
                results.append(result)
        else:
            print("Please specify test in format: test_file.py::test_name")
            return 1
    
    elif args.module:
        # Profile a specific module
        profile_module(args.module, args.output_dir)
    
    elif args.all_slow:
        # Profile all slow tests
        slow_tests = get_slow_tests()
        print(f"Found {len(slow_tests)} tests marked as slow.")
        
        for test_file, test_name in slow_tests:
            result = profile_test(test_file, test_name, args.output_dir)
            if result:
                results.append(result)
    
    else:
        # Default to profiling the most commonly slow tests based on test_profiler.py
        default_slow_tests = [
            ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_basic"),
            ("tests/test_methods/test_unconditional.py", "test_exact_ci_unconditional_small_counts"),
            ("tests/test_integration.py", "test_readme_example"),
            ("tests/test_methods/test_midp.py", "test_exact_ci_midp_basic"),
            ("tests/test_methods/test_blaker.py", "test_exact_ci_blaker_basic"),
        ]
        
        print("Profiling default set of known slow tests...")
        for test_file, test_name in default_slow_tests:
            result = profile_test(test_file, test_name, args.output_dir)
            if result:
                results.append(result)
    
    # Print summary of results
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
