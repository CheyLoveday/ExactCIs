#!/usr/bin/env python3
"""
ExactCIs Profiling Enhancements

This script enhances the existing profiling tools with additional command-line options
and functionality to fully support the profiling plan.

Usage:
    python profiling_enhancements.py benchmark [--compare-baseline] [--save-baseline]
    python profiling_enhancements.py profile [--method METHOD] [--scenario SCENARIO]
    python profiling_enhancements.py line-profile [--method METHOD] [--function FUNCTION] [--test-case TEST_CASE]
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import existing profiling tools
from performance_benchmark import PerformanceBenchmark
from performance_profiler import PerformanceProfiler
from line_profiler import DetailedProfiler

# Import ExactCIs methods
try:
    from exactcis.methods.conditional import exact_ci_conditional
    from exactcis.methods.midp import exact_ci_midp
    from exactcis.methods.blaker import exact_ci_blaker
    from exactcis.methods.unconditional import exact_ci_unconditional
    from exactcis.methods.wald import ci_wald_haldane
    from exactcis.methods.clopper_pearson import exact_ci_clopper_pearson
except ImportError as e:
    print(f"Error importing ExactCIs methods: {e}")
    print("Make sure you're running this script from the repository root.")
    sys.exit(1)

# Define standard test scenarios
SCENARIOS = {
    "small": [
        (10, 10, 10, 10, "small_balanced"),
        (2, 18, 3, 17, "small_imbalanced"),
        (0, 20, 5, 15, "small_zero_cell"),
    ],
    "medium": [
        (50, 50, 50, 50, "medium_balanced"),
        (10, 90, 30, 70, "medium_imbalanced"),
        (2, 98, 5, 95, "medium_rare_events"),
    ],
    "large": [
        (250, 250, 250, 250, "large_balanced"),
        (50, 450, 100, 400, "large_imbalanced"),
        (5, 495, 10, 490, "large_very_rare"),
    ],
    "edge": [
        (20, 0, 0, 20, "edge_perfect_separation"),
        (0, 0, 10, 10, "edge_zero_marginals"),
        (1, 999, 2, 998, "edge_large_imbalance"),
    ],
}

# Define methods
METHODS = {
    "conditional": exact_ci_conditional,
    "midp": exact_ci_midp,
    "blaker": exact_ci_blaker,
    "unconditional": lambda a, b, c, d, alpha=0.05: exact_ci_unconditional(a, b, c, d, alpha, timeout=60),
    "wald_haldane": ci_wald_haldane,
    "clopper_pearson": exact_ci_clopper_pearson,
}

def parse_test_case(test_case: str) -> Tuple[int, int, int, int]:
    """Parse a test case string into a tuple of integers."""
    try:
        a, b, c, d = map(int, test_case.split(","))
        return (a, b, c, d)
    except ValueError:
        raise ValueError(f"Invalid test case format: {test_case}. Expected format: 'a,b,c,d'")

def enhanced_benchmark(args: argparse.Namespace) -> None:
    """Run enhanced benchmark with additional options."""
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    benchmark.save_results(results)
    
    # Generate and save report
    report = benchmark.generate_performance_report(results)
    
    report_file = benchmark.output_dir / "performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        baseline_file = benchmark.output_dir / "baseline_performance.json"
        if not baseline_file.exists():
            print(f"No baseline file found at {baseline_file}")
        else:
            comparison = benchmark.compare_with_baseline(baseline_file)
            print("\nComparison with baseline:")
            for method_name, method_comparison in comparison.get('method_comparisons', {}).items():
                status = method_comparison['status']
                improvement = method_comparison.get('improvement_percent', 0)
                print(f"  {method_name}: {status.upper()} ({improvement:.2f}% change)")
    
    # Save as baseline if requested
    if args.save_baseline:
        baseline_file = benchmark.output_dir / "baseline_performance.json"
        benchmark.save_results(results, "baseline_performance.json")
        print(f"\nSaved as baseline: {baseline_file}")

def enhanced_profile(args: argparse.Namespace) -> None:
    """Run enhanced profiling with additional options."""
    profiler = PerformanceProfiler()
    
    # Filter methods if specified
    if args.method:
        if args.method not in METHODS:
            print(f"Error: Method '{args.method}' not found. Available methods: {', '.join(METHODS.keys())}")
            sys.exit(1)
        methods = {args.method: METHODS[args.method]}
    else:
        methods = METHODS
    
    # Filter scenarios if specified
    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Error: Scenario '{args.scenario}' not found. Available scenarios: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS
    
    # Run timing analysis for selected methods and scenarios
    timing_results = {}
    for method_name, method_func in methods.items():
        print(f"\nProfiling {method_name}...")
        method_results = {}
        
        for scenario_name, scenario_list in scenarios.items():
            print(f"  Testing {scenario_name} scenarios...")
            for a, b, c, d, case_name in scenario_list:
                try:
                    result = profiler.time_method(method_func, a, b, c, d, method_name, case_name)
                    method_results[case_name] = result
                    if result['status'] == 'success':
                        print(f"    {case_name}: {result['mean_time']:.4f}s")
                    else:
                        print(f"    {case_name}: FAILED - {result.get('errors', ['Unknown error'])[0]}")
                except Exception as e:
                    print(f"    {case_name}: ERROR - {e}")
        
        timing_results[method_name] = method_results
    
    # Save results
    output_file = Path(profiler.output_dir) / "enhanced_timing_results.json"
    with open(output_file, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate report
    report = ["# ExactCIs Enhanced Profiling Report", "=" * 50, ""]
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for method_name, method_results in timing_results.items():
        report.append(f"## {method_name.upper()}")
        report.append("-" * 40)
        
        # Group results by scenario
        scenario_results = {}
        for case_name, result in method_results.items():
            for scenario_name, scenario_list in scenarios.items():
                if any(case[4] == case_name for case in scenario_list):
                    if scenario_name not in scenario_results:
                        scenario_results[scenario_name] = []
                    scenario_results[scenario_name].append((case_name, result))
        
        # Report results by scenario
        for scenario_name, results in scenario_results.items():
            report.append(f"\n### {scenario_name.capitalize()} Scenarios")
            report.append(f"{'Case':<25} {'Status':<10} {'Mean Time':<12} {'Min Time':<12} {'Max Time':<12}")
            report.append("-" * 75)
            
            for case_name, result in results:
                if result['status'] == 'success':
                    report.append(f"{case_name:<25} {'SUCCESS':<10} {result['mean_time']:<12.6f} {result['min_time']:<12.6f} {result['max_time']:<12.6f}")
                else:
                    report.append(f"{case_name:<25} {'FAILED':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        
        report.append("\n")
    
    # Save report
    report_file = Path(profiler.output_dir) / "enhanced_profiling_report.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Report saved to: {report_file}")

def enhanced_line_profile(args: argparse.Namespace) -> None:
    """Run enhanced line profiling with additional options."""
    profiler = DetailedProfiler()
    
    # Validate method
    if args.method not in METHODS:
        print(f"Error: Method '{args.method}' not found. Available methods: {', '.join(METHODS.keys())}")
        sys.exit(1)
    
    method_func = METHODS[args.method]
    
    # Parse test case if provided
    if args.test_case:
        try:
            test_case = parse_test_case(args.test_case)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Use a default test case
        test_case = (10, 10, 10, 10)
    
    print(f"Line profiling {args.method} with test case {test_case}...")
    
    # If function is specified, profile only that function
    if args.function:
        # Get the module name for the method
        module_name = method_func.__module__
        
        # Import the module
        try:
            module = __import__(module_name, fromlist=[''])
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            sys.exit(1)
        
        # Get the function from the module
        try:
            func = getattr(module, args.function)
        except AttributeError:
            print(f"Error: Function '{args.function}' not found in module {module_name}")
            sys.exit(1)
        
        # Profile the function
        print(f"Profiling function {args.function}...")
        profiler.time_function_calls(func, *test_case)
    else:
        # Profile the method and all its dependencies
        print(f"Profiling method {args.method} and its dependencies...")
        profiler.analyze_method_performance(args.method, method_func, [(test_case, f"test_case_{args.method}")])
    
    print("Line profiling complete. Results saved to profiling directory.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="ExactCIs Profiling Enhancements")
    subparsers = parser.add_subparsers(dest="command", help="Profiling command")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run comprehensive benchmark")
    benchmark_parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline")
    benchmark_parser.add_argument("--save-baseline", action="store_true", help="Save as baseline")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Run performance profiling")
    profile_parser.add_argument("--method", help="Method to profile")
    profile_parser.add_argument("--scenario", help="Scenario to profile")
    
    # Line profile command
    line_profile_parser = subparsers.add_parser("line-profile", help="Run line-by-line profiling")
    line_profile_parser.add_argument("--method", required=True, help="Method to profile")
    line_profile_parser.add_argument("--function", help="Specific function to profile")
    line_profile_parser.add_argument("--test-case", help="Test case in format 'a,b,c,d'")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        enhanced_benchmark(args)
    elif args.command == "profile":
        enhanced_profile(args)
    elif args.command == "line-profile":
        enhanced_line_profile(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()