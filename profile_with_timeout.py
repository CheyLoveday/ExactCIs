#!/usr/bin/env python
"""
Profile ExactCIs with timeout functionality.

This script tests the exact confidence interval methods with timeouts, allowing
for automatic identification of slow calculations. It iterates through test cases
and records which calculations timeout, making it easier to focus optimization efforts.
"""

import time
import logging
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
import json
from datetime import datetime

# Import methods from exactcis
from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories for results
os.makedirs("profiling_results", exist_ok=True)


def run_with_timeout(method_func, a, b, c, d, alpha, timeout_seconds, **kwargs):
    """Run a CI method with a timeout and return result or None if timed out."""
    start_time = time.time()
    
    try:
        # Call the method with timeout parameter
        if method_func.__name__ == 'exact_ci_unconditional':
            # Add timeout parameter for unconditional method
            result = method_func(a, b, c, d, alpha, timeout=timeout_seconds, **kwargs)
        else:
            # For other methods without timeout parameter, use a simple time check
            result = method_func(a, b, c, d, alpha, **kwargs)
        
        elapsed = time.time() - start_time
        timed_out = elapsed >= timeout_seconds
        
        return {
            "result": result,
            "elapsed": elapsed,
            "timed_out": timed_out,
            "status": "timeout" if timed_out else "success"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "result": None,
            "elapsed": elapsed,
            "timed_out": elapsed >= timeout_seconds,
            "status": "error",
            "error": str(e)
        }


def generate_test_cases(num_cases=20, max_count=100, random_seed=42):
    """Generate test cases with varying table sizes and ratios."""
    np.random.seed(random_seed)
    
    test_cases = []
    
    # Add boundary cases
    test_cases.extend([
        (0, 10, 0, 10, 0.05),  # Empty corners
        (10, 0, 10, 0, 0.05),  # Empty corners
        (0, 10, 10, 0, 0.05),  # Empty corners
        (10, 0, 0, 10, 0.05),  # Empty corners
        (1, 0, 0, 1, 0.05),    # Small diagonal
        (0, 1, 1, 0, 0.05),    # Small diagonal
    ])
    
    # Add small tables (for fast cases)
    for _ in range(num_cases // 4):
        a = np.random.randint(0, min(10, max_count))
        b = np.random.randint(1, min(10, max_count))
        c = np.random.randint(0, min(10, max_count))
        d = np.random.randint(1, min(10, max_count))
        test_cases.append((a, b, c, d, 0.05))
    
    # Add medium tables
    for _ in range(num_cases // 4):
        a = np.random.randint(5, min(30, max_count-1))
        b = np.random.randint(5, min(30, max_count-1))
        c = np.random.randint(5, min(30, max_count-1))
        d = np.random.randint(5, min(30, max_count-1))
        test_cases.append((a, b, c, d, 0.05))
    
    # Add large tables (likely to be slow)
    for _ in range(num_cases // 4):
        if max_count > 21:  # Ensure we have a valid range
            a = np.random.randint(20, max_count)
            b = np.random.randint(20, max_count)
            c = np.random.randint(20, max_count)
            d = np.random.randint(20, max_count)
            test_cases.append((a, b, c, d, 0.05))
    
    # Add extreme ratio tables (likely to be slow)
    for _ in range(num_cases // 4):
        if max_count > 51:  # Ensure we have a valid range
            a = np.random.randint(1, 5)
            b = np.random.randint(50, max_count)
            c = np.random.randint(50, max_count)
            d = np.random.randint(1, 5)
            test_cases.append((a, b, c, d, 0.05))
        else:
            # Create alternate extreme ratio cases for smaller max_count
            a = np.random.randint(1, 3)
            b = np.random.randint(max(5, max_count//2), max_count)
            c = np.random.randint(max(5, max_count//2), max_count)
            d = np.random.randint(1, 3)
            test_cases.append((a, b, c, d, 0.05))
    
    return test_cases


def profile_methods_with_timeout(test_cases, timeout_seconds=30, grid_size=20):
    """Profile methods with timeout and collect results."""
    results = []
    
    total_cases = len(test_cases)
    print(f"\nRunning {total_cases} test cases with {timeout_seconds}s timeout...")
    
    for i, case in enumerate(test_cases):
        a, b, c, d, alpha = case
        
        case_id = f"case_{i+1}"
        print(f"\nCase {i+1}/{total_cases}: ({a}, {b}, {c}, {d}) with alpha={alpha}")
        
        # Calculate table stats
        n1 = a + b
        n2 = c + d
        table_size = n1 + n2
        ratio = max(a/max(1, n1), b/max(1, n1), c/max(1, n2), d/max(1, n2))
        
        # Run all methods with timing and timeout
        methods = {
            "blaker": (exact_ci_blaker, {}),
            "conditional": (exact_ci_conditional, {}),
            "midp": (exact_ci_midp, {}),
            "unconditional": (exact_ci_unconditional, {"grid_size": grid_size}),
            "wald": (ci_wald_haldane, {})
        }
        
        case_results = {
            "case_id": case_id,
            "a": a, "b": b, "c": c, "d": d,
            "alpha": alpha,
            "table_size": table_size,
            "n1": n1, "n2": n2,
            "max_ratio": ratio
        }
        
        for method_name, (method_func, method_kwargs) in methods.items():
            print(f"  Running {method_name}...", end="", flush=True)
            
            # Run method with timeout
            result = run_with_timeout(
                method_func, a, b, c, d, alpha, 
                timeout_seconds=timeout_seconds,
                **method_kwargs
            )
            
            # Store results
            ci_result = result["result"]
            if ci_result is not None and not isinstance(ci_result, tuple):
                ci_result = None
                
            case_results[f"{method_name}_lower"] = ci_result[0] if ci_result else None
            case_results[f"{method_name}_upper"] = ci_result[1] if ci_result else None
            case_results[f"{method_name}_time"] = result["elapsed"]
            case_results[f"{method_name}_status"] = result["status"]
            
            status_message = {
                "success": f"done in {result['elapsed']:.4f}s",
                "timeout": f"TIMEOUT after {result['elapsed']:.4f}s",
                "error": f"ERROR after {result['elapsed']:.4f}s: {result.get('error', 'Unknown error')}"
            }.get(result["status"], "unknown status")
            
            print(f" {status_message}")
        
        results.append(case_results)
        
        # Print summary for this case
        print("  Summary:")
        for method_name in methods:
            status = case_results[f"{method_name}_status"]
            time = case_results[f"{method_name}_time"]
            lower = case_results[f"{method_name}_lower"]
            upper = case_results[f"{method_name}_upper"]
            
            ci_text = f"({lower:.4f}, {upper:.4f})" if lower is not None and upper is not None else "N/A"
            print(f"    {method_name.ljust(15)}: {ci_text} - {time:.4f}s - {status}")
    
    return results


def analyze_results(results):
    """Analyze results and generate insights."""
    df = pd.DataFrame(results)
    
    # Calculate success rates
    methods = ["blaker", "conditional", "midp", "unconditional", "wald"]
    success_rates = {}
    avg_times = {}
    
    print("\n=== Method Performance Summary ===")
    print(f"Total cases: {len(df)}")
    
    for method in methods:
        success_count = sum(df[f"{method}_status"] == "success")
        success_rate = success_count / len(df) * 100
        success_rates[method] = success_rate
        
        # Average time for successful runs
        success_times = df[df[f"{method}_status"] == "success"][f"{method}_time"]
        avg_time = success_times.mean() if len(success_times) > 0 else float('nan')
        avg_times[method] = avg_time
        
        print(f"{method.ljust(15)}: {success_rate:.1f}% success rate, avg time: {avg_time:.4f}s")
    
    # Identify characteristics of timeout cases
    timeout_analysis = {}
    for method in methods:
        timeout_cases = df[df[f"{method}_status"] == "timeout"]
        if len(timeout_cases) > 0:
            timeout_analysis[method] = {
                "count": len(timeout_cases),
                "avg_table_size": timeout_cases["table_size"].mean(),
                "avg_ratio": timeout_cases["max_ratio"].mean(),
                "example_cases": timeout_cases[["case_id", "a", "b", "c", "d"]].head(3).to_dict('records')
            }
    
    print("\n=== Timeout Analysis ===")
    for method, analysis in timeout_analysis.items():
        print(f"{method} timeouts ({analysis['count']} cases):")
        print(f"  Average table size: {analysis['avg_table_size']:.1f}")
        print(f"  Average max ratio: {analysis['avg_ratio']:.2f}")
        print("  Example cases:")
        for case in analysis["example_cases"]:
            print(f"    {case['case_id']}: ({case['a']}, {case['b']}, {case['c']}, {case['d']})")
    
    return {
        "success_rates": success_rates,
        "avg_times": avg_times,
        "timeout_analysis": timeout_analysis
    }


def save_results(results, analysis, timeout_seconds, output_dir="profiling_results"):
    """Save results and analysis to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"timeout_profiling_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save analysis
    analysis_path = os.path.join(output_dir, f"timeout_analysis_{timestamp}.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to {csv_path}")
    print(f"Analysis saved to {analysis_path}")
    
    # Generate improvement recommendations
    recommendations = []
    
    # Check for unconditional method issues
    unconditional_timeout_rate = 100 - analysis["success_rates"]["unconditional"]
    if unconditional_timeout_rate > 10:
        recommendations.append(
            f"The unconditional method timed out on {unconditional_timeout_rate:.1f}% of cases. "
            "Consider implementing grid size reduction for large tables or early stopping when convergence is slow."
        )
    
    # Check for blaker method issues
    blaker_timeout_rate = 100 - analysis["success_rates"]["blaker"]
    if blaker_timeout_rate > 10:
        recommendations.append(
            f"Blaker's method timed out on {blaker_timeout_rate:.1f}% of cases. "
            "Consider more aggressive caching or parallel processing for p-value calculations."
        )
    
    # Generate recommendations file
    if recommendations:
        recommendations_path = os.path.join(output_dir, f"improvement_recommendations_{timestamp}.txt")
        with open(recommendations_path, "w") as f:
            f.write(f"# Improvement Recommendations (timeout={timeout_seconds}s)\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
        print(f"Recommendations saved to {recommendations_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile ExactCIs methods with timeout functionality")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each method (default: 60)")
    parser.add_argument("--num-cases", type=int, default=40, help="Number of test cases to generate (default: 40)")
    parser.add_argument("--max-count", type=int, default=100, help="Maximum count for table cells (default: 100)")
    parser.add_argument("--grid-size", type=int, default=20, help="Grid size for unconditional method (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test case generation (default: 42)")
    args = parser.parse_args()
    
    # Generate test cases
    test_cases = generate_test_cases(
        num_cases=args.num_cases,
        max_count=args.max_count,
        random_seed=args.seed
    )
    
    # Run profiling with timeout
    results = profile_methods_with_timeout(
        test_cases, 
        timeout_seconds=args.timeout,
        grid_size=args.grid_size
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Save results
    save_results(results, analysis, args.timeout)
    
    print("\nProfiling with timeout completed successfully!")
    print(f"You can now examine the results to identify methods that need optimization.")


if __name__ == "__main__":
    main()
