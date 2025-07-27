#!/usr/bin/env python
"""
Optimize the unconditional CI calculation.

This script analyzes and optimizes the performance of the unconditional CI method,
trying different strategies to improve calculation speed while maintaining accuracy.
"""

import time
import logging
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from functools import partial

# Import methods from exactcis
from exactcis.methods import exact_ci_unconditional
from exactcis.core import find_root_log, find_plateau_edge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories for results
os.makedirs("optimization_results", exist_ok=True)


def benchmark_unconditional(a, b, c, d, alpha=0.05, timeout=60, **kwargs):
    """Benchmark the unconditional method with varying parameters."""
    # Base parameters
    base_args = {
        "grid_size": 20,  # Default grid size
        "refine": True,   # Default refinement setting
        "timeout": timeout
    }
    
    # Update with any additional kwargs
    base_args.update(kwargs)
    
    # Start timing
    start_time = time.time()
    success = True
    error = None
    
    try:
        result = exact_ci_unconditional(a, b, c, d, alpha, **base_args)
        ci_lower, ci_upper = result
    except Exception as e:
        success = False
        error = str(e)
        ci_lower, ci_upper = None, None
    
    elapsed = time.time() - start_time
    timed_out = elapsed >= timeout
    
    return {
        "success": success and not timed_out,
        "timed_out": timed_out,
        "elapsed": elapsed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "error": error
    }


def optimize_grid_size(a, b, c, d, alpha=0.05, timeout=60):
    """Find optimal grid size by testing different values."""
    results = []
    
    print(f"\nOptimizing grid size for case ({a}, {b}, {c}, {d})...")
    
    # Test different grid sizes
    grid_sizes = [5, 10, 15, 20, 30, 50]
    
    for grid_size in grid_sizes:
        print(f"  Testing grid_size={grid_size}...", end="", flush=True)
        
        result = benchmark_unconditional(
            a, b, c, d, alpha, 
            timeout=timeout,
            grid_size=grid_size
        )
        
        results.append({
            "grid_size": grid_size,
            "elapsed": result["elapsed"],
            "success": result["success"],
            "timed_out": result["timed_out"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"]
        })
        
        status = "SUCCESS" if result["success"] else "TIMEOUT" if result["timed_out"] else f"ERROR: {result['error']}"
        print(f" {status} - {result['elapsed']:.4f}s")
    
    # Find the optimal grid size
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        optimal_result = min(successful_results, key=lambda x: x["elapsed"])
        optimal_grid_size = optimal_result["grid_size"]
        print(f"  Optimal grid size: {optimal_grid_size} ({optimal_result['elapsed']:.4f}s)")
    else:
        optimal_grid_size = None
        print("  No successful grid size found!")
    
    return results, optimal_grid_size


def optimize_refine_setting(a, b, c, d, alpha=0.05, timeout=60, grid_size=20):
    """Test the impact of the refine setting."""
    results = []
    
    print(f"\nTesting refine setting for case ({a}, {b}, {c}, {d})...")
    
    for refine in [True, False]:
        print(f"  Testing refine={refine}...", end="", flush=True)
        
        result = benchmark_unconditional(
            a, b, c, d, alpha, 
            timeout=timeout,
            grid_size=grid_size,
            refine=refine
        )
        
        results.append({
            "refine": refine,
            "elapsed": result["elapsed"],
            "success": result["success"],
            "timed_out": result["timed_out"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"]
        })
        
        status = "SUCCESS" if result["success"] else "TIMEOUT" if result["timed_out"] else f"ERROR: {result['error']}"
        print(f" {status} - {result['elapsed']:.4f}s")
    
    # Compare performance with and without refine
    if len(results) == 2 and results[0]["success"] and results[1]["success"]:
        time_diff = results[0]["elapsed"] - results[1]["elapsed"]
        percent_diff = (time_diff / results[0]["elapsed"]) * 100
        
        print(f"  Time difference: {abs(time_diff):.4f}s ({abs(percent_diff):.1f}%)")
        print(f"  Refine=False is {'faster' if time_diff > 0 else 'slower'} than Refine=True")
    
    return results


def generate_optimization_cases():
    """Generate cases suitable for optimization testing."""
    cases = [
        # Small tables (likely fast)
        (2, 8, 3, 7, 0.05),
        (5, 5, 5, 5, 0.05),
        
        # Medium tables
        (10, 15, 12, 18, 0.05),
        (20, 25, 15, 30, 0.05),
        
        # Large tables (likely slow)
        (30, 40, 35, 45, 0.05),
        (50, 60, 55, 65, 0.05),
        
        # Extreme ratio tables (likely slow)
        (2, 50, 45, 3, 0.05),
        (1, 30, 25, 2, 0.05)
    ]
    
    return cases


def run_comprehensive_optimization(timeout=60):
    """Run comprehensive optimization tests."""
    cases = generate_optimization_cases()
    results = {}
    
    for i, case in enumerate(cases):
        a, b, c, d, alpha = case
        case_id = f"case_{i+1}"
        
        print(f"\n=== Case {i+1}/{len(cases)}: ({a}, {b}, {c}, {d}) with alpha={alpha} ===")
        
        # First, find optimal grid size
        grid_results, optimal_grid_size = optimize_grid_size(a, b, c, d, alpha, timeout)
        
        if optimal_grid_size is None:
            print(f"Skipping refine optimization for case {i+1} as no successful grid size found")
            refine_results = []
        else:
            # Then test refine setting with optimal grid size
            refine_results = optimize_refine_setting(a, b, c, d, alpha, timeout, optimal_grid_size)
        
        # Store results
        results[case_id] = {
            "case": (a, b, c, d, alpha),
            "grid_optimization": grid_results,
            "refine_optimization": refine_results,
            "optimal_grid_size": optimal_grid_size
        }
    
    return results


def analyze_optimization_results(results):
    """Analyze optimization results and provide recommendations."""
    grid_size_recommendations = []
    refine_recommendations = []
    
    # Collect statistics on optimal grid sizes
    optimal_grid_sizes = [results[case]["optimal_grid_size"] for case in results 
                         if results[case]["optimal_grid_size"] is not None]
    
    if optimal_grid_sizes:
        avg_optimal_grid = sum(optimal_grid_sizes) / len(optimal_grid_sizes)
        
        grid_size_recommendations.append(
            f"The average optimal grid size is {avg_optimal_grid:.1f}. "
            f"Consider using a grid size of {round(avg_optimal_grid)} as the default."
        )
    else:
        grid_size_recommendations.append(
            "Could not determine optimal grid size due to timeouts or errors."
        )
    
    # Analyze refine setting impact
    refine_faster_count = 0
    no_refine_faster_count = 0
    
    for case in results:
        refine_results = results[case]["refine_optimization"]
        
        if len(refine_results) == 2 and refine_results[0]["success"] and refine_results[1]["success"]:
            time_diff = refine_results[0]["elapsed"] - refine_results[1]["elapsed"]
            
            if time_diff > 0:  # refine=True is slower than refine=False
                no_refine_faster_count += 1
            else:
                refine_faster_count += 1
    
    if refine_faster_count + no_refine_faster_count > 0:
        if refine_faster_count > no_refine_faster_count:
            refine_recommendations.append(
                f"In {refine_faster_count}/{refine_faster_count + no_refine_faster_count} cases, "
                f"using refine=True was faster. Keep refine=True as the default."
            )
        else:
            refine_recommendations.append(
                f"In {no_refine_faster_count}/{refine_faster_count + no_refine_faster_count} cases, "
                f"using refine=False was faster. Consider changing the default to refine=False."
            )
    else:
        refine_recommendations.append(
            "Could not determine optimal refine setting due to timeouts or errors."
        )
    
    # Combine recommendations
    recommendations = {
        "grid_size": grid_size_recommendations,
        "refine": refine_recommendations
    }
    
    return recommendations


def plot_grid_size_performance(results, output_dir="optimization_results"):
    """Plot grid size vs. performance for cases that finished successfully."""
    plt.figure(figsize=(10, 6))
    
    for case_id, case_results in results.items():
        grid_results = case_results["grid_optimization"]
        
        # Only include successful results
        successful_results = [(r["grid_size"], r["elapsed"]) for r in grid_results if r["success"]]
        
        if successful_results:
            grid_sizes, times = zip(*successful_results)
            
            # Get the case details for the label
            a, b, c, d, alpha = case_results["case"]
            label = f"Case {case_id}: ({a},{b},{c},{d})"
            
            plt.plot(grid_sizes, times, 'o-', label=label)
    
    plt.xlabel('Grid Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Grid Size vs. Execution Time')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"grid_size_performance_{timestamp}.png")
    plt.savefig(plot_path)
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Optimize unconditional CI calculation")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each method (default: 60)")
    args = parser.parse_args()
    
    # Run comprehensive optimization
    results = run_comprehensive_optimization(timeout=args.timeout)
    
    # Analyze results
    recommendations = analyze_optimization_results(results)
    
    # Plot results
    plot_path = plot_grid_size_performance(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join("optimization_results", f"unconditional_optimization_{timestamp}.json")
    
    with open(results_path, "w") as f:
        json.dump({
            "optimization_results": results,
            "recommendations": recommendations
        }, f, indent=2, default=str)
    
    print("\n=== Optimization Complete ===")
    print(f"Results saved to: {results_path}")
    print(f"Performance plot saved to: {plot_path}")
    
    print("\n=== Recommendations ===")
    for category, recs in recommendations.items():
        print(f"\n{category.upper()} RECOMMENDATIONS:")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
