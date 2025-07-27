#!/usr/bin/env python3
"""
Unified Comparison Script for ExactCIs

This script consolidates multiple comparison scripts into a single, configurable tool
for comparing different confidence interval methods. It supports various comparison modes,
test cases, and output formats.

Usage:
    python comparison_methods.py [options]

Examples:
    # Compare all methods on standard test cases
    python comparison_methods.py --mode=all
    
    # Compare specific methods (midp and blaker)
    python comparison_methods.py --methods=midp,blaker
    
    # Run comprehensive comparison with plots
    python comparison_methods.py --comprehensive --plot
    
    # Compare with external implementations (R, SciPy)
    python comparison_methods.py --external
    
    # Run analysis on extreme or edge cases
    python comparison_methods.py --extreme-cases
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the parent directory to sys.path to import exactcis
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.exactcis import compute_all_cis


class ComparisonEngine:
    """Unified engine for comparing confidence interval methods."""
    
    def __init__(self, output_dir: str = "analysis/plots"):
        """Initialize the comparison engine."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard test cases
        self.standard_cases = [
            (10, 5, 7, 8),  # Common case
            (100, 50, 70, 80),  # Large numbers
            (5, 2, 3, 4),  # Small numbers
            (0, 5, 0, 10),  # Zero in a cell
            (12, 5, 8, 10),  # Known problematic case
        ]
        
        # Extreme cases for testing edge behavior
        self.extreme_cases = [
            (1000, 0, 1000, 0),  # Perfect separation
            (0, 0, 10, 10),  # Double zeros
            (1, 0, 0, 1),  # Diagonal pattern
            (100000, 50000, 70000, 30000),  # Very large numbers
            (1, 1, 1, 1),  # All ones
        ]
        
        # Available methods in exactcis
        self.available_methods = [
            "midp", "blaker", "conditional", "unconditional", "wald"
        ]
        
    def run_comparison(self, args: argparse.Namespace) -> Dict:
        """Run comparison based on provided arguments."""
        results = {}
        
        # Determine which methods to compare
        methods = self.available_methods
        if args.methods:
            methods = args.methods.split(",")
            for method in methods:
                if method not in self.available_methods:
                    print(f"Warning: Method '{method}' is not recognized.")
        
        # Determine which test cases to use
        test_cases = self.standard_cases
        if args.extreme_cases:
            test_cases.extend(self.extreme_cases)
        
        # Run the comparison
        print(f"Comparing methods: {', '.join(methods)}")
        for i, case in enumerate(test_cases):
            print(f"Case {i+1}: {case}")
            a, b, c, d = case
            case_results = {}
            
            for method in methods:
                try:
                    start_time = time.time()
                    ci = compute_all_cis(a, b, c, d, methods=[method])
                    elapsed = time.time() - start_time
                    
                    case_results[method] = {
                        "ci": ci[method],
                        "time": elapsed
                    }
                except Exception as e:
                    case_results[method] = {
                        "error": str(e)
                    }
            
            results[str(case)] = case_results
            
            # Print results for this case
            for method, result in case_results.items():
                if "error" in result:
                    print(f"  {method}: ERROR - {result['error']}")
                else:
                    ci_str = f"({result['ci'][0]:.4f}, {result['ci'][1]:.4f})"
                    print(f"  {method}: {ci_str} ({result['time']:.4f}s)")
            print()
        
        # Save results if requested
        if args.save:
            results_file = os.path.join(self.output_dir, "comparison_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_file}")
        
        # Generate plots if requested
        if args.plot:
            self._generate_plots(results, methods)
        
        # Compare with external implementations if requested
        if args.external:
            external_results = self._compare_with_external(test_cases, methods)
            results["external"] = external_results
        
        return results
    
    def _generate_plots(self, results: Dict, methods: List[str]) -> None:
        """Generate comparison plots."""
        # Performance comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_labels = [f"Case {i+1}" for i in range(len(results) - (1 if "external" in results else 0))]
        x_pos = np.arange(len(x_labels))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            times = []
            for case in results:
                if case == "external":
                    continue
                    
                if method in results[case] and "time" in results[case][method]:
                    times.append(results[case][method]["time"])
                else:
                    times.append(0)
            
            ax.bar(x_pos + i * width - 0.4 + width/2, times, width, label=method)
        
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_comparison.png"))
        print(f"Plot saved to {os.path.join(self.output_dir, 'performance_comparison.png')}")
        
        # CI width comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            widths = []
            for case in results:
                if case == "external":
                    continue
                    
                if method in results[case] and "ci" in results[case][method]:
                    ci = results[case][method]["ci"]
                    if ci[1] == float('inf'):
                        widths.append(5)  # Placeholder for infinite width
                    else:
                        widths.append(ci[1] - ci[0])
                else:
                    widths.append(0)
            
            ax.bar(x_pos + i * width - 0.4 + width/2, widths, width, label=method)
        
        ax.set_ylabel("CI Width")
        ax.set_title("CI Width Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "width_comparison.png"))
        print(f"Plot saved to {os.path.join(self.output_dir, 'width_comparison.png')}")
    
    def _compare_with_external(self, test_cases: List[Tuple[int, int, int, int]], 
                              methods: List[str]) -> Dict:
        """Compare with external implementations (R, SciPy)."""
        print("\nComparing with external implementations...")
        
        try:
            from rpy2 import robjects
            from rpy2.robjects.packages import importr
            has_r = True
            
            # Import R packages
            stats = importr('stats')
            exact2x2 = importr('exact2x2')
            print("Successfully imported R packages")
        except Exception as e:
            print(f"Warning: Could not import R packages: {str(e)}")
            has_r = False
        
        try:
            from scipy.stats import fisher_exact
            has_scipy = True
        except ImportError:
            print("Warning: Could not import SciPy")
            has_scipy = False
            
        external_results = {}
        
        for i, case in enumerate(test_cases):
            a, b, c, d = case
            case_results = {}
            
            if has_scipy:
                try:
                    start_time = time.time()
                    oddsratio, pvalue = fisher_exact([[a, b], [c, d]])
                    elapsed = time.time() - start_time
                    
                    case_results["scipy_fisher"] = {
                        "pvalue": pvalue,
                        "oddsratio": oddsratio,
                        "time": elapsed
                    }
                except Exception as e:
                    case_results["scipy_fisher"] = {"error": str(e)}
            
            if has_r:
                for method in methods:
                    if method == "midp":
                        try:
                            start_time = time.time()
                            result = exact2x2.exact2x2(a, b, c, d, midp=True, tsmethod="central")
                            elapsed = time.time() - start_time
                            
                            case_results["r_midp"] = {
                                "ci": (result.rx2('conf.int')[0], result.rx2('conf.int')[1]),
                                "time": elapsed
                            }
                        except Exception as e:
                            case_results["r_midp"] = {"error": str(e)}
                    
                    if method == "conditional":
                        try:
                            start_time = time.time()
                            result = exact2x2.exact2x2(a, b, c, d, tsmethod="central")
                            elapsed = time.time() - start_time
                            
                            case_results["r_conditional"] = {
                                "ci": (result.rx2('conf.int')[0], result.rx2('conf.int')[1]),
                                "time": elapsed
                            }
                        except Exception as e:
                            case_results["r_conditional"] = {"error": str(e)}
                    
                    if method == "unconditional":
                        try:
                            start_time = time.time()
                            result = exact2x2.uncondExact2x2(a, b, c, d)
                            elapsed = time.time() - start_time
                            
                            case_results["r_unconditional"] = {
                                "ci": (result.rx2('conf.int')[0], result.rx2('conf.int')[1]),
                                "time": elapsed
                            }
                        except Exception as e:
                            case_results["r_unconditional"] = {"error": str(e)}
                            
            external_results[str(case)] = case_results
            
        return external_results


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description="Unified comparison tool for ExactCIs methods")
    
    # Basic options
    parser.add_argument("--methods", type=str, help="Comma-separated list of methods to compare")
    parser.add_argument("--mode", choices=["all", "basic", "comprehensive"], 
                        default="basic", help="Comparison mode")
    
    # Output options
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--output-dir", type=str, default="analysis/plots", 
                        help="Output directory for plots and results")
    
    # Special comparisons
    parser.add_argument("--external", action="store_true", 
                        help="Compare with external implementations (R, SciPy)")
    parser.add_argument("--extreme-cases", action="store_true", 
                        help="Include extreme test cases")
    parser.add_argument("--comprehensive", action="store_true", 
                        help="Run comprehensive comparison (implies --save --plot --external)")
    
    args = parser.parse_args()
    
    # Handle comprehensive mode
    if args.comprehensive or args.mode == "comprehensive":
        args.save = True
        args.plot = True
        args.external = True
        args.extreme_cases = True
    
    # Handle all mode
    if args.mode == "all":
        args.methods = None  # Use all methods
    
    # Run the comparison
    engine = ComparisonEngine(output_dir=args.output_dir)
    results = engine.run_comparison(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
