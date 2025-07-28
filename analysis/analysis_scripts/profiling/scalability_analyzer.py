#!/usr/bin/env python
"""
Scalability analyzer for ExactCIs methods.

This script performs detailed scalability analysis to understand:
1. Computational complexity curves for each method
2. Performance scaling with different table dimensions
3. Parameter sensitivity analysis (grid sizes, timeouts)
4. Break-even points where methods become infeasible
5. Comparative scaling behavior across methods
"""

import sys
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import warnings

# Import ExactCIs methods
from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

@dataclass
class ScalabilityResult:
    """Data class for storing scalability test results."""
    table_size: int
    n1: int
    n2: int
    a: int
    b: int
    c: int
    d: int
    execution_time: float
    success: bool
    result: Optional[Tuple[float, float]]
    error: Optional[str]
    parameters: Dict[str, Any]


class ScalabilityAnalyzer:
    """Scalability analyzer for ExactCIs methods."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Methods to analyze
        self.methods = {
            "conditional": exact_ci_conditional,
            "midp": exact_ci_midp,
            "blaker": exact_ci_blaker,
            "unconditional": exact_ci_unconditional,
            "wald": ci_wald_haldane
        }
        
        # Method-specific parameters
        self.method_params = {
            "conditional": {},
            "midp": {},
            "blaker": {},
            "unconditional": {"grid_size": 20, "timeout": 300},
            "wald": {}
        }

    def generate_test_tables(self, sizes: List[int], table_types: List[str] = None) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Generate test tables of different types and sizes."""
        if table_types is None:
            table_types = ["balanced", "unbalanced", "extreme", "sparse"]
        
        test_tables = {table_type: [] for table_type in table_types}
        
        for size in sizes:
            for table_type in table_types:
                if table_type == "balanced":
                    # Roughly equal margins
                    n1 = size // 2
                    n2 = size - n1
                    a = n1 // 2
                    b = n1 - a
                    c = n2 // 2
                    d = n2 - c
                
                elif table_type == "unbalanced":
                    # Unequal margins
                    n1 = size // 3
                    n2 = size - n1
                    a = n1 // 2
                    b = n1 - a
                    c = n2 // 3
                    d = n2 - c
                
                elif table_type == "extreme":
                    # Very unequal margins with extreme ratios
                    n1 = max(1, size // 10)
                    n2 = size - n1
                    a = max(0, n1 // 2)
                    b = n1 - a
                    c = max(1, n2 // 20)
                    d = n2 - c
                
                elif table_type == "sparse":
                    # Tables with some zero or very small cells
                    n1 = size // 2
                    n2 = size - n1
                    a = 0 if size > 10 else 1
                    b = n1 - a
                    c = max(1, min(2, n2 // 2))
                    d = n2 - c
                
                # Ensure valid table
                if all(x >= 0 for x in [a, b, c, d]) and (a + b > 0) and (c + d > 0):
                    test_tables[table_type].append((a, b, c, d))
        
        return test_tables

    def run_scalability_test(self, method_name: str, test_case: Tuple[int, int, int, int], 
                           timeout: float = 300, **method_kwargs) -> ScalabilityResult:
        """Run a single scalability test."""
        a, b, c, d = test_case
        table_size = a + b + c + d
        
        method_func = self.methods[method_name]
        
        # Prepare parameters
        params = self.method_params[method_name].copy()
        params.update(method_kwargs)
        
        start_time = time.time()
        
        try:
            if method_name == "unconditional":
                # Use timeout for unconditional method
                result = method_func(a, b, c, d, 0.05, timeout=timeout, **params)
            else:
                result = method_func(a, b, c, d, 0.05, **params)
            
            execution_time = time.time() - start_time
            success = True
            error = None
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = None
            success = False
            error = str(e)
        
        return ScalabilityResult(
            table_size=table_size,
            n1=a+b, n2=c+d,
            a=a, b=b, c=c, d=d,
            execution_time=execution_time,
            success=success,
            result=result,
            error=error,
            parameters=params
        )

    def analyze_computational_complexity(self, method_name: str, max_size: int = 300, 
                                       size_step: int = 20) -> Dict[str, Any]:
        """Analyze computational complexity by testing increasing table sizes."""
        print(f"\n[+] Computational complexity analysis: {method_name}")
        
        # Generate size progression
        if method_name == "unconditional":
            # More conservative sizing for unconditional method
            sizes = list(range(10, min(max_size, 150), max(size_step // 2, 5)))
        else:
            sizes = list(range(10, max_size + 1, size_step))
        
        # Generate test tables for each size
        test_tables = self.generate_test_tables(sizes, ["balanced"])
        balanced_tables = test_tables["balanced"]
        
        complexity_results = []
        
        for i, (a, b, c, d) in enumerate(balanced_tables):
            table_size = a + b + c + d
            print(f"  Testing size {table_size}: ({a},{b},{c},{d})")
            
            # Adjust parameters based on size for unconditional method
            if method_name == "unconditional":
                grid_size = max(10, min(30, table_size // 5))
                timeout = min(300, table_size * 2)
                result = self.run_scalability_test(method_name, (a, b, c, d), 
                                                 timeout=timeout, grid_size=grid_size)
            else:
                result = self.run_scalability_test(method_name, (a, b, c, d))
            
            complexity_results.append(result)
            
            print(f"    Time: {result.execution_time:.3f}s, Success: {result.success}")
            
            # Stop if method becomes too slow or starts failing consistently
            if result.execution_time > 180:  # 3 minutes
                print(f"    Method too slow at size {table_size}, stopping complexity analysis")
                break
            
            if not result.success:
                print(f"    Method failed at size {table_size}: {result.error}")
                # Continue for a few more sizes to see if it's consistent
                if i > 3:  # Allow some initial failures
                    recent_failures = sum(1 for r in complexity_results[-3:] if not r.success)
                    if recent_failures >= 2:
                        print("    Too many recent failures, stopping")
                        break
        
        # Analyze complexity pattern
        complexity_analysis = self._analyze_complexity_pattern(complexity_results)
        
        return {
            "method": method_name,
            "results": [self._result_to_dict(r) for r in complexity_results],
            "complexity_analysis": complexity_analysis,
            "max_feasible_size": max((r.table_size for r in complexity_results if r.success), default=0)
        }

    def _analyze_complexity_pattern(self, results: List[ScalabilityResult]) -> Dict[str, Any]:
        """Analyze the computational complexity pattern from results."""
        successful_results = [r for r in results if r.success]
        
        if len(successful_results) < 3:
            return {"error": "Insufficient data for complexity analysis"}
        
        sizes = np.array([r.table_size for r in successful_results])
        times = np.array([r.execution_time for r in successful_results])
        
        # Remove outliers (times > 3 standard deviations)
        mean_time = np.mean(times)
        std_time = np.std(times)
        outlier_mask = np.abs(times - mean_time) <= 3 * std_time
        sizes_clean = sizes[outlier_mask]
        times_clean = times[outlier_mask]
        
        if len(sizes_clean) < 3:
            sizes_clean, times_clean = sizes, times
        
        complexity_fits = {}
        
        try:
            # Linear fit: T = a*n + b
            linear_coeffs = np.polyfit(sizes_clean, times_clean, 1)
            linear_pred = np.polyval(linear_coeffs, sizes_clean)
            linear_r2 = 1 - np.sum((times_clean - linear_pred)**2) / np.sum((times_clean - np.mean(times_clean))**2)
            complexity_fits["linear"] = {
                "coefficients": linear_coeffs.tolist(),
                "r_squared": float(linear_r2),
                "formula": f"T = {linear_coeffs[0]:.3e}*n + {linear_coeffs[1]:.3e}"
            }
            
            # Quadratic fit: T = a*n^2 + b*n + c
            if len(sizes_clean) >= 4:
                quad_coeffs = np.polyfit(sizes_clean, times_clean, 2)
                quad_pred = np.polyval(quad_coeffs, sizes_clean)
                quad_r2 = 1 - np.sum((times_clean - quad_pred)**2) / np.sum((times_clean - np.mean(times_clean))**2)
                complexity_fits["quadratic"] = {
                    "coefficients": quad_coeffs.tolist(),
                    "r_squared": float(quad_r2),
                    "formula": f"T = {quad_coeffs[0]:.3e}*n² + {quad_coeffs[1]:.3e}*n + {quad_coeffs[2]:.3e}"
                }
            
            # Exponential fit: T = a*exp(b*n)
            if len(sizes_clean) >= 4:
                try:
                    # Fit log(T) = log(a) + b*n
                    log_times = np.log(np.maximum(times_clean, 1e-10))
                    exp_coeffs = np.polyfit(sizes_clean, log_times, 1)
                    exp_pred = np.exp(np.polyval(exp_coeffs, sizes_clean))
                    exp_r2 = 1 - np.sum((times_clean - exp_pred)**2) / np.sum((times_clean - np.mean(times_clean))**2)
                    complexity_fits["exponential"] = {
                        "coefficients": [np.exp(exp_coeffs[1]), exp_coeffs[0]],
                        "r_squared": float(exp_r2),
                        "formula": f"T = {np.exp(exp_coeffs[1]):.3e}*exp({exp_coeffs[0]:.3e}*n)"
                    }
                except:
                    pass  # Skip exponential fit if it fails
            
            # Determine best fit
            best_fit = max(complexity_fits.keys(), key=lambda k: complexity_fits[k]["r_squared"])
            
            return {
                "fits": complexity_fits,
                "best_fit": best_fit,
                "best_r_squared": complexity_fits[best_fit]["r_squared"],
                "complexity_class": self._classify_complexity(best_fit, complexity_fits[best_fit])
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze complexity: {e}"}

    def _classify_complexity(self, best_fit: str, fit_data: Dict) -> str:
        """Classify the computational complexity based on the best fit."""
        if best_fit == "linear":
            return "O(n)"
        elif best_fit == "quadratic":
            return "O(n²)"
        elif best_fit == "exponential":
            return "O(exp(n))"
        else:
            return "unknown"

    def analyze_parameter_sensitivity(self, method_name: str) -> Dict[str, Any]:
        """Analyze sensitivity to method-specific parameters."""
        print(f"\n[+] Parameter sensitivity analysis: {method_name}")
        
        if method_name != "unconditional":
            return {"message": f"Parameter sensitivity analysis only applicable to unconditional method"}
        
        # Test case for parameter sensitivity
        test_case = (15, 25, 20, 30)  # Medium-sized balanced table
        
        # Test different grid sizes
        grid_sizes = [5, 10, 15, 20, 30, 50, 75, 100]
        grid_results = []
        
        print("  Testing grid size sensitivity...")
        for grid_size in grid_sizes:
            print(f"    Grid size: {grid_size}")
            result = self.run_scalability_test(method_name, test_case, grid_size=grid_size, timeout=120)
            grid_results.append({
                "grid_size": grid_size,
                "execution_time": result.execution_time,
                "success": result.success,
                "result": result.result,
                "error": result.error
            })
            
            if not result.success:
                print(f"      Failed: {result.error}")
            else:
                print(f"      Time: {result.execution_time:.3f}s")
        
        # Test different timeout values
        timeout_values = [30, 60, 120, 300, 600]
        timeout_results = []
        
        print("  Testing timeout sensitivity...")
        for timeout in timeout_values:
            print(f"    Timeout: {timeout}s")
            result = self.run_scalability_test(method_name, test_case, grid_size=20, timeout=timeout)
            timeout_results.append({
                "timeout": timeout,
                "execution_time": result.execution_time,
                "success": result.success,
                "result": result.result,
                "error": result.error
            })
            
            if not result.success:
                print(f"      Failed: {result.error}")
            else:
                print(f"      Time: {result.execution_time:.3f}s")
        
        return {
            "method": method_name,
            "test_case": test_case,
            "grid_size_analysis": {
                "results": grid_results,
                "sensitivity": self._analyze_grid_sensitivity(grid_results)
            },
            "timeout_analysis": {
                "results": timeout_results,
                "sensitivity": self._analyze_timeout_sensitivity(timeout_results)
            }
        }

    def _analyze_grid_sensitivity(self, grid_results: List[Dict]) -> Dict[str, Any]:
        """Analyze grid size sensitivity."""
        successful_results = [r for r in grid_results if r["success"]]
        
        if len(successful_results) < 2:
            return {"error": "Insufficient successful results"}
        
        grid_sizes = [r["grid_size"] for r in successful_results]
        times = [r["execution_time"] for r in successful_results]
        
        # Find optimal grid size (fastest with successful completion)
        optimal_idx = np.argmin(times)
        optimal_grid_size = grid_sizes[optimal_idx]
        
        # Calculate time scaling with grid size
        if len(grid_sizes) >= 3:
            try:
                scaling_coeffs = np.polyfit(grid_sizes, times, 1)
                scaling_slope = scaling_coeffs[0]
                scaling_type = "linear" if abs(scaling_slope) > 0.001 else "constant"
            except:
                scaling_slope = None
                scaling_type = "unknown"
        else:
            scaling_slope = None
            scaling_type = "insufficient_data"
        
        return {
            "optimal_grid_size": optimal_grid_size,
            "optimal_time": times[optimal_idx],
            "scaling_slope": scaling_slope,
            "scaling_type": scaling_type,
            "time_range": [min(times), max(times)]
        }

    def _analyze_timeout_sensitivity(self, timeout_results: List[Dict]) -> Dict[str, Any]:
        """Analyze timeout sensitivity."""
        # Find minimum timeout that allows completion
        successful_results = [r for r in timeout_results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful completions"}
        
        min_timeout = min(r["timeout"] for r in successful_results)
        completion_rate = len(successful_results) / len(timeout_results)
        
        return {
            "minimum_successful_timeout": min_timeout,
            "completion_rate": completion_rate,
            "timeout_effectiveness": "good" if completion_rate > 0.8 else "moderate" if completion_rate > 0.5 else "poor"
        }

    def compare_method_scalability(self, max_size: int = 200) -> Dict[str, Any]:
        """Compare scalability across all methods."""
        print(f"\n[+] Method scalability comparison")
        
        # Generate comparable test cases
        sizes = list(range(10, max_size + 1, 20))
        test_tables = self.generate_test_tables(sizes, ["balanced"])
        
        method_comparisons = {}
        
        for method_name in self.methods.keys():
            print(f"  Analyzing {method_name}...")
            
            method_results = []
            
            for a, b, c, d in test_tables["balanced"]:
                table_size = a + b + c + d
                
                if method_name == "unconditional":
                    # Use adaptive parameters
                    grid_size = max(10, min(25, table_size // 6))
                    timeout = min(180, table_size * 1.5)
                    result = self.run_scalability_test(method_name, (a, b, c, d), 
                                                     grid_size=grid_size, timeout=timeout)
                else:
                    result = self.run_scalability_test(method_name, (a, b, c, d))
                
                method_results.append(result)
                
                # Stop if method becomes too slow for comparison
                if result.execution_time > 120:
                    print(f"    Stopping {method_name} at size {table_size} (too slow for comparison)")
                    break
            
            method_comparisons[method_name] = {
                "results": [self._result_to_dict(r) for r in method_results],
                "max_size_tested": max((r.table_size for r in method_results), default=0),
                "success_rate": sum(1 for r in method_results if r.success) / len(method_results) if method_results else 0,
                "avg_time_successful": np.mean([r.execution_time for r in method_results if r.success]) if any(r.success for r in method_results) else None
            }
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(method_comparisons)
        
        return {
            "method_comparisons": method_comparisons,
            "summary": comparison_summary
        }

    def _generate_comparison_summary(self, method_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of method comparison."""
        summary = {
            "speed_ranking": [],
            "reliability_ranking": [],
            "scalability_ranking": []
        }
        
        # Speed ranking (average time for successful runs)
        speed_data = []
        for method, data in method_comparisons.items():
            if data["avg_time_successful"] is not None:
                speed_data.append((method, data["avg_time_successful"]))
        summary["speed_ranking"] = sorted(speed_data, key=lambda x: x[1])
        
        # Reliability ranking (success rate)
        reliability_data = [(method, data["success_rate"]) for method, data in method_comparisons.items()]
        summary["reliability_ranking"] = sorted(reliability_data, key=lambda x: x[1], reverse=True)
        
        # Scalability ranking (max size successfully tested)
        scalability_data = [(method, data["max_size_tested"]) for method, data in method_comparisons.items()]
        summary["scalability_ranking"] = sorted(scalability_data, key=lambda x: x[1], reverse=True)
        
        return summary

    def _result_to_dict(self, result: ScalabilityResult) -> Dict[str, Any]:
        """Convert ScalabilityResult to dictionary."""
        return {
            "table_size": result.table_size,
            "n1": result.n1,
            "n2": result.n2,
            "a": result.a,
            "b": result.b,
            "c": result.c,
            "d": result.d,
            "execution_time": result.execution_time,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "parameters": result.parameters
        }

    def run_complete_scalability_analysis(self, 
                                        methods_to_analyze: Optional[List[str]] = None,
                                        max_size: int = 300,
                                        include_complexity: bool = True,
                                        include_parameters: bool = True,
                                        include_comparison: bool = True) -> Dict[str, Any]:
        """Run complete scalability analysis."""
        
        if methods_to_analyze is None:
            methods_to_analyze = ["conditional", "midp", "blaker", "unconditional"]
        
        print("=== ExactCIs Scalability Analysis ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Methods to analyze: {methods_to_analyze}")
        print(f"Max size: {max_size}")
        
        analysis_results = {
            "timestamp": self.timestamp,
            "methods_analyzed": methods_to_analyze,
            "max_size": max_size,
            "complexity_analysis": {},
            "parameter_sensitivity": {},
            "method_comparison": {},
            "summary": {}
        }
        
        # 1. Computational complexity analysis
        if include_complexity:
            print("\n" + "="*60)
            print("1. COMPUTATIONAL COMPLEXITY ANALYSIS")
            print("="*60)
            
            for method_name in methods_to_analyze:
                try:
                    complexity_data = self.analyze_computational_complexity(method_name, max_size)
                    analysis_results["complexity_analysis"][method_name] = complexity_data
                except Exception as e:
                    print(f"Error in complexity analysis for {method_name}: {e}")
                    analysis_results["complexity_analysis"][method_name] = {"error": str(e)}
        
        # 2. Parameter sensitivity analysis
        if include_parameters:
            print("\n" + "="*60)
            print("2. PARAMETER SENSITIVITY ANALYSIS")
            print("="*60)
            
            try:
                param_data = self.analyze_parameter_sensitivity("unconditional")
                analysis_results["parameter_sensitivity"] = param_data
            except Exception as e:
                print(f"Error in parameter sensitivity analysis: {e}")
                analysis_results["parameter_sensitivity"] = {"error": str(e)}
        
        # 3. Method comparison
        if include_comparison:
            print("\n" + "="*60)
            print("3. METHOD SCALABILITY COMPARISON")
            print("="*60)
            
            try:
                comparison_data = self.compare_method_scalability(max_size)
                analysis_results["method_comparison"] = comparison_data
            except Exception as e:
                print(f"Error in method comparison: {e}")
                analysis_results["method_comparison"] = {"error": str(e)}
        
        # 4. Generate summary
        print("\n" + "="*60)
        print("4. SCALABILITY ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self._generate_scalability_summary(analysis_results)
        analysis_results["summary"] = summary
        
        # Save results
        results_file = self.output_dir / f"scalability_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\nScalability analysis saved to: {results_file}")
        return analysis_results

    def _generate_scalability_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of scalability analysis."""
        summary = {
            "complexity_classes": {},
            "scalability_limits": {},
            "performance_recommendations": [],
            "parameter_recommendations": {}
        }
        
        # Extract complexity classes
        for method, data in results["complexity_analysis"].items():
            if "error" not in data and "complexity_analysis" in data:
                complexity_info = data["complexity_analysis"]
                if "error" not in complexity_info:
                    summary["complexity_classes"][method] = {
                        "class": complexity_info.get("complexity_class", "unknown"),
                        "best_fit": complexity_info.get("best_fit", "unknown"),
                        "r_squared": complexity_info.get("best_r_squared", 0)
                    }
            
            # Extract scalability limits
            if "error" not in data:
                summary["scalability_limits"][method] = data.get("max_feasible_size", 0)
        
        # Extract parameter recommendations
        if "parameter_sensitivity" in results and "error" not in results["parameter_sensitivity"]:
            param_data = results["parameter_sensitivity"]
            if "grid_size_analysis" in param_data:
                grid_analysis = param_data["grid_size_analysis"].get("sensitivity", {})
                if "optimal_grid_size" in grid_analysis:
                    summary["parameter_recommendations"]["optimal_grid_size"] = grid_analysis["optimal_grid_size"]
        
        # Generate performance recommendations
        recommendations = []
        
        # Check for poor scalability
        poor_scalability = [method for method, limit in summary["scalability_limits"].items() if limit < 100]
        if poor_scalability:
            recommendations.append(f"Methods with poor scalability (limit <100): {', '.join(poor_scalability)}")
        
        # Check for exponential complexity
        exponential_methods = [method for method, data in summary["complexity_classes"].items()
                             if data.get("class") == "O(exp(n))"]
        if exponential_methods:
            recommendations.append(f"Methods with exponential complexity: {', '.join(exponential_methods)}")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing early stopping criteria for slow convergence",
            "Use adaptive parameter selection based on table size",
            "Implement progress monitoring for long-running calculations"
        ])
        
        summary["performance_recommendations"] = recommendations
        
        # Print summary
        print("\nComplexity Classes:")
        for method, data in summary["complexity_classes"].items():
            print(f"  {method}: {data['class']} (R² = {data['r_squared']:.3f})")
        
        print("\nScalability Limits:")
        for method, limit in summary["scalability_limits"].items():
            print(f"  {method}: max size ~{limit}")
        
        if summary["parameter_recommendations"]:
            print("\nParameter Recommendations:")
            for param, value in summary["parameter_recommendations"].items():
                print(f"  {param}: {value}")
        
        print("\nPerformance Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Scalability analyzer for ExactCIs")
    parser.add_argument("--methods", nargs="*", 
                       choices=["conditional", "midp", "blaker", "unconditional", "wald"],
                       help="Methods to analyze (default: conditional, midp, blaker, unconditional)")
    parser.add_argument("--max-size", type=int, default=300,
                       help="Maximum table size to test (default: 300)")
    parser.add_argument("--no-complexity", action="store_true",
                       help="Skip complexity analysis")
    parser.add_argument("--no-parameters", action="store_true",
                       help="Skip parameter sensitivity analysis")
    parser.add_argument("--no-comparison", action="store_true", 
                       help="Skip method comparison")
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ScalabilityAnalyzer(args.output_dir)
    
    # Run analysis
    try:
        results = analyzer.run_complete_scalability_analysis(
            methods_to_analyze=args.methods,
            max_size=args.max_size,
            include_complexity=not args.no_complexity,
            include_parameters=not args.no_parameters,
            include_comparison=not args.no_comparison
        )
        print("\n" + "="*60)
        print("SCALABILITY ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nError during scalability analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())