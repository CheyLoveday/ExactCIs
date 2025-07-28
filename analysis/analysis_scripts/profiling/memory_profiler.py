#!/usr/bin/env python
"""
Memory profiler for ExactCIs methods.

This script analyzes memory usage patterns across different CI methods,
focusing on:
1. Peak memory usage for different table sizes
2. Memory allocation patterns in grid-based methods
3. Memory efficiency of core probability calculations
4. Memory scaling with table size parameters
"""

import sys
import os
import time
import argparse
import json
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Try to import memory_profiler
try:
    from memory_profiler import profile as memory_profile, LineProfiler as MemoryLineProfiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("memory_profiler not available. Install with: pip install memory-profiler")

# Import ExactCIs methods
from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

class MemoryProfiler:
    """Memory profiler for ExactCIs methods."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Methods to profile
        self.methods = {
            "conditional": exact_ci_conditional,
            "midp": exact_ci_midp,
            "blaker": exact_ci_blaker,
            "unconditional": exact_ci_unconditional,
            "wald": ci_wald_haldane
        }

    def measure_memory_usage(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure memory usage of a function call."""
        # Start tracing memory allocations
        tracemalloc.start()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else final_memory
        
        # Get tracemalloc statistics
        current, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "result": result,
            "success": success,
            "error": error,
            "execution_time": end_time - start_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": peak_memory,
            "memory_delta_mb": final_memory - initial_memory,
            "peak_traced_mb": peak_trace / 1024 / 1024,
            "current_traced_mb": current / 1024 / 1024
        }

    def profile_memory_scaling(self, method_name: str, max_table_size: int = 200) -> Dict[str, Any]:
        """Analyze how memory usage scales with table size."""
        print(f"\n[+] Memory scaling analysis: {method_name}")
        
        method_func = self.methods[method_name]
        scaling_data = []
        
        # Generate tables of increasing size
        if method_name == "unconditional":
            # Use smaller sizes for unconditional method due to computational cost
            table_sizes = [10, 20, 30, 50, 75, 100]
        else:
            table_sizes = [10, 25, 50, 75, 100, 150, 200]
        
        # Filter sizes based on max_table_size
        table_sizes = [size for size in table_sizes if size <= max_table_size]
        
        for size in table_sizes:
            print(f"  Testing size ~{size}...")
            
            # Generate a representative table of this size
            n1 = size // 2
            n2 = size - n1
            a = min(n1 // 3, n2 // 3)  # Make it somewhat unbalanced
            b = n1 - a
            c = min(n2 // 3, n1 // 3)
            d = n2 - c
            
            # Measure memory usage
            if method_name == "unconditional":
                # Use adaptive parameters for unconditional method
                grid_size = max(10, min(20, size // 4))
                timeout = min(120, size * 2)  # Scale timeout with size
                memory_data = self.measure_memory_usage(
                    method_func, a, b, c, d, 0.05, 
                    grid_size=grid_size, timeout=timeout
                )
            else:
                memory_data = self.measure_memory_usage(
                    method_func, a, b, c, d, 0.05
                )
            
            scaling_entry = {
                "table_size": size,
                "n1": n1,
                "n2": n2,
                "a": a, "b": b, "c": c, "d": d,
                **memory_data
            }
            
            scaling_data.append(scaling_entry)
            
            print(f"    Memory: {memory_data['memory_delta_mb']:.2f}MB delta, "
                  f"{memory_data['peak_traced_mb']:.2f}MB peak, "
                  f"{memory_data['execution_time']:.2f}s")
            
            # Stop if method fails or becomes too slow/memory intensive
            if not memory_data['success']:
                print(f"    Method failed at size {size}: {memory_data['error']}")
                break
            elif memory_data['execution_time'] > 300:  # 5 minutes
                print(f"    Method too slow at size {size}, stopping")
                break
            elif memory_data['peak_traced_mb'] > 500:  # 500MB
                print(f"    Memory usage too high at size {size}, stopping")
                break
        
        return {
            "method": method_name,
            "scaling_data": scaling_data,
            "max_size_tested": max(d["table_size"] for d in scaling_data) if scaling_data else 0,
            "memory_efficiency": self._analyze_memory_efficiency(scaling_data)
        }

    def _analyze_memory_efficiency(self, scaling_data: List[Dict]) -> Dict[str, Any]:
        """Analyze memory efficiency from scaling data."""
        if len(scaling_data) < 2:
            return {"error": "Insufficient data for efficiency analysis"}
        
        # Extract data for analysis
        sizes = [d["table_size"] for d in scaling_data if d["success"]]
        peak_memory = [d["peak_traced_mb"] for d in scaling_data if d["success"]]
        
        if len(sizes) < 2:
            return {"error": "Insufficient successful runs for analysis"}
        
        # Fit polynomial to understand scaling behavior
        try:
            # Linear fit: memory = a * size + b
            linear_coeffs = np.polyfit(sizes, peak_memory, 1)
            
            # Quadratic fit: memory = a * size^2 + b * size + c
            if len(sizes) >= 3:
                quad_coeffs = np.polyfit(sizes, peak_memory, 2)
                quad_r2 = np.corrcoef(peak_memory, np.polyval(quad_coeffs, sizes))[0, 1] ** 2
            else:
                quad_coeffs = None
                quad_r2 = None
            
            # Calculate R-squared for linear fit
            linear_r2 = np.corrcoef(peak_memory, np.polyval(linear_coeffs, sizes))[0, 1] ** 2
            
            return {
                "linear_slope_mb_per_size": linear_coeffs[0],
                "linear_intercept_mb": linear_coeffs[1],
                "linear_r_squared": linear_r2,
                "quadratic_coeffs": quad_coeffs.tolist() if quad_coeffs is not None else None,
                "quadratic_r_squared": quad_r2,
                "memory_growth_type": "quadratic" if quad_r2 and quad_r2 > linear_r2 + 0.1 else "linear",
                "efficiency_rating": "good" if linear_coeffs[0] < 1.0 else "moderate" if linear_coeffs[0] < 5.0 else "poor"
            }
        except Exception as e:
            return {"error": f"Failed to analyze efficiency: {e}"}

    def profile_grid_method_memory(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Specifically profile memory usage of grid-based methods with different grid sizes."""
        print(f"\n[+] Grid method memory analysis")
        
        if grid_sizes is None:
            grid_sizes = [5, 10, 20, 30, 50, 75, 100]
        
        # Use a representative medium-sized table
        a, b, c, d = 15, 35, 20, 30  # Total size = 100
        
        grid_analysis = []
        
        for grid_size in grid_sizes:
            print(f"  Testing grid_size={grid_size}...")
            
            memory_data = self.measure_memory_usage(
                exact_ci_unconditional, a, b, c, d, 0.05,
                grid_size=grid_size, timeout=120
            )
            
            grid_entry = {
                "grid_size": grid_size,
                "table_case": f"({a},{b},{c},{d})",
                **memory_data
            }
            
            grid_analysis.append(grid_entry)
            
            print(f"    Memory: {memory_data['memory_delta_mb']:.2f}MB delta, "
                  f"{memory_data['peak_traced_mb']:.2f}MB peak, "
                  f"{memory_data['execution_time']:.2f}s")
            
            if not memory_data['success']:
                print(f"    Failed with grid_size={grid_size}: {memory_data['error']}")
                break
        
        return {
            "grid_analysis": grid_analysis,
            "memory_vs_grid_correlation": self._analyze_grid_memory_correlation(grid_analysis)
        }

    def _analyze_grid_memory_correlation(self, grid_analysis: List[Dict]) -> Dict[str, Any]:
        """Analyze correlation between grid size and memory usage."""
        successful_runs = [run for run in grid_analysis if run["success"]]
        
        if len(successful_runs) < 2:
            return {"error": "Insufficient successful runs"}
        
        grid_sizes = [run["grid_size"] for run in successful_runs]
        peak_memory = [run["peak_traced_mb"] for run in successful_runs]
        execution_times = [run["execution_time"] for run in successful_runs]
        
        # Calculate correlations
        memory_correlation = np.corrcoef(grid_sizes, peak_memory)[0, 1]
        time_correlation = np.corrcoef(grid_sizes, execution_times)[0, 1]
        
        # Find optimal grid size (balance between speed and memory)
        memory_per_second = [m / max(t, 0.001) for m, t in zip(peak_memory, execution_times)]
        optimal_idx = np.argmin(memory_per_second)
        optimal_grid_size = grid_sizes[optimal_idx]
        
        return {
            "memory_grid_correlation": memory_correlation,
            "time_grid_correlation": time_correlation,
            "optimal_grid_size": optimal_grid_size,
            "memory_scaling": "linear" if abs(memory_correlation) > 0.8 else "non-linear"
        }

    def compare_method_memory_efficiency(self, test_cases: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Compare memory efficiency across all methods."""
        print(f"\n[+] Method memory efficiency comparison")
        
        comparison_data = {}
        
        for method_name in self.methods.keys():
            print(f"  Profiling {method_name}...")
            
            method_results = []
            
            for i, (a, b, c, d) in enumerate(test_cases):
                print(f"    Case {i+1}: ({a},{b},{c},{d})")
                
                if method_name == "unconditional":
                    # Use reasonable parameters for unconditional method
                    table_size = a + b + c + d
                    grid_size = max(10, min(20, table_size // 5))
                    memory_data = self.measure_memory_usage(
                        self.methods[method_name], a, b, c, d, 0.05,
                        grid_size=grid_size, timeout=60
                    )
                else:
                    memory_data = self.measure_memory_usage(
                        self.methods[method_name], a, b, c, d, 0.05
                    )
                
                case_result = {
                    "case": f"({a},{b},{c},{d})",
                    "table_size": a + b + c + d,
                    **memory_data
                }
                
                method_results.append(case_result)
                
                if not memory_data['success']:
                    print(f"      Failed: {memory_data['error']}")
                else:
                    print(f"      Memory: {memory_data['peak_traced_mb']:.2f}MB, "
                          f"Time: {memory_data['execution_time']:.2f}s")
            
            comparison_data[method_name] = {
                "results": method_results,
                "summary": self._summarize_method_memory(method_results)
            }
        
        return comparison_data

    def _summarize_method_memory(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize memory usage for a method across test cases."""
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful runs"}
        
        peak_memories = [r["peak_traced_mb"] for r in successful_results]
        execution_times = [r["execution_time"] for r in successful_results]
        
        return {
            "success_rate": len(successful_results) / len(results),
            "avg_peak_memory_mb": np.mean(peak_memories),
            "max_peak_memory_mb": np.max(peak_memories),
            "min_peak_memory_mb": np.min(peak_memories),
            "std_peak_memory_mb": np.std(peak_memories),
            "avg_execution_time": np.mean(execution_times),
            "memory_efficiency_score": np.mean(peak_memories) / np.mean(execution_times)  # Lower is better
        }

    def run_complete_memory_analysis(self,
                                     max_table_size: int = 200,
                                     include_grid_analysis: bool = True,
                                     include_method_comparison: bool = True) -> Dict[str, Any]:
        """Run complete memory profiling analysis."""
        
        print("=== ExactCIs Memory Profiling Analysis ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Max table size: {max_table_size}")
        
        analysis_results = {
            "timestamp": self.timestamp,
            "max_table_size": max_table_size,
            "memory_scaling": {},
            "grid_analysis": {},
            "method_comparison": {},
            "summary": {}
        }
        
        # 1. Memory scaling analysis
        print("\n" + "="*60)
        print("1. MEMORY SCALING ANALYSIS")
        print("="*60)
        
        methods_to_test = ["conditional", "midp", "blaker", "unconditional"]
        
        for method_name in methods_to_test:
            try:
                scaling_data = self.profile_memory_scaling(method_name, max_table_size)
                analysis_results["memory_scaling"][method_name] = scaling_data
            except Exception as e:
                print(f"Error in memory scaling for {method_name}: {e}")
                analysis_results["memory_scaling"][method_name] = {"error": str(e)}
        
        # 2. Grid-specific analysis
        if include_grid_analysis:
            print("\n" + "="*60)
            print("2. GRID METHOD MEMORY ANALYSIS")
            print("="*60)
            
            try:
                grid_data = self.profile_grid_method_memory()
                analysis_results["grid_analysis"] = grid_data
            except Exception as e:
                print(f"Error in grid analysis: {e}")
                analysis_results["grid_analysis"] = {"error": str(e)}
        
        # 3. Method comparison
        if include_method_comparison:
            print("\n" + "="*60)
            print("3. METHOD MEMORY COMPARISON")
            print("="*60)
            
            # Test cases of varying sizes
            test_cases = [
                (5, 15, 8, 12),    # Small
                (10, 25, 15, 20),  # Medium
                (20, 30, 25, 35),  # Large
                (2, 48, 3, 47),    # Extreme ratio
            ]
            
            try:
                comparison_data = self.compare_method_memory_efficiency(test_cases)
                analysis_results["method_comparison"] = comparison_data
            except Exception as e:
                print(f"Error in method comparison: {e}")
                analysis_results["method_comparison"] = {"error": str(e)}
        
        # 4. Generate summary
        print("\n" + "="*60)
        print("4. MEMORY ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self._generate_memory_summary(analysis_results)
        analysis_results["summary"] = summary
        
        # Save results
        results_file = self.output_dir / f"memory_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\nMemory analysis saved to: {results_file}")
        return analysis_results

    def _generate_memory_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of memory analysis."""
        summary = {
            "memory_efficiency_ranking": [],
            "memory_bottlenecks": [],
            "optimization_recommendations": [],
            "scaling_characteristics": {}
        }
        
        # Analyze method comparison data
        if "method_comparison" in results and "error" not in results["method_comparison"]:
            method_efficiencies = {}
            for method, data in results["method_comparison"].items():
                if "summary" in data and "error" not in data["summary"]:
                    summary_data = data["summary"]
                    # Lower memory efficiency score is better (less memory per second)
                    method_efficiencies[method] = summary_data.get("memory_efficiency_score", float('inf'))
            
            # Rank methods by memory efficiency
            summary["memory_efficiency_ranking"] = sorted(method_efficiencies.items(), key=lambda x: x[1])
        
        # Analyze scaling characteristics
        for method, scaling_data in results["memory_scaling"].items():
            if "error" not in scaling_data and "memory_efficiency" in scaling_data:
                efficiency = scaling_data["memory_efficiency"]
                if "error" not in efficiency:
                    summary["scaling_characteristics"][method] = {
                        "growth_type": efficiency.get("memory_growth_type", "unknown"),
                        "efficiency_rating": efficiency.get("efficiency_rating", "unknown"),
                        "linear_slope": efficiency.get("linear_slope_mb_per_size", 0)
                    }
        
        # Generate recommendations
        recommendations = []
        
        # Check for poor memory efficiency
        poor_methods = [method for method, rating in summary["scaling_characteristics"].items() 
                       if rating.get("efficiency_rating") == "poor"]
        if poor_methods:
            recommendations.append(f"Methods with poor memory efficiency: {', '.join(poor_methods)}")
        
        # Check for quadratic scaling
        quadratic_methods = [method for method, rating in summary["scaling_characteristics"].items()
                           if rating.get("growth_type") == "quadratic"]
        if quadratic_methods:
            recommendations.append(f"Methods with quadratic memory growth: {', '.join(quadratic_methods)}")
        
        # Grid-specific recommendations
        if "grid_analysis" in results and "error" not in results["grid_analysis"]:
            grid_data = results["grid_analysis"]
            if "memory_vs_grid_correlation" in grid_data:
                corr_data = grid_data["memory_vs_grid_correlation"]
                if "optimal_grid_size" in corr_data:
                    recommendations.append(f"Optimal grid size for memory efficiency: {corr_data['optimal_grid_size']}")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing streaming computation for large tables",
            "Investigate memory pooling for repeated calculations",
            "Profile garbage collection patterns during long computations"
        ])
        
        summary["optimization_recommendations"] = recommendations
        
        # Print summary
        print("\nMemory Efficiency Ranking (most to least efficient):")
        for method, score in summary["memory_efficiency_ranking"]:
            print(f"  {method}: {score:.2f} MB/second")
        
        print("\nMemory Scaling Characteristics:")
        for method, chars in summary["scaling_characteristics"].items():
            print(f"  {method}: {chars['growth_type']} growth, {chars['efficiency_rating']} efficiency")
        
        print("\nMemory Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Memory profiler for ExactCIs")
    parser.add_argument("--max-size", type=int, default=200,
                       help="Maximum table size to test (default: 200)")
    parser.add_argument("--no-grid-analysis", action="store_true",
                       help="Skip grid-specific memory analysis")
    parser.add_argument("--no-method-comparison", action="store_true", 
                       help="Skip method comparison analysis")
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create memory profiler
    profiler = MemoryProfiler(args.output_dir)
    
    # Check if we have required tools
    if not HAS_MEMORY_PROFILER:
        print("Warning: memory_profiler package not found. Some features may be limited.")
    
    # Run analysis
    try:
        results = profiler.run_complete_memory_analysis(
            max_table_size=args.max_size,
            include_grid_analysis=not args.no_grid_analysis,
            include_method_comparison=not args.no_method_comparison
        )
        print("\n" + "="*60)
        print("MEMORY PROFILING COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nError during memory profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())