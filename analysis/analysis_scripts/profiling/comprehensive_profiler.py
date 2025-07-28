#!/usr/bin/env python
"""
Comprehensive profiler for ExactCIs with multi-level analysis.

This script implements a complete profiling strategy:
1. Function-level profiling using cProfile
2. Line-level profiling using line_profiler 
3. Targeted analysis of computational hotspots
4. Performance scaling analysis
5. Method comparison and optimization recommendations
"""

import cProfile
import pstats
import io
import subprocess
import sys
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pstats import SortKey

# Try to import line_profiler
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    print("line_profiler not available. Install with: pip install line_profiler")

# Import ExactCIs methods for direct profiling
from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)
from exactcis.core import pmf_weights, support, logsumexp

class ComprehensiveProfiler:
    """Comprehensive profiling class for ExactCIs methods."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test case categories
        self.test_categories = {
            "small": self._generate_small_tables(),
            "medium": self._generate_medium_tables(), 
            "large": self._generate_large_tables(),
            "extreme_ratio": self._generate_extreme_ratio_tables(),
            "edge_cases": self._generate_edge_cases()
        }
        
        # Methods to profile
        self.methods = {
            "conditional": exact_ci_conditional,
            "midp": exact_ci_midp,
            "blaker": exact_ci_blaker,
            "unconditional": exact_ci_unconditional,
            "wald": ci_wald_haldane
        }
        
        self.results = {}
        
    def _generate_small_tables(self) -> List[Tuple[int, int, int, int]]:
        """Generate small test tables (total count ≤ 20)."""
        return [
            (1, 2, 3, 4),
            (2, 3, 1, 5),
            (0, 5, 2, 8),
            (3, 1, 7, 2),
            (1, 9, 4, 6)
        ]
    
    def _generate_medium_tables(self) -> List[Tuple[int, int, int, int]]:
        """Generate medium test tables (total count 20-100)."""
        return [
            (10, 15, 12, 18),
            (5, 25, 8, 22),
            (20, 10, 15, 15),
            (2, 28, 7, 23),
            (18, 12, 6, 24)
        ]
    
    def _generate_large_tables(self) -> List[Tuple[int, int, int, int]]:
        """Generate large test tables (total count 100-300)."""
        return [
            (45, 55, 38, 62),
            (20, 80, 30, 70),
            (60, 40, 45, 55),
            (10, 90, 25, 75),
            (35, 65, 40, 60)
        ]
    
    def _generate_extreme_ratio_tables(self) -> List[Tuple[int, int, int, int]]:
        """Generate tables with extreme ratios."""
        return [
            (1, 99, 50, 50),
            (45, 5, 5, 45),
            (2, 48, 48, 2),
            (1, 19, 1, 19),
            (0, 50, 25, 25)
        ]
    
    def _generate_edge_cases(self) -> List[Tuple[int, int, int, int]]:
        """Generate edge case tables."""
        return [
            (0, 10, 0, 10),
            (10, 0, 10, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1)
        ]

    def profile_function_level(self, method_name: str, test_cases: List[Tuple], alpha: float = 0.05) -> Dict[str, Any]:
        """Profile at function level using cProfile."""
        print(f"\n[+] Function-level profiling: {method_name}")
        
        method_func = self.methods[method_name]
        profile_data = {}
        
        for category, cases in test_cases.items():
            print(f"  Testing {category} tables...")
            
            # Create profiler
            profiler = cProfile.Profile()
            
            # Profile the method across all test cases in this category
            profiler.enable()
            start_time = time.time()
            
            for a, b, c, d in cases:
                try:
                    if method_name == "unconditional":
                        # Use smaller grid size and timeout for profiling
                        result = method_func(a, b, c, d, alpha, grid_size=10, timeout=30)
                    else:
                        result = method_func(a, b, c, d, alpha)
                except Exception as e:
                    print(f"    Error with table ({a},{b},{c},{d}): {e}")
                    continue
            
            elapsed_time = time.time() - start_time
            profiler.disable()
            
            # Analyze profiler output
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats(SortKey.CUMULATIVE)
            ps.print_stats(20)
            
            profile_data[category] = {
                "elapsed_time": elapsed_time,
                "num_cases": len(cases),
                "avg_time_per_case": elapsed_time / len(cases),
                "profile_text": s.getvalue()
            }
            
            # Save detailed profile
            profile_file = self.output_dir / f"{method_name}_{category}_profile_{self.timestamp}.prof"
            ps.dump_stats(str(profile_file))
        
        return profile_data

    def profile_line_level(self, method_name: str, target_functions: List[str], test_case: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
        """Profile at line level using line_profiler."""
        if not HAS_LINE_PROFILER:
            print(f"Skipping line-level profiling for {method_name} (line_profiler not available)")
            return None
            
        print(f"\n[+] Line-level profiling: {method_name}")
        
        method_func = self.methods[method_name]
        a, b, c, d = test_case
        
        # Create line profiler
        profiler = LineProfiler()
        
        # Add target functions to profiler
        profiler.add_function(method_func)
        
        # Add specific functions based on method
        if method_name == "unconditional":
            try:
                from exactcis.methods.unconditional import (
                    _build_adaptive_grid, 
                    _optimize_grid_size,
                    _log_binom_pmf
                )
                profiler.add_function(_build_adaptive_grid)
                profiler.add_function(_optimize_grid_size)
                profiler.add_function(_log_binom_pmf)
            except ImportError:
                pass
                
        elif method_name == "blaker":
            try:
                from exactcis.methods.blaker import blaker_p_value, blaker_acceptability
                profiler.add_function(blaker_p_value)
                profiler.add_function(blaker_acceptability)
            except ImportError:
                pass
        
        # Add core functions
        profiler.add_function(pmf_weights)
        profiler.add_function(support)
        profiler.add_function(logsumexp)
        
        # Run profiling
        try:
            if method_name == "unconditional":
                profiler.enable_by_count()
                result = method_func(a, b, c, d, 0.05, grid_size=15, timeout=60)
                profiler.disable_by_count()
            else:
                profiler.enable_by_count()
                result = method_func(a, b, c, d, 0.05)
                profiler.disable_by_count()
        except Exception as e:
            print(f"  Error in line profiling: {e}")
            return None
        
        # Capture output
        s = io.StringIO()
        profiler.print_stats(stream=s)
        
        # Save line profile output
        line_profile_file = self.output_dir / f"{method_name}_line_profile_{self.timestamp}.txt"
        with open(line_profile_file, 'w') as f:
            f.write(s.getvalue())
        
        return {
            "test_case": test_case,
            "result": result,
            "profile_output": s.getvalue(),
            "profile_file": str(line_profile_file)
        }

    def profile_scalability(self, method_name: str, max_timeout: int = 120) -> Dict[str, Any]:
        """Analyze how performance scales with table size."""
        print(f"\n[+] Scalability analysis: {method_name}")
        
        method_func = self.methods[method_name]
        scalability_data = []
        
        # Generate tables of increasing size
        table_sizes = [10, 20, 50, 100, 200]
        if method_name == "unconditional":
            # Reduce max size for unconditional method to avoid very long runtimes
            table_sizes = [10, 20, 30, 50]
        
        for size in table_sizes:
            print(f"  Testing size ~{size}...")
            
            # Generate a representative table of this size
            n1 = size // 2
            n2 = size - n1
            a = min(n1 // 2, n2 // 2)
            b = n1 - a
            c = min(n2 // 2, n1 // 2)  
            d = n2 - c
            
            start_time = time.time()
            try:
                if method_name == "unconditional":
                    # Use adaptive grid size and timeout
                    grid_size = max(10, min(20, size // 3))
                    result = method_func(a, b, c, d, 0.05, grid_size=grid_size, timeout=max_timeout)
                else:
                    result = method_func(a, b, c, d, 0.05)
                elapsed_time = time.time() - start_time
                success = True
            except Exception as e:
                elapsed_time = time.time() - start_time
                result = None
                success = False
                print(f"    Failed: {e}")
            
            scalability_data.append({
                "table_size": size,
                "a": a, "b": b, "c": c, "d": d,
                "elapsed_time": elapsed_time,
                "success": success,
                "result": result
            })
            
            # Stop if method becomes too slow
            if elapsed_time > max_timeout:
                print(f"    Method too slow at size {size}, stopping scalability test")
                break
        
        return {
            "method": method_name,
            "scalability_data": scalability_data,
            "max_size_tested": max(d["table_size"] for d in scalability_data),
            "feasible_max_size": max(d["table_size"] for d in scalability_data if d["success"])
        }

    def run_comprehensive_analysis(self, 
                                   methods_to_profile: Optional[List[str]] = None,
                                   include_line_profiling: bool = True,
                                   include_scalability: bool = True) -> Dict[str, Any]:
        """Run comprehensive profiling analysis."""
        
        if methods_to_profile is None:
            methods_to_profile = ["conditional", "midp", "blaker", "unconditional"]
            # Skip wald as it's typically very fast
        
        print("=== ExactCIs Comprehensive Profiling Analysis ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Methods to profile: {methods_to_profile}")
        print(f"Include line profiling: {include_line_profiling and HAS_LINE_PROFILER}")
        print(f"Include scalability: {include_scalability}")
        
        analysis_results = {
            "timestamp": self.timestamp,
            "methods_profiled": methods_to_profile,
            "function_level": {},
            "line_level": {},
            "scalability": {},
            "summary": {}
        }
        
        # 1. Function-level profiling
        print("\n" + "="*60)
        print("1. FUNCTION-LEVEL PROFILING")
        print("="*60)
        
        for method_name in methods_to_profile:
            try:
                profile_data = self.profile_function_level(method_name, self.test_categories)
                analysis_results["function_level"][method_name] = profile_data
            except Exception as e:
                print(f"Error profiling {method_name}: {e}")
                analysis_results["function_level"][method_name] = {"error": str(e)}
        
        # 2. Line-level profiling (on representative cases)
        if include_line_profiling and HAS_LINE_PROFILER:
            print("\n" + "="*60)
            print("2. LINE-LEVEL PROFILING")
            print("="*60)
            
            # Use a medium-sized representative case
            representative_case = (10, 15, 12, 18)
            
            for method_name in methods_to_profile:
                try:
                    if method_name in ["blaker", "unconditional"]:  # Focus on slower methods
                        line_data = self.profile_line_level(method_name, [], representative_case)
                        analysis_results["line_level"][method_name] = line_data
                except Exception as e:
                    print(f"Error in line profiling {method_name}: {e}")
                    analysis_results["line_level"][method_name] = {"error": str(e)}
        
        # 3. Scalability analysis
        if include_scalability:
            print("\n" + "="*60)
            print("3. SCALABILITY ANALYSIS")
            print("="*60)
            
            for method_name in methods_to_profile:
                try:
                    scalability_data = self.profile_scalability(method_name)
                    analysis_results["scalability"][method_name] = scalability_data
                except Exception as e:
                    print(f"Error in scalability analysis {method_name}: {e}")
                    analysis_results["scalability"][method_name] = {"error": str(e)}
        
        # 4. Generate summary and recommendations
        print("\n" + "="*60)
        print("4. ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self._generate_summary(analysis_results)
        analysis_results["summary"] = summary
        
        # Save complete results
        results_file = self.output_dir / f"comprehensive_analysis_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to JSON-serializable types
            json_safe_results = self._make_json_safe(analysis_results)
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\nComplete analysis saved to: {results_file}")
        return analysis_results

    def _make_json_safe(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary and optimization recommendations."""
        summary = {
            "performance_ranking": [],
            "bottlenecks_identified": [],
            "optimization_recommendations": [],
            "scalability_limits": {}
        }
        
        # Analyze function-level results
        method_speeds = {}
        for method_name, data in results["function_level"].items():
            if "error" not in data:
                # Calculate average time across all categories
                total_time = sum(cat_data.get("elapsed_time", 0) for cat_data in data.values())
                total_cases = sum(cat_data.get("num_cases", 0) for cat_data in data.values())
                avg_time = total_time / total_cases if total_cases > 0 else float('inf')
                method_speeds[method_name] = avg_time
        
        # Rank methods by speed
        summary["performance_ranking"] = sorted(method_speeds.items(), key=lambda x: x[1])
        
        # Identify scalability limits
        for method_name, data in results["scalability"].items():
            if "error" not in data:
                summary["scalability_limits"][method_name] = data.get("feasible_max_size", "unknown")
        
        # Generate recommendations
        if "unconditional" in method_speeds and method_speeds["unconditional"] > 1.0:
            summary["optimization_recommendations"].append(
                "Unconditional method is slow - consider reducing grid_size for large tables"
            )
        
        if "blaker" in method_speeds and method_speeds["blaker"] > 0.5:
            summary["optimization_recommendations"].append(
                "Blaker method shows performance issues - consider optimizing p-value calculations"
            )
        
        # Add general recommendations
        summary["optimization_recommendations"].extend([
            "Consider implementing caching for frequently called core functions",
            "Profile memory usage for large tables to identify allocation bottlenecks",
            "Investigate parallel processing opportunities for grid-based methods"
        ])
        
        # Print summary
        print("\nPerformance Ranking (fastest to slowest):")
        for method, avg_time in summary["performance_ranking"]:
            print(f"  {method}: {avg_time:.4f}s average per case")
        
        print("\nScalability Limits:")
        for method, max_size in summary["scalability_limits"].items():
            print(f"  {method}: max feasible size ~{max_size}")
        
        print("\nOptimization Recommendations:")
        for i, rec in enumerate(summary["optimization_recommendations"], 1):
            print(f"  {i}. {rec}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive profiler for ExactCIs")
    parser.add_argument("--methods", nargs="*", 
                       choices=["conditional", "midp", "blaker", "unconditional", "wald"],
                       help="Methods to profile (default: conditional, midp, blaker, unconditional)")
    parser.add_argument("--no-line-profiling", action="store_true",
                       help="Skip line-level profiling")
    parser.add_argument("--no-scalability", action="store_true", 
                       help="Skip scalability analysis")
    parser.add_argument("--output-dir", default="profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = ComprehensiveProfiler(args.output_dir)
    
    # Run analysis
    try:
        results = profiler.run_comprehensive_analysis(
            methods_to_profile=args.methods,
            include_line_profiling=not args.no_line_profiling,
            include_scalability=not args.no_scalability
        )
        print("\n" + "="*60)
        print("PROFILING COMPLETED SUCCESSFULLY")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())