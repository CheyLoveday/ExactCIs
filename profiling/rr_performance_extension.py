"""
Extension to performance_profiler.py for Relative Risk methods.

This script extends the existing OR profiling infrastructure to include 
all 6 RR methods with RR-specific test scenarios and analysis.
"""

import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import RR methods
from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)

# RR Method definitions
RR_METHODS = {
    'wald_rr': ci_wald_rr,
    'wald_katz_rr': ci_wald_katz_rr,
    'wald_corr_rr': ci_wald_correlated_rr,
    'score_rr': ci_score_rr,
    'score_cc_rr': ci_score_cc_rr,
    'ustat_rr': ci_ustat_rr
}

# RR-specific test scenarios focusing on recent fixes and edge cases
RR_TEST_SCENARIOS = [
    # Standard epidemiological cases
    (20, 80, 10, 90, "epi_standard"),       # RR ≈ 2.0
    (90, 910, 10, 990, "epi_smoking"),     # RR ≈ 9.0 (smoking-lung cancer)
    (15, 85, 25, 75, "clinical_trial"),    # RR ≈ 0.6 (protective effect)
    
    # Edge cases for RR-specific algorithms  
    (0, 100, 5, 95, "zero_exposed"),       # RR = 0
    (5, 5, 0, 10, "zero_unexposed"),       # RR = ∞, tests wald_corr fix
    (3, 7, 2, 8, "small_sample"),          # Small n, legitimate finite bounds
    
    # Score method stress tests (cases that previously failed)
    (15, 5, 10, 10, "score_fix_case"),     # Previously returned (1.18, inf)
    (1, 1, 1, 1, "minimal_counts"),        # Extreme sparsity
    
    # Performance stress cases
    (150, 50, 100, 100, "moderate_large"), # n=400
    (500, 500, 300, 700, "large_balanced"), # n=2000
]

class RRPerformanceProfiler:
    """Extended profiler specifically for RR methods."""
    
    def __init__(self):
        self.methods = RR_METHODS
        self.scenarios = RR_TEST_SCENARIOS
        self.results = {}
        
    def benchmark_all_rr_methods(self) -> Dict[str, Any]:
        """
        Benchmark all RR methods across all scenarios.
        Returns timing and success data.
        """
        print("=== Relative Risk Methods Performance Benchmark ===")
        
        results = {
            'method_timings': {},
            'scenario_analysis': {},
            'convergence_stats': {},
            'total_runtime': 0
        }
        
        start_time = time.time()
        
        for method_name, method_func in self.methods.items():
            print(f"\nBenchmarking {method_name}...")
            method_results = self._benchmark_single_method(method_name, method_func)
            results['method_timings'][method_name] = method_results
            
        results['total_runtime'] = time.time() - start_time
        
        # Generate scenario analysis
        results['scenario_analysis'] = self._analyze_scenarios()
        
        return results
    
    def _benchmark_single_method(self, method_name: str, method_func) -> Dict[str, Any]:
        """Benchmark a single RR method across all scenarios."""
        method_results = {
            'scenario_times': {},
            'total_time': 0,
            'success_count': 0,
            'error_count': 0,
            'infinite_bounds': 0,
            'convergence_issues': []
        }
        
        method_start = time.time()
        
        for a, b, c, d, description in self.scenarios:
            scenario_start = time.time()
            
            try:
                # Run the method
                if method_name == 'score_cc_rr':
                    # score_cc has different signature with delta parameter
                    lower, upper = method_func(a, b, c, d, alpha=0.05)
                else:
                    lower, upper = method_func(a, b, c, d, alpha=0.05)
                
                scenario_time = time.time() - scenario_start
                method_results['scenario_times'][description] = {
                    'time_ms': scenario_time * 1000,
                    'bounds': (lower, upper),
                    'status': 'success'
                }
                method_results['success_count'] += 1
                
                # Check for infinite bounds
                if upper == float('inf'):
                    method_results['infinite_bounds'] += 1
                    
                    # Log specific cases for analysis
                    if description in ['score_fix_case', 'small_sample']:
                        print(f"  Note: {method_name} returned inf upper bound for {description}")
                
            except Exception as e:
                scenario_time = time.time() - scenario_start
                method_results['scenario_times'][description] = {
                    'time_ms': scenario_time * 1000,
                    'status': 'error',
                    'error': str(e)
                }
                method_results['error_count'] += 1
                print(f"  Error in {description}: {str(e)}")
        
        method_results['total_time'] = time.time() - method_start
        return method_results
    
    def _analyze_scenarios(self) -> Dict[str, Any]:
        """Analyze performance patterns across scenarios."""
        # This would be filled by the main benchmark run
        return {
            'fastest_methods': {},
            'most_reliable_methods': {},
            'scenario_difficulty_ranking': {}
        }
    
    def benchmark_score_method_fixes(self) -> Dict[str, Any]:
        """
        Focus specifically on validating the score method fixes.
        Tests the cases that previously returned infinite bounds.
        """
        print("\n=== Score Method Fix Validation ===")
        
        # Test cases that were specifically fixed
        fix_test_cases = [
            (15, 5, 10, 10, "primary_fix_case"),    # Main fix target
            (12, 8, 6, 14, "secondary_case"),       # Similar pattern
            (20, 30, 15, 35, "larger_sample")       # Scaling test
        ]
        
        results = {}
        
        for method_name in ['score_rr', 'score_cc_rr']:
            method_func = self.methods[method_name]
            results[method_name] = {}
            
            print(f"\nTesting {method_name} fixes:")
            
            for a, b, c, d, description in fix_test_cases:
                start_time = time.time()
                
                try:
                    lower, upper = method_func(a, b, c, d)
                    runtime = (time.time() - start_time) * 1000
                    
                    # Check if we get finite bounds (the main fix objective)
                    has_finite_bounds = upper < float('inf')
                    
                    results[method_name][description] = {
                        'bounds': (lower, upper),
                        'runtime_ms': runtime,
                        'finite_upper_bound': has_finite_bounds,
                        'fix_successful': has_finite_bounds
                    }
                    
                    status = "✅ FIXED" if has_finite_bounds else "❌ Still infinite"
                    print(f"  {description}: {lower:.4f}, {upper} - {status} ({runtime:.2f}ms)")
                    
                except Exception as e:
                    results[method_name][description] = {
                        'error': str(e),
                        'fix_successful': False
                    }
                    print(f"  {description}: ❌ ERROR - {str(e)}")
        
        return results
    
    def benchmark_zero_cell_handling(self) -> Dict[str, Any]:
        """
        Test performance of zero-cell detection and delegation logic.
        Particularly for wald_correlated_rr improvements.
        """
        print("\n=== Zero Cell Handling Performance ===")
        
        zero_cell_cases = [
            (0, 10, 5, 5, "zero_a"),
            (5, 5, 0, 10, "zero_c_wald_corr_fix"),  # Specific fix case
            (0, 100, 0, 100, "zero_both"),
            (10, 0, 5, 5, "zero_b"),
        ]
        
        results = {}
        
        for method_name, method_func in self.methods.items():
            results[method_name] = {}
            
            for a, b, c, d, description in zero_cell_cases:
                start_time = time.time()
                
                try:
                    lower, upper = method_func(a, b, c, d)
                    runtime = (time.time() - start_time) * 1000
                    
                    # Analyze zero-cell behavior
                    upper_is_large = upper > 100 or upper == float('inf')
                    
                    results[method_name][description] = {
                        'bounds': (lower, upper),
                        'runtime_ms': runtime,
                        'upper_appropriately_large': upper_is_large,
                        'delegation_working': True  # Assuming no errors means delegation worked
                    }
                    
                except Exception as e:
                    results[method_name][description] = {
                        'error': str(e),
                        'delegation_working': False
                    }
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report = [
            "# Relative Risk Methods Performance Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- Total methods tested: {len(self.methods)}",
            f"- Total scenarios: {len(self.scenarios)}",
            f"- Total runtime: {results['total_runtime']:.2f} seconds",
            "",
            "## Method Performance Summary"
        ]
        
        for method_name, method_data in results['method_timings'].items():
            success_rate = method_data['success_count'] / len(self.scenarios) * 100
            avg_time = method_data['total_time'] / len(self.scenarios) * 1000
            
            report.extend([
                f"### {method_name}",
                f"- Success rate: {success_rate:.1f}%",
                f"- Average time: {avg_time:.2f}ms",
                f"- Infinite bounds: {method_data['infinite_bounds']} cases",
                f"- Errors: {method_data['error_count']} cases",
                ""
            ])
        
        return "\n".join(report)

def run_rr_profiling():
    """Main function to run RR performance profiling."""
    profiler = RRPerformanceProfiler()
    
    # Run comprehensive benchmark
    print("Starting RR performance profiling...")
    results = profiler.benchmark_all_rr_methods()
    
    # Run specific fix validation
    fix_results = profiler.benchmark_score_method_fixes()
    results['fix_validation'] = fix_results
    
    # Run zero-cell handling tests
    zero_cell_results = profiler.benchmark_zero_cell_handling()
    results['zero_cell_analysis'] = zero_cell_results
    
    # Generate report
    report = profiler.generate_performance_report(results)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "rr_performance_results.txt", "w") as f:
        f.write(report)
    
    # Save raw data as JSON
    import json
    with open(output_dir / "rr_performance_data.json", "w") as f:
        # Convert any non-serializable values
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # Test serializability
                serializable_results[key] = value
            except:
                serializable_results[key] = str(value)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nProfiling complete! Results saved to:")
    print(f"- Report: {output_dir / 'rr_performance_results.txt'}")
    print(f"- Data: {output_dir / 'rr_performance_data.json'}")
    
    return results

if __name__ == "__main__":
    run_rr_profiling()