"""
Line-by-line profiling for ExactCIs methods.

This script provides detailed line-level profiling using Python's built-in tools
and custom timing analysis to identify performance bottlenecks.
"""

import time
import sys
import cProfile
import pstats
import io
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import functools

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from exactcis.methods.conditional import exact_ci_conditional
    from exactcis.methods.midp import exact_ci_midp
    from exactcis.methods.blaker import exact_ci_blaker
    from exactcis.methods.unconditional import exact_ci_unconditional
    from exactcis.methods.wald import ci_wald_haldane
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test cases that showed performance issues
CHALLENGING_CASES = [
    (100, 200, 150, 300, "very_large"),
    (50, 75, 60, 90, "large_balanced"),
    (25, 125, 75, 175, "large_imbalanced"),
    (10, 15, 12, 18, "medium_balanced"),
]

class DetailedProfiler:
    """Detailed line-by-line performance profiler."""
    
    def __init__(self, output_dir: str = "profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timing_data = {}
        
    def time_function_calls(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Time individual function calls within a method execution.
        """
        call_times = {}
        original_functions = {}
        
        # Get the module containing the function
        if hasattr(func, '__module__'):
            module = sys.modules[func.__module__]
        else:
            return None, {}
        
        def create_timer_wrapper(original_func, func_name):
            @functools.wraps(original_func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = original_func(*args, **kwargs)
                end = time.perf_counter()
                
                if func_name not in call_times:
                    call_times[func_name] = []
                call_times[func_name].append(end - start)
                return result
            return wrapper
        
        # Instrument key functions based on the method being profiled
        functions_to_instrument = self._get_functions_to_instrument(func.__name__)
        
        # Apply instrumentation
        for func_name in functions_to_instrument:
            if hasattr(module, func_name):
                original_func = getattr(module, func_name)
                if callable(original_func):
                    original_functions[func_name] = original_func
                    setattr(module, func_name, create_timer_wrapper(original_func, func_name))
        
        try:
            # Run the function
            result = func(*args, **kwargs)
            
            # Calculate summary statistics
            summary_times = {}
            for func_name, times in call_times.items():
                summary_times[func_name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
            
            return result, summary_times
            
        finally:
            # Restore original functions
            for func_name, original_func in original_functions.items():
                setattr(module, func_name, original_func)
    
    def _get_functions_to_instrument(self, method_name: str) -> List[str]:
        """Get list of functions to instrument based on the method being profiled."""
        
        # Common core functions across all methods
        core_functions = [
            'log_nchg_pmf', 'log_binom_coeff', 'logsumexp', 'find_root_log',
            'find_smallest_theta', 'validate_counts', 'support'
        ]
        
        method_specific = {
            'exact_ci_conditional': [
                'fisher_lower_bound', 'fisher_upper_bound', 'zero_cell_upper_bound',
                'zero_cell_lower_bound', 'validate_bounds'
            ],
            'exact_ci_midp': [
                'midp_pval_func'  # This is defined inline, so we can't instrument it easily
            ],
            'exact_ci_blaker': [
                'blaker_acceptability', 'blaker_p_value', '_clear_blaker_cache',
                '_cache_key_for_pmf'
            ],
            'exact_ci_unconditional': [
                '_process_grid_point', '_build_adaptive_grid', '_optimize_grid_size',
                '_log_binom_pmf', 'calculate_log_p_value_barnard'
            ],
            'ci_wald_haldane': []  # Very simple, no complex internal functions
        }
        
        return core_functions + method_specific.get(method_name, [])
    
    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> Tuple[Any, pstats.Stats]:
        """Profile using cProfile for detailed call statistics."""
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Create stats object
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return result, stats
    
    def analyze_method_performance(self, method_name: str, method_func: Callable, 
                                 test_cases: List[Tuple]) -> Dict[str, Any]:
        """Comprehensive performance analysis of a single method."""
        print(f"\n=== Analyzing {method_name} ===")
        
        method_results = {
            'method_name': method_name,
            'test_cases': {},
            'summary': {}
        }
        
        total_time = 0
        total_calls = 0
        
        for a, b, c, d, case_name in test_cases:
            print(f"  Testing case: {case_name} (a={a}, b={b}, c={c}, d={d})")
            
            case_results = {}
            
            try:
                # Basic timing
                start_time = time.perf_counter()
                if method_name == 'exact_ci_unconditional':
                    result = method_func(a, b, c, d, timeout=30)
                else:
                    result = method_func(a, b, c, d)
                end_time = time.perf_counter()
                
                basic_time = end_time - start_time
                total_time += basic_time
                
                case_results['basic_timing'] = {
                    'execution_time': basic_time,
                    'result': result
                }
                
                # Detailed timing with function instrumentation
                print(f"    Basic time: {basic_time:.4f}s")
                print(f"    Running detailed analysis...")
                
                if method_name == 'exact_ci_unconditional':
                    detailed_result, func_times = self.time_function_calls(method_func, a, b, c, d, timeout=30)
                else:
                    detailed_result, func_times = self.time_function_calls(method_func, a, b, c, d)
                
                case_results['detailed_timing'] = func_times
                
                # cProfile analysis
                if method_name == 'exact_ci_unconditional':
                    profile_result, stats = self.profile_with_cprofile(method_func, a, b, c, d, timeout=30)
                else:
                    profile_result, stats = self.profile_with_cprofile(method_func, a, b, c, d)
                
                # Extract key statistics from cProfile
                case_results['profile_stats'] = self._extract_profile_data(stats)
                
                print(f"    Analysis complete")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                case_results['error'] = str(e)
                case_results['traceback'] = traceback.format_exc()
            
            method_results['test_cases'][case_name] = case_results
        
        # Calculate summary statistics
        method_results['summary'] = {
            'total_time': total_time,
            'average_time': total_time / len(test_cases) if test_cases else 0,
            'successful_cases': len([case for case in method_results['test_cases'].values() 
                                   if 'error' not in case])
        }
        
        return method_results
    
    def _extract_profile_data(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Extract key data from cProfile stats."""
        # Get the stats dictionary directly from the stats object
        stats_dict = stats.stats
        
        function_data = []
        total_time = 0
        total_calls = 0
        
        for func_key, (cc, nc, tt, ct, callers) in stats_dict.items():
            filename, line_num, func_name = func_key
            
            # Only include functions from our package
            if 'exactcis' in filename or 'profiling' in filename:
                function_data.append({
                    'function': func_name,
                    'file': Path(filename).name,
                    'line': line_num,
                    'calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt / nc if nc > 0 else 0,
                    'cum_time_per_call': ct / nc if nc > 0 else 0,
                })
                
                total_time += tt
                total_calls += nc
        
        # Sort by total time
        function_data.sort(key=lambda x: x['total_time'], reverse=True)
        
        return {
            'total_time': total_time,
            'total_calls': total_calls,
            'functions': function_data[:30]  # Top 30 functions
        }
    
    def save_detailed_report(self, method_results: Dict[str, Any]):
        """Save detailed performance report for a method."""
        method_name = method_results['method_name']
        report_file = self.output_dir / f"detailed_profile_{method_name}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Detailed Performance Analysis: {method_name}\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            summary = method_results['summary']
            f.write("## Summary\n\n")
            f.write(f"- Total execution time: {summary['total_time']:.4f}s\n")
            f.write(f"- Average time per case: {summary['average_time']:.4f}s\n")
            f.write(f"- Successful cases: {summary['successful_cases']}\n\n")
            
            # Per-case analysis
            f.write("## Per-Case Analysis\n\n")
            
            for case_name, case_data in method_results['test_cases'].items():
                f.write(f"### Case: {case_name}\n\n")
                
                if 'error' in case_data:
                    f.write(f"**ERROR**: {case_data['error']}\n\n")
                    f.write("```\n")
                    f.write(case_data.get('traceback', ''))
                    f.write("```\n\n")
                    continue
                
                # Basic timing
                basic = case_data.get('basic_timing', {})
                f.write(f"**Execution Time**: {basic.get('execution_time', 0):.4f}s\n")
                f.write(f"**Result**: {basic.get('result', 'N/A')}\n\n")
                
                # Function timing details
                detailed = case_data.get('detailed_timing', {})
                if detailed:
                    f.write("**Function Call Analysis**:\n\n")
                    f.write("| Function | Calls | Total Time | Avg Time | Max Time |\n")
                    f.write("|----------|-------|------------|----------|----------|\n")
                    
                    # Sort by total time
                    sorted_funcs = sorted(detailed.items(), 
                                        key=lambda x: x[1]['total_time'], reverse=True)
                    
                    for func_name, timing in sorted_funcs:
                        f.write(f"| {func_name} | {timing['calls']} | "
                               f"{timing['total_time']:.4f}s | "
                               f"{timing['avg_time']:.6f}s | "
                               f"{timing['max_time']:.6f}s |\n")
                    f.write("\n")
                
                # Profile statistics
                profile_stats = case_data.get('profile_stats', {})
                if profile_stats and 'functions' in profile_stats:
                    f.write("**Top Time-Consuming Functions (cProfile)**:\n\n")
                    f.write("| Function | File | Calls | Total Time | Time/Call |\n")
                    f.write("|----------|------|-------|------------|----------|\n")
                    
                    for func_data in profile_stats['functions'][:15]:
                        f.write(f"| {func_data['function']} | {func_data['file']} | "
                               f"{func_data['calls']} | {func_data['total_time']:.4f}s | "
                               f"{func_data['time_per_call']:.6f}s |\n")
                    f.write("\n")
        
        print(f"Detailed report saved: {report_file}")

def main():
    """Main line profiling execution."""
    print("ExactCIs Line-by-Line Profiler")
    print("=" * 50)
    
    profiler = DetailedProfiler()
    
    # Based on initial results, focus on the slowest methods
    methods_to_profile = {
        'exact_ci_blaker': exact_ci_blaker,
        'exact_ci_conditional': exact_ci_conditional,
        'exact_ci_unconditional': exact_ci_unconditional,
    }
    
    for method_name, method_func in methods_to_profile.items():
        print(f"\nProfiling {method_name}...")
        
        try:
            results = profiler.analyze_method_performance(
                method_name, method_func, CHALLENGING_CASES
            )
            
            profiler.save_detailed_report(results)
            
        except Exception as e:
            print(f"Failed to profile {method_name}: {e}")
            traceback.print_exc()
    
    print(f"\n✅ Line profiling complete! Results saved in: {profiler.output_dir}")

if __name__ == "__main__":
    main()