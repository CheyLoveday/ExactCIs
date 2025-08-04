"""
Comprehensive performance profiling for ExactCIs methods.

This script profiles all five CI methods across various scenarios to identify
computational bottlenecks and performance characteristics.
"""

import time
import cProfile
import pstats
import io
import json
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import traceback
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from exactcis.methods.conditional import exact_ci_conditional
    from exactcis.methods.midp import exact_ci_midp
    from exactcis.methods.blaker import exact_ci_blaker
    from exactcis.methods.unconditional import exact_ci_unconditional
    from exactcis.methods.wald import ci_wald_haldane
    from exactcis import compute_all_cis
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Test scenarios: (a, b, c, d, description)
TEST_SCENARIOS = [
    # Small balanced tables
    (2, 3, 4, 5, "small_balanced"),
    (5, 5, 5, 5, "small_symmetric"),
    (1, 2, 3, 4, "very_small"),
    
    # Medium tables
    (10, 15, 12, 18, "medium_balanced"),
    (8, 22, 15, 25, "medium_imbalanced"),
    
    # Large tables
    (50, 75, 60, 90, "large_balanced"),
    (25, 125, 75, 175, "large_imbalanced"),
    
    # Zero cells (challenging cases)
    (0, 5, 3, 7, "zero_a"),
    (2, 0, 4, 6, "zero_b"),
    (3, 5, 0, 8, "zero_c"),
    (4, 6, 2, 0, "zero_d"),
    
    # Extreme cases
    (1, 99, 2, 98, "extreme_imbalanced"),
    (100, 200, 150, 300, "very_large"),
    
    # Rare events
    (1, 499, 2, 498, "rare_events"),
]

# Methods to test
METHODS = {
    'conditional': exact_ci_conditional,
    'midp': exact_ci_midp,
    'blaker': exact_ci_blaker,
    'unconditional': lambda a, b, c, d, alpha=0.05: exact_ci_unconditional(a, b, c, d, alpha, timeout=30),
    'wald_haldane': ci_wald_haldane,
}

class PerformanceProfiler:
    """Comprehensive performance profiler for ExactCIs methods."""
    
    def __init__(self, output_dir: str = "profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.timing_results: Dict[str, Dict[str, List[float]]] = {}
        self.error_results: Dict[str, Dict[str, List[str]]] = {}
        self.profile_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize storage structures
        for method_name in METHODS.keys():
            self.timing_results[method_name] = {}
            self.error_results[method_name] = {}
            self.profile_stats[method_name] = {}
    
    def time_method(self, method_func, a: int, b: int, c: int, d: int, 
                   method_name: str, scenario_name: str, repeats: int = 3) -> Tuple[float, Optional[Tuple[float, float]], Optional[str]]:
        """Time a single method execution with error handling."""
        times = []
        result = None
        error = None
        
        for _ in range(repeats):
            try:
                start_time = time.perf_counter()
                result = method_func(a, b, c, d)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                times.append(float('inf'))  # Mark failed runs
                break
        
        # Use minimum time from successful runs
        valid_times = [t for t in times if t != float('inf')]
        avg_time = min(valid_times) if valid_times else float('inf')
        
        return avg_time, result, error
    
    def profile_method_detailed(self, method_func, a: int, b: int, c: int, d: int, 
                              method_name: str, scenario_name: str) -> Optional[pstats.Stats]:
        """Perform detailed profiling of a method."""
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run the method
            result = method_func(a, b, c, d)
            
            profiler.disable()
            
            # Create stats object
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            
            return stats
            
        except Exception as e:
            print(f"Profiling failed for {method_name} on {scenario_name}: {e}")
            return None
    
    def run_timing_analysis(self):
        """Run timing analysis across all methods and scenarios."""
        print("=== TIMING ANALYSIS ===")
        print(f"Testing {len(METHODS)} methods across {len(TEST_SCENARIOS)} scenarios...")
        
        for i, (a, b, c, d, scenario_name) in enumerate(TEST_SCENARIOS):
            print(f"\n[{i+1}/{len(TEST_SCENARIOS)}] Testing scenario: {scenario_name} (a={a}, b={b}, c={c}, d={d})")
            
            scenario_results = {}
            
            for method_name, method_func in METHODS.items():
                print(f"  Testing {method_name}...", end=" ", flush=True)
                
                exec_time, result, error = self.time_method(
                    method_func, a, b, c, d, method_name, scenario_name
                )
                
                # Store results
                if scenario_name not in self.timing_results[method_name]:
                    self.timing_results[method_name][scenario_name] = []
                    self.error_results[method_name][scenario_name] = []
                
                self.timing_results[method_name][scenario_name].append(exec_time)
                if error:
                    self.error_results[method_name][scenario_name].append(error)
                    print(f"ERROR: {error}")
                else:
                    self.error_results[method_name][scenario_name].append(None)
                    if exec_time == float('inf'):
                        print("TIMEOUT")
                    else:
                        print(f"{exec_time:.4f}s")
                
                scenario_results[method_name] = {
                    'time': exec_time,
                    'result': result,
                    'error': error
                }
        
        self._save_timing_results()
    
    def run_detailed_profiling(self, slowest_methods: List[str], top_scenarios: List[str]):
        """Run detailed profiling on the slowest methods."""
        print(f"\n=== DETAILED PROFILING ===")
        print(f"Profiling methods: {slowest_methods}")
        print(f"On scenarios: {top_scenarios}")
        
        for method_name in slowest_methods:
            method_func = METHODS[method_name]
            
            for scenario_name in top_scenarios:
                # Find the scenario details
                scenario_data = None
                for a, b, c, d, name in TEST_SCENARIOS:
                    if name == scenario_name:
                        scenario_data = (a, b, c, d)
                        break
                
                if not scenario_data:
                    continue
                
                a, b, c, d = scenario_data
                print(f"\nDetailed profiling: {method_name} on {scenario_name}")
                
                stats = self.profile_method_detailed(method_func, a, b, c, d, method_name, scenario_name)
                
                if stats:
                    # Save detailed profile
                    profile_key = f"{method_name}_{scenario_name}"
                    
                    # Save raw stats to file
                    stats_file = self.output_dir / f"profile_{profile_key}.txt"
                    with open(stats_file, 'w') as f:
                        stats.print_stats(50, file=f)  # Top 50 functions
                    
                    # Extract key metrics
                    stats_dict = self._extract_profile_metrics(stats)
                    self.profile_stats[method_name][scenario_name] = stats_dict
                    
                    print(f"  Profile saved to {stats_file}")
                    print(f"  Total calls: {stats_dict.get('total_calls', 'N/A')}")
                    print(f"  Total time: {stats_dict.get('total_time', 'N/A'):.4f}s")
    
    def _extract_profile_metrics(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Extract key metrics from profile stats."""
        # Get the raw stats
        stats_dict = stats.get_stats()
        
        # Calculate total time and calls
        total_time = 0
        total_calls = 0
        function_details = []
        
        for func_key, (cc, nc, tt, ct, callers) in stats_dict.items():
            total_calls += nc
            total_time += tt
            
            filename, line_num, func_name = func_key
            function_details.append({
                'function': func_name,
                'filename': os.path.basename(filename),
                'line': line_num,
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0,
            })
        
        # Sort by total time
        function_details.sort(key=lambda x: x['total_time'], reverse=True)
        
        return {
            'total_time': total_time,
            'total_calls': total_calls,
            'top_functions': function_details[:20],  # Top 20 functions
        }
    
    def _save_timing_results(self):
        """Save timing results to JSON files."""
        # Save raw timing data
        timing_file = self.output_dir / "timing_results.json"
        with open(timing_file, 'w') as f:
            # Convert inf values to string for JSON serialization
            serializable_results = {}
            for method, scenarios in self.timing_results.items():
                serializable_results[method] = {}
                for scenario, times in scenarios.items():
                    serializable_results[method][scenario] = [
                        "inf" if t == float('inf') else t for t in times
                    ]
            json.dump(serializable_results, f, indent=2)
        
        # Save error data
        error_file = self.output_dir / "error_results.json"
        with open(error_file, 'w') as f:
            json.dump(self.error_results, f, indent=2)
        
        print(f"\nTiming results saved to {timing_file}")
        print(f"Error results saved to {error_file}")
    
    def analyze_results(self) -> Tuple[List[str], List[str]]:
        """Analyze results to identify slowest methods and most challenging scenarios."""
        print("\n=== RESULTS ANALYSIS ===")
        
        # Calculate average times per method
        method_avg_times = {}
        for method_name, scenarios in self.timing_results.items():
            total_time = 0
            valid_scenarios = 0
            
            for scenario_times in scenarios.values():
                valid_times = [t for t in scenario_times if t != float('inf')]
                if valid_times:
                    total_time += min(valid_times)
                    valid_scenarios += 1
            
            method_avg_times[method_name] = total_time / valid_scenarios if valid_scenarios > 0 else float('inf')
        
        # Sort methods by average time
        sorted_methods = sorted(method_avg_times.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMethod Performance Ranking (slowest to fastest):")
        for i, (method, avg_time) in enumerate(sorted_methods, 1):
            if avg_time == float('inf'):
                print(f"  {i}. {method}: FAILED")
            else:
                print(f"  {i}. {method}: {avg_time:.4f}s average")
        
        # Calculate scenario difficulty
        scenario_avg_times = {}
        for scenario_name in [name for _, _, _, _, name in TEST_SCENARIOS]:
            total_time = 0
            valid_methods = 0
            
            for method_times in self.timing_results.values():
                if scenario_name in method_times:
                    valid_times = [t for t in method_times[scenario_name] if t != float('inf')]
                    if valid_times:
                        total_time += min(valid_times)
                        valid_methods += 1
            
            scenario_avg_times[scenario_name] = total_time / valid_methods if valid_methods > 0 else float('inf')
        
        # Sort scenarios by difficulty
        sorted_scenarios = sorted(scenario_avg_times.items(), key=lambda x: x[1], reverse=True)
        
        print("\nScenario Difficulty Ranking (most challenging to easiest):")
        for i, (scenario, avg_time) in enumerate(sorted_scenarios, 1):
            if avg_time == float('inf'):
                print(f"  {i}. {scenario}: FAILED")
            else:
                print(f"  {i}. {scenario}: {avg_time:.4f}s average")
        
        # Identify top candidates for detailed profiling
        slowest_methods = [method for method, _ in sorted_methods[:3]]  # Top 3 slowest
        most_challenging_scenarios = [scenario for scenario, _ in sorted_scenarios[:5]]  # Top 5 scenarios
        
        return slowest_methods, most_challenging_scenarios
    
    def generate_report(self):
        """Generate a comprehensive performance report."""
        report_file = self.output_dir / "performance_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# ExactCIs Performance Analysis Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Method comparison
            f.write("## Method Performance Comparison\n\n")
            f.write("| Method | Avg Time (s) | Success Rate | Notes |\n")
            f.write("|--------|--------------|--------------|-------|\n")
            
            for method_name in METHODS.keys():
                scenarios = self.timing_results[method_name]
                
                # Calculate statistics
                all_times = []
                error_count = 0
                total_runs = 0
                
                for scenario_times in scenarios.values():
                    for time_val in scenario_times:
                        total_runs += 1
                        if time_val == float('inf'):
                            error_count += 1
                        else:
                            all_times.append(time_val)
                
                avg_time = sum(all_times) / len(all_times) if all_times else float('inf')
                success_rate = (total_runs - error_count) / total_runs if total_runs > 0 else 0
                
                notes = []
                if error_count > 0:
                    notes.append(f"{error_count} failures")
                if avg_time > 10:
                    notes.append("Very slow")
                elif avg_time > 1:
                    notes.append("Slow")
                
                f.write(f"| {method_name} | {avg_time:.4f} | {success_rate:.1%} | {', '.join(notes) or 'OK'} |\n")
            
            # Scenario analysis
            f.write("\n## Scenario Analysis\n\n")
            f.write("| Scenario | Description | Avg Time (s) | Failure Rate |\n")
            f.write("|----------|-------------|--------------|-------------|\n")
            
            for a, b, c, d, scenario_name in TEST_SCENARIOS:
                # Calculate scenario stats across all methods
                scenario_times = []
                scenario_errors = 0
                scenario_total = 0
                
                for method_name in METHODS.keys():
                    if scenario_name in self.timing_results[method_name]:
                        for time_val in self.timing_results[method_name][scenario_name]:
                            scenario_total += 1
                            if time_val == float('inf'):
                                scenario_errors += 1
                            else:
                                scenario_times.append(time_val)
                
                avg_time = sum(scenario_times) / len(scenario_times) if scenario_times else float('inf')
                error_rate = scenario_errors / scenario_total if scenario_total > 0 else 1.0
                
                description = f"({a},{b},{c},{d})"
                
                f.write(f"| {scenario_name} | {description} | {avg_time:.4f} | {error_rate:.1%} |\n")
            
            # Detailed profiling results
            if self.profile_stats:
                f.write("\n## Detailed Profiling Results\n\n")
                for method_name, scenarios in self.profile_stats.items():
                    f.write(f"### {method_name}\n\n")
                    for scenario_name, stats in scenarios.items():
                        f.write(f"#### Scenario: {scenario_name}\n\n")
                        f.write(f"- Total time: {stats.get('total_time', 0):.4f}s\n")
                        f.write(f"- Total calls: {stats.get('total_calls', 0)}\n")
                        f.write(f"- Top time-consuming functions:\n\n")
                        
                        for i, func in enumerate(stats.get('top_functions', [])[:10], 1):
                            f.write(f"  {i}. `{func['function']}` ({func['filename']}:{func['line']})\n")
                            f.write(f"     - {func['calls']} calls, {func['total_time']:.4f}s total, {func['time_per_call']:.6f}s per call\n")
                        f.write("\n")
        
        print(f"\nComprehensive report generated: {report_file}")

def main():
    """Main profiling execution."""
    print("ExactCIs Performance Profiler")
    print("=" * 50)
    
    profiler = PerformanceProfiler()
    
    try:
        # Run timing analysis
        profiler.run_timing_analysis()
        
        # Analyze results to identify bottlenecks
        slowest_methods, challenging_scenarios = profiler.analyze_results()
        
        # Run detailed profiling on the worst performers
        profiler.run_detailed_profiling(slowest_methods, challenging_scenarios)
        
        # Generate comprehensive report
        profiler.generate_report()
        
        print(f"\n✅ Profiling complete! Results saved in: {profiler.output_dir}")
        print("\nKey files generated:")
        print(f"  - timing_results.json: Raw timing data")
        print(f"  - error_results.json: Error information")
        print(f"  - performance_report.md: Human-readable report")
        print(f"  - profile_*.txt: Detailed profiling for slowest methods")
        
    except KeyboardInterrupt:
        print("\n\n❌ Profiling interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Profiling failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()