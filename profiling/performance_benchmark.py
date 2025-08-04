"""
Performance benchmark suite for ExactCIs optimization validation.

This script provides comprehensive performance benchmarking to validate
optimization improvements and detect performance regressions.
"""

import time
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
import numpy as np

# Add src to path
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

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "profiling"):
        self.output_dir = Path(output_dir)
        self.results = {}
        
        # Benchmark scenarios grouped by complexity
        self.scenarios = {
            'small': [
                (2, 3, 4, 5, "small_balanced"),
                (1, 2, 3, 4, "very_small"),
                (5, 5, 5, 5, "small_symmetric"),
            ],
            'medium': [
                (10, 15, 12, 18, "medium_balanced"),
                (8, 22, 15, 25, "medium_imbalanced"),
            ],
            'large': [
                (50, 75, 60, 90, "large_balanced"),
                (25, 125, 75, 175, "large_imbalanced"),
            ],
            'very_large': [
                (100, 200, 150, 300, "very_large"),
            ],
            'extreme': [
                (200, 400, 300, 600, "extreme_case"),
            ]
        }
        
        # Methods to benchmark
        self.methods = {
            'conditional': exact_ci_conditional,
            'midp': exact_ci_midp,
            'blaker': exact_ci_blaker,
            'unconditional': lambda a, b, c, d, alpha=0.05: exact_ci_unconditional(a, b, c, d, alpha, timeout=60),
            'wald_haldane': ci_wald_haldane,
        }
    
    def time_method_execution(self, method_func: Callable, args: Tuple, 
                             repeats: int = 5) -> Dict[str, Any]:
        """Time method execution with statistical analysis."""
        times = []
        results = []
        errors = []
        
        for i in range(repeats):
            try:
                start_time = time.perf_counter()
                result = method_func(*args)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                results.append(result)
                
            except Exception as e:
                errors.append(str(e))
                times.append(float('inf'))
        
        # Filter out failed runs for statistics
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {
                'status': 'failed',
                'errors': errors,
                'times': times
            }
        
        return {
            'status': 'success',
            'times': valid_times,
            'min_time': min(valid_times),
            'max_time': max(valid_times),
            'mean_time': statistics.mean(valid_times),
            'median_time': statistics.median(valid_times),
            'std_time': statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            'success_rate': len(valid_times) / repeats,
            'result_sample': results[0] if results else None,
            'errors': errors
        }
    
    def benchmark_scaling_behavior(self, method_name: str, 
                                 method_func: Callable) -> Dict[str, Any]:
        """Analyze scaling behavior across different problem sizes."""
        scaling_results = {}
        
        for size_category, scenarios in self.scenarios.items():
            category_times = []
            category_sizes = []
            
            print(f"  Testing {size_category} scenarios...")
            
            for a, b, c, d, scenario_name in scenarios:
                total_n = a + b + c + d
                category_sizes.append(total_n)
                
                try:
                    if method_name == 'unconditional':
                        result = self.time_method_execution(method_func, (a, b, c, d))
                    else:
                        result = self.time_method_execution(method_func, (a, b, c, d))
                    
                    if result['status'] == 'success':
                        category_times.append(result['min_time'])
                    else:
                        category_times.append(float('inf'))
                        print(f"    {scenario_name}: FAILED")
                        continue
                    
                    print(f"    {scenario_name} (N={total_n}): {result['min_time']:.4f}s")
                    
                except Exception as e:
                    print(f"    {scenario_name}: ERROR - {e}")
                    category_times.append(float('inf'))
            
            scaling_results[size_category] = {
                'times': category_times,
                'sizes': category_sizes,
                'avg_time': statistics.mean([t for t in category_times if t != float('inf')]) if any(t != float('inf') for t in category_times) else float('inf')
            }
        
        return scaling_results
    
    def analyze_performance_characteristics(self, method_name: str, 
                                          scaling_results: Dict) -> Dict[str, Any]:
        """Analyze performance characteristics and scaling behavior."""
        
        # Collect all valid time/size pairs
        all_times = []
        all_sizes = []
        
        for category_data in scaling_results.values():
            for time_val, size_val in zip(category_data['times'], category_data['sizes']):
                if time_val != float('inf'):
                    all_times.append(time_val)
                    all_sizes.append(size_val)
        
        if len(all_times) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate scaling coefficient (rough approximation)
        # Using log-log regression to estimate scaling exponent
        log_times = np.log(all_times)
        log_sizes = np.log(all_sizes)
        
        # Simple linear regression in log space
        n = len(log_times)
        sum_log_x = np.sum(log_sizes)
        sum_log_y = np.sum(log_times)
        sum_log_xy = np.sum(log_sizes * log_times)
        sum_log_x2 = np.sum(log_sizes ** 2)
        
        if n * sum_log_x2 - sum_log_x ** 2 != 0:
            scaling_exponent = (n * sum_log_xy - sum_log_x * sum_log_y) / (n * sum_log_x2 - sum_log_x ** 2)
        else:
            scaling_exponent = 0
        
        # Performance categories
        avg_time = statistics.mean(all_times)
        
        if avg_time < 0.001:
            performance_category = "Excellent"
        elif avg_time < 0.01:
            performance_category = "Good"  
        elif avg_time < 0.1:
            performance_category = "Moderate"
        elif avg_time < 1.0:
            performance_category = "Poor"
        else:
            performance_category = "Very Poor"
        
        return {
            'status': 'success',
            'scaling_exponent': scaling_exponent,
            'avg_execution_time': avg_time,
            'performance_category': performance_category,
            'time_range': (min(all_times), max(all_times)),
            'size_range': (min(all_sizes), max(all_sizes)),
            'data_points': len(all_times)
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all methods and scenarios."""
        print("Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        benchmark_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methods': {}
        }
        
        for method_name, method_func in self.methods.items():
            print(f"\nBenchmarking {method_name}...")
            
            try:
                # Get scaling behavior data
                scaling_results = self.benchmark_scaling_behavior(method_name, method_func)
                
                # Analyze performance characteristics
                characteristics = self.analyze_performance_characteristics(method_name, scaling_results)
                
                benchmark_results['methods'][method_name] = {
                    'scaling_results': scaling_results,
                    'characteristics': characteristics
                }
                
                if characteristics.get('status') == 'success':
                    print(f"  Performance: {characteristics['performance_category']}")
                    print(f"  Scaling exponent: {characteristics['scaling_exponent']:.2f}")
                    print(f"  Average time: {characteristics['avg_execution_time']:.4f}s")
                
            except Exception as e:
                print(f"  FAILED: {e}")
                benchmark_results['methods'][method_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return benchmark_results
    
    def compare_with_baseline(self, baseline_file: str = None) -> Dict[str, Any]:
        """Compare current performance with baseline measurements."""
        if baseline_file is None:
            baseline_file = self.output_dir / "baseline_performance.json"
        
        if not Path(baseline_file).exists():
            print(f"No baseline file found at {baseline_file}")
            return {}
        
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        current_results = self.run_comprehensive_benchmark()
        
        comparison = {
            'baseline_timestamp': baseline_data.get('timestamp', 'unknown'),
            'current_timestamp': current_results['timestamp'],
            'method_comparisons': {}
        }
        
        for method_name in self.methods.keys():
            if (method_name in baseline_data.get('methods', {}) and 
                method_name in current_results['methods']):
                
                baseline_method = baseline_data['methods'][method_name]
                current_method = current_results['methods'][method_name]
                
                if (baseline_method.get('characteristics', {}).get('status') == 'success' and
                    current_method.get('characteristics', {}).get('status') == 'success'):
                    
                    baseline_time = baseline_method['characteristics']['avg_execution_time']
                    current_time = current_method['characteristics']['avg_execution_time']
                    
                    improvement_ratio = baseline_time / current_time
                    improvement_percent = (improvement_ratio - 1) * 100
                    
                    comparison['method_comparisons'][method_name] = {
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'improvement_ratio': improvement_ratio,
                        'improvement_percent': improvement_percent,
                        'status': 'improved' if improvement_ratio > 1.05 else 'degraded' if improvement_ratio < 0.95 else 'unchanged'
                    }
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_file = self.output_dir / filename
        
        # Convert numpy types to native Python for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable performance report."""
        report = ["ExactCIs Performance Benchmark Report", "=" * 50, ""]
        report.append(f"Generated: {results['timestamp']}\n")
        
        # Summary table
        report.append("Method Performance Summary:")
        report.append("-" * 40)
        report.append(f"{'Method':<15} {'Category':<12} {'Avg Time':<12} {'Scaling':<10}")
        report.append("-" * 40)
        
        for method_name, method_data in results['methods'].items():
            if method_data.get('characteristics', {}).get('status') == 'success':
                chars = method_data['characteristics']
                avg_time = chars['avg_execution_time']
                category = chars['performance_category']
                scaling = f"{chars['scaling_exponent']:.2f}"
                
                report.append(f"{method_name:<15} {category:<12} {avg_time:<12.4f} {scaling:<10}")
            else:
                report.append(f"{method_name:<15} {'FAILED':<12} {'N/A':<12} {'N/A':<10}")
        
        report.append("")
        
        # Detailed analysis
        report.append("Detailed Analysis:")
        report.append("-" * 20)
        
        for method_name, method_data in results['methods'].items():
            report.append(f"\n{method_name.upper()}:")
            
            if method_data.get('characteristics', {}).get('status') == 'success':
                chars = method_data['characteristics']
                scaling_data = method_data['scaling_results']
                
                report.append(f"  Performance Category: {chars['performance_category']}")
                report.append(f"  Average Execution Time: {chars['avg_execution_time']:.4f}s")
                report.append(f"  Time Range: {chars['time_range'][0]:.4f}s - {chars['time_range'][1]:.4f}s")
                report.append(f"  Scaling Exponent: {chars['scaling_exponent']:.2f}")
                
                # Size category breakdown
                report.append("  Performance by Size Category:")
                for category, data in scaling_data.items():
                    if data['avg_time'] != float('inf'):
                        report.append(f"    {category}: {data['avg_time']:.4f}s average")
                    else:
                        report.append(f"    {category}: FAILED")
            else:
                report.append(f"  Status: {method_data.get('status', 'unknown')}")
                if 'error' in method_data:
                    report.append(f"  Error: {method_data['error']}")
        
        return "\n".join(report)

def main():
    """Main benchmark execution."""
    print("ExactCIs Performance Benchmark Suite")
    print("=" * 50)
    
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
    
    # Display summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(report)
    
    # Save as baseline if requested
    baseline_file = benchmark.output_dir / "baseline_performance.json"
    if not baseline_file.exists():
        benchmark.save_results(results, "baseline_performance.json")
        print(f"\nSaved as baseline: {baseline_file}")

if __name__ == "__main__":
    main()