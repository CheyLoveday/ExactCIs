"""
Core performance profiling engine for ExactCIs methods.

Provides comprehensive benchmarking capabilities with user-friendly API
for performance analysis and method comparison.
"""

import time
import tracemalloc
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Core imports for timing and memory profiling
import psutil
import os

@dataclass 
class BenchmarkResult:
    """
    Results from benchmarking a single method on a single table.
    
    Attributes
    ----------
    method : str
        Name of the method benchmarked
    runtime_ms : float
        Execution time in milliseconds
    bounds : tuple of float
        Confidence interval bounds (lower, upper)
    memory_mb : float
        Peak memory usage in megabytes
    success : bool
        Whether the method completed without errors
    error : str, optional
        Error message if method failed
    convergence_info : dict, optional
        Additional convergence information for iterative methods
    """
    method: str
    runtime_ms: float
    bounds: Tuple[float, float]
    memory_mb: float
    success: bool
    error: Optional[str] = None
    convergence_info: Optional[Dict[str, Any]] = None
    
    @property
    def is_fast(self) -> bool:
        """True if runtime is below 10ms threshold."""
        return self.runtime_ms < 10.0
    
    @property 
    def has_finite_bounds(self) -> bool:
        """True if both bounds are finite."""
        return all(abs(bound) < float('inf') for bound in self.bounds)
    
    @property
    def width(self) -> float:
        """Confidence interval width."""
        if self.has_finite_bounds:
            return self.bounds[1] - self.bounds[0]
        return float('inf')
    
    def __str__(self) -> str:
        if self.success:
            return f"{self.method}: {self.runtime_ms:.2f}ms, CI=({self.bounds[0]:.3f}, {self.bounds[1]:.3f})"
        else:
            return f"{self.method}: FAILED - {self.error}"

@dataclass
class BenchmarkResults:
    """
    Container for multiple benchmark results with analysis capabilities.
    """
    results: List[BenchmarkResult]
    table: Tuple[int, int, int, int]
    scenario_name: Optional[str] = None
    
    @property
    def successful_results(self) -> List[BenchmarkResult]:
        """Results that completed successfully."""
        return [r for r in self.results if r.success]
    
    @property
    def fastest_method(self) -> str:
        """Name of fastest successful method."""
        successful = self.successful_results
        if not successful:
            return "No successful methods"
        return min(successful, key=lambda r: r.runtime_ms).method
    
    @property
    def slowest_method(self) -> str:
        """Name of slowest successful method.""" 
        successful = self.successful_results
        if not successful:
            return "No successful methods"
        return max(successful, key=lambda r: r.runtime_ms).method
    
    @property
    def narrowest_ci(self) -> str:
        """Method with narrowest confidence interval."""
        finite_results = [r for r in self.successful_results if r.has_finite_bounds]
        if not finite_results:
            return "No finite CIs"
        return min(finite_results, key=lambda r: r.width).method
    
    @property
    def success_rate(self) -> float:
        """Fraction of methods that completed successfully."""
        if not self.results:
            return 0.0
        return len(self.successful_results) / len(self.results)
    
    def get_result(self, method: str) -> Optional[BenchmarkResult]:
        """Get result for specific method."""
        for result in self.results:
            if result.method == method:
                return result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'table': self.table,
            'scenario_name': self.scenario_name,
            'summary': {
                'fastest_method': self.fastest_method,
                'slowest_method': self.slowest_method, 
                'narrowest_ci': self.narrowest_ci,
                'success_rate': self.success_rate
            },
            'results': [asdict(r) for r in self.results]
        }
    
    def plot(self, show_errors: bool = True):
        """
        Generate performance visualization.
        
        Creates scatter plot of runtime vs CI width with method labels.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required for plotting. Install with: pip install matplotlib")
            return None
            
        successful = self.successful_results
        if not successful:
            print("No successful results to plot")
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Runtime comparison
        methods = [r.method for r in successful]
        runtimes = [r.runtime_ms for r in successful]
        
        bars = ax1.bar(methods, runtimes, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Runtime (ms)')
        ax1.set_title('Method Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight fastest method
        fastest_idx = runtimes.index(min(runtimes))
        bars[fastest_idx].set_color('green')
        
        # Plot 2: Runtime vs CI Width scatter
        finite_results = [r for r in successful if r.has_finite_bounds]
        if finite_results:
            runtimes_finite = [r.runtime_ms for r in finite_results]
            widths = [r.width for r in finite_results]
            methods_finite = [r.method for r in finite_results]
            
            scatter = ax2.scatter(runtimes_finite, widths, alpha=0.7, s=100)
            
            for i, method in enumerate(methods_finite):
                ax2.annotate(method, (runtimes_finite[i], widths[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('Runtime (ms)')
            ax2.set_ylabel('CI Width')
            ax2.set_title('Performance vs Precision Trade-off')
            ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.show()
        return fig

class PerformanceProfiler:
    """
    Built-in profiling engine for ExactCIs methods.
    
    Provides comprehensive benchmarking with memory profiling,
    method recommendations, and cross-method comparisons.
    """
    
    def __init__(self):
        # Import all available methods dynamically
        self.or_methods = self._get_or_methods()
        self.rr_methods = self._get_rr_methods() 
        self.all_methods = {**self.or_methods, **self.rr_methods}
        
        # Get process for memory monitoring
        self.process = psutil.Process()
    
    def _get_or_methods(self) -> Dict[str, callable]:
        """Get all OR methods from exactcis."""
        try:
            from ..methods.conditional import exact_ci_conditional
            from ..methods.midp import exact_ci_midp
            from ..methods.blaker import exact_ci_blaker
            from ..methods.unconditional import exact_ci_unconditional
            from ..methods.wald import ci_wald_haldane
            
            return {
                'conditional': exact_ci_conditional,
                'midp': exact_ci_midp,
                'blaker': exact_ci_blaker,
                'unconditional': exact_ci_unconditional, 
                'wald_haldane': ci_wald_haldane
            }
        except ImportError as e:
            warnings.warn(f"Could not import OR methods: {e}")
            return {}
    
    def _get_rr_methods(self) -> Dict[str, callable]:
        """Get all RR methods from exactcis."""
        try:
            from ..methods.relative_risk import (
                ci_wald_rr,
                ci_wald_katz_rr, 
                ci_wald_correlated_rr,
                ci_score_rr,
                ci_score_cc_rr,
                ci_ustat_rr
            )
            
            return {
                'wald_rr': ci_wald_rr,
                'wald_katz_rr': ci_wald_katz_rr,
                'wald_corr_rr': ci_wald_correlated_rr,
                'score_rr': ci_score_rr,
                'score_cc_rr': ci_score_cc_rr,
                'ustat_rr': ci_ustat_rr
            }
        except ImportError as e:
            warnings.warn(f"Could not import RR methods: {e}")
            return {}
    
    def benchmark_table(self, a: int, b: int, c: int, d: int, 
                       methods: Optional[List[str]] = None,
                       alpha: float = 0.05,
                       warmup_runs: int = 1) -> BenchmarkResults:
        """
        Benchmark methods on specific 2x2 table.
        
        Parameters
        ----------
        a, b, c, d : int
            Cell counts of 2x2 contingency table
        methods : list of str, optional
            Methods to benchmark. If None, uses all available methods.
        alpha : float, default 0.05
            Significance level for confidence intervals
        warmup_runs : int, default 1
            Number of warmup runs to stabilize timing
            
        Returns
        -------
        BenchmarkResults
            Comprehensive results with timing, memory, and analysis
        """
        if methods is None:
            methods = list(self.all_methods.keys())
        
        # Validate methods
        invalid_methods = [m for m in methods if m not in self.all_methods]
        if invalid_methods:
            raise ValueError(f"Unknown methods: {invalid_methods}. "
                           f"Available: {list(self.all_methods.keys())}")
        
        results = []
        
        for method_name in methods:
            method_func = self.all_methods[method_name]
            result = self._benchmark_single_method(
                method_name, method_func, a, b, c, d, alpha, warmup_runs
            )
            results.append(result)
        
        return BenchmarkResults(results, table=(a, b, c, d))
    
    def _benchmark_single_method(self, method_name: str, method_func: callable,
                                a: int, b: int, c: int, d: int, alpha: float,
                                warmup_runs: int) -> BenchmarkResult:
        """Benchmark single method with memory and timing profiling."""
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                method_func(a, b, c, d, alpha)
            except:
                pass  # Ignore warmup errors
        
        # Start memory tracking
        memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # Actual benchmark run
        start_time = time.perf_counter()
        
        try:
            bounds = method_func(a, b, c, d, alpha)
            success = True
            error = None
            
            # Validate bounds
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"Method returned invalid bounds: {bounds}")
                
        except Exception as e:
            end_time = time.perf_counter()
            bounds = (float('nan'), float('nan'))
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        runtime_ms = (end_time - start_time) * 1000
        
        # Memory usage
        memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = max(0, memory_after - memory_before)
        
        return BenchmarkResult(
            method=method_name,
            runtime_ms=runtime_ms,
            bounds=bounds,
            memory_mb=memory_used,
            success=success,
            error=error
        )
    
    def recommend_method(self, a: int, b: int, c: int, d: int, 
                        priority: str = 'balanced',
                        method_type: str = 'auto') -> str:
        """
        Recommend optimal method based on data characteristics.
        
        Parameters
        ----------
        a, b, c, d : int
            Cell counts of 2x2 contingency table
        priority : {'balanced', 'speed', 'accuracy'}, default 'balanced'
            Optimization priority
        method_type : {'auto', 'or', 'rr'}, default 'auto'
            Whether to recommend OR methods, RR methods, or auto-detect
            
        Returns
        -------
        str
            Method recommendation with rationale
        """
        n = a + b + c + d
        min_cell = min(a, b, c, d)
        has_zero = min_cell == 0
        
        # Auto-detect method type if not specified
        if method_type == 'auto':
            # Simple heuristic: if this looks like a case-control study or
            # the user specifically wants OR/RR, we'd need more context
            # For now, default to OR methods which are more established
            method_type = 'or'
        
        # Get available methods for the type
        if method_type == 'or':
            available_methods = self.or_methods
            method_prefix = ""
        elif method_type == 'rr': 
            available_methods = self.rr_methods
            method_prefix = ""
        else:
            raise ValueError(f"Unknown method_type: {method_type}")
        
        # Rule-based recommendations
        if has_zero:
            if method_type == 'rr' and a == 0:
                return "wald_rr: Zero exposed events, handles continuity correction"
            elif method_type == 'rr' and c == 0:
                return "wald_rr: Zero unexposed events, appropriate infinite bounds"
            else:
                return "conditional: Zero cells require exact methods with guaranteed coverage"
        
        elif n < 40:
            if method_type == 'rr':
                return "score_rr: Small samples benefit from score-based inference"
            else:
                return "conditional: Small samples need exact coverage guarantees"
        
        elif n > 200:
            if priority == 'speed':
                return "wald_haldane: Large samples with asymptotic approximation" if method_type == 'or' else "wald_rr: Fastest RR method for large samples"
            elif priority == 'accuracy':
                return "blaker: Optimal coverage properties" if method_type == 'or' else "score_cc_rr: Most accurate RR method"
            else:
                return "midp: Good balance for large samples" if method_type == 'or' else "score_cc_rr: Balanced speed/accuracy for RR"
        
        else:  # Medium samples
            if priority == 'speed':
                return "wald_haldane: Fast approximation" if method_type == 'or' else "wald_katz_rr: Fast RR approximation"
            elif priority == 'accuracy':
                return "blaker: Most accurate coverage" if method_type == 'or' else "score_cc_rr: Accurate RR with continuity correction"
            else:
                return "midp: Best balanced choice" if method_type == 'or' else "score_cc_rr: Best balanced RR method"
    
    def compare_or_vs_rr(self, a: int, b: int, c: int, d: int) -> Dict[str, Any]:
        """
        Compare OR vs RR methods on the same data.
        
        Useful for understanding the trade-offs between odds ratios
        and relative risk calculations.
        """
        or_results = self.benchmark_table(a, b, c, d, list(self.or_methods.keys()))
        rr_results = self.benchmark_table(a, b, c, d, list(self.rr_methods.keys()))
        
        return {
            'or_results': or_results,
            'rr_results': rr_results,
            'summary': {
                'fastest_or': or_results.fastest_method,
                'fastest_rr': rr_results.fastest_method,
                'or_success_rate': or_results.success_rate,
                'rr_success_rate': rr_results.success_rate
            }
        }