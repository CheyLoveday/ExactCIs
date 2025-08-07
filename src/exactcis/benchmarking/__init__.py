"""
Built-in performance benchmarking for ExactCIs.

This module provides first-class profiling and performance comparison functionality
as part of the core package, enabling users to:

- Benchmark methods on their specific data
- Get method recommendations based on data characteristics  
- Compare ExactCIs performance with R packages
- Access transparent, reproducible performance evidence

Examples
--------
>>> from exactcis import benchmark_table, recommend_method
>>> 
>>> # Quick benchmark of your 2x2 table
>>> results = benchmark_table(a=20, b=80, c=10, d=90)
>>> print(f"Fastest method: {results.fastest_method}")
>>> print(f"Most accurate: {results.most_accurate}")
>>> 
>>> # Get method recommendation 
>>> recommendation = recommend_method(a=20, b=80, c=10, d=90, priority='speed')
>>> print(recommendation)  # "wald_haldane: Large samples with speed priority"
>>>
>>> # Compare performance across scenarios
>>> comparison = compare_performance(table_size='medium', zero_cells=False)
>>> comparison.plot()  # Interactive performance vs accuracy chart
"""

from .profiler import PerformanceProfiler, BenchmarkResult, BenchmarkResults
from .comparator import compare_performance, MethodComparison
from .scenarios import StandardScenarios, EPIDEMIOLOGICAL_CASES, CLINICAL_TRIAL_CASES
from .reporter import PerformanceReport, generate_benchmark_report

# Convenience functions for direct import
def benchmark_table(a: int, b: int, c: int, d: int, methods=None, alpha: float = 0.05):
    """
    Benchmark all methods on specific 2x2 table.
    
    Parameters
    ----------
    a, b, c, d : int
        Cell counts of 2x2 contingency table
    methods : list of str, optional
        Methods to benchmark. If None, benchmarks all available methods.
    alpha : float, default 0.05
        Significance level for confidence intervals
        
    Returns
    -------
    BenchmarkResults
        Results object with .fastest_method, .most_accurate properties
        and plotting/analysis capabilities
        
    Examples
    --------
    >>> results = benchmark_table(20, 80, 10, 90)
    >>> print(f"Fastest: {results.fastest_method}")
    >>> print(f"Most accurate: {results.most_accurate}")
    >>> results.plot()  # Performance vs accuracy scatter plot
    """
    profiler = PerformanceProfiler()
    return profiler.benchmark_table(a, b, c, d, methods, alpha)

def recommend_method(a: int, b: int, c: int, d: int, priority: str = 'balanced'):
    """
    Get method recommendation based on data characteristics.
    
    Uses evidence-based rules considering sample size, zero cells,
    and user priorities to recommend the optimal method.
    
    Parameters
    ----------
    a, b, c, d : int
        Cell counts of 2x2 contingency table  
    priority : {'balanced', 'speed', 'accuracy'}, default 'balanced'
        Optimization priority:
        - 'speed': Fastest method that meets accuracy requirements
        - 'accuracy': Most accurate method regardless of speed
        - 'balanced': Best trade-off of speed and accuracy
        
    Returns
    -------
    str
        Method recommendation with rationale
        
    Examples
    --------
    >>> recommend_method(2, 3, 1, 4, priority='accuracy')
    'conditional: Small samples need exact coverage'
    
    >>> recommend_method(200, 800, 100, 900, priority='speed')  
    'wald_haldane: Large samples with speed priority'
    """
    profiler = PerformanceProfiler()
    return profiler.recommend_method(a, b, c, d, priority)

def run_full_benchmark_suite(output_path: str = None, include_r_comparison: bool = False):
    """
    Run comprehensive benchmark suite across all standard scenarios.
    
    This function runs the complete performance analysis used for
    package documentation and performance claims validation.
    
    Parameters
    ----------
    output_path : str, optional
        Path to save detailed benchmark results (JSON format)
    include_r_comparison : bool, default False
        Whether to include R package comparisons (requires rpy2)
        
    Returns
    -------
    PerformanceReport
        Comprehensive performance analysis with plots and recommendations
    """
    from .reporter import run_comprehensive_benchmarks
    return run_comprehensive_benchmarks(output_path, include_r_comparison)

# Version info for benchmark compatibility
__version__ = "1.0.0"
__benchmark_version__ = "2025.1"  # Tracks benchmark format changes

# Export public API
__all__ = [
    # Core classes
    'PerformanceProfiler', 'BenchmarkResult', 'BenchmarkResults',
    'MethodComparison', 'StandardScenarios', 'PerformanceReport',
    
    # Convenience functions  
    'benchmark_table', 'recommend_method', 'compare_performance',
    'run_full_benchmark_suite',
    
    # Standard scenarios
    'EPIDEMIOLOGICAL_CASES', 'CLINICAL_TRIAL_CASES'
]