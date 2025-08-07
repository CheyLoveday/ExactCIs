"""
Example showing how performance profiling integrates with ExactCIs main package.

This demonstrates the user-facing API and how benchmarking becomes a first-class
feature accessible through the main exactcis import.
"""

# This would be added to src/exactcis/__init__.py
"""
Enhanced __init__.py with built-in profiling support:

# Existing imports
from .methods.conditional import exact_ci_conditional
from .methods.midp import exact_ci_midp
from .methods.blaker import exact_ci_blaker
from .methods.unconditional import exact_ci_unconditional
from .methods.wald import ci_wald_haldane
from .methods.relative_risk import *

# NEW: Built-in profiling imports
from .benchmarking import (
    benchmark_table, 
    recommend_method, 
    compare_performance,
    PerformanceProfiler,
    StandardScenarios
)

# Enhanced __all__ exports
__all__ = [
    # Existing method exports
    'compute_all_cis', 'exact_ci_conditional', 'exact_ci_midp', 
    'exact_ci_blaker', 'exact_ci_unconditional', 'ci_wald_haldane',
    'ci_wald_rr', 'ci_wald_katz_rr', 'ci_wald_correlated_rr',
    'ci_score_rr', 'ci_score_cc_rr', 'ci_ustat_rr',
    
    # NEW: Profiling exports for trust-building and method selection
    'benchmark_table', 'recommend_method', 'compare_performance',
    'PerformanceProfiler', 'StandardScenarios'
]
"""

def example_user_workflows():
    """Examples of how users would interact with built-in profiling."""
    
    print("=== ExactCIs Built-in Profiling Examples ===\n")
    
    # Example 1: Quick method recommendation
    print("1. Getting method recommendations:")
    print(">>> from exactcis import recommend_method")
    print(">>> recommendation = recommend_method(20, 80, 10, 90, priority='speed')")
    print(">>> print(recommendation)")
    print("# Output: 'wald_haldane: Large samples with asymptotic approximation'")
    print()
    
    # Example 2: Benchmark specific table
    print("2. Benchmarking your specific data:")
    print(">>> from exactcis import benchmark_table")
    print(">>> results = benchmark_table(a=15, b=85, c=25, d=75)")
    print(">>> print(f'Fastest method: {results.fastest_method}')")
    print(">>> print(f'Most accurate: {results.narrowest_ci}')")
    print(">>> results.plot()  # Interactive performance visualization")
    print()
    
    # Example 3: Compare OR vs RR methods
    print("3. Comparing OR vs RR approaches:")
    print(">>> from exactcis import PerformanceProfiler")
    print(">>> profiler = PerformanceProfiler()")
    print(">>> comparison = profiler.compare_or_vs_rr(20, 80, 10, 90)")
    print(">>> print(f'Fastest OR method: {comparison[\"summary\"][\"fastest_or\"]}')")
    print(">>> print(f'Fastest RR method: {comparison[\"summary\"][\"fastest_rr\"]}')")
    print()
    
    # Example 4: Validate Python vs R performance claims
    print("4. Validating performance claims (requires rpy2):")
    print(">>> from exactcis.benchmarking import run_full_benchmark_suite")
    print(">>> report = run_full_benchmark_suite(include_r_comparison=True)")
    print(">>> print(report.python_vs_r_summary())")
    print("# Shows evidence that ExactCIs is faster than R equivalents")
    print()
    
    # Example 5: Method selection based on data characteristics
    print("5. Data-driven method selection:")
    print(">>> from exactcis import StandardScenarios, benchmark_table")
    print(">>> scenarios = StandardScenarios()")
    print(">>> ")
    print(">>> # Test your table against similar scenarios")
    print(">>> similar_tables = scenarios.get('epidemiological').tables")
    print(">>> for table in similar_tables:")
    print(">>>     results = benchmark_table(*table)")
    print(">>>     print(f'{table}: fastest = {results.fastest_method}')")
    print()

def example_documentation_integration():
    """Example of how performance data would appear in documentation."""
    
    doc_example = """
Method Selection Guide
======================

ExactCIs provides transparent, reproducible performance benchmarks to help you 
choose the optimal method for your data. All benchmarks can be reproduced via:

.. code-block:: python

   from exactcis import benchmark_table, StandardScenarios
   scenarios = StandardScenarios()
   benchmark_table(*scenarios.get('epidemiological').tables[0])

Performance Overview
--------------------

.. performance-summary::
   :data-source: automated-benchmarks
   :update-frequency: weekly

+------------------+-------------------+-------------------+------------------+
| Sample Size      | Recommended (OR)  | Recommended (RR)  | Avg Runtime      |
+==================+===================+===================+==================+
| Small (n < 50)   | conditional       | score_rr          | 2.1ms            |
+------------------+-------------------+-------------------+------------------+
| Medium (50-200)  | midp              | score_cc_rr       | 0.8ms            |
+------------------+-------------------+-------------------+------------------+
| Large (n > 200)  | wald_haldane      | wald_katz_rr      | 0.002ms          |
+------------------+-------------------+-------------------+------------------+

Method Comparison: Small Samples
---------------------------------

For small samples (n < 50), exact methods are essential for proper coverage:

.. benchmark-chart::
   :scenario: small_sample
   :methods: [conditional, midp, blaker, score_rr]
   :metrics: [runtime, accuracy, ci_width]
   
**Key Findings:**
- conditional: Guaranteed coverage, conservative (5.2ms avg)
- midp: Less conservative, faster (2.1ms avg) 
- score_rr: Best for RR estimation (3.8ms avg)

Python vs R Performance
------------------------

ExactCIs consistently outperforms R equivalents:

.. speed-comparison::
   :python-methods: [conditional, midp, blaker]
   :r-packages: [exact2x2, Exact]
   :scenarios: [epidemiological, clinical_trial]

**Results Summary:**
- 2-5x faster than R exact2x2 package
- 3-8x faster than R Exact package  
- Identical numerical results (validated)

Interactive Method Selection
----------------------------

Use built-in profiling to find the best method for your data:

.. code-block:: python

   from exactcis import benchmark_table, recommend_method
   
   # Your 2x2 table
   a, b, c, d = 20, 80, 10, 90
   
   # Get recommendation  
   recommendation = recommend_method(a, b, c, d, priority='balanced')
   print(recommendation)  # "midp: Good balance for medium samples"
   
   # Benchmark all methods
   results = benchmark_table(a, b, c, d)
   print(f"Fastest: {results.fastest_method}")
   print(f"Narrowest CI: {results.narrowest_ci}")
   
   # Interactive plot
   results.plot()  # Shows performance vs accuracy trade-offs

All benchmark data is automatically updated weekly via CI/CD and reflects
the latest optimizations and algorithm improvements.
"""
    
    print("=== Documentation Integration Example ===")
    print(doc_example)

def example_ci_integration():
    """Example of CI integration for automated performance monitoring."""
    
    ci_workflow = """
# .github/workflows/performance-monitoring.yml
name: Performance Monitoring
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly performance checks
  push:
    paths: ['src/exactcis/methods/**']

jobs:
  performance-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev,benchmarking]"
          
      - name: Run performance benchmarks
        run: |
          python -c "
          from exactcis.benchmarking import run_full_benchmark_suite
          report = run_full_benchmark_suite(
              output_path='performance_report.json',
              compare_with_baseline=True
          )
          
          # Check for performance regressions
          if report.has_significant_regression(threshold=0.20):
              print('❌ Performance regression detected!')
              exit(1)
          else:
              print('✅ Performance benchmarks passed')
          "
          
      - name: Update documentation benchmarks
        run: |
          python -c "
          from exactcis.benchmarking.reporter import update_docs_benchmarks
          update_docs_benchmarks('docs/source/performance/')
          "
          
      - name: Commit updated benchmarks
        if: github.event_name == 'schedule'
        run: |
          git config --local user.email 'action@github.com'
          git config --local user.name 'GitHub Action'
          git add docs/source/performance/
          git commit -m '🚀 Auto-update performance benchmarks' || exit 0
          git push
          
      - name: Create performance report
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '⚠️ Performance Regression Detected',
              body: `
              Performance benchmarks failed. Please investigate:
              - Check recent changes to method implementations
              - Review benchmark results in workflow logs
              - Consider optimizations if regression is genuine
              `
            })
"""
    
    print("=== CI Integration Example ===")
    print(ci_workflow)

if __name__ == "__main__":
    example_user_workflows()
    print("\n" + "="*70 + "\n")
    example_documentation_integration() 
    print("\n" + "="*70 + "\n")
    example_ci_integration()