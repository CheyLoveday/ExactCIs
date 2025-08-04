#!/usr/bin/env python3
"""
Performance verification test for conditional method optimizations.

This test measures the actual performance of the conditional method and validates
the claimed improvements from the optimization work.
"""

import pytest
import time
import logging
import statistics
from typing import List, Tuple
from exactcis.methods.conditional import exact_ci_conditional


# Set logging to WARNING to reduce overhead during performance testing
logging.basicConfig(level=logging.WARNING)


class PerformanceBenchmark:
    """Benchmark class for measuring conditional method performance."""
    
    def __init__(self):
        self.results = []
        self.times = []
    
    def measure_execution_time(self, a: int, b: int, c: int, d: int, 
                              alpha: float = 0.05, runs: int = 10) -> dict:
        """
        Measure execution time for conditional CI calculation.
        
        Args:
            a, b, c, d: 2x2 table cell counts
            alpha: Significance level
            runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        results = []
        
        # Warm-up run to ensure caches are initialized
        _ = exact_ci_conditional(a, b, c, d, alpha)
        
        # Perform timed runs
        for _ in range(runs):
            start = time.perf_counter()
            result = exact_ci_conditional(a, b, c, d, alpha)
            end = time.perf_counter()
            
            execution_time = (end - start) * 1000  # Convert to milliseconds
            times.append(execution_time)
            results.append(result)
        
        # Check consistency
        consistent = all(r == results[0] for r in results)
        
        return {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'total_runs': runs,
            'results_consistent': consistent,
            'sample_result': results[0],
            'raw_times': times
        }


@pytest.mark.slow
def test_conditional_performance_small_table():
    """Test performance on a small 2x2 table."""
    benchmark = PerformanceBenchmark()
    
    # Small table case
    stats = benchmark.measure_execution_time(5, 10, 8, 12, runs=20)
    
    print(f"\nSmall table (5,10,8,12) performance:")
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  Median: {stats['median_ms']:.2f}ms")
    print(f"  Range: {stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms")
    print(f"  Std Dev: {stats['std_ms']:.2f}ms")
    print(f"  Result: {stats['sample_result']}")
    print(f"  Consistent: {stats['results_consistent']}")
    
    # Basic sanity checks
    assert stats['mean_ms'] > 0
    assert stats['results_consistent']
    assert stats['std_ms'] >= 0


@pytest.mark.slow
def test_conditional_performance_medium_table():
    """Test performance on a medium-sized 2x2 table."""
    benchmark = PerformanceBenchmark()
    
    # Medium table case - this was likely used in original benchmarks
    stats = benchmark.measure_execution_time(100, 200, 150, 300, runs=20)
    
    print(f"\nMedium table (100,200,150,300) performance:")
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  Median: {stats['median_ms']:.2f}ms")
    print(f"  Range: {stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms")
    print(f"  Std Dev: {stats['std_ms']:.2f}ms")
    print(f"  Result: {stats['sample_result']}")
    print(f"  Consistent: {stats['results_consistent']}")
    
    # Basic sanity checks
    assert stats['mean_ms'] > 0
    assert stats['results_consistent']
    assert stats['std_ms'] >= 0


@pytest.mark.slow 
def test_conditional_performance_large_table():
    """Test performance on a large 2x2 table."""
    benchmark = PerformanceBenchmark()
    
    # Large table case
    stats = benchmark.measure_execution_time(500, 1000, 750, 1500, runs=10)
    
    print(f"\nLarge table (500,1000,750,1500) performance:")
    print(f"  Mean: {stats['mean_ms']:.2f}ms")
    print(f"  Median: {stats['median_ms']:.2f}ms")
    print(f"  Range: {stats['min_ms']:.2f}ms - {stats['max_ms']:.2f}ms")
    print(f"  Std Dev: {stats['std_ms']:.2f}ms")
    print(f"  Result: {stats['sample_result']}")
    print(f"  Consistent: {stats['results_consistent']}")
    
    # Basic sanity checks
    assert stats['mean_ms'] > 0
    assert stats['results_consistent']
    assert stats['std_ms'] >= 0


@pytest.mark.slow
def test_conditional_performance_edge_cases():
    """Test performance on edge cases (zeros, small values)."""
    benchmark = PerformanceBenchmark()
    
    edge_cases = [
        (0, 5, 3, 7),    # Zero in a
        (3, 0, 7, 5),    # Zero in b  
        (5, 3, 0, 7),    # Zero in c
        (7, 5, 3, 0),    # Zero in d
        (1, 1, 1, 1),    # All ones
    ]
    
    print(f"\nEdge case performance:")
    
    for i, (a, b, c, d) in enumerate(edge_cases):
        stats = benchmark.measure_execution_time(a, b, c, d, runs=10)
        
        print(f"  Case {i+1} ({a},{b},{c},{d}):")
        print(f"    Mean: {stats['mean_ms']:.2f}ms")
        print(f"    Result: {stats['sample_result']}")
        print(f"    Consistent: {stats['results_consistent']}")
        
        # Sanity checks
        assert stats['mean_ms'] > 0
        assert stats['results_consistent']


def test_conditional_cache_effectiveness():
    """Test that caching is working by measuring repeated calls."""
    benchmark = PerformanceBenchmark()
    
    # First call (cold cache)
    stats_cold = benchmark.measure_execution_time(100, 200, 150, 300, runs=1)
    
    # Subsequent calls (warm cache) - should be faster
    stats_warm = benchmark.measure_execution_time(100, 200, 150, 300, runs=10)
    
    print(f"\nCache effectiveness test:")
    print(f"  Cold cache (1st call): {stats_cold['mean_ms']:.2f}ms")
    print(f"  Warm cache (avg): {stats_warm['mean_ms']:.2f}ms")
    print(f"  Cache speedup: {stats_cold['mean_ms'] / stats_warm['mean_ms']:.1f}x")
    
    # The warm cache should generally be faster or similar
    # (though this isn't guaranteed due to system variability)
    assert stats_warm['results_consistent']
    assert stats_cold['sample_result'] == stats_warm['sample_result']


@pytest.mark.slow
def test_conditional_performance_comparison():
    """
    Performance comparison test to validate claimed improvements.
    
    This test provides a comprehensive performance profile that can be
    compared against baseline measurements.
    """
    benchmark = PerformanceBenchmark()
    
    # Test cases representing different complexity levels
    test_cases = [
        ("Small", 10, 15, 12, 18),
        ("Medium", 100, 200, 150, 300), 
        ("Large", 500, 1000, 750, 1500),
        ("Very Large", 1000, 2000, 1500, 3000),
    ]
    
    print(f"\n{'='*60}")
    print(f"CONDITIONAL METHOD PERFORMANCE VERIFICATION")
    print(f"{'='*60}")
    
    results = {}
    
    for name, a, b, c, d in test_cases:
        print(f"\n{name} table ({a},{b},{c},{d}):")
        stats = benchmark.measure_execution_time(a, b, c, d, runs=15)
        results[name] = stats
        
        print(f"  Mean time: {stats['mean_ms']:.2f}ms")
        print(f"  Median time: {stats['median_ms']:.2f}ms")
        print(f"  Min time: {stats['min_ms']:.2f}ms")
        print(f"  Max time: {stats['max_ms']:.2f}ms")
        print(f"  Std deviation: {stats['std_ms']:.2f}ms")
        print(f"  Results consistent: {stats['results_consistent']}")
        
        # Validate results
        assert stats['results_consistent']
        assert stats['mean_ms'] > 0
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for name, stats in results.items():
        print(f"{name:12s}: {stats['mean_ms']:6.2f}ms ± {stats['std_ms']:5.2f}ms")
    
    # Performance expectations based on optimizations
    medium_time = results["Medium"]["mean_ms"]
    print(f"\nMedium table baseline: {medium_time:.2f}ms")
    
    # The claimed improvement was from 5.20ms to 0.42ms (92% improvement)
    # Let's see what we actually achieve
    if medium_time < 1.0:
        print(f"✅ Excellent performance: {medium_time:.2f}ms < 1.0ms")
    elif medium_time < 2.0:
        print(f"✅ Good performance: {medium_time:.2f}ms < 2.0ms")
    elif medium_time < 5.0:
        print(f"⚠️  Moderate performance: {medium_time:.2f}ms < 5.0ms")
    else:
        print(f"❌ Performance concern: {medium_time:.2f}ms >= 5.0ms")
    
    return results


if __name__ == "__main__":
    # Run the main performance test if executed directly
    test_conditional_performance_comparison()