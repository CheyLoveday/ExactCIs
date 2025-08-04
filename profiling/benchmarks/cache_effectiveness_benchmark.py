#!/usr/bin/env python3
"""
Cache effectiveness benchmark for Stage 1 parallel optimization.

This benchmark specifically measures the impact of shared inter-process caching
on batch processing performance.
"""

import time
import logging
from exactcis.methods.conditional import exact_ci_conditional_batch
from exactcis.methods.blaker import exact_ci_blaker_batch
from exactcis.utils.shared_cache import get_shared_cache

def benchmark_cache_effectiveness():
    """Benchmark cache effectiveness for shared inter-process caching."""
    
    # Set up logging to see cache statistics
    logging.getLogger().setLevel(logging.INFO)
    
    print("Stage 1: Shared Inter-Process Cache Effectiveness Benchmark")
    print("=" * 65)
    
    # Test scenario: Tables with similar marginals should benefit most from caching
    # as they will have repeated CDF/SF calculations during root finding
    
    # Scenario 1: Identical tables (maximum cache benefit)
    identical_tables = [(8, 12, 10, 15)] * 20
    
    # Scenario 2: Similar marginals (high cache benefit)
    similar_tables = [(8+i, 12+i, 10+i, 15+i) for i in range(0, 20)]
    
    # Scenario 3: Random tables (low cache benefit)
    import random
    random.seed(42)
    random_tables = [(random.randint(5, 15), random.randint(5, 15),
                     random.randint(5, 15), random.randint(5, 15)) for _ in range(20)]
    
    scenarios = {
        "identical_tables": identical_tables,
        "similar_marginals": similar_tables, 
        "random_tables": random_tables
    }
    
    for scenario_name, tables in scenarios.items():
        print(f"\n{scenario_name.upper()} ({len(tables)} tables):")
        print("-" * 50)
        
        # Reset shared cache before each test for fair comparison
        from exactcis.utils.shared_cache import reset_shared_cache
        reset_shared_cache()
        
        # Test conditional method (uses shared cache)
        print("  Testing conditional method with shared cache...")
        start_time = time.perf_counter()
        conditional_results = exact_ci_conditional_batch(tables, alpha=0.05, max_workers=4)
        end_time = time.perf_counter()
        
        conditional_time = (end_time - start_time) * 1000
        cache = get_shared_cache()
        cache_stats = cache.get_stats()
        
        print(f"    Time: {conditional_time:.1f}ms ({conditional_time/len(tables):.2f}ms per table)")
        print(f"    Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"    Total lookups: {cache_stats['total_lookups']}")
        print(f"    Cache sizes: CDF={cache_stats['cache_sizes']['cdf']}, SF={cache_stats['cache_sizes']['sf']}")
        
        # Verify results are valid
        success_count = sum(1 for result in conditional_results if result != (0.0, float('inf')))
        print(f"    Success rate: {success_count}/{len(tables)} ({100*success_count/len(tables):.1f}%)")
        
        # Reset cache and test Blaker method
        reset_shared_cache()
        
        print("  Testing Blaker method with shared cache...")
        start_time = time.perf_counter()
        blaker_results = exact_ci_blaker_batch(tables, alpha=0.05, max_workers=4)
        end_time = time.perf_counter()
        
        blaker_time = (end_time - start_time) * 1000
        cache = get_shared_cache()  
        cache_stats = cache.get_stats()
        
        print(f"    Time: {blaker_time:.1f}ms ({blaker_time/len(tables):.2f}ms per table)")
        print(f"    Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"    Total lookups: {cache_stats['total_lookups']}")
        print(f"    Cache sizes: PMF={cache_stats['cache_sizes']['pmf']}, Support={cache_stats['cache_sizes']['support']}")
        
        # Verify results are valid
        success_count = sum(1 for result in blaker_results if result != (0.0, float('inf')))
        print(f"    Success rate: {success_count}/{len(tables)} ({100*success_count/len(tables):.1f}%)")
    
    print(f"\n{'='*65}")
    print("CACHE EFFECTIVENESS ANALYSIS:")
    print(f"{'='*65}")
    print("Expected results:")
    print("- Identical tables: Very high cache hit rates (>80%)")
    print("- Similar marginals: High cache hit rates (50-80%)")  
    print("- Random tables: Low cache hit rates (<30%)")
    print("\nCache effectiveness validates shared inter-process optimization.")

if __name__ == "__main__":
    benchmark_cache_effectiveness()