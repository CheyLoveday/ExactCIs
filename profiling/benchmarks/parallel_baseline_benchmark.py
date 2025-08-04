#!/usr/bin/env python3
"""
Baseline benchmark for parallel batch processing before optimization.
"""

import time
import logging
import random
from exactcis.methods.blaker import exact_ci_blaker_batch
from exactcis.methods.midp import exact_ci_midp_batch

def benchmark_batch_processing():
    """Benchmark current batch processing performance."""
    
    # Suppress logging noise during benchmark
    logging.getLogger().setLevel(logging.WARNING)
    
    print("Parallel Batch Processing - Baseline Benchmark")
    print("=" * 55)
    
    # Generate test tables with some similar marginals to test cache benefits
    random.seed(42)  # Reproducible results
    
    # Create scenarios with different characteristics
    scenarios = {
        "small_homogeneous": [(5, 8, 6, 9) for _ in range(20)],  # Same table repeated
        "small_varied": [(random.randint(1, 10), random.randint(1, 10), 
                         random.randint(1, 10), random.randint(1, 10)) for _ in range(20)],
        "medium_similar": [(10+i, 15+i, 12+i, 18+i) for i in range(50)],  # Similar marginals
        "medium_varied": [(random.randint(5, 25), random.randint(5, 25),
                          random.randint(5, 25), random.randint(5, 25)) for _ in range(50)],
    }
    
    results = {}
    
    for scenario_name, tables in scenarios.items():
        print(f"\n{scenario_name.upper()} ({len(tables)} tables):")
        print("-" * 40)
        
        # Test Blaker method
        try:
            start_time = time.perf_counter()
            blaker_results = exact_ci_blaker_batch(tables, alpha=0.05, max_workers=4)
            end_time = time.perf_counter()
            
            blaker_time = (end_time - start_time) * 1000
            print(f"  Blaker batch:  {blaker_time:.1f}ms ({blaker_time/len(tables):.2f}ms per table)")
            
            # Verify we got results
            success_count = sum(1 for result in blaker_results if result != (0.0, float('inf')))
            print(f"    Success rate: {success_count}/{len(tables)} ({100*success_count/len(tables):.1f}%)")
            
        except Exception as e:
            print(f"  Blaker batch:  FAILED - {e}")
            blaker_time = None
        
        # Test Mid-P method  
        try:
            start_time = time.perf_counter()
            midp_results = exact_ci_midp_batch(tables, alpha=0.05, max_workers=4)
            end_time = time.perf_counter()
            
            midp_time = (end_time - start_time) * 1000
            print(f"  Mid-P batch:   {midp_time:.1f}ms ({midp_time/len(tables):.2f}ms per table)")
            
            # Verify we got results
            success_count = sum(1 for result in midp_results if result != (0.0, float('inf')))
            print(f"    Success rate: {success_count}/{len(tables)} ({100*success_count/len(tables):.1f}%)")
            
        except Exception as e:
            print(f"  Mid-P batch:   FAILED - {e}")
            midp_time = None
            
        results[scenario_name] = {'blaker': blaker_time, 'midp': midp_time, 'table_count': len(tables)}
    
    # Summary
    print(f"\n{'='*55}")
    print("BASELINE RESULTS SUMMARY:")
    print(f"{'='*55}")
    
    total_tables = sum(r['table_count'] for r in results.values())
    total_blaker_time = sum(r['blaker'] for r in results.values() if r['blaker'] is not None)
    total_midp_time = sum(r['midp'] for r in results.values() if r['midp'] is not None)
    
    print(f"Total tables processed: {total_tables}")
    print(f"Total Blaker time:      {total_blaker_time:.1f}ms")
    print(f"Total Mid-P time:       {total_midp_time:.1f}ms")
    print(f"Average per table:")
    print(f"  Blaker: {total_blaker_time/total_tables:.2f}ms")
    print(f"  Mid-P:  {total_midp_time/total_tables:.2f}ms")
    
    return results

if __name__ == "__main__":
    results = benchmark_batch_processing()