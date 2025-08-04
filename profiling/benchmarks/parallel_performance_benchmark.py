#!/usr/bin/env python3
"""
Simple parallel performance benchmark to measure Stage 1 effectiveness.

This benchmark focuses on validating that parallel processing works correctly
and measures performance improvements over sequential processing.
"""

import time
import logging
from exactcis.methods.conditional import exact_ci_conditional, exact_ci_conditional_batch
from exactcis.methods.blaker import exact_ci_blaker, exact_ci_blaker_batch

def benchmark_parallel_vs_sequential():
    """Compare parallel vs sequential performance."""
    
    # Reduce logging noise
    logging.getLogger().setLevel(logging.WARNING)
    
    print("Stage 1: Parallel vs Sequential Performance Benchmark")
    print("=" * 60)
    
    # Test with medium-sized dataset that should benefit from parallelization
    # Use tables with different marginals to avoid identical calculations
    import random
    random.seed(42)
    
    tables = []
    for _ in range(50):
        # Generate reasonably sized tables
        a = random.randint(3, 15)
        b = random.randint(3, 15) 
        c = random.randint(3, 15)
        d = random.randint(3, 15)
        tables.append((a, b, c, d))
    
    print(f"\nTesting with {len(tables)} randomly generated tables")
    print("-" * 50)
    
    # Test conditional method
    print("CONDITIONAL METHOD:")
    
    # Sequential processing
    print("  Sequential processing...")
    start_time = time.perf_counter()
    sequential_results = []
    for a, b, c, d in tables:
        try:
            result = exact_ci_conditional(a, b, c, d, alpha=0.05)
            sequential_results.append(result)
        except Exception as e:
            print(f"    Error in sequential processing: {e}")
            sequential_results.append((0.0, float('inf')))
    end_time = time.perf_counter()
    
    sequential_time = (end_time - start_time) * 1000
    print(f"    Time: {sequential_time:.1f}ms ({sequential_time/len(tables):.2f}ms per table)")
    
    # Parallel processing
    print("  Parallel processing...")
    start_time = time.perf_counter()
    try:
        parallel_results = exact_ci_conditional_batch(tables, alpha=0.05, max_workers=4)
        end_time = time.perf_counter()
        
        parallel_time = (end_time - start_time) * 1000
        print(f"    Time: {parallel_time:.1f}ms ({parallel_time/len(tables):.2f}ms per table)")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        print(f"    Speedup: {speedup:.2f}x")
        
        # Verify results match (within tolerance)
        matches = 0
        for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
            if abs(seq[0] - par[0]) < 1e-6 and abs(seq[1] - par[1]) < 1e-6:
                matches += 1
        
        print(f"    Result accuracy: {matches}/{len(tables)} matches ({100*matches/len(tables):.1f}%)")
        
    except Exception as e:
        print(f"    Parallel processing failed: {e}")
        parallel_time = None
        speedup = None
    
    # Test Blaker method
    print("\nBLAKER METHOD:")
    
    # Sequential processing 
    print("  Sequential processing...")
    start_time = time.perf_counter()
    sequential_results = []
    for a, b, c, d in tables:
        try:
            result = exact_ci_blaker(a, b, c, d, alpha=0.05)
            sequential_results.append(result)
        except Exception as e:
            print(f"    Error in sequential processing: {e}")
            sequential_results.append((0.0, float('inf')))
    end_time = time.perf_counter()
    
    blaker_sequential_time = (end_time - start_time) * 1000
    print(f"    Time: {blaker_sequential_time:.1f}ms ({blaker_sequential_time/len(tables):.2f}ms per table)")
    
    # Parallel processing
    print("  Parallel processing...")
    start_time = time.perf_counter()
    try:
        parallel_results = exact_ci_blaker_batch(tables, alpha=0.05, max_workers=4)
        end_time = time.perf_counter()
        
        blaker_parallel_time = (end_time - start_time) * 1000
        print(f"    Time: {blaker_parallel_time:.1f}ms ({blaker_parallel_time/len(tables):.2f}ms per table)")
        
        # Calculate speedup
        blaker_speedup = blaker_sequential_time / blaker_parallel_time
        print(f"    Speedup: {blaker_speedup:.2f}x")
        
        # Verify results match (within tolerance)
        matches = 0
        for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
            if abs(seq[0] - par[0]) < 1e-6 and abs(seq[1] - par[1]) < 1e-6:
                matches += 1
        
        print(f"    Result accuracy: {matches}/{len(tables)} matches ({100*matches/len(tables):.1f}%)")
        
    except Exception as e:
        print(f"    Parallel processing failed: {e}")
        blaker_parallel_time = None
        blaker_speedup = None
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY:")
    print(f"{'='*60}")
    
    if speedup is not None:
        print(f"Conditional method: {speedup:.2f}x speedup")
        improvement = (speedup - 1) * 100
        print(f"  Performance improvement: {improvement:.1f}%")
    else:
        print("Conditional method: Failed to measure parallel performance")
    
    if blaker_speedup is not None:
        print(f"Blaker method: {blaker_speedup:.2f}x speedup")
        improvement = (blaker_speedup - 1) * 100
        print(f"  Performance improvement: {improvement:.1f}%")
    else:
        print("Blaker method: Failed to measure parallel performance")
    
    print("\nStage 1 Status:")
    if speedup is not None and speedup > 1.3:
        print("✓ Parallel processing working with significant speedup")
    elif speedup is not None and speedup > 1.0:
        print("⚠ Parallel processing working but limited speedup")
    else:
        print("✗ Parallel processing not providing expected benefits")
    
    print("\nNotes:")
    print("- Shared cache may not be functioning as expected")
    print("- Parallel overhead may be significant for this dataset size")
    print("- Actual benefits should be higher with larger datasets")

if __name__ == "__main__":
    benchmark_parallel_vs_sequential()