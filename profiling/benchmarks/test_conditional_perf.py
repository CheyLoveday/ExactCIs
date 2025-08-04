#!/usr/bin/env python3
"""
Test script to measure conditional method performance across multiple runs.
"""

from exactcis.methods.conditional import exact_ci_conditional
import time
import logging

# Set logging level to reduce output
logging.basicConfig(level=logging.WARNING)

def main():
    """Run multiple executions and measure performance."""
    print("Testing conditional method performance...")
    print("Running 5 executions of exact_ci_conditional(100, 200, 150, 300)")
    
    times = []
    results = []
    
    # Warm-up run
    print("\nWarm-up run...")
    _ = exact_ci_conditional(100, 200, 150, 300)
    
    # Timed runs
    for i in range(5):
        print(f"\nRun {i+1}:")
        start = time.perf_counter()
        result = exact_ci_conditional(100, 200, 150, 300)
        end = time.perf_counter()
        
        execution_time = (end - start) * 1000  # Convert to ms
        times.append(execution_time)
        results.append(result)
        
        print(f"  Time: {execution_time:.2f}ms")
        print(f"  Result: {result}")
    
    # Summary
    print("\nSummary:")
    print(f"  Average time: {sum(times)/len(times):.2f}ms")
    print(f"  Min time: {min(times):.2f}ms")
    print(f"  Max time: {max(times):.2f}ms")
    
    # Check if all results are consistent
    all_same = all(r == results[0] for r in results)
    print(f"  Results consistent: {'Yes' if all_same else 'No'}")

if __name__ == "__main__":
    main()