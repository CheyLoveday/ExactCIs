"""
Test script to verify performance improvements in the Blaker method.

This script compares the performance of the optimized Blaker method with
large and very large tables.
"""

import time
import numpy as np
from exactcis.methods.blaker import exact_ci_blaker

def time_execution(func, *args, **kwargs):
    """Time the execution of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, (end_time - start_time) * 1000  # Convert to milliseconds

def test_large_table():
    """Test performance with a large table (50,75,60,90)."""
    a, b, c, d = 50, 75, 60, 90
    print(f"\nTesting large table: ({a},{b},{c},{d})")
    
    # Run multiple times to get average performance
    times = []
    for i in range(5):
        _, execution_time = time_execution(exact_ci_blaker, a, b, c, d)
        times.append(execution_time)
        print(f"Run {i+1}: {execution_time:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"Average execution time: {avg_time:.2f} ms")
    return avg_time

def test_very_large_table():
    """Test performance with a very large table (100,200,150,300)."""
    a, b, c, d = 100, 200, 150, 300
    print(f"\nTesting very large table: ({a},{b},{c},{d})")
    
    # Run multiple times to get average performance
    times = []
    for i in range(3):  # Fewer runs for very large table
        _, execution_time = time_execution(exact_ci_blaker, a, b, c, d)
        times.append(execution_time)
        print(f"Run {i+1}: {execution_time:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"Average execution time: {avg_time:.2f} ms")
    return avg_time

def test_medium_table():
    """Test performance with a medium table (20,30,25,35)."""
    a, b, c, d = 20, 30, 25, 35
    print(f"\nTesting medium table: ({a},{b},{c},{d})")
    
    # Run multiple times to get average performance
    times = []
    for i in range(10):  # More runs for medium table
        _, execution_time = time_execution(exact_ci_blaker, a, b, c, d)
        times.append(execution_time)
        print(f"Run {i+1}: {execution_time:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"Average execution time: {avg_time:.2f} ms")
    return avg_time

def test_small_table():
    """Test performance with a small table (5,10,8,12)."""
    a, b, c, d = 5, 10, 8, 12
    print(f"\nTesting small table: ({a},{b},{c},{d})")
    
    # Run multiple times to get average performance
    times = []
    for i in range(20):  # More runs for small table
        _, execution_time = time_execution(exact_ci_blaker, a, b, c, d)
        times.append(execution_time)
        print(f"Run {i+1}: {execution_time:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"Average execution time: {avg_time:.2f} ms")
    return avg_time

def main():
    """Run all performance tests."""
    print("Testing Blaker method performance after optimizations")
    print("====================================================")
    
    # Test with different table sizes
    small_time = test_small_table()
    medium_time = test_medium_table()
    large_time = test_large_table()
    very_large_time = test_very_large_table()
    
    # Print summary
    print("\nPerformance Summary")
    print("==================")
    print(f"Small table:      {small_time:.2f} ms")
    print(f"Medium table:     {medium_time:.2f} ms")
    print(f"Large table:      {large_time:.2f} ms")
    print(f"Very large table: {very_large_time:.2f} ms")
    
    # Compare with expected performance from issue description
    print("\nComparison with Expected Performance")
    print("==================================")
    print("Before Optimization (Original):")
    print("- Very large table: 270 ms")
    print("- Large table: 51 ms")
    print("\nAfter Optimization (Current):")
    print(f"- Very large table: {very_large_time:.2f} ms")
    print(f"- Large table: {large_time:.2f} ms")
    
    # Calculate improvement
    large_improvement = (51 - large_time) / 51 * 100
    very_large_improvement = (270 - very_large_time) / 270 * 100
    
    print("\nImprovement:")
    print(f"- Very large table: {very_large_improvement:.1f}%")
    print(f"- Large table: {large_improvement:.1f}%")
    
    # Check if we met the target
    print("\nTarget Achievement:")
    target_large = 90
    target_very_large = 90
    
    if large_improvement >= target_large:
        print(f"✅ Large table: {large_improvement:.1f}% improvement (Target: {target_large}%)")
    else:
        print(f"❌ Large table: {large_improvement:.1f}% improvement (Target: {target_large}%)")
    
    if very_large_improvement >= target_very_large:
        print(f"✅ Very large table: {very_large_improvement:.1f}% improvement (Target: {target_very_large}%)")
    else:
        print(f"❌ Very large table: {very_large_improvement:.1f}% improvement (Target: {target_very_large}%)")

if __name__ == "__main__":
    main()