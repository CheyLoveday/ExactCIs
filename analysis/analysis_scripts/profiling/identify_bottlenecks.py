#!/usr/bin/env python3
"""
Targeted bottleneck identification for remaining performance issues
"""

import cProfile
import pstats
import time
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.unconditional import exact_ci_unconditional  
from exactcis.methods.conditional import exact_ci_conditional


def profile_method_details(method_name, method_func, test_cases):
    """Profile a specific method and identify top bottlenecks"""
    
    print(f"\n=== Detailed Profiling: {method_name.upper()} ===")
    
    # Profile the method
    profiler = cProfile.Profile()
    profiler.enable()
    
    total_time = 0
    for case in test_cases:
        start_time = time.time()
        try:
            result = method_func(*case)
            end_time = time.time()
            case_time = end_time - start_time
            total_time += case_time
            print(f"  Case {case}: {case_time:.4f}s -> CI = [{result[0]:.3f}, {result[1]:.3f}]")
        except Exception as e:
            print(f"  Case {case}: ERROR - {str(e)[:50]}...")
    
    profiler.disable()
    
    # Analyze the profile
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    
    print(f"\nTotal time: {total_time:.4f}s")
    print(f"Average per case: {total_time/len(test_cases):.4f}s")
    
    print(f"\n--- TOP 15 BOTTLENECKS ({method_name}) ---")
    stats.print_stats(15)
    
    return total_time, stats


def main():
    """Run targeted profiling to identify remaining bottlenecks"""
    
    # Test cases for analysis
    test_cases = [
        (12, 5, 8, 10),   # Medium table
        (20, 10, 8, 15),  # Larger table
        (8, 12, 15, 5),   # Different ratio
    ]
    
    print("=== TARGETED BOTTLENECK ANALYSIS ===")
    print("Identifying remaining performance issues after initial optimizations")
    
    methods = {
        'conditional': exact_ci_conditional,
        'blaker': exact_ci_blaker, 
        'unconditional': exact_ci_unconditional
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            total_time, stats = profile_method_details(method_name, method_func, test_cases)
            results[method_name] = total_time
        except Exception as e:
            print(f"Error profiling {method_name}: {e}")
            results[method_name] = float('inf')
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Sort by performance
    sorted_methods = sorted(results.items(), key=lambda x: x[1])
    
    for i, (method, total_time) in enumerate(sorted_methods):
        if total_time != float('inf'):
            avg_time = total_time / len(test_cases)
            status = "✅ FAST" if avg_time < 0.05 else ("⚠️ MODERATE" if avg_time < 0.2 else "🔴 SLOW")
            print(f"{i+1}. {method:12}: {total_time:.4f}s total, {avg_time:.4f}s avg {status}")
    
    # Identify bottlenecks needing attention
    slow_methods = [method for method, time in results.items() 
                   if time != float('inf') and time/len(test_cases) > 0.05]
    
    if slow_methods:
        print(f"\n🎯 Methods needing optimization: {', '.join(slow_methods)}")
        print("📊 Focus on functions with highest 'tottime' in profiles above")
    else:
        print("\n✅ All methods performing well (< 0.05s average)")
    
    return results


if __name__ == "__main__":
    main()