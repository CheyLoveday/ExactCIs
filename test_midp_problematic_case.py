#!/usr/bin/env python
"""
Test script for the problematic mid-p case (20,80,40,60).
This script verifies that the improved mid-p calculations produce valid results.
"""

import sys
import os
import time

# Add the project root to the path so we can import exactcis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

def main():
    """Test the problematic case (20,80,40,60)."""
    a, b, c, d = 20, 80, 40, 60
    
    print(f"\nTesting problematic case: a={a}, b={b}, c={c}, d={d}")
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    print(f"Odds ratio: {odds_ratio:.6f}")
    
    # Calculate CIs using different methods
    methods = {
        "wald": ci_wald_haldane,
        "conditional": exact_ci_conditional,
        "midp": exact_ci_midp,
        "unconditional": exact_ci_unconditional
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] Starting {method_name} CI calculation...")
        try:
            start_time = time.time()
            
            # Call the function
            lower, upper = method_func(a, b, c, d)
            
            elapsed = time.time() - start_time
            ci_width = upper - lower
            
            results[method_name] = {
                "lower": lower,
                "upper": upper,
                "width": ci_width,
                "contains_or": lower <= odds_ratio <= upper,
                "valid": lower <= upper
            }
            
            print(f"[{time.strftime('%H:%M:%S')}] Completed {method_name} CI calculation in {elapsed:.2f} seconds:")
            print(f"  CI: ({lower:.6f}, {upper:.6f})")
            print(f"  Width: {ci_width:.6f}")
            print(f"  Contains OR: {results[method_name]['contains_or']}")
            print(f"  Valid: {results[method_name]['valid']}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error in {method_name} CI calculation: {str(e)}")
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"Odds ratio: {odds_ratio:.6f}")
    print("\nMethod      | Lower      | Upper      | Width      | Contains OR | Valid")
    print("-" * 80)
    
    for method_name, result in results.items():
        if "lower" in result:
            print(f"{method_name:11} | {result['lower']:.6f} | {result['upper']:.6f} | {result['width']:.6f} | {result['contains_or']!s:11} | {result['valid']!s}")
    
if __name__ == "__main__":
    main()