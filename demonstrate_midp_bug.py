#!/usr/bin/env python3
"""
Demonstration of Critical MidP Bug

This script demonstrates the critical bug in the MidP method where
it returns mathematically impossible confidence intervals.
"""

from exactcis.methods.midp import exact_ci_midp
from exactcis import compute_all_cis

def demonstrate_critical_bug():
    """
    Demonstrate the critical MidP bug with the user's scenario.
    """
    print("DEMONSTRATION OF CRITICAL MidP BUG")
    print("="*50)
    
    # User's scenario: 20/100 vs 40/100
    # This translates to table: [[20, 80], [40, 60]]
    a, b, c, d = 20, 80, 40, 60
    alpha = 0.05
    
    print(f"Test case: Table [[{a}, {b}], [{c}, {d}]]")
    print(f"True OR = ({a}×{d})/({b}×{c}) = {(a*d)/(b*c):.6f}")
    print()
    
    # Test MidP method
    print("Testing MidP method:")
    try:
        lower, upper = exact_ci_midp(a, b, c, d, alpha=alpha, haldane=True)
        width = upper - lower
        contains_or = lower <= (a*d)/(b*c) <= upper
        
        print(f"  MidP CI: ({lower:.6f}, {upper:.6f})")
        print(f"  Width: {width:.6f}")
        print(f"  Contains true OR: {contains_or}")
        
        # Check for mathematical impossibility
        if lower > upper:
            print("  🚨 CRITICAL BUG: Lower bound > Upper bound!")
            print("     This is mathematically impossible for a confidence interval!")
        
        if width < 0:
            print("  🚨 CRITICAL BUG: Negative width!")
            print("     Confidence intervals cannot have negative width!")
            
        if not contains_or:
            print("  🚨 STATISTICAL FAILURE: CI excludes true parameter!")
            print("     This violates the fundamental property of confidence intervals!")
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print()
    print("Comparing with other methods:")
    print("-" * 30)
    
    try:
        results = compute_all_cis(a, b, c, d, alpha=alpha, grid_size=10)
        
        for method, (lower, upper) in results.items():
            width = upper - lower if upper != float('inf') else float('inf')
            contains_or = lower <= (a*d)/(b*c) <= upper if upper != float('inf') else (a*d)/(b*c) >= lower
            
            status = "✅ VALID" if lower <= upper and contains_or else "❌ INVALID"
            
            print(f"  {method:12s}: ({lower:.6f}, {upper:.6f}) {status}")
            
            if method == "midp" and lower > upper:
                print(f"  {'':14s}   ↑ MATHEMATICAL IMPOSSIBILITY!")
                
    except Exception as e:
        print(f"  ERROR in comparison: {e}")

if __name__ == "__main__":
    demonstrate_critical_bug()