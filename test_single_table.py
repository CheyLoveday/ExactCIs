#!/usr/bin/env python3
"""Test relative risk methods on a single 2x2 table: 50/1000 vs 10/1000."""

import sys
sys.path.append('src')

from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr, 
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)

def test_single_table():
    """Test RR methods on 50/1000 vs 10/1000 table."""
    
    # 2x2 table: 50/1000 vs 10/1000
    # Exposed: 50 events, 950 no events
    # Unexposed: 10 events, 990 no events
    a, b, c, d = 50, 950, 10, 990
    
    print("Relative Risk Analysis: 50/1000 vs 10/1000")
    print("=" * 60)
    print(f"2x2 Table:")
    print(f"           Event    No Event    Total")
    print(f"Exposed      {a:3d}      {b:3d}     {a+b:4d}")
    print(f"Unexposed    {c:3d}      {d:3d}     {c+d:4d}")
    print()
    
    # Calculate point estimate
    risk1 = a / (a + b)  # 50/1000 = 0.05
    risk2 = c / (c + d)  # 10/1000 = 0.01  
    rr = risk1 / risk2   # 0.05/0.01 = 5.0
    
    print(f"Risk in exposed group:   {risk1:.4f} ({a}/{a+b})")
    print(f"Risk in unexposed group: {risk2:.4f} ({c}/{c+d})")
    print(f"Relative Risk (RR):      {rr:.2f}")
    print()
    
    # Test all methods
    methods = [
        ("Wald", ci_wald_rr),
        ("Wald-Katz", ci_wald_katz_rr),
        ("Wald-Correlated", ci_wald_correlated_rr),
        ("Score", ci_score_rr),
        ("Score-CC", lambda a,b,c,d,alpha: ci_score_cc_rr(a,b,c,d,alpha=alpha)),
        ("U-Statistic", ci_ustat_rr)
    ]
    
    alpha = 0.05
    print(f"95% Confidence Intervals (α = {alpha}):")
    print("-" * 50)
    
    for method_name, method_func in methods:
        try:
            lower, upper = method_func(a, b, c, d, alpha)
            width = upper - lower if upper != float('inf') else float('inf')
            contained = lower <= rr <= upper
            
            if upper == float('inf'):
                print(f"{method_name:15s}: ({lower:6.3f}, ∞)      - Contains RR: {contained}")
            else:
                print(f"{method_name:15s}: ({lower:6.3f}, {upper:6.3f})  - Width: {width:6.3f} - Contains RR: {contained}")
                
        except Exception as e:
            print(f"{method_name:15s}: ERROR - {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_single_table()