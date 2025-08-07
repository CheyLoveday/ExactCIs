#!/usr/bin/env python3
"""Test script to verify the score-based relative risk fixes."""

import sys
sys.path.append('src')

from exactcis.methods.relative_risk import (
    ci_score_rr,
    ci_score_cc_rr,
    score_statistic,
    corrected_score_statistic
)

def test_score_methods():
    """Test the corrected score methods."""
    print("Testing Score-Based Relative Risk Fixes")
    print("=" * 50)
    
    # Test case from the problem description: RR=5.0 should be contained
    a, b, c, d = 20, 5, 4, 20  # RR = (20/25)/(4/24) = 0.8/0.167 ≈ 4.8
    
    print(f"\nTest case: a={a}, b={b}, c={c}, d={d}")
    
    # Calculate point estimate
    rr = (a/(a+b)) / (c/(c+d))
    print(f"Point estimate RR: {rr:.3f}")
    
    # Test score statistic at true RR (should be close to 0)
    score_at_rr = score_statistic(a, b, c, d, rr)
    print(f"Score statistic at RR={rr:.3f}: {score_at_rr:.6f}")
    
    # Test corrected score statistic at true RR
    corrected_score_at_rr = corrected_score_statistic(a, b, c, d, rr)
    print(f"Corrected score statistic at RR={rr:.3f}: {corrected_score_at_rr:.6f}")
    
    # Calculate confidence intervals
    try:
        lower_score, upper_score = ci_score_rr(a, b, c, d, alpha=0.05)
        print(f"Score CI: ({lower_score:.3f}, {upper_score:.3f})")
        
        # Check if RR is contained
        contained_score = lower_score <= rr <= upper_score
        print(f"RR contained in score CI: {contained_score}")
        
    except Exception as e:
        print(f"Score CI failed: {e}")
        
    try:
        lower_cc, upper_cc = ci_score_cc_rr(a, b, c, d, alpha=0.05)
        print(f"Score CC CI: ({lower_cc:.3f}, {upper_cc:.3f})")
        
        # Check if RR is contained
        contained_cc = lower_cc <= rr <= upper_cc
        print(f"RR contained in score CC CI: {contained_cc}")
        
    except Exception as e:
        print(f"Score CC CI failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Additional test cases
    test_cases = [
        (15, 5, 10, 10),  # Standard case
        (5, 5, 2, 8),     # Smaller numbers
        (1, 9, 1, 9),     # Very small counts
    ]
    
    for i, (a, b, c, d) in enumerate(test_cases, 1):
        print(f"\nTest case {i}: a={a}, b={b}, c={c}, d={d}")
        rr = (a/(a+b)) / (c/(c+d))
        print(f"Point estimate RR: {rr:.3f}")
        
        try:
            lower, upper = ci_score_rr(a, b, c, d)
            contained = lower <= rr <= upper
            print(f"Score CI: ({lower:.3f}, {upper:.3f}) - Contains RR: {contained}")
        except Exception as e:
            print(f"Score CI failed: {e}")
            
        try:
            lower, upper = ci_score_cc_rr(a, b, c, d)
            contained = lower <= rr <= upper
            print(f"Score CC CI: ({lower:.3f}, {upper:.3f}) - Contains RR: {contained}")
        except Exception as e:
            print(f"Score CC CI failed: {e}")

if __name__ == "__main__":
    test_score_methods()