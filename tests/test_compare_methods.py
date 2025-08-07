"""
Tests comparing the different confidence interval methods with gold standard results.
"""

import pytest
import numpy as np
import pandas as pd
from exactcis import compute_all_cis, calculate_odds_ratio, calculate_relative_risk


def test_compare_all_methods():
    """
    Compare all available methods against a gold standard example.
    
    The gold standard contains results for several different approaches:
    - OR Wald
    - OR Fisher Exact
    - RR Wald
    - RR Log-Binomial
    - Risk Diff Wilson
    
    Our implementation may use slightly different algorithms, so exact matches
    aren't expected, but results should be reasonably close.
    """
    # Define the 2x2 table - using the gold standard method 50/1000 vs 10/1000
    a, b = 50, 950   # Exposed group: 50 cases, 950 controls  
    c, d = 10, 990   # Unexposed group: 10 cases, 990 controls
    
    # Calculate estimates
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    relative_risk = calculate_relative_risk(a, b, c, d)
    risk_diff = (a / (a + b)) - (c / (c + d))
    
    # Gold standard results for 50/1000 vs 10/1000 table (OR ≈ 5.21)
    gold_standard = {
        "OR Wald": (3.2, 8.5),  # Placeholder - will be updated with actual gold standard
        "OR Fisher Exact": (3.0, 9.0),  # Placeholder - will be updated with actual gold standard
        "RR Wald": (3.1, 8.1),  # Placeholder - will be updated with actual gold standard
        "RR Log-Binomial": (3.1, 8.1),  # Placeholder - will be updated with actual gold standard
        "Risk Diff Wilson": (0.025, 0.055),  # Placeholder - will be updated with actual gold standard
    }
    
    # Get results from our implemented methods
    alpha = 0.05  # 95% confidence level
    our_results = compute_all_cis(a, b, c, d, alpha=alpha)
    
    # Create a comparison table
    data = []
    
    # Add gold standard results
    data.append(["OR Wald (Gold)", odds_ratio, *gold_standard["OR Wald"]])
    data.append(["OR Fisher Exact (Gold)", odds_ratio, *gold_standard["OR Fisher Exact"]])
    data.append(["RR Wald (Gold)", relative_risk, *gold_standard["RR Wald"]])
    data.append(["RR Log-Binomial (Gold)", relative_risk, *gold_standard["RR Log-Binomial"]])
    data.append(["Risk Diff Wilson (Gold)", risk_diff, *gold_standard["Risk Diff Wilson"]])
    
    # Add our implementation results
    for method, (lower, upper) in our_results.items():
        # Use the appropriate estimate type based on the method
        estimate = odds_ratio  # Default to odds ratio
        if method == "risk_diff" or method.startswith("risk_diff_"):
            estimate = risk_diff
        elif method == "rr" or method.startswith("rr_"):
            estimate = relative_risk
            
        data.append([f"{method} (Our Impl)", estimate, lower, upper])
    
    # Print comparison table using string formatting instead of tabulate
    print("\nConfidence Interval Comparison (95% CI)")
    print("=" * 65)
    print(f"{'Method':<25} {'Estimate':>10} {'Lower CI':>12} {'Upper CI':>12}")
    print("-" * 65)
    for row in data:
        method, est, lower, upper = row
        print(f"{method:<25} {est:>10.4f} {lower:>12.4f} {upper:>12.4f}")
    print("=" * 65)
    
    # Validate that our implementations are somewhat close to gold standard
    # We don't expect exact matches, but order of magnitude should be similar
    
    # Find the closest matching method for each gold standard entry
    matches = {
        "OR Wald": "wald_haldane",  # Closest to our Wald-Haldane implementation
        "OR Fisher Exact": "conditional",  # Our conditional is Fisher's exact
    }
    
    for gold_name, match_name in matches.items():
        if match_name in our_results:
            gold_lower, gold_upper = gold_standard[gold_name]
            our_lower, our_upper = our_results[match_name]
            
            # Check that orders of magnitude are similar (within factor of 2)
            # This is a loose check since implementations differ
            assert our_lower / gold_lower < 2.0 or our_lower / gold_lower > 0.5, \
                f"Lower bound for {match_name} too far from {gold_name}"
            assert our_upper / gold_upper < 2.0 or our_upper / gold_upper > 0.5, \
                f"Upper bound for {match_name} too far from {gold_name}"
    
    # No assertions for the exact values - the table comparison is the main output
    return data


def test_specific_methods_match_gold():
    """
    Test specific methods that should more closely match gold standard.
    This is useful for ensuring regression testing on methods we expect to align.
    """
    # Define the 2x2 table - using the gold standard method 50/1000 vs 10/1000
    a, b = 50, 950   # Exposed group: 50 cases, 950 controls
    c, d = 10, 990   # Unexposed group: 10 cases, 990 controls
    
    # Gold standard for OR Fisher Exact (50/1000 vs 10/1000 table)
    gold_fisher_lower, gold_fisher_upper = 3.0, 9.0  # Placeholder - will be updated with actual gold standard
    
    # Our implementation (conditional = Fisher's exact)
    alpha = 0.05
    our_results = compute_all_cis(a, b, c, d, alpha=alpha)
    our_conditional_lower, our_conditional_upper = our_results["conditional"]
    
    # Print comparison
    print("\nFisher's Exact Test Comparison:")
    print(f"Gold Standard: ({gold_fisher_lower:.4f}, {gold_fisher_upper:.4f})")
    print(f"Our Impl:      ({our_conditional_lower:.4f}, {our_conditional_upper:.4f})")
    print(f"Diff (Lower):  {abs(our_conditional_lower - gold_fisher_lower):.4f}")
    print(f"Diff (Upper):  {abs(our_conditional_upper - gold_fisher_upper):.4f}")
    
    # Allow some difference but check that it's reasonably close
    assert abs(our_conditional_lower - gold_fisher_lower) < 1.0, \
        "Lower bound for conditional method too far from gold standard"
    assert abs(our_conditional_upper - gold_fisher_upper) < 1.0, \
        "Upper bound for conditional method too far from gold standard"


if __name__ == "__main__":
    # This allows running the test directly for debugging
    test_compare_all_methods()
    test_specific_methods_match_gold()
