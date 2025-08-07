"""
Simple script to compare ExactCIs methods with gold standard results.
This compares the confidence intervals calculated by our library with 
provided gold standard values.
"""

import os
import sys
import numpy as np

# Add the package to the path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exactcis import (
    compute_all_cis,
    calculate_odds_ratio,
    calculate_relative_risk
)

def print_header():
    """Print the comparison table header."""
    print("\nConfidence Interval Comparison (95% CI)")
    print("=" * 70)
    print(f"{'Method':<25} {'Estimate':>10} {'Lower CI':>12} {'Upper CI':>12} {'Notes':>10}")
    print("-" * 70)


def print_row(method, estimate, lower, upper, note=""):
    """Print a row in the comparison table."""
    print(f"{method:<25} {estimate:>10.4f} {lower:>12.4f} {upper:>12.4f} {note:>10}")


def main():
    """Run the comparison between our methods and gold standards."""
    # Define the 2x2 table with the correct values (50 in 1000 vs 10 in 1000)
    a, b = 50, 950  # Exposed group: 50 cases, 950 controls (50 in 1000)
    c, d = 10, 990  # Unexposed group: 10 cases, 990 controls (10 in 1000)
    
    # Calculate estimates
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    relative_risk = calculate_relative_risk(a, b, c, d)
    risk_diff = (a / (a + b)) - (c / (c + d))
    
    print(f"Table: [ {a}, {b} ; {c}, {d} ]")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"Relative Risk: {relative_risk:.4f}")
    print(f"Risk Difference: {risk_diff:.4f}")
    
    # Gold standard results (provided by user)
    gold_standard = {
        "OR Wald": (2.6272, 10.3340),
        "OR Fisher Exact": (2.4533, 11.0665),
        "RR Wald": (2.5502, 9.8032),
        "RR Log-Binomial": (2.5502, 9.8032),
        "Risk Diff Wilson": (0.0252, 0.0548),
    }
    
    # Get results from our implemented methods
    alpha = 0.05  # 95% confidence level
    our_results = compute_all_cis(a, b, c, d, alpha=alpha)
    
    # Print the comparison table
    print_header()
    
    # Print gold standard results
    print_row("OR Wald (Gold)", odds_ratio, *gold_standard["OR Wald"], "Gold Std")
    print_row("OR Fisher Exact (Gold)", odds_ratio, *gold_standard["OR Fisher Exact"], "Gold Std")
    print_row("RR Wald (Gold)", relative_risk, *gold_standard["RR Wald"], "Gold Std")
    print_row("RR Log-Binomial (Gold)", relative_risk, *gold_standard["RR Log-Binomial"], "Gold Std")
    print_row("Risk Diff Wilson (Gold)", risk_diff, *gold_standard["Risk Diff Wilson"], "Gold Std")
    
    print("-" * 70)
    
    # Add our implementation results
    for method, (lower, upper) in our_results.items():
        # Use the appropriate estimate type based on the method
        estimate = odds_ratio  # Default to odds ratio
        if method == "risk_diff" or method.startswith("risk_diff_"):
            estimate = risk_diff
        elif method == "rr" or method.startswith("rr_"):
            estimate = relative_risk
            
        # Add a note about which gold standard this might correspond to
        note = ""
        if method == "conditional":
            note = "≈ Fisher"
        elif method == "wald_haldane":
            note = "≈ OR Wald"
            
        print_row(f"{method}", estimate, lower, upper, note)

    print("=" * 70)


if __name__ == "__main__":
    main()
