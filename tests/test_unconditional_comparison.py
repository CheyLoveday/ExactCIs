#!/usr/bin/env python3
"""
Test script for unconditional exact confidence interval method.

This script tests the unconditional method for small counts and compares
the results with the mid-p method to validate that the implementation
works correctly and provides wider confidence intervals as expected.
"""

import sys
import os
import numpy as np
from tabulate import tabulate

# Add parent directory to path to import exactcis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.exactcis.methods.midp import exact_ci_midp
from src.exactcis.methods.unconditional import exact_ci_unconditional


def compare_methods(a, b, c, d, alpha=0.05, label=""):
    """Compare unconditional and mid-p methods for a 2x2 table."""
    
    print(f"\nTable {label}: [{a}, {b}, {c}, {d}]")
    print(f"Margins: n1={a+b}, n2={c+d}, m1={a+c}")
    
    # Calculate odds ratio
    if b == 0 or c == 0:
        odds_ratio = float('inf')
    else:
        odds_ratio = (a * d) / (b * c)
    print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Calculate confidence intervals
    midp_ci = exact_ci_midp(a, b, c, d, alpha=alpha)
    uc_ci = exact_ci_unconditional(a, b, c, d, alpha=alpha)
    
    # Calculate interval widths (in log space for better comparison)
    if midp_ci[0] <= 0 or midp_ci[1] == float('inf'):
        midp_width = float('inf')
    else:
        midp_width = np.log(midp_ci[1]) - np.log(midp_ci[0])
        
    if uc_ci[0] <= 0 or uc_ci[1] == float('inf'):
        uc_width = float('inf')
    else:
        uc_width = np.log(uc_ci[1]) - np.log(uc_ci[0])
    
    # Determine if unconditional is wider
    if midp_width == float('inf') and uc_width == float('inf'):
        width_comp = "Both infinite"
    elif uc_width > midp_width:
        width_comp = "UC wider"
    elif uc_width < midp_width:
        width_comp = "Mid-P wider (!)"
    else:
        width_comp = "Equal width"
    
    # Format results
    results = [
        ["Method", "Lower Bound", "Upper Bound", "Width (log scale)"],
        ["Mid-P", f"{midp_ci[0]:.4f}", f"{midp_ci[1]:.4f}", f"{midp_width:.4f}"],
        ["Unconditional", f"{uc_ci[0]:.4f}", f"{uc_ci[1]:.4f}", f"{uc_width:.4f}"]
    ]
    
    print(tabulate(results, headers="firstrow", tablefmt="grid"))
    print(f"Comparison: {width_comp}")
    
    return {
        "midp": midp_ci, 
        "unconditional": uc_ci,
        "wider": width_comp
    }


def main():
    """Run tests on unconditional method."""
    print("Testing Unconditional vs Mid-P CI Methods")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        # Small counts
        (1, 9, 1, 9, "small 1"),
        (2, 8, 1, 9, "small 2"),
        (0, 10, 1, 9, "zero cell"),
        (5, 5, 5, 5, "balanced"),
        
        # Target 2x2 table from the prompt
        (12, 5, 8, 10, "target table"),
        
        # Medium counts
        (20, 30, 15, 35, "medium"),
        
        # Large counts
        (50, 950, 10, 990, "large")
    ]
    
    results = []
    for a, b, c, d, label in test_cases:
        result = compare_methods(a, b, c, d, label=label)
        results.append((a, b, c, d, label, result))
    
    # Summary
    print("\nSummary of Results")
    print("=" * 50)
    summary = [
        ["Table", "Mid-P CI", "Unconditional CI", "Comparison"]
    ]
    
    for a, b, c, d, label, result in results:
        midp_ci = result["midp"]
        uc_ci = result["unconditional"]
        summary.append([
            f"{label} [{a},{b},{c},{d}]",
            f"({midp_ci[0]:.4f}, {midp_ci[1]:.4f})",
            f"({uc_ci[0]:.4f}, {uc_ci[1]:.4f})",
            result["wider"]
        ])
    
    print(tabulate(summary, headers="firstrow", tablefmt="grid"))


if __name__ == "__main__":
    main()
