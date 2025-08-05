"""
Tests for the unconditional exact confidence interval method using profile likelihood.

This module tests the profile likelihood implementation of Barnard's unconditional
exact confidence interval method for the odds ratio of a 2x2 contingency table.
"""

import sys
import os
import time
import math
import numpy as np
from typing import Tuple, List, Dict, Optional

# Add the src directory to the Python path if needed
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from exactcis.methods.unconditional_profile import (
    exact_ci_unconditional_profile,
    find_mle_p1,
    _log_likelihood
)
from exactcis.core import calculate_odds_ratio


def test_find_mle_p1():
    """Test the MLE optimization for p1."""
    # Simple test case
    a, b, c, d = 10, 20, 5, 25
    n1 = a + b
    n2 = c + d
    theta = 2.5
    
    # Find MLE
    p1_mle = find_mle_p1(a, c, n1, n2, theta)
    
    # Check that it's within valid range
    assert 0 < p1_mle < 1
    
    # Calculate p2 from p1 and theta
    p2_mle = (theta * p1_mle) / (1 - p1_mle + theta * p1_mle)
    
    # Check that p2 is within valid range
    assert 0 < p2_mle < 1
    
    # Check that the MLE maximizes the likelihood
    log_lik_mle = _log_likelihood(p1_mle, a, c, n1, n2, theta)
    
    # Check nearby values
    for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
        p1_test = max(0.001, min(0.999, p1_mle + delta))
        log_lik_test = _log_likelihood(p1_test, a, c, n1, n2, theta)
        assert log_lik_test <= log_lik_mle + 1e-6, f"MLE check failed: {log_lik_test} > {log_lik_mle} at p1={p1_test}"


def test_large_sample_case():
    """
    Test the unconditional method with a large sample case (50/1000 vs 25/1000).
    
    This case previously returned [0.767, 10.263] with the root-finding approach,
    and [0.297, 0.732] with the grid search approach, neither of which contained
    the odds ratio of 2.05. The profile likelihood approach should produce a
    confidence interval that contains the odds ratio.
    """
    # Define the 2x2 table
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate the odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    
    # Calculate the confidence interval
    start_time = time.time()
    lower, upper = exact_ci_unconditional_profile(a, b, c, d, alpha=0.05)
    elapsed_time = time.time() - start_time
    
    # Print the results
    print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
    print(f"Odds ratio: {odds_ratio:.3f}")
    print(f"Confidence interval: [{lower:.3f}, {upper:.3f}] (width: {upper-lower:.3f})")
    print(f"Computation time: {elapsed_time:.2f} seconds")
    
    # Check if the result is reasonable
    # The interval should be finite and not too wide
    assert lower > 0, "Lower bound should be positive"
    assert upper < float('inf'), "Upper bound should be finite"
    assert upper - lower < 5, "Interval width should be reasonable (< 5)"
    
    # The interval should include the odds ratio
    assert lower <= odds_ratio <= upper, f"Interval [{lower:.3f}, {upper:.3f}] should include the odds ratio {odds_ratio:.3f}"
    
    return lower, upper, elapsed_time


def test_multiple_large_sample_cases():
    """
    Test the unconditional method with multiple large sample cases.
    """
    # Define several 2x2 tables with large sample sizes
    tables = [
        (50, 950, 25, 975),    # 50/1000 vs 25/1000
        (100, 900, 50, 950),   # 100/1000 vs 50/1000
        (200, 800, 100, 900),  # 200/1000 vs 100/1000
        (300, 700, 150, 850),  # 300/1000 vs 150/1000
        (400, 600, 200, 800)   # 400/1000 vs 200/1000
    ]
    
    results = []
    
    for a, b, c, d in tables:
        # Calculate the odds ratio
        odds_ratio = calculate_odds_ratio(a, b, c, d)
        
        # Calculate the confidence interval
        start_time = time.time()
        lower, upper = exact_ci_unconditional_profile(a, b, c, d, alpha=0.05)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
        print(f"Odds ratio: {odds_ratio:.3f}")
        print(f"Confidence interval: [{lower:.3f}, {upper:.3f}] (width: {upper-lower:.3f})")
        print(f"Computation time: {elapsed_time:.2f} seconds")
        
        # Check if the result is reasonable
        assert lower > 0, "Lower bound should be positive"
        assert upper < float('inf'), "Upper bound should be finite"
        assert lower <= odds_ratio <= upper, f"Interval [{lower:.3f}, {upper:.3f}] should include the odds ratio {odds_ratio:.3f}"
        
        results.append((a, b, c, d, lower, upper, elapsed_time))
    
    return results


def test_small_sample_cases():
    """
    Test the unconditional method with small sample cases.
    """
    # Define several 2x2 tables with small sample sizes
    tables = [
        (5, 15, 2, 18),    # 5/20 vs 2/20
        (10, 20, 5, 25),   # 10/30 vs 5/30
        (2, 8, 1, 9),      # 2/10 vs 1/10
        (0, 10, 5, 5),     # 0/10 vs 5/10 (zero cell)
        (1, 9, 0, 10)      # 1/10 vs 0/10 (zero cell)
    ]
    
    results = []
    
    for a, b, c, d in tables:
        # Calculate the odds ratio (handle zero cells)
        try:
            odds_ratio = calculate_odds_ratio(a, b, c, d)
        except:
            odds_ratio = float('inf') if (a * d > 0 and b * c == 0) else 0.0
        
        # Calculate the confidence interval
        start_time = time.time()
        lower, upper = exact_ci_unconditional_profile(a, b, c, d, alpha=0.05)
        elapsed_time = time.time() - start_time
        
        # Print the results
        print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
        print(f"Odds ratio: {odds_ratio if odds_ratio != float('inf') else 'inf'}")
        print(f"Confidence interval: [{lower:.3f}, {upper if upper != float('inf') else 'inf'}] (width: {upper-lower if upper != float('inf') else 'inf'})")
        print(f"Computation time: {elapsed_time:.2f} seconds")
        
        # For non-degenerate cases, check that the odds ratio is in the interval
        if 0 < odds_ratio < float('inf'):
            assert lower <= odds_ratio <= upper, f"Interval [{lower:.3f}, {upper:.3f}] should include the odds ratio {odds_ratio:.3f}"
        
        results.append((a, b, c, d, lower, upper, elapsed_time))
    
    return results


def test_compare_with_grid_search():
    """
    Compare the profile likelihood approach with the grid search approach.
    """
    try:
        from exactcis.methods.unconditional import exact_ci_unconditional
    except ImportError:
        print("Could not import exact_ci_unconditional from unconditional.py, skipping comparison")
        return
    
    # Define several 2x2 tables
    tables = [
        (50, 950, 25, 975),    # 50/1000 vs 25/1000
        (10, 20, 5, 25),       # 10/30 vs 5/30
        (2, 8, 1, 9)           # 2/10 vs 1/10
    ]
    
    results = []
    
    for a, b, c, d in tables:
        # Calculate the odds ratio
        odds_ratio = calculate_odds_ratio(a, b, c, d)
        
        # Calculate the confidence interval using grid search
        start_time = time.time()
        lower_grid, upper_grid = exact_ci_unconditional(a, b, c, d, alpha=0.05)
        elapsed_time_grid = time.time() - start_time
        
        # Calculate the confidence interval using profile likelihood
        start_time = time.time()
        lower_profile, upper_profile = exact_ci_unconditional_profile(a, b, c, d, alpha=0.05)
        elapsed_time_profile = time.time() - start_time
        
        # Print the results
        print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
        print(f"Odds ratio: {odds_ratio:.3f}")
        print(f"Grid search CI: [{lower_grid:.3f}, {upper_grid:.3f}] (width: {upper_grid-lower_grid:.3f}), time: {elapsed_time_grid:.2f}s")
        print(f"Profile likelihood CI: [{lower_profile:.3f}, {upper_profile:.3f}] (width: {upper_profile-lower_profile:.3f}), time: {elapsed_time_profile:.2f}s")
        
        # Check if the odds ratio is in the profile likelihood interval
        assert lower_profile <= odds_ratio <= upper_profile, f"Profile likelihood interval [{lower_profile:.3f}, {upper_profile:.3f}] should include the odds ratio {odds_ratio:.3f}"
        
        results.append((a, b, c, d, odds_ratio, lower_grid, upper_grid, lower_profile, upper_profile))
    
    return results


if __name__ == "__main__":
    print("Testing unconditional method with profile likelihood approach")
    print("===========================================================")
    
    # Test the MLE optimization
    print("\nTesting MLE optimization...")
    test_find_mle_p1()
    print("MLE optimization test passed")
    
    # Test the large sample case
    print("\nTesting large sample case...")
    lower, upper, elapsed_time = test_large_sample_case()
    
    # Test multiple large sample cases
    print("\nTesting multiple large sample cases...")
    results_large = test_multiple_large_sample_cases()
    
    # Test small sample cases
    print("\nTesting small sample cases...")
    results_small = test_small_sample_cases()
    
    # Compare with grid search
    print("\nComparing with grid search...")
    results_compare = test_compare_with_grid_search()
    
    print("\nSummary of results:")
    print("------------------")
    print("Large sample cases:")
    for a, b, c, d, lower, upper, elapsed_time in results_large:
        odds_ratio = calculate_odds_ratio(a, b, c, d)
        print(f"{a}/{a+b} vs {c}/{c+d}: OR={odds_ratio:.3f}, CI=[{lower:.3f}, {upper:.3f}], width={upper-lower:.3f}, time={elapsed_time:.2f}s")
    
    print("\nSmall sample cases:")
    for a, b, c, d, lower, upper, elapsed_time in results_small:
        try:
            odds_ratio = calculate_odds_ratio(a, b, c, d)
            or_str = f"{odds_ratio:.3f}"
        except:
            or_str = "inf" if (a * d > 0 and b * c == 0) else "0.0"
        
        upper_str = f"{upper:.3f}" if upper != float('inf') else "inf"
        width_str = f"{upper-lower:.3f}" if upper != float('inf') else "inf"
        
        print(f"{a}/{a+b} vs {c}/{c+d}: OR={or_str}, CI=[{lower:.3f}, {upper_str}], width={width_str}, time={elapsed_time:.2f}s")
    
    print("\nComparison with grid search:")
    for a, b, c, d, odds_ratio, lower_grid, upper_grid, lower_profile, upper_profile in results_compare:
        print(f"{a}/{a+b} vs {c}/{c+d}: OR={odds_ratio:.3f}")
        print(f"  Grid search: [{lower_grid:.3f}, {upper_grid:.3f}], width={upper_grid-lower_grid:.3f}")
        print(f"  Profile likelihood: [{lower_profile:.3f}, {upper_profile:.3f}], width={upper_profile-lower_profile:.3f}")
        
        # Check if odds ratio is in each interval
        in_grid = lower_grid <= odds_ratio <= upper_grid
        in_profile = lower_profile <= odds_ratio <= upper_profile
        
        print(f"  OR in grid search interval: {in_grid}")
        print(f"  OR in profile likelihood interval: {in_profile}")