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

from exactcis.methods.unconditional import (
    exact_ci_unconditional,
    exact_ci_unconditional_batch,
    calculate_unconditional_pvalue_or,
    or_test_statistic
)
from exactcis.core import calculate_odds_ratio


def test_or_test_statistic():
    """Test the odds ratio test statistic function."""
    # Simple test case
    a, n1, n2, m1 = 10, 30, 30, 15
    p1, p2 = 0.4, 0.3
    
    # Calculate test statistic
    stat = or_test_statistic(a, n1, n2, m1, p1, p2)
    
    # Check that it's a valid number
    assert not math.isnan(stat), "Test statistic should not be NaN"
    assert stat >= 0, "Test statistic should be non-negative"
    
    # Test edge cases
    stat_edge = or_test_statistic(0, n1, n2, m1, p1, p2)
    assert not math.isnan(stat_edge), "Test statistic should handle zero cells"


def test_unconditional_pvalue():
    """Test the unconditional p-value calculation."""
    # Simple test case
    a_obs, n1, n2, m1 = 10, 30, 30, 15
    theta = 2.0
    
    # Calculate p-value
    pvalue = calculate_unconditional_pvalue_or(a_obs, n1, n2, m1, theta)
    
    # Check that it's a valid probability
    assert 0 <= pvalue <= 1, f"P-value should be between 0 and 1, got {pvalue}"
    
    # Test with different theta values
    for theta_test in [0.5, 1.0, 2.0, 5.0]:
        pvalue_test = calculate_unconditional_pvalue_or(a_obs, n1, n2, m1, theta_test)
        assert 0 <= pvalue_test <= 1, f"P-value should be between 0 and 1 for theta={theta_test}"


def test_batch_processing():
    """Test batch processing of multiple tables."""
    tables = [
        (10, 20, 5, 25),
        (2, 8, 1, 9),
        (50, 950, 25, 975)
    ]
    
    # Test batch processing
    results = exact_ci_unconditional_batch(tables, alpha=0.05)
    
    # Check that we get results for all tables
    assert len(results) == len(tables), "Should get results for all tables"
    
    # Check that all results are valid
    for i, (lower, upper) in enumerate(results):
        assert lower >= 0, f"Lower bound should be non-negative for table {i}"
        assert upper > lower, f"Upper bound should be greater than lower bound for table {i}"


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
    lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
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
        lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
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
        lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
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


if __name__ == "__main__":
    print("Testing unconditional method")
    print("============================")
    
    # Test basic functions
    print("\nTesting basic functions...")
    test_or_test_statistic()
    test_unconditional_pvalue()
    test_batch_processing()
    print("Basic function tests passed")
    
    # Test the large sample case
    print("\nTesting large sample case...")
    lower, upper, elapsed_time = test_large_sample_case()
    
    # Test multiple large sample cases
    print("\nTesting multiple large sample cases...")
    results_large = test_multiple_large_sample_cases()
    
    # Test small sample cases
    print("\nTesting small sample cases...")
    results_small = test_small_sample_cases()
    
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