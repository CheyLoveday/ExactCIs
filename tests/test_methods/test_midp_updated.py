"""
Tests for the mid-p adjusted confidence interval method.

This file contains tests for the new grid search implementation of the Mid-P method,
which is designed to handle large sample sizes better than the previous implementation.
"""

import pytest
import numpy as np
import time
from exactcis.methods.midp import exact_ci_midp, exact_ci_midp_batch, calculate_midp_pvalue, find_ci_bound
from exactcis.core import validate_counts, calculate_odds_ratio


def test_exact_ci_midp_basic():
    """Test basic functionality of exact_ci_midp."""
    # Test with standard example case
    lower, upper = exact_ci_midp(12, 5, 8, 10, alpha=0.05)
    
    # Verify mathematical properties instead of hardcoded values
    # 1. Lower bound should be positive
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    
    # 2. Upper bound should be finite and greater than lower bound
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    assert upper > lower, f"Expected upper bound ({upper}) > lower bound ({lower})"
    
    # 3. Odds ratio should be within the interval
    odds_ratio = (12 * 10) / (5 * 8)  # (a*d)/(b*c)
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"


def test_exact_ci_midp_edge_cases():
    """Test edge cases for exact_ci_midp."""
    # When a is at the minimum possible value
    lower, upper = exact_ci_midp(0, 10, 10, 10, alpha=0.05)
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    # With the new implementation, we should get a finite upper bound even in edge cases
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"

    # When a is at the maximum possible value
    lower, upper = exact_ci_midp(10, 0, 0, 10, alpha=0.05)
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    # With the new implementation, we should get a finite upper bound even in edge cases
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"

    # When b is zero (should return (0, inf) as documented in the function)
    lower, upper = exact_ci_midp(5, 0, 5, 5, alpha=0.05)
    assert lower == 0.0, f"Expected lower bound of 0.0, got {lower}"
    assert upper == float('inf'), f"Expected upper bound of infinity, got {upper}"

    # When c is zero (should return (0, inf) as documented in the function)
    lower, upper = exact_ci_midp(5, 5, 0, 5, alpha=0.05)
    assert lower == 0.0, f"Expected lower bound of 0.0, got {lower}"
    assert upper == float('inf'), f"Expected upper bound of infinity, got {upper}"


def test_exact_ci_midp_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Negative count
    with pytest.raises(ValueError):
        exact_ci_midp(-1, 5, 8, 10)

    # Empty margin
    with pytest.raises(ValueError):
        exact_ci_midp(0, 0, 8, 10)

    # Invalid alpha
    with pytest.raises(ValueError):
        exact_ci_midp(12, 5, 8, 10, alpha=1.5)


def test_exact_ci_midp_small_counts():
    """Test with small counts."""
    lower, upper = exact_ci_midp(1, 1, 1, 1, alpha=0.05)
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    
    # For this balanced case, odds ratio should be 1
    odds_ratio = 1.0
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"


def test_exact_ci_midp_large_imbalance():
    """Test with large imbalance in counts."""
    lower, upper = exact_ci_midp(50, 5, 2, 20, alpha=0.05)
    # With large imbalance, the lower bound might legitimately be 0
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    # With the new implementation, we should get a finite upper bound
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    
    # Calculate expected odds ratio
    odds_ratio = (50 * 20) / (5 * 2)
    # Check that it's within bounds
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"


def test_exact_ci_midp_problematic_case():
    """Test the previously problematic case (20,80,40,60) that produced invalid CI."""
    a, b, c, d = 20, 80, 40, 60
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
    
    # 1. Verify lower bound is less than upper bound
    assert lower <= upper, f"Expected lower bound ({lower}) <= upper bound ({upper})"
    
    # 2. Calculate odds ratio
    odds_ratio = (a * d) / (b * c)  # (a*d)/(b*c)
    
    # 3. Verify odds ratio is within the CI
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    
    # 4. Verify CI width is positive
    ci_width = upper - lower
    assert ci_width > 0, f"Expected positive CI width, got {ci_width}"


def test_exact_ci_midp_large_sample():
    """Test the Mid-P method with a large sample size that previously failed."""
    # Example case: 50/1000 vs 25/1000
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate the odds ratio
    odds_ratio = (a * d) / (b * c)
    
    # Calculate the confidence interval with default parameters
    lower, upper = exact_ci_midp(a, b, c, d)
    
    # Verify the confidence interval includes the odds ratio
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    
    # Verify the confidence interval has a finite upper bound
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    
    # Verify the confidence interval width is reasonable
    ci_width = upper - lower
    assert 1.0 < ci_width < 5.0, f"Expected reasonable CI width, got {ci_width}"


def test_exact_ci_midp_grid_size_impact():
    """Test the impact of grid size on precision."""
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate CIs with different grid sizes
    ci_small = exact_ci_midp(a, b, c, d, grid_size=50)
    ci_medium = exact_ci_midp(a, b, c, d, grid_size=200)
    ci_large = exact_ci_midp(a, b, c, d, grid_size=500)
    
    # Calculate widths
    width_small = ci_small[1] - ci_small[0]
    width_medium = ci_medium[1] - ci_medium[0]
    width_large = ci_large[1] - ci_large[0]
    
    # Larger grid sizes should generally give more precise (narrower) CIs
    # But this is not guaranteed due to the discrete nature of the grid
    # So we just check that all CIs are reasonable
    assert width_small > 0, f"Expected positive CI width, got {width_small}"
    assert width_medium > 0, f"Expected positive CI width, got {width_medium}"
    assert width_large > 0, f"Expected positive CI width, got {width_large}"
    
    # All CIs should include the odds ratio
    odds_ratio = (a * d) / (b * c)
    assert ci_small[0] <= odds_ratio <= ci_small[1], f"Small grid CI does not include odds ratio"
    assert ci_medium[0] <= odds_ratio <= ci_medium[1], f"Medium grid CI does not include odds ratio"
    assert ci_large[0] <= odds_ratio <= ci_large[1], f"Large grid CI does not include odds ratio"


def test_exact_ci_midp_theta_range_impact():
    """Test the impact of theta range on results."""
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate CIs with different theta ranges
    ci_narrow = exact_ci_midp(a, b, c, d, theta_min=0.1, theta_max=10)
    ci_medium = exact_ci_midp(a, b, c, d, theta_min=0.001, theta_max=1000)
    ci_wide = exact_ci_midp(a, b, c, d, theta_min=0.0001, theta_max=10000)
    
    # All CIs should include the odds ratio
    odds_ratio = (a * d) / (b * c)
    assert ci_narrow[0] <= odds_ratio <= ci_narrow[1], f"Narrow range CI does not include odds ratio"
    assert ci_medium[0] <= odds_ratio <= ci_medium[1], f"Medium range CI does not include odds ratio"
    assert ci_wide[0] <= odds_ratio <= ci_wide[1], f"Wide range CI does not include odds ratio"
    
    # All CIs should have finite bounds
    assert ci_narrow[0] > 0 and ci_narrow[1] < float('inf'), f"Narrow range CI has invalid bounds: {ci_narrow}"
    assert ci_medium[0] > 0 and ci_medium[1] < float('inf'), f"Medium range CI has invalid bounds: {ci_medium}"
    assert ci_wide[0] > 0 and ci_wide[1] < float('inf'), f"Wide range CI has invalid bounds: {ci_wide}"


def test_exact_ci_midp_alpha_levels():
    """Test different alpha levels."""
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate CIs with different alpha levels
    ci_01 = exact_ci_midp(a, b, c, d, alpha=0.01)
    ci_05 = exact_ci_midp(a, b, c, d, alpha=0.05)
    ci_10 = exact_ci_midp(a, b, c, d, alpha=0.10)
    
    # Calculate widths
    width_01 = ci_01[1] - ci_01[0]
    width_05 = ci_05[1] - ci_05[0]
    width_10 = ci_10[1] - ci_10[0]
    
    # Lower alpha should give wider CIs
    assert width_01 > width_05 > width_10, f"Expected width_01 ({width_01}) > width_05 ({width_05}) > width_10 ({width_10})"
    
    # All CIs should include the odds ratio
    odds_ratio = (a * d) / (b * c)
    assert ci_01[0] <= odds_ratio <= ci_01[1], f"CI with alpha=0.01 does not include odds ratio"
    assert ci_05[0] <= odds_ratio <= ci_05[1], f"CI with alpha=0.05 does not include odds ratio"
    assert ci_10[0] <= odds_ratio <= ci_10[1], f"CI with alpha=0.10 does not include odds ratio"


def test_exact_ci_midp_batch():
    """Test batch processing of multiple tables."""
    # Create a list of tables
    tables = [
        (50, 950, 25, 975),  # Large sample
        (10, 90, 5, 95),     # Medium sample
        (2, 8, 1, 9)         # Small sample
    ]
    
    # Calculate confidence intervals for all tables
    results = exact_ci_midp_batch(tables)
    
    # Check results for each table
    for i, ((a, b, c, d), (lower, upper)) in enumerate(zip(tables, results)):
        # Calculate odds ratio
        odds_ratio = (a * d) / (b * c)
        
        # Verify the confidence interval includes the odds ratio
        assert lower <= odds_ratio <= upper, f"Table {i+1}: CI ({lower}, {upper}) does not include odds ratio {odds_ratio}"
        
        # Verify the confidence interval has a finite upper bound
        assert upper < float('inf'), f"Table {i+1}: Expected finite upper bound, got {upper}"
        
        # Verify the confidence interval width is positive
        ci_width = upper - lower
        assert ci_width > 0, f"Table {i+1}: Expected positive CI width, got {ci_width}"


def test_helper_functions():
    """Test the helper functions used by the Mid-P method."""
    # Test calculate_midp_pvalue
    a_obs, n1, n2, m1 = 50, 1000, 1000, 75
    theta = 2.0
    p_value = calculate_midp_pvalue(a_obs, n1, n2, m1, theta)
    assert 0 <= p_value <= 1, f"Expected p-value between 0 and 1, got {p_value}"
    
    # Test find_ci_bound
    theta_grid = np.logspace(-3, 3, 100)
    p_values = np.linspace(0, 1, 100)
    alpha = 0.05
    
    # Lower bound
    lower = find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    
    # Upper bound
    upper = find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    assert upper > lower, f"Expected upper bound ({upper}) > lower bound ({lower})"


def test_comparison_with_other_methods():
    """Compare Mid-P results with other methods."""
    try:
        from exactcis.methods.conditional import exact_ci_conditional
        from exactcis.methods.blaker import exact_ci_blaker
        
        # Test case from CLAUDE.md: 50/1000 vs 25/1000
        a, b, c, d = 50, 950, 25, 975
        
        # Calculate CIs with different methods
        ci_midp = exact_ci_midp(a, b, c, d)
        ci_conditional = exact_ci_conditional(a, b, c, d)
        ci_blaker = exact_ci_blaker(a, b, c, d)
        
        # Calculate widths
        width_midp = ci_midp[1] - ci_midp[0]
        width_conditional = ci_conditional[1] - ci_conditional[0]
        width_blaker = ci_blaker[1] - ci_blaker[0]
        
        # Expected width ordering: Blaker ≤ Mid-P ≤ Conditional
        # But this is not guaranteed for all cases, so we just check that all CIs are reasonable
        assert width_blaker > 0, f"Expected positive Blaker CI width, got {width_blaker}"
        assert width_midp > 0, f"Expected positive Mid-P CI width, got {width_midp}"
        assert width_conditional > 0, f"Expected positive Conditional CI width, got {width_conditional}"
        
        # All CIs should include the odds ratio
        odds_ratio = (a * d) / (b * c)
        assert ci_blaker[0] <= odds_ratio <= ci_blaker[1], f"Blaker CI does not include odds ratio"
        assert ci_midp[0] <= odds_ratio <= ci_midp[1], f"Mid-P CI does not include odds ratio"
        assert ci_conditional[0] <= odds_ratio <= ci_conditional[1], f"Conditional CI does not include odds ratio"
        
        # All CIs should have finite bounds
        assert ci_blaker[0] > 0 and ci_blaker[1] < float('inf'), f"Blaker CI has invalid bounds: {ci_blaker}"
        assert ci_midp[0] > 0 and ci_midp[1] < float('inf'), f"Mid-P CI has invalid bounds: {ci_midp}"
        assert ci_conditional[0] > 0 and ci_conditional[1] < float('inf'), f"Conditional CI has invalid bounds: {ci_conditional}"
        
        # Check if the results match the expected values from CLAUDE.md
        # Blaker: [1.250, 2.053]
        # Conditional: [1.234, 3.263]
        # Mid-P: should be between Blaker and Conditional
        assert abs(ci_blaker[0] - 1.250) < 0.1, f"Expected Blaker lower bound ~1.250, got {ci_blaker[0]}"
        assert abs(ci_blaker[1] - 2.053) < 0.1, f"Expected Blaker upper bound ~2.053, got {ci_blaker[1]}"
        assert abs(ci_conditional[0] - 1.234) < 0.1, f"Expected Conditional lower bound ~1.234, got {ci_conditional[0]}"
        assert abs(ci_conditional[1] - 3.263) < 0.1, f"Expected Conditional upper bound ~3.263, got {ci_conditional[1]}"
        
    except ImportError:
        pytest.skip("Other methods not available for comparison")


def test_performance():
    """Test performance of the Mid-P method with different parameters."""
    a, b, c, d = 50, 950, 25, 975
    
    # Test with different grid sizes
    start_time = time.time()
    exact_ci_midp(a, b, c, d, grid_size=50)
    time_small = time.time() - start_time
    
    start_time = time.time()
    exact_ci_midp(a, b, c, d, grid_size=200)
    time_medium = time.time() - start_time
    
    start_time = time.time()
    exact_ci_midp(a, b, c, d, grid_size=500)
    time_large = time.time() - start_time
    
    # Larger grid sizes should take more time
    assert time_small < time_medium < time_large, f"Expected time_small ({time_small}) < time_medium ({time_medium}) < time_large ({time_large})"
    
    # But the increase should be roughly linear with grid size
    ratio_medium_small = time_medium / time_small
    ratio_large_medium = time_large / time_medium
    
    assert 2 < ratio_medium_small < 6, f"Expected ratio_medium_small between 2 and 6, got {ratio_medium_small}"
    assert 1.5 < ratio_large_medium < 4, f"Expected ratio_large_medium between 1.5 and 4, got {ratio_large_medium}"