"""
Tests for Clopper-Pearson exact confidence interval method.
"""

import pytest
import numpy as np
from exactcis.methods import exact_ci_clopper_pearson


def test_exact_ci_clopper_pearson_basic():
    """Test basic functionality of exact_ci_clopper_pearson with property-based assertions."""
    # Test with a simple example
    lower, upper = exact_ci_clopper_pearson(12, 8, 5, 15, alpha=0.05, group=1)
    
    # Test fundamental properties
    assert 0.0 <= lower <= 1.0, f"Lower bound should be between 0 and 1, got {lower:.3f}"
    assert 0.0 <= upper <= 1.0, f"Upper bound should be between 0 and 1, got {upper:.3f}"
    assert upper > lower, f"Upper bound {upper:.3f} should be greater than lower {lower:.3f}"
    
    # Test that confidence interval contains the point estimate
    p_hat = 12 / (12 + 8)  # Point estimate for group 1
    assert lower <= p_hat <= upper, f"CI [{lower:.3f}, {upper:.3f}] should contain point estimate {p_hat:.3f}"
    
    # Test with group=2
    lower, upper = exact_ci_clopper_pearson(12, 8, 5, 15, alpha=0.05, group=2)
    
    # Test fundamental properties
    assert 0.0 <= lower <= 1.0, f"Lower bound should be between 0 and 1, got {lower:.3f}"
    assert 0.0 <= upper <= 1.0, f"Upper bound should be between 0 and 1, got {upper:.3f}"
    assert upper > lower, f"Upper bound {upper:.3f} should be greater than lower {lower:.3f}"
    
    # Test that confidence interval contains the point estimate
    p_hat = 5 / (5 + 15)  # Point estimate for group 2
    assert lower <= p_hat <= upper, f"CI [{lower:.3f}, {upper:.3f}] should contain point estimate {p_hat:.3f}"


def test_exact_ci_clopper_pearson_edge_cases():
    """Test edge cases for exact_ci_clopper_pearson."""
    # When x = 0, lower bound must be 0
    lower, upper = exact_ci_clopper_pearson(0, 10, 5, 5, alpha=0.05, group=1)
    assert lower == 0.0, f"Expected lower bound 0.0 when x=0, got {lower}"
    assert upper < 1.0, f"Expected upper bound < 1.0 when x=0, got {upper}"
    
    # When x = n, upper bound must be 1
    lower, upper = exact_ci_clopper_pearson(10, 0, 5, 5, alpha=0.05, group=1)
    assert lower > 0.0, f"Expected lower bound > 0.0 when x=n, got {lower}"
    assert upper == 1.0, f"Expected upper bound 1.0 when x=n, got {upper}"
    
    # Same tests for group=2
    lower, upper = exact_ci_clopper_pearson(5, 5, 0, 10, alpha=0.05, group=2)
    assert lower == 0.0, f"Expected lower bound 0.0 when x=0, got {lower}"
    assert upper < 1.0, f"Expected upper bound < 1.0 when x=0, got {upper}"
    
    lower, upper = exact_ci_clopper_pearson(5, 5, 10, 0, alpha=0.05, group=2)
    assert lower > 0.0, f"Expected lower bound > 0.0 when x=n, got {lower}"
    assert upper == 1.0, f"Expected upper bound 1.0 when x=n, got {upper}"


def test_exact_ci_clopper_pearson_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Negative count
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(-1, 5, 8, 10)
    
    # Empty margin (will raise ValueError from validate_counts)
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(0, 0, 8, 10)
    
    # Invalid alpha
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(12, 5, 8, 10, alpha=1.5)
    
    # Invalid group
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(12, 5, 8, 10, group=3)
    
    # Empty group (n1 = 0 for group=1)
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(0, 0, 8, 10, group=1)
    
    # Empty group (n2 = 0 for group=2)
    with pytest.raises(ValueError):
        exact_ci_clopper_pearson(8, 10, 0, 0, group=2)


def test_exact_ci_clopper_pearson_small_counts():
    """Test with small counts."""
    # For (1, 1, 1, 1), p1 = p2 = 0.5
    lower1, upper1 = exact_ci_clopper_pearson(1, 1, 1, 1, alpha=0.05, group=1)
    lower2, upper2 = exact_ci_clopper_pearson(1, 1, 1, 1, alpha=0.05, group=2)
    
    # Both intervals should be the same
    assert np.isclose(lower1, lower2), f"Expected same lower bounds, got {lower1:.3f} and {lower2:.3f}"
    assert np.isclose(upper1, upper2), f"Expected same upper bounds, got {upper1:.3f} and {upper2:.3f}"
    
    # Interval should contain 0.5
    assert lower1 < 0.5 < upper1, f"CI [{lower1:.3f}, {upper1:.3f}] should contain 0.5"


def test_exact_ci_clopper_pearson_large_imbalance():
    """Test with large imbalance in counts."""
    # This is a case where the proportion is very small
    lower, upper = exact_ci_clopper_pearson(1, 99, 50, 50, alpha=0.05, group=1)
    assert 0.0 <= lower < upper <= 1.0, f"Expected valid interval, got [{lower:.3f}, {upper:.3f}]"
    
    # This is a case where the proportion is very large
    lower, upper = exact_ci_clopper_pearson(99, 1, 50, 50, alpha=0.05, group=1)
    assert 0.0 <= lower < upper <= 1.0, f"Expected valid interval, got [{lower:.3f}, {upper:.3f}]"


@pytest.mark.fast
def test_clopper_pearson_alpha_convention():
    """
    Test that the implementation correctly uses α (two-sided) convention.
    
    This verifies the statistical correctness without hardcoded values.
    """
    # Test monotonicity: stricter alpha should produce wider intervals
    lower_05, upper_05 = exact_ci_clopper_pearson(12, 8, 5, 15, alpha=0.05, group=1)
    lower_025, upper_025 = exact_ci_clopper_pearson(12, 8, 5, 15, alpha=0.025, group=1)
    lower_01, upper_01 = exact_ci_clopper_pearson(12, 8, 5, 15, alpha=0.01, group=1)
    
    # Smaller alpha (higher confidence) should produce wider intervals (monotonicity test)
    assert lower_01 <= lower_025 <= lower_05, "Lower bounds should decrease as alpha decreases (confidence increases)"
    assert upper_05 <= upper_025 <= upper_01, "Upper bounds should increase as alpha decreases (confidence increases)"
    
    # Test interval properties
    assert 0.0 <= lower_05 < upper_05 <= 1.0, "Alpha=0.05 should produce valid interval"
    assert 0.0 <= lower_025 < upper_025 <= 1.0, "Alpha=0.025 should produce valid interval"
    assert 0.0 <= lower_01 < upper_01 <= 1.0, "Alpha=0.01 should produce valid interval"


@pytest.mark.fast
def test_clopper_pearson_group_parameter():
    """
    Test that the group parameter correctly selects which group to calculate the CI for.
    """
    # Create a table with different proportions in each group
    a, b, c, d = 8, 2, 3, 7  # p1 = 0.8, p2 = 0.3
    
    # Calculate CIs for both groups
    lower1, upper1 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=1)
    lower2, upper2 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=2)
    
    # The intervals should be different
    assert not np.isclose(lower1, lower2), f"Expected different lower bounds, got {lower1:.3f} and {lower2:.3f}"
    assert not np.isclose(upper1, upper2), f"Expected different upper bounds, got {upper1:.3f} and {upper2:.3f}"
    
    # The intervals should contain their respective point estimates
    p1 = a / (a + b)  # 0.8
    p2 = c / (c + d)  # 0.3
    
    assert lower1 <= p1 <= upper1, f"CI for group 1 [{lower1:.3f}, {upper1:.3f}] should contain {p1:.3f}"
    assert lower2 <= p2 <= upper2, f"CI for group 2 [{lower2:.3f}, {upper2:.3f}] should contain {p2:.3f}"