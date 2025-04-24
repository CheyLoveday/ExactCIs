"""
Tests for the conditional (Fisher) confidence interval method.
"""

import pytest
from exactcis.methods import exact_ci_conditional


def test_exact_ci_conditional_basic():
    """Test basic functionality of exact_ci_conditional."""
    # Example from the README
    lower, upper = exact_ci_conditional(12, 5, 8, 10, alpha=0.05)
    assert round(lower, 3) == 1.059, f"Expected lower bound 1.059, got {lower:.3f}"
    assert round(upper, 3) == 8.726, f"Expected upper bound 8.726, got {upper:.3f}"


def test_exact_ci_conditional_edge_cases():
    """Test edge cases for exact_ci_conditional."""
    # When a is at the minimum possible value
    lower, upper = exact_ci_conditional(0, 10, 10, 10, alpha=0.05)
    assert lower == 0.0, f"Expected lower bound 0.0, got {lower}"
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    
    # When a is at the maximum possible value
    lower, upper = exact_ci_conditional(10, 0, 0, 10, alpha=0.05)
    assert lower > 0.0, f"Expected positive lower bound, got {lower}"
    assert upper == float('inf'), f"Expected infinite upper bound, got {upper}"


def test_exact_ci_conditional_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Negative count
    with pytest.raises(ValueError):
        exact_ci_conditional(-1, 5, 8, 10)
    
    # Empty margin
    with pytest.raises(ValueError):
        exact_ci_conditional(0, 0, 8, 10)
    
    # Invalid alpha
    with pytest.raises(ValueError):
        exact_ci_conditional(12, 5, 8, 10, alpha=1.5)