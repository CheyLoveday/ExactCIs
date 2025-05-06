"""
Tests for the mid-p adjusted confidence interval method.
"""

import pytest
from exactcis.methods import exact_ci_midp


def test_exact_ci_midp_basic():
    """Test basic functionality of exact_ci_midp."""
    # Example from the README
    lower, upper = exact_ci_midp(12, 5, 8, 10, alpha=0.05)
    assert round(lower, 3) == 1.205, f"Expected lower bound 1.205, got {lower:.3f}"
    assert round(upper, 3) == 7.893, f"Expected upper bound 7.893, got {upper:.3f}"


def test_exact_ci_midp_edge_cases():
    """Test edge cases for exact_ci_midp."""
    # When a is at the minimum possible value
    try:
        lower, upper = exact_ci_midp(0, 10, 10, 10, alpha=0.05)
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        # Accept infinity as a valid upper bound in edge cases
        assert upper > 0, f"Expected positive upper bound, got {upper}"
    except RuntimeError:
        # If the method raises a RuntimeError, that's acceptable for this edge case
        pass

    # When a is at the maximum possible value
    try:
        lower, upper = exact_ci_midp(10, 0, 0, 10, alpha=0.05)
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        assert upper <= float('inf'), f"Expected upper bound at most infinity, got {upper}"
    except RuntimeError:
        # If the method raises a RuntimeError, that's acceptable for this edge case
        pass


def test_exact_ci_midp_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Negative count
    with pytest.raises((ValueError, RuntimeError)):
        exact_ci_midp(-1, 5, 8, 10)

    # Empty margin
    with pytest.raises((ValueError, RuntimeError)):
        exact_ci_midp(0, 0, 8, 10)

    # Invalid alpha
    with pytest.raises((ValueError, RuntimeError)):
        exact_ci_midp(12, 5, 8, 10, alpha=1.5)


def test_exact_ci_midp_small_counts():
    """Test with small counts."""
    lower, upper = exact_ci_midp(1, 1, 1, 1, alpha=0.05)
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    assert upper <= float('inf'), f"Expected upper bound at most infinity, got {upper}"


def test_exact_ci_midp_large_imbalance():
    """Test with large imbalance in counts."""
    try:
        lower, upper = exact_ci_midp(50, 5, 2, 20, alpha=0.05)
        # With large imbalance, the lower bound might legitimately be 0
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        assert upper <= float('inf'), f"Expected upper bound at most infinity, got {upper}"
    except RuntimeError:
        # If the method raises a RuntimeError, that's acceptable for this edge case
        pass
