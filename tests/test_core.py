"""
Tests for the core functionality of the ExactCIs package.
"""

import pytest
from exactcis.core import validate_counts, support, pmf, pmf_weights, find_root, find_smallest_theta


def test_validate_counts_valid():
    """Test that validate_counts accepts valid inputs."""
    validate_counts(1, 2, 3, 4)  # Should not raise


def test_validate_counts_negative():
    """Test that validate_counts rejects negative counts."""
    with pytest.raises(ValueError):
        validate_counts(-1, 2, 3, 4)


def test_validate_counts_empty_margin():
    """Test that validate_counts rejects tables with empty margins."""
    with pytest.raises(ValueError):
        validate_counts(0, 0, 3, 4)  # Empty row
    with pytest.raises(ValueError):
        validate_counts(1, 2, 0, 0)  # Empty row
    with pytest.raises(ValueError):
        validate_counts(0, 2, 0, 4)  # Empty column
    with pytest.raises(ValueError):
        validate_counts(1, 0, 3, 0)  # Empty column


def test_support():
    """Test the support function."""
    assert support(5, 5, 5) == (0, 1, 2, 3, 4, 5)
    assert support(3, 2, 2) == (0, 1, 2)
    assert support(2, 3, 2) == (0, 1, 2)
    assert support(2, 2, 3) == (1, 2)


def test_pmf_sums_to_one():
    """Test that the PMF sums to 1."""
    n1, n2, m = 5, 5, 5
    theta = 2.0
    supp = support(n1, n2, m)
    total = sum(pmf(k, n1, n2, m, theta) for k in supp)
    assert abs(total - 1.0) < 1e-10


def test_pmf_weights():
    """Test the pmf_weights function."""
    # Test with simple parameters
    n1, n2, m = 5, 5, 5
    theta = 2.0
    supp, probs = pmf_weights(n1, n2, m, theta)

    # Check that support is correct
    assert supp == (0, 1, 2, 3, 4, 5)

    # Check that probabilities sum to 1
    assert abs(sum(probs) - 1.0) < 1e-10

    # Check that probabilities are non-negative
    assert all(p >= 0 for p in probs)

    # Check that the length of support and probabilities match
    assert len(supp) == len(probs)

    # Test with theta = 0
    supp, probs = pmf_weights(5, 5, 5, 0.0)
    assert probs[0] == 1.0  # All probability mass at the minimum value
    assert all(p == 0.0 for p in probs[1:])

    # Test that pmf and pmf_weights are consistent
    n1, n2, m = 4, 6, 3
    theta = 1.5
    supp, probs = pmf_weights(n1, n2, m, theta)
    for i, k in enumerate(supp):
        assert abs(probs[i] - pmf(k, n1, n2, m, theta)) < 1e-10


def test_find_root():
    """Test the find_root function."""
    # Test with a simple function: f(x) = x^2 - 4
    f = lambda x: x**2 - 4
    root = find_root(f, 0, 3)
    assert abs(root - 2.0) < 1e-8


def test_find_smallest_theta():
    """Test the find_smallest_theta function."""
    # Create a simple test function that returns p-value = alpha when theta = 2.0
    def test_func(theta):
        return 0.025 if theta >= 2.0 else 0.05

    # Test with two-sided=True (default)
    theta = find_smallest_theta(test_func, alpha=0.05, lo=1.0, hi=3.0)
    assert abs(theta - 2.0) < 1e-4

    # Test with two-sided=False
    theta = find_smallest_theta(test_func, alpha=0.05, lo=1.0, hi=3.0, two_sided=False)
    assert abs(theta - 2.0) < 1e-4

    # Test with a continuous function
    def continuous_func(theta):
        return 0.05 * (theta / 2.0)

    theta = find_smallest_theta(continuous_func, alpha=0.05, lo=1.0, hi=3.0)
    assert abs(theta - 2.0) < 1e-4
