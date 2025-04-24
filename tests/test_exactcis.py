"""
Tests for the main ExactCIs package functionality.
"""

import pytest
from exactcis import compute_all_cis


def test_compute_all_cis():
    """Test that compute_all_cis returns results for all methods."""
    # Example from the README
    results = compute_all_cis(12, 5, 8, 10, alpha=0.05, grid_size=200)

    # Check that all methods are included
    expected_methods = {"conditional", "midp", "blaker", "unconditional", "wald_haldane"}
    assert set(results.keys()) == expected_methods

    # Check that each result is a tuple of two floats
    for method, ci in results.items():
        assert isinstance(ci, tuple), f"Result for {method} is not a tuple"
        assert len(ci) == 2, f"Result for {method} does not have 2 elements"
        assert all(isinstance(x, float) for x in ci), f"Result for {method} contains non-float values"
        assert ci[0] <= ci[1], f"Lower bound is greater than upper bound for {method}"


def test_compute_all_cis_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Negative count
    with pytest.raises(ValueError):
        compute_all_cis(-1, 5, 8, 10)

    # Empty margin
    with pytest.raises(ValueError):
        compute_all_cis(0, 0, 8, 10)


def test_compute_all_cis_matches_readme_example():
    """Test that the results match the example in the README."""
    results = compute_all_cis(12, 5, 8, 10, alpha=0.05, grid_size=500)

    # Expected values from the README
    expected = {
        "conditional": (1.059, 8.726),
        "midp": (1.205, 7.893),
        "blaker": (1.114, 8.312),
        "unconditional": (1.132, 8.204),
        "wald_haldane": (1.024, 8.658)
    }

    # Check that each result is close to the expected value
    for method, (expected_lower, expected_upper) in expected.items():
        actual_lower, actual_upper = results[method]
        assert abs(actual_lower - expected_lower) < 0.01, \
            f"Lower bound for {method} differs from expected: {actual_lower:.3f} vs {expected_lower:.3f}"
        assert abs(actual_upper - expected_upper) < 0.01, \
            f"Upper bound for {method} differs from expected: {actual_upper:.3f} vs {expected_upper:.3f}"
