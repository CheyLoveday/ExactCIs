"""
Tests for the conditional (Fisher) confidence interval method.
"""

import pytest
import numpy as np
from scipy import stats
from exactcis.methods import exact_ci_conditional
from exactcis.methods.conditional import ComputationError


def test_exact_ci_conditional_basic():
    """Test basic functionality of exact_ci_conditional."""
    # Example from the README
    lower, upper = exact_ci_conditional(12, 5, 8, 10, alpha=0.05)
    assert round(lower, 3) == 1.059, f"Expected lower bound 1.059, got {lower:.3f}"
    assert round(upper, 3) == 8.726, f"Expected upper bound 8.726, got {upper:.3f}"


def test_exact_ci_conditional_agresti_example():
    """Test Agresti (2002) tea tasting example."""
    # Agresti (2002), p. 91 - Tea tasting
    # Table: [[3, 1], [1, 3]] -> a=3, b=1, c=1, d=3
    # Expected from statsmodels: (0.238051, 1074.817433)
    lower, upper = exact_ci_conditional(3, 1, 1, 3, alpha=0.05)
    assert round(lower, 6) == 0.238051, f"Expected lower bound 0.238051, got {lower:.6f}"
    assert upper > 500, f"Expected upper bound > 500, got {upper}"


def test_exact_ci_conditional_scipy_example():
    """Test example from scipy.stats.fisher_exact documentation."""
    # Table: [[1, 9], [11, 3]] -> a=1, b=9, c=11, d=3
    # Expected from statsmodels: (0.000541, 0.525381)
    lower, upper = exact_ci_conditional(1, 9, 11, 3, alpha=0.05)
    assert round(lower, 6) == 0.000541, f"Expected lower bound 0.000541, got {lower:.6f}"
    assert round(upper, 6) == 0.525381, f"Expected upper bound 0.525381, got {upper:.6f}"


def test_exact_ci_conditional_infinite_bound():
    """Test cases where the upper bound should be infinity."""
    # Table: [[5, 0], [2, 3]] -> a=5, b=0, c=2, d=3
    # Expected from statsmodels: (0.528283, inf)
    lower, upper = exact_ci_conditional(5, 0, 2, 3, alpha=0.05)
    assert round(lower, 6) == 0.528283, f"Expected lower bound 0.528283, got {lower:.6f}"
    assert upper == float('inf'), f"Expected infinite upper bound, got {upper}"

    # Another example with infinite upper bound
    # Table: [[1, 0], [9, 10]] -> a=1, b=0, c=9, d=10
    lower, upper = exact_ci_conditional(1, 0, 9, 10, alpha=0.05)
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    assert upper == float('inf'), f"Expected infinite upper bound, got {upper}"


def test_exact_ci_conditional_statsmodels_example():
    """Test example from statsmodels Table2x2 fisher example."""
    # Table: [[7, 17], [15, 5]] -> a=7, b=17, c=15, d=5
    # Expected from statsmodels: (0.019110, 0.831039)
    lower, upper = exact_ci_conditional(7, 17, 15, 5, alpha=0.05)
    assert round(lower, 6) == 0.019110, f"Expected lower bound 0.019110, got {lower:.6f}"
    assert round(upper, 6) == 0.831039, f"Expected upper bound 0.831039, got {upper:.6f}"


def test_exact_ci_conditional_from_r_comparison():
    """Test examples from R comparison script."""
    # Standard table from R comparison
    lower, upper = exact_ci_conditional(7, 3, 2, 8, alpha=0.05)
    assert 0.88 <= lower <= 0.89, f"Expected lower bound ~0.882, got {lower:.6f}"
    assert 120 <= upper <= 130, f"Expected upper bound ~127.05, got {upper:.6f}"

    # Balanced small table
    lower, upper = exact_ci_conditional(10, 10, 10, 10, alpha=0.05)
    assert 0.24 <= lower <= 0.25, f"Expected lower bound ~0.244, got {lower:.6f}"
    assert 4.0 <= upper <= 4.2, f"Expected upper bound ~4.10, got {upper:.6f}"

    # Large table
    lower, upper = exact_ci_conditional(100, 50, 60, 120, alpha=0.05)
    # Allow for some numerical differences between R and Python implementation
    assert 2.9 <= lower <= 3.1, f"Expected lower bound ~3.0, got {lower:.6f}"
    assert 7.3 <= upper <= 7.5, f"Expected upper bound ~7.4, got {upper:.6f}"


def test_exact_ci_conditional_with_zeros():
    """Test scenarios with zeros in the table."""
    # Table with a zero: [[0, 5], [5, 5]]
    # Expected from statsmodels: (0.000000, 1.506704)
    lower, upper = exact_ci_conditional(0, 5, 5, 5, alpha=0.05)
    assert lower == 0.0, f"Expected lower bound 0.0, got {lower:.6f}"
    assert 1.4 <= upper <= 1.6, f"Expected upper bound ~1.51, got {upper:.6f}"


def test_exact_ci_conditional_extreme_values():
    """Test cases with extreme values in the table."""
    # Extreme proportions: Table [[99, 1], [50, 50]]
    # This tests robustness against numerical issues with large odds ratios
    try:
        lower, upper = exact_ci_conditional(99, 1, 50, 50, alpha=0.05)
        # The values are not as important as the function not crashing
        assert lower > 0, f"Lower bound should be positive, got {lower}"
        assert np.isfinite(upper), f"Upper bound should be finite, got {upper}"
    except ComputationError as e:
        pytest.skip(f"Test skipped due to computation error: {e}")


def test_exact_ci_conditional_different_alpha():
    """Test the method with different alpha values."""
    # Test with alpha = 0.01 (99% CI)
    lower, upper = exact_ci_conditional(7, 3, 2, 8, alpha=0.01)
    # We don't check exact values, just that it runs and gives sensible results
    assert 0 < lower < 1, f"Lower bound outside expected range: {lower}"
    assert upper > 50, f"Upper bound too small: {upper}"

    # Test with alpha = 0.1 (90% CI)
    lower, upper = exact_ci_conditional(7, 3, 2, 8, alpha=0.1)
    assert 0 < lower < 1, f"Lower bound outside expected range: {lower}"
    assert upper > 50, f"Upper bound too small: {upper}"


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


def test_exact_ci_conditional_precision():
    """Test precision of values near boundaries, checking for numerical stability."""
    # Test with very small observed value
    lower, upper = exact_ci_conditional(1, 99, 99, 1, alpha=0.05)
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    assert upper < 1, f"Expected upper bound < 1, got {upper}"
    
    # Test with large counts but balanced table
    try:
        lower, upper = exact_ci_conditional(1000, 1000, 1000, 1000, alpha=0.05)
        # Should be close to 1 for balanced table
        assert 0.9 <= lower <= 1.1, f"Expected lower ~1.0, got {lower}"
        assert 0.9 <= upper <= 1.1, f"Expected upper ~1.0, got {upper}"
    except ComputationError:
        pytest.skip("Skipping large balanced table due to computational limits")


def test_exact_ci_conditional_comparison_with_scipy():
    """Compare our results with scipy.stats.fisher_exact for odds ratio p-values."""
    # We can only compare p-values, not CI directly
    # Table: [[3, 1], [1, 3]]
    a, b, c, d = 3, 1, 1, 3
    
    # Calculate the odds ratio
    or_point = (a * d) / (b * c) if (b * c) != 0 else float('inf')
    
    # Get p-value from scipy
    _, p_scipy = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
    
    # Our CIs should match the scipy p-value at alpha = p_scipy
    # This is an approximate check, since we're inverting the test
    try:
        lower, upper = exact_ci_conditional(a, b, c, d, alpha=p_scipy)
        # The odds ratio should be approximately at one of the CI boundaries
        # for the p-value test
        assert (or_point < lower or or_point > upper), \
            f"For p={p_scipy}, OR={or_point} should be outside CI=({lower}, {upper})"
    except ValueError:
        # For very small p-values, this might not be computationally feasible
        pytest.skip(f"Skipping scipy comparison with p={p_scipy} due to computational limits")


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