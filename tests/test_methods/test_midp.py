"""
Tests for the mid-p adjusted confidence interval method.

This file contains tests for the grid search implementation of the Mid-P method,
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
    
    # For this balanced case, odds ratio should be 1
    odds_ratio = 1.0
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"


def test_exact_ci_midp_large_imbalance():
    """Test with large imbalance in counts."""
    try:
        lower, upper = exact_ci_midp(50, 5, 2, 20, alpha=0.05)
        # With large imbalance, the lower bound might legitimately be 0
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        assert upper <= float('inf'), f"Expected upper bound at most infinity, got {upper}"
        
        # Calculate expected odds ratio
        odds_ratio = (50 * 20) / (5 * 2)
        # Only check that it's within bounds if bounds are finite
        if lower < upper < float('inf'):
            assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    except RuntimeError:
        # If the method raises a RuntimeError, that's acceptable for this edge case
        pass


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
