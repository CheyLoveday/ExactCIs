"""
Tests for the odds ratio utility functions in ExactCIs.
"""

import pytest
from exactcis import (
    calculate_odds_ratio,
    calculate_odds_ratio_with_correction,
    add_haldane_correction,
    calculate_standard_error
)


def test_calculate_odds_ratio():
    """Test the basic odds ratio calculation."""
    assert calculate_odds_ratio(10, 5, 2, 8) == 8.0  # (10*8)/(5*2) = 8
    
    # Test with zero cells
    assert calculate_odds_ratio(10, 0, 5, 8) == float('inf')
    assert calculate_odds_ratio(10, 5, 0, 8) == float('inf')
    assert calculate_odds_ratio(0, 5, 2, 0) == 0.0  # (0*0)/(5*2) = 0
    assert calculate_odds_ratio(0, 5, 0, 8) == 1.0  # Indeterminate form 0/0


def test_add_haldane_correction():
    """Test applying Haldane's correction."""
    a, b, c, d = 10, 5, 2, 8
    a_corr, b_corr, c_corr, d_corr = add_haldane_correction(a, b, c, d)
    
    assert a_corr == a + 0.5
    assert b_corr == b + 0.5
    assert c_corr == c + 0.5
    assert d_corr == d + 0.5


def test_calculate_odds_ratio_with_correction():
    """Test odds ratio calculation with correction."""
    # Without correction
    or_without_corr = calculate_odds_ratio_with_correction(10, 5, 2, 8, add_correction=False)
    assert or_without_corr == 8.0
    
    # With correction
    or_with_corr = calculate_odds_ratio_with_correction(10, 5, 2, 8, add_correction=True)
    a_corr, b_corr, c_corr, d_corr = add_haldane_correction(10, 5, 2, 8)
    expected_or = (a_corr * d_corr) / (b_corr * c_corr)
    assert or_with_corr == pytest.approx(expected_or)
    
    # Test with zero cells
    or_zero_cells = calculate_odds_ratio_with_correction(0, 5, 0, 8, add_correction=True)
    assert or_zero_cells != 1.0  # Should not be indeterminate with correction
    

def test_calculate_standard_error():
    """Test standard error calculation."""
    # With positive values
    se = calculate_standard_error(10, 5, 2, 8, add_correction=False)
    expected_se = (1/10 + 1/5 + 1/2 + 1/8) ** 0.5
    assert se == pytest.approx(expected_se)
    
    # With zero cells and correction
    se_zero = calculate_standard_error(0, 5, 0, 8, add_correction=True)
    # Should use correction values instead of zeros
    a_corr, b_corr, c_corr, d_corr = add_haldane_correction(0, 5, 0, 8)
    expected_se_zero = (1/a_corr + 1/b_corr + 1/c_corr + 1/d_corr) ** 0.5
    assert se_zero == pytest.approx(expected_se_zero)
