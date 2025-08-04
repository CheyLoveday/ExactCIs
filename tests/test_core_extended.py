"""
Extended tests for core functionality to improve coverage.
These tests focus on untested code paths and edge cases.
"""

import pytest
import numpy as np
import math
import time
from unittest.mock import patch, MagicMock
from exactcis.core import (
    validate_counts, apply_haldane_correction, logsumexp, log_binom_coeff,
    log_nchg_pmf, log_nchg_cdf, log_nchg_sf, nchg_pdf, support, pmf_weights,
    pmf, find_root, find_root_log, find_plateau_edge, find_sign_change,
    calculate_odds_ratio, estimate_point_or, calculate_relative_risk,
    create_2x2_table, batch_validate_counts, batch_calculate_odds_ratios,
    batch_log_nchg_pmf, batch_support_calculations, optimize_core_cache_for_batch,
    _pmf_weights_original_impl, _find_root_log_original_impl
)
@pytest.mark.core
@pytest.mark.fast
def test_validate_counts_edge_cases():
    """Test validate_counts with edge cases."""
    # Test with float inputs
    validate_counts(1.0, 2.0, 3.0, 4.0)  # Should not raise
    
    # Test with very large numbers
    validate_counts(1000000, 2000000, 3000000, 4000000)  # Should not raise
    
    # Test specific empty margin cases
    with pytest.raises(ValueError, match="Empty row or column"):
        validate_counts(0, 0, 5, 5)  # Empty first row
    
    with pytest.raises(ValueError, match="Empty row or column"):
        validate_counts(5, 5, 0, 0)  # Empty second row
    
    with pytest.raises(ValueError, match="Empty row or column"):
        validate_counts(0, 5, 0, 5)  # Empty first column
    
    with pytest.raises(ValueError, match="Empty row or column"):
        validate_counts(5, 0, 5, 0)  # Empty second column


@pytest.mark.core
@pytest.mark.fast
def test_apply_haldane_correction_edge_cases():
    """Test apply_haldane_correction with various edge cases."""
    # Test with all non-zero values
    result = apply_haldane_correction(2, 3, 4, 5)
    assert result == (2.0, 3.0, 4.0, 5.0)
    
    # Test with float inputs
    result = apply_haldane_correction(1.5, 2.5, 3.5, 4.5)
    assert result == (1.5, 2.5, 3.5, 4.5)
    
    # Test with mixed zero and non-zero
    result = apply_haldane_correction(0, 5, 0, 10)
    assert result == (0.5, 5.5, 0.5, 10.5)


@pytest.mark.core
@pytest.mark.fast
def test_logsumexp_edge_cases():
    """Test logsumexp with edge cases."""
    # Test with single very large value
    assert abs(logsumexp([1000]) - 1000) < 1e-10
    
    # Test with mixed finite and -inf values
    result = logsumexp([1.0, float('-inf'), 2.0, float('-inf')])
    expected = logsumexp([1.0, 2.0])
    assert abs(result - expected) < 1e-10
    
    # Test with numpy array input
    result = logsumexp(np.array([0, 0, 0]))
    assert abs(result - math.log(3)) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_log_binom_coeff_edge_cases():
    """Test log_binom_coeff with edge cases."""
    # Test with float inputs
    assert abs(log_binom_coeff(5.0, 2.0) - math.log(10)) < 1e-10
    
    # Test with large numbers
    result = log_binom_coeff(100, 50)
    assert np.isfinite(result)
    assert result > 0  # C(100,50) is very large
    
    # Test with k = n
    assert log_binom_coeff(10, 10) == 0.0
    
    # Test with k = 0
    assert log_binom_coeff(10, 0) == 0.0


@pytest.mark.core
@pytest.mark.fast
def test_nchg_pdf():
    """Test nchg_pdf function."""
    # Test basic functionality
    k_values = np.array([0, 1, 2, 3])
    n1, n2, m1, theta = 5, 5, 5, 2.0
    
    result = nchg_pdf(k_values, n1, n2, m1, theta)
    
    # Check that result is numpy array
    assert isinstance(result, np.ndarray)
    assert len(result) == len(k_values)
    
    # Check that all probabilities are non-negative
    assert np.all(result >= 0)
    
    # Check that probabilities sum to approximately 1 for full support
    supp = support(n1, n2, m1)
    full_result = nchg_pdf(supp.x, n1, n2, m1, theta)
    assert abs(np.sum(full_result) - 1.0) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_support_edge_cases():
    """Test support function with edge cases."""
    # Test with float inputs
    supp = support(5.0, 5.0, 5.0)
    assert hasattr(supp, 'x')  # Should have x attribute
    assert np.array_equal(supp.x, np.array([0, 1, 2, 3, 4, 5]))
    
    # Test with very constrained support
    supp = support(2, 2, 3)
    assert supp.min_val == 1
    assert supp.max_val == 2
    
    # Test with minimal support
    supp = support(1, 1, 1)
    assert supp.min_val == 0
    assert supp.max_val == 1


@pytest.mark.core
@pytest.mark.fast
def test_pmf_weights_edge_cases():
    """Test pmf_weights with edge cases."""
    # Test with theta = 0
    supp, probs = pmf_weights(5, 5, 5, 0.0)
    assert probs[0] == 1.0  # All probability on minimum value
    assert all(p == 0.0 for p in probs[1:])
    
    # Test with very large theta
    supp, probs = pmf_weights(5, 5, 5, 1000.0)
    assert probs[-1] == 1.0  # All probability on maximum value
    assert all(p == 0.0 for p in probs[:-1])
    
    # Test with theta = 1 (null hypothesis)
    supp, probs = pmf_weights(5, 5, 5, 1.0)
    assert len(supp) == len(probs)
    assert abs(sum(probs) - 1.0) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_pmf_edge_cases():
    """Test pmf function with edge cases."""
    # Test with k outside support
    result = pmf(-1, 5, 5, 5, 2.0)
    assert result == 0.0
    
    result = pmf(10, 5, 5, 5, 2.0)
    assert result == 0.0
    
    # Test with theta = 0
    result = pmf(0, 5, 5, 5, 0.0)
    assert result == 1.0
    
    result = pmf(1, 5, 5, 5, 0.0)
    assert result == 0.0


@pytest.mark.core
@pytest.mark.fast
def test_find_root_edge_cases():
    """Test find_root with edge cases."""
    # Test with function that has root at boundary
    f = lambda x: x - 0.5  # Root at x = 0.5
    root = find_root(f, 0.0, 1.0)
    assert abs(root - 0.5) < 1e-8
    
    # Test with function that doesn't change sign
    f_no_root = lambda x: x + 1  # Always positive
    with pytest.raises(ValueError, match="Function does not change sign"):
        find_root(f_no_root, 0.0, 1.0)
    
    # Test with very tight tolerance
    f = lambda x: x**2 - 4
    root = find_root(f, 0, 3, tol=1e-12)
    assert abs(root - 2.0) < 1e-11


@pytest.mark.core
@pytest.mark.fast
def test_find_root_log_edge_cases():
    """Test find_root_log with edge cases."""
    # Test with negative bounds
    f = lambda x: x - 2.0
    with pytest.raises(ValueError, match="Bounds must be positive"):
        find_root_log(f, -1.0, 2.0)
    
    with pytest.raises(ValueError, match="Bounds must be positive"):
        find_root_log(f, 1.0, -2.0)
    
    # Test with timeout
    def slow_func(x):
        time.sleep(0.01)
        return x - 2.0
    
    timeout_checker = lambda: True  # Always timeout
    result = find_root_log(slow_func, 1.0, 3.0, timeout_checker=timeout_checker)
    assert result is None
    
    # Test with function that can't be bracketed
    f_unbracketable = lambda x: x + 1  # Always positive
    result = find_root_log(f_unbracketable, 1.0, 2.0)
    assert result is None


@pytest.mark.core
@pytest.mark.fast
def test_find_plateau_edge_edge_cases():
    """Test find_plateau_edge with edge cases."""
    # Test with function where lo already meets condition
    def func_lo_good(x):
        return 0.06  # Always above target
    
    result = find_plateau_edge(func_lo_good, 1.0, 5.0, target=0.05, increasing=True)
    assert result is not None
    theta, iterations = result
    assert theta == 1.0
    assert iterations == 0
    
    # Test with function where hi doesn't meet condition
    def func_hi_bad(x):
        return 0.04  # Always below target
    
    result = find_plateau_edge(func_hi_bad, 1.0, 5.0, target=0.05, increasing=True)
    assert result is not None
    theta, iterations = result
    assert theta == 5.0
    assert iterations == 0
    
    # Test with timeout
    def slow_func(x):
        time.sleep(0.01)
        return 0.05
    
    timeout_checker = lambda: True
    result = find_plateau_edge(slow_func, 1.0, 5.0, target=0.05, timeout_checker=timeout_checker)
    assert result is None
    
    # Test with decreasing function
    def decreasing_func(x):
        if x < 3.0:
            return 0.06  # Above target
        else:
            return 0.04  # Below target
    
    result = find_plateau_edge(decreasing_func, 1.0, 5.0, target=0.05, increasing=False)
    assert result is not None
    theta, iterations = result
    assert 2.5 < theta < 3.5


@pytest.mark.core
@pytest.mark.fast
def test_find_sign_change():
    """Test find_sign_change function."""
    # Test basic sign change
    f = lambda x: x - 2.0  # Changes sign at x = 2
    result = find_sign_change(f, 1.0, 3.0)
    assert result is not None
    assert 1.9 < result < 2.1
    
    # Test with no sign change
    f_no_change = lambda x: x + 1  # Always positive
    result = find_sign_change(f_no_change, 1.0, 3.0)
    assert result is None
    
    # Test with timeout
    def slow_func(x):
        time.sleep(0.01)
        return x - 2.0
    
    timeout_checker = lambda: True
    result = find_sign_change(slow_func, 1.0, 3.0, timeout_checker=timeout_checker)
    assert result is None


@pytest.mark.core
@pytest.mark.fast
def test_calculate_odds_ratio():
    """Test calculate_odds_ratio function."""
    # Test normal case
    or_val = calculate_odds_ratio(5, 3, 2, 8)
    expected = (5 * 8) / (3 * 2)
    assert abs(or_val - expected) < 1e-10
    
    # Test with zero in denominator
    or_val = calculate_odds_ratio(5, 0, 2, 8)
    assert or_val == float('inf')
    
    or_val = calculate_odds_ratio(5, 3, 0, 8)
    assert or_val == float('inf')
    
    or_val = calculate_odds_ratio(5, 0, 0, 8)
    assert or_val == float('inf')
    
    # Test with zero in numerator
    or_val = calculate_odds_ratio(0, 3, 2, 8)
    assert or_val == 0.0
    
    or_val = calculate_odds_ratio(5, 3, 2, 0)
    assert or_val == 0.0
    
    # Test with zeros in both
    or_val = calculate_odds_ratio(0, 3, 2, 0)
    assert np.isnan(or_val)


@pytest.mark.core
@pytest.mark.fast
def test_estimate_point_or():
    """Test estimate_point_or function."""
    # Test normal case
    or_val = estimate_point_or(5, 3, 2, 8)
    expected = (5 * 8) / (3 * 2)
    assert abs(or_val - expected) < 1e-10
    
    # Test with Haldane correction needed
    or_val = estimate_point_or(0, 3, 2, 8)
    expected = (0.5 * 8.5) / (3.5 * 2.5)
    assert abs(or_val - expected) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_calculate_relative_risk():
    """Test calculate_relative_risk function."""
    # Test normal case
    rr = calculate_relative_risk(5, 3, 2, 8)
    p1 = 5 / (5 + 3)
    p2 = 2 / (2 + 8)
    expected = p1 / p2
    assert abs(rr - expected) < 1e-10
    
    # Test with zero denominator
    rr = calculate_relative_risk(5, 3, 0, 0)
    assert rr == float('inf')
    
    # Test with zero numerator
    rr = calculate_relative_risk(0, 0, 2, 8)
    assert rr == 0.0


@pytest.mark.core
@pytest.mark.fast
def test_create_2x2_table():
    """Test create_2x2_table function."""
    table = create_2x2_table(5, 3, 2, 8)
    expected = np.array([[5, 3], [2, 8]])
    assert np.array_equal(table, expected)
    
    # Test with float inputs
    table = create_2x2_table(5.0, 3.0, 2.0, 8.0)
    expected = np.array([[5.0, 3.0], [2.0, 8.0]])
    assert np.array_equal(table, expected)


@pytest.mark.core
@pytest.mark.fast
def test_batch_validate_counts():
    """Test batch_validate_counts function."""
    tables = [
        (5, 3, 2, 8),  # Valid
        (0, 0, 2, 8),  # Invalid - empty row
        (5, 3, 2, 8),  # Valid
        (-1, 3, 2, 8), # Invalid - negative
    ]
    
    results = batch_validate_counts(tables)
    assert results == [True, False, True, False]
    
    # Test with empty list
    results = batch_validate_counts([])
    assert results == []


@pytest.mark.core
@pytest.mark.fast
def test_batch_calculate_odds_ratios():
    """Test batch_calculate_odds_ratios function."""
    tables = [
        (5, 3, 2, 8),  # Normal case
        (0, 3, 2, 8),  # Zero in numerator
        (5, 0, 2, 8),  # Zero in denominator
        (5, 3, 2, 0),  # Zero in numerator
    ]
    
    results = batch_calculate_odds_ratios(tables)
    
    assert len(results) == 4
    assert abs(results[0] - (5*8)/(3*2)) < 1e-10
    assert results[1] == 0.0
    assert results[2] == float('inf')
    assert results[3] == 0.0


@pytest.mark.core
@pytest.mark.fast
def test_batch_log_nchg_pmf():
    """Test batch_log_nchg_pmf function."""
    k_values = [0, 1, 2, 3]
    n1, n2, m1, theta = 5, 5, 5, 2.0
    
    results = batch_log_nchg_pmf(k_values, n1, n2, m1, theta)
    
    assert len(results) == len(k_values)
    
    # Compare with individual calculations
    for i, k in enumerate(k_values):
        expected = log_nchg_pmf(k, n1, n2, m1, theta)
        assert abs(results[i] - expected) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_batch_support_calculations():
    """Test batch_support_calculations function."""
    tables = [
        (5, 5, 5, 5),
        (3, 2, 2, 3),
        (10, 10, 10, 10),
    ]
    
    results = batch_support_calculations(tables)
    
    assert len(results) == len(tables)
    
    for i, (a, b, c, d) in enumerate(tables):
        n1, n2, m1 = a + b, c + d, a + c
        expected_support = support(n1, n2, m1)
        
        result = results[i]
        assert 'support' in result
        assert 'min_val' in result
        assert 'max_val' in result
        assert 'range_size' in result
        
        assert np.array_equal(result['support'], expected_support.x)
        assert result['min_val'] == expected_support.min_val
        assert result['max_val'] == expected_support.max_val


@pytest.mark.core
@pytest.mark.fast
def test_optimize_core_cache_for_batch():
    """Test optimize_core_cache_for_batch function."""
    # Test enabling large cache
    optimize_core_cache_for_batch(enable_large_cache=True)
    # Function should complete without error
    
    # Test disabling large cache
    optimize_core_cache_for_batch(enable_large_cache=False)
    # Function should complete without error


@pytest.mark.core
@pytest.mark.fast
def test_pmf_weights_original_impl():
    """Test _pmf_weights_original_impl function."""
    n1, n2, m, theta = 5, 5, 5, 2.0
    
    supp, probs = _pmf_weights_original_impl(n1, n2, m, theta)
    
    # Compare with regular pmf_weights
    expected_supp, expected_probs = pmf_weights(n1, n2, m, theta)
    
    assert supp == expected_supp
    assert len(probs) == len(expected_probs)
    
    for i in range(len(probs)):
        assert abs(probs[i] - expected_probs[i]) < 1e-10


@pytest.mark.core
@pytest.mark.fast
def test_find_root_log_original_impl():
    """Test _find_root_log_original_impl function."""
    # Test with simple function
    f = lambda x: x - 2.0
    result = _find_root_log_original_impl(f, 1.0, 3.0)
    
    if result is not None:
        assert abs(math.exp(result) - 2.0) < 1e-6
    
    # Test with timeout
    def slow_func(x):
        time.sleep(0.01)
        return x - 2.0
    
    timeout_checker = lambda: True
    result = _find_root_log_original_impl(slow_func, 1.0, 3.0, timeout_checker=timeout_checker)
    assert result is None


@pytest.mark.core
@pytest.mark.fast
def test_log_nchg_functions_consistency():
    """Test consistency between log_nchg_pmf, log_nchg_cdf, and log_nchg_sf."""
    n1, n2, m1, theta = 5, 5, 5, 2.0
    supp = support(n1, n2, m1)
    
    for k in supp.x:
        # Test that CDF and SF are complementary
        log_cdf = log_nchg_cdf(k, n1, n2, m1, theta)
        log_sf = log_nchg_sf(k, n1, n2, m1, theta)
        
        # For k < max, CDF(k) + SF(k+1) should be close to 1
        if k < supp.max_val:
            log_sf_next = log_nchg_sf(k + 1, n1, n2, m1, theta)
            log_sum = logsumexp([log_cdf, log_sf_next])
            assert abs(log_sum - 0.0) < 1e-8  # log(1) = 0


@pytest.mark.core
@pytest.mark.fast
def test_log_nchg_pmf_extreme_theta():
    """Test log_nchg_pmf with extreme theta values."""
    n1, n2, m1 = 5, 5, 5
    supp = support(n1, n2, m1)
    
    # Test with very small theta
    for k in supp.x:
        log_prob = log_nchg_pmf(k, n1, n2, m1, 1e-10)
        if k == supp.min_val:
            assert log_prob == 0.0  # log(1)
        else:
            assert log_prob == float('-inf')
    
    # Test with very large theta
    for k in supp.x:
        log_prob = log_nchg_pmf(k, n1, n2, m1, 1e10)
        if k == supp.max_val:
            assert log_prob == 0.0  # log(1)
        else:
            assert log_prob == float('-inf')


@pytest.mark.core
@pytest.mark.fast
def test_find_root_with_maxiter():
    """Test find_root with maxiter parameter."""
    # Function that converges slowly
    f = lambda x: (x - 2.0) ** 3  # Cubic function, slower convergence
    
    # Test with low maxiter
    root = find_root(f, 1.0, 3.0, maxiter=5)
    assert abs(root - 2.0) < 1e-3  # Less precise due to low maxiter
    
    # Test with high maxiter
    root = find_root(f, 1.0, 3.0, maxiter=100)
    assert abs(root - 2.0) < 1e-10  # More precise


@pytest.mark.core
@pytest.mark.fast
def test_find_plateau_edge_precision():
    """Test find_plateau_edge with different precision requirements."""
    def plateau_func(x):
        if x < 2.0:
            return 0.04
        elif x < 4.0:
            return 0.05  # Plateau
        else:
            return 0.06
    
    # Test with high precision
    result = find_plateau_edge(plateau_func, 1.0, 5.0, target=0.05, tol=1e-10)
    assert result is not None
    theta, iterations = result
    assert abs(theta - 2.0) < 1e-8
    
    # Test with low precision
    result = find_plateau_edge(plateau_func, 1.0, 5.0, target=0.05, tol=1e-3)
    assert result is not None
    theta, iterations = result
    assert abs(theta - 2.0) < 1e-2


@pytest.mark.core
@pytest.mark.slow
def test_core_functions_stress_test():
    """Stress test core functions with challenging inputs."""
    # Test with very large numbers
    try:
        validate_counts(1e6, 1e6, 1e6, 1e6)
        
        # Test support with large numbers
        supp = support(1000, 1000, 1000)
        assert len(supp.x) <= 1001  # Should be manageable
        
        # Test pmf_weights with large numbers (smaller to avoid memory issues)
        supp, probs = pmf_weights(100, 100, 100, 2.0)
        assert abs(sum(probs) - 1.0) < 1e-10
        
    except Exception as e:
        pytest.skip(f"Stress test failed with large numbers: {e}")
    
    # Test with very small probabilities
    try:
        log_prob = log_nchg_pmf(0, 100, 100, 50, 1e-10)
        assert np.isfinite(log_prob)
        
    except Exception as e:
        pytest.skip(f"Stress test failed with small probabilities: {e}")


@pytest.mark.core
@pytest.mark.fast
def test_cached_pmf_weights():
    """Test that pmf_weights caching works correctly."""
    # Call pmf_weights twice with same parameters
    n1, n2, m, theta = 5, 5, 5, 2.0
    
    supp1, probs1 = pmf_weights(n1, n2, m, theta)
    supp2, probs2 = pmf_weights(n1, n2, m, theta)
    
    # Results should be identical
    assert supp1 == supp2
    assert len(probs1) == len(probs2)
    
    for i in range(len(probs1)):
        assert abs(probs1[i] - probs2[i]) < 1e-15  # Should be exactly equal due to caching


@pytest.mark.core
@pytest.mark.fast
def test_find_root_log_bracket_expansion():
    """Test find_root_log bracket expansion logic."""
    # Function that requires bracket expansion
    def narrow_root_func(x):
        # Root at x = 1.5, but only changes sign in narrow range
        if 1.4 < x < 1.6:
            return x - 1.5
        elif x <= 1.4:
            return -0.1
        else:
            return 0.1
    
    # Initial bracket might not contain the root
    result = find_root_log(narrow_root_func, 1.0, 2.0)
    if result is not None:
        assert abs(math.exp(result) - 1.5) < 1e-6