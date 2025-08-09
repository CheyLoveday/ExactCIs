"""
Unit tests for exactcis.utils.mathops module.

Tests safe mathematical operations, numerical stability functions,
and edge case handling for all mathematical utilities.
"""

import math
import pytest
from typing import List

from exactcis.utils.mathops import (
    safe_division, safe_log, safe_log_ratio, exp_safe, clip_probability,
    log_sum_exp, log_diff_exp, log_mean_exp,
    wald_variance_or, wald_variance_rr, pooled_variance_estimate,
    apply_correction_if_needed, zero_cell_adjustment,
    cached_log_factorial, stirling_log_factorial,
    relative_error, is_numerically_close
)


class TestBasicSafeOperations:
    """Tests for basic safe mathematical operations."""
    
    def test_safe_division_normal(self):
        """Test safe division with normal inputs."""
        assert safe_division(10.0, 2.0) == 5.0
        assert safe_division(-6.0, 3.0) == -2.0
        assert safe_division(0.0, 5.0) == 0.0
    
    def test_safe_division_zero_denominator(self):
        """Test safe division with zero denominator."""
        assert safe_division(5.0, 0.0) == 0.0  # Default
        assert safe_division(5.0, 0.0, default=float('inf')) == float('inf')
        assert safe_division(5.0, 1e-20, default=0.0) == 0.0  # Near-zero
    
    def test_safe_log_positive(self):
        """Test safe logarithm with positive inputs."""
        assert abs(safe_log(1.0) - 0.0) < 1e-10
        assert abs(safe_log(math.e) - 1.0) < 1e-10
        assert abs(safe_log(10.0) - math.log(10)) < 1e-10
    
    def test_safe_log_nonpositive(self):
        """Test safe logarithm with non-positive inputs."""
        default_val = -50.0
        assert safe_log(0.0, default_val) == default_val
        assert safe_log(-5.0, default_val) == default_val
    
    def test_safe_log_ratio_valid(self):
        """Test safe log ratio with valid inputs."""
        assert abs(safe_log_ratio(4.0, 2.0) - math.log(2)) < 1e-10
        assert abs(safe_log_ratio(1.0, math.e) + 1.0) < 1e-10
        assert abs(safe_log_ratio(10.0, 10.0) - 0.0) < 1e-10
    
    def test_safe_log_ratio_invalid(self):
        """Test safe log ratio with invalid inputs."""
        default_val = -100.0
        assert safe_log_ratio(0.0, 2.0, default_val) == default_val
        assert safe_log_ratio(2.0, 0.0, default_val) == default_val
        assert safe_log_ratio(-1.0, 2.0, default_val) == default_val
        assert safe_log_ratio(2.0, -1.0, default_val) == default_val
    
    def test_exp_safe_normal(self):
        """Test safe exponential with normal inputs."""
        assert abs(exp_safe(0.0) - 1.0) < 1e-10
        assert abs(exp_safe(1.0) - math.e) < 1e-10
        assert abs(exp_safe(-1.0) - 1/math.e) < 1e-10
    
    def test_exp_safe_extreme(self):
        """Test safe exponential with extreme inputs."""
        assert exp_safe(-1000.0) == 0.0  # Underflow to zero
        assert math.isinf(exp_safe(1000.0))  # Overflow to infinity
    
    def test_clip_probability(self):
        """Test probability clipping."""
        assert clip_probability(0.5) == 0.5  # Normal case
        assert clip_probability(-0.1) > 0  # Clipped to minimum
        assert clip_probability(1.5) < 1.0  # Clipped to maximum
        assert clip_probability(0.0) > 0  # Avoid exact zero
        assert clip_probability(1.0) < 1.0  # Avoid exact one


class TestLogSpaceArithmetic:
    """Tests for log-space arithmetic functions."""
    
    def test_log_sum_exp_simple(self):
        """Test log-sum-exp with simple inputs."""
        # log(exp(0) + exp(1)) = log(1 + e) ≈ 1.313
        result = log_sum_exp([0.0, 1.0])
        expected = math.log(1 + math.e)
        assert abs(result - expected) < 1e-10
    
    def test_log_sum_exp_large_values(self):
        """Test log-sum-exp with large values (overflow protection)."""
        # Should not overflow even with large inputs
        large_vals = [100.0, 101.0, 102.0]
        result = log_sum_exp(large_vals)
        
        # Should be approximately 102 + log(1 + exp(-1) + exp(-2))
        assert result > 102.0
        assert math.isfinite(result)
    
    def test_log_sum_exp_empty(self):
        """Test log-sum-exp with empty input."""
        result = log_sum_exp([])
        assert result < -20  # Should return very negative value
    
    def test_log_diff_exp_normal(self):
        """Test log-diff-exp with normal inputs."""
        # log(exp(2) - exp(1)) = log(e^2 - e^1) = log(e(e-1)) = 1 + log(e-1)
        result = log_diff_exp(2.0, 1.0)
        expected = 1.0 + math.log(math.e - 1)
        assert abs(result - expected) < 1e-10
    
    def test_log_diff_exp_invalid_order(self):
        """Test log-diff-exp with invalid ordering."""
        # log_a <= log_b should return very negative value
        result = log_diff_exp(1.0, 2.0)
        assert result < -20
    
    def test_log_mean_exp(self):
        """Test log-mean-exp calculation."""
        values = [0.0, math.log(2), math.log(4)]  # log(1), log(2), log(4)
        result = log_mean_exp(values)
        
        # Mean of [1, 2, 4] = 7/3, so result should be log(7/3)
        expected = math.log(7/3)
        assert abs(result - expected) < 1e-10


class TestVarianceCalculations:
    """Tests for statistical variance calculations."""
    
    def test_wald_variance_or_normal(self):
        """Test Wald variance for odds ratio with normal inputs."""
        a, b, c, d = 10, 5, 8, 12
        
        # Without Haldane correction
        var_no_corr = wald_variance_or(a, b, c, d, haldane_corrected=False)
        expected = 1/10.5 + 1/5.5 + 1/8.5 + 1/12.5  # After adding 0.5
        assert abs(var_no_corr - expected) < 1e-10
        
        # With Haldane correction already applied
        var_corr = wald_variance_or(10.5, 5.5, 8.5, 12.5, haldane_corrected=True)
        assert abs(var_corr - expected) < 1e-10
    
    def test_wald_variance_or_zero_cells(self):
        """Test Wald variance with zero cells."""
        var_result = wald_variance_or(0, 5, 8, 12)
        # Should be finite after Haldane correction
        assert math.isfinite(var_result)
        assert var_result > 0
    
    def test_wald_variance_rr_standard(self):
        """Test standard Wald variance for relative risk."""
        a, b, c, d = 12, 5, 8, 10
        
        var_result = wald_variance_rr(a, b, c, d, method="standard")
        
        # Expected: b/(a*(a+b)) + d/(c*(c+d)) = 5/(12*17) + 10/(8*18)
        expected = 5/(12*17) + 10/(8*18)
        assert abs(var_result - expected) < 1e-10
    
    def test_wald_variance_rr_katz(self):
        """Test Katz-adjusted Wald variance for relative risk."""
        a, b, c, d = 0, 17, 8, 10  # Zero in 'a' cell
        
        var_result = wald_variance_rr(a, b, c, d, method="katz")
        
        # Should apply Katz adjustment: a becomes 0.5
        assert math.isfinite(var_result)
        assert var_result > 0
    
    def test_wald_variance_rr_correlated(self):
        """Test correlated Wald variance for matched pairs."""
        a, b, c, d = 12, 5, 8, 10
        
        var_result = wald_variance_rr(a, b, c, d, method="correlated")
        
        # Expected: (b+d)/(a+c)^2 = (5+10)/(12+8)^2 = 15/400
        expected = 15/400
        assert abs(var_result - expected) < 1e-10
    
    def test_wald_variance_rr_invalid_method(self):
        """Test Wald variance with invalid method."""
        with pytest.raises(ValueError, match="Unknown variance method"):
            wald_variance_rr(12, 5, 8, 10, method="invalid")
    
    def test_pooled_variance_estimate(self):
        """Test pooled variance estimation."""
        n1, n2 = 100, 150
        p_pooled = 0.3
        
        result = pooled_variance_estimate(n1, n2, p_pooled)
        
        # Expected: p*(1-p)*(1/n1 + 1/n2) = 0.3*0.7*(1/100 + 1/150)
        expected = 0.3 * 0.7 * (1/100 + 1/150)
        assert abs(result - expected) < 1e-10
    
    def test_pooled_variance_invalid_inputs(self):
        """Test pooled variance with invalid inputs."""
        assert math.isinf(pooled_variance_estimate(0, 100, 0.5))
        assert math.isinf(pooled_variance_estimate(100, 0, 0.5))


class TestCorrectionFunctions:
    """Tests for correction application functions."""
    
    def test_zero_cell_adjustment_haldane(self):
        """Test Haldane zero-cell adjustment."""
        a, b, c, d = 0, 5, 8, 12
        
        result = zero_cell_adjustment(a, b, c, d, adjustment_type="haldane")
        expected = (0.5, 5.5, 8.5, 12.5)
        
        assert result == expected
    
    def test_zero_cell_adjustment_reciprocal(self):
        """Test reciprocal zero-cell adjustment."""
        a, b, c, d = 0, 5, 8, 12
        total = 25
        adj = 1.0 / total
        
        result = zero_cell_adjustment(a, b, c, d, adjustment_type="reciprocal")
        expected = (adj, 5+adj, 8+adj, 12+adj)
        
        assert all(abs(r - e) < 1e-10 for r, e in zip(result, expected))
    
    def test_zero_cell_adjustment_none(self):
        """Test no zero-cell adjustment."""
        a, b, c, d = 0, 5, 8, 12
        
        result = zero_cell_adjustment(a, b, c, d, adjustment_type="none")
        expected = (0.0, 5.0, 8.0, 12.0)
        
        assert result == expected
    
    def test_zero_cell_adjustment_invalid(self):
        """Test invalid adjustment type."""
        with pytest.raises(ValueError, match="Unknown adjustment type"):
            zero_cell_adjustment(0, 5, 8, 12, adjustment_type="invalid")


class TestNumericalHelpers:
    """Tests for numerical stability helper functions."""
    
    def test_cached_log_factorial(self):
        """Test cached log factorial computation."""
        assert cached_log_factorial(0) == 0.0
        assert cached_log_factorial(1) == 0.0
        assert abs(cached_log_factorial(3) - math.log(6)) < 1e-10
        assert abs(cached_log_factorial(5) - math.log(120)) < 1e-10
    
    def test_stirling_log_factorial_small(self):
        """Test Stirling approximation for small values (should use exact)."""
        # Should use exact calculation for n < 10
        assert stirling_log_factorial(5) == cached_log_factorial(5)
    
    def test_stirling_log_factorial_large(self):
        """Test Stirling approximation for large values."""
        n = 100
        result = stirling_log_factorial(n)
        
        # Should be close to exact value but computed via approximation
        # For n=100, exact log(100!) ≈ 363.74
        assert 360 < result < 370  # Reasonable range
        assert math.isfinite(result)
    
    def test_relative_error_normal(self):
        """Test relative error calculation with normal values."""
        assert relative_error(10.0, 10.5) == 0.05  # 5% error
        assert relative_error(100.0, 99.0) == 0.01  # 1% error
        assert abs(relative_error(-5.0, -5.1) - 0.02) < 1e-10   # 2% error (floating point precision)
    
    def test_relative_error_near_zero(self):
        """Test relative error with expected value near zero."""
        # Should return absolute error when expected is near zero (< EPS)
        # When expected = 1e-15 (< EPS ≈ 1e-15), return abs(actual)
        result = relative_error(1e-16, 0.1)  # 1e-16 is definitely < EPS
        assert result == 0.1  # Should return abs(actual) = abs(0.1) = 0.1
    
    def test_is_numerically_close_within_tolerance(self):
        """Test numerical closeness within tolerance."""
        assert is_numerically_close(1.0, 1.0001, rel_tol=1e-3)
        assert is_numerically_close(1000.0, 1000.1, rel_tol=1e-3)
        assert is_numerically_close(1e-10, 2e-10, abs_tol=1e-9)
    
    def test_is_numerically_close_outside_tolerance(self):
        """Test numerical closeness outside tolerance."""
        assert not is_numerically_close(1.0, 1.1, rel_tol=1e-3)
        assert not is_numerically_close(1000.0, 1100.0, rel_tol=1e-3)
        assert not is_numerically_close(1e-5, 1e-4, abs_tol=1e-6, rel_tol=1e-6)


class TestIntegration:
    """Integration tests combining multiple mathematical operations."""
    
    def test_odds_ratio_calculation_chain(self):
        """Test complete odds ratio calculation using mathops functions."""
        a, b, c, d = 12, 5, 8, 10
        
        # Apply Haldane correction
        a_c, b_c, c_c, d_c = zero_cell_adjustment(a, b, c, d, "haldane")
        
        # Calculate log odds ratio safely
        log_or = safe_log_ratio(a_c * d_c, b_c * c_c)
        
        # Calculate variance
        var_log_or = wald_variance_or(a_c, b_c, c_c, d_c, haldane_corrected=True)
        
        # Calculate confidence bounds
        se = math.sqrt(var_log_or)
        z = 1.96  # Approximate 95% CI
        
        lower_log = log_or - z * se
        upper_log = log_or + z * se
        
        # Transform back to original scale
        or_estimate = exp_safe(log_or)
        lower_bound = exp_safe(lower_log)
        upper_bound = exp_safe(upper_log)
        
        # Sanity checks
        assert or_estimate > 0
        assert lower_bound > 0
        assert upper_bound > lower_bound
        assert math.isfinite(or_estimate)
        assert math.isfinite(lower_bound) 
        assert math.isfinite(upper_bound)
    
    def test_relative_risk_calculation_chain(self):
        """Test complete relative risk calculation using mathops functions."""
        a, b, c, d = 20, 30, 15, 35
        
        # Calculate risks safely
        n1, n2 = a + b, c + d
        risk1 = safe_division(a, n1, 0.0)
        risk2 = safe_division(c, n2, 0.0)
        
        # Calculate log relative risk
        log_rr = safe_log_ratio(risk1, risk2)
        
        # Calculate variance
        var_log_rr = wald_variance_rr(a, b, c, d, method="standard")
        
        # Should produce reasonable results
        assert math.isfinite(log_rr)
        assert math.isfinite(var_log_rr)
        assert var_log_rr > 0
    
    def test_edge_case_handling_pipeline(self):
        """Test edge case handling through complete calculation pipeline."""
        # Table with zero in first cell
        a, b, c, d = 0, 15, 10, 25
        
        # Apply correction
        a_c, b_c, c_c, d_c = apply_correction_if_needed(a, b, c, d, "wald", "auto")
        
        # Should handle gracefully
        assert a_c > 0  # Correction applied
        
        # Calculate variance  
        var_or = wald_variance_or(a_c, b_c, c_c, d_c, haldane_corrected=True)
        var_rr = wald_variance_rr(a, b, c, d, method="katz")
        
        # Should be finite
        assert math.isfinite(var_or)
        assert math.isfinite(var_rr)
        assert var_or > 0
        assert var_rr > 0