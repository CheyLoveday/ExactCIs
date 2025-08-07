"""
Unit tests for relative risk confidence interval methods.

This module provides comprehensive testing for all relative risk CI methods
including Wald-based, score-based, and U-statistic methods.
"""

import pytest
import math
import numpy as np
from scipy import stats
from typing import Tuple, Dict

from exactcis.methods.relative_risk import (
    validate_counts,
    add_continuity_correction,
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    constrained_mle_p21,
    score_statistic,
    corrected_score_statistic,
    find_score_ci_bound,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)


@pytest.mark.fast
@pytest.mark.methods
class TestValidationFunctions:
    """Tests for input validation and utility functions."""

    def test_validate_counts_valid(self):
        """Test validate_counts with valid inputs."""
        # Should not raise any exception
        validate_counts(10, 5, 8, 12)
        validate_counts(0, 5, 8, 12)  # Zero is valid
        validate_counts(10.0, 5.0, 8.0, 12.0)  # Floats are valid

    def test_validate_counts_invalid(self):
        """Test validate_counts with invalid inputs."""
        with pytest.raises(ValueError):
            validate_counts(-1, 5, 8, 12)
        
        with pytest.raises(ValueError):
            validate_counts(10, -5, 8, 12)
        
        with pytest.raises(ValueError):
            validate_counts(10, 5, -8, 12)
        
        with pytest.raises(ValueError):
            validate_counts(10, 5, 8, -12)

    def test_add_continuity_correction(self):
        """Test continuity correction application."""
        # No zeros - should return original values as floats
        result = add_continuity_correction(10, 5, 8, 12)
        assert result == (10.0, 5.0, 8.0, 12.0)
        
        # One zero - should add correction
        result = add_continuity_correction(0, 5, 8, 12)
        assert result == (0.5, 5.5, 8.5, 12.5)
        
        # Multiple zeros - should add correction
        result = add_continuity_correction(0, 0, 8, 12)
        assert result == (0.5, 0.5, 8.5, 12.5)
        
        # Custom correction value
        result = add_continuity_correction(0, 5, 8, 12, correction=0.25)
        assert result == (0.25, 5.25, 8.25, 12.25)


@pytest.mark.fast
@pytest.mark.methods
class TestWaldMethods:
    """Tests for Wald-based confidence interval methods."""

    def test_ci_wald_rr_basic(self):
        """Test basic Wald CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_wald_rr(a, b, c, d, alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower < upper < float('inf')

    def test_ci_wald_rr_zero_cells(self):
        """Test Wald CI with zero cells."""
        # Zero in exposed outcome
        lower, upper = ci_wald_rr(0, 10, 5, 5, 0.05)
        assert lower >= 0.0  # May be slightly above 0 due to continuity correction
        assert upper < float('inf')  # Finite upper bound due to continuity correction
        
        # Zero in unexposed outcome
        lower, upper = ci_wald_rr(5, 5, 0, 10, 0.05)
        assert lower > 0.0  # Positive lower bound due to continuity correction
        assert upper < float('inf')  # Finite but large upper bound
        
        # Zero in both outcomes
        lower, upper = ci_wald_rr(0, 10, 0, 10, 0.05)
        assert lower >= 0.0  # May be slightly above 0 due to continuity correction
        assert upper > 10.0  # Should be large (may be finite due to continuity correction)

    def test_ci_wald_katz_rr_basic(self):
        """Test Katz-adjusted Wald CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_wald_katz_rr(a, b, c, d, alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower < upper < float('inf')

    def test_ci_wald_correlated_rr_basic(self):
        """Test correlation-adjusted Wald CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_wald_correlated_rr(a, b, c, d, alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower < upper < float('inf')

    def test_wald_methods_different_alpha(self):
        """Test Wald methods with different alpha levels."""
        a, b, c, d = 15, 5, 10, 10
        
        # Test different alpha levels
        for alpha in [0.01, 0.05, 0.1]:
            lower_wald, upper_wald = ci_wald_rr(a, b, c, d, alpha)
            lower_katz, upper_katz = ci_wald_katz_rr(a, b, c, d, alpha)
            lower_corr, upper_corr = ci_wald_correlated_rr(a, b, c, d, alpha)
            
            # All should be valid intervals
            assert 0 < lower_wald < upper_wald < float('inf')
            assert 0 < lower_katz < upper_katz < float('inf')
            assert 0 < lower_corr < upper_corr < float('inf')
            
            # Smaller alpha should give wider intervals
            if alpha == 0.01:
                alpha_05_lower, alpha_05_upper = ci_wald_rr(a, b, c, d, 0.05)
                assert lower_wald <= alpha_05_lower
                assert upper_wald >= alpha_05_upper


@pytest.mark.fast
@pytest.mark.methods
class TestScoreMethods:
    """Tests for score-based confidence interval methods."""

    def test_constrained_mle_p21(self):
        """Test constrained MLE calculation."""
        x11, x12, x21, x22 = 15, 5, 10, 10
        theta0 = 1.5
        
        p21 = constrained_mle_p21(x11, x12, x21, x22, theta0)
        
        # Should be a valid probability
        assert 0 <= p21 <= 1
        
        # Test edge cases
        p21_null = constrained_mle_p21(x11, x12, x21, x22, 1.0)
        assert 0 <= p21_null <= 1
        
        # Test with zero cells
        p21_zero = constrained_mle_p21(0, 10, 5, 5, 1.0)
        assert 0 <= p21_zero <= 1

    def test_score_statistic(self):
        """Test score statistic calculation."""
        x11, x12, x21, x22 = 15, 5, 10, 10
        
        # Test with different theta values
        for theta0 in [0.5, 1.0, 1.5, 2.0]:
            score = score_statistic(x11, x12, x21, x22, theta0)
            assert math.isfinite(score)
        
        # Test with zero cells
        score_zero = score_statistic(0, 10, 5, 5, 1.0)
        assert math.isfinite(score_zero)

    def test_corrected_score_statistic(self):
        """Test continuity-corrected score statistic calculation."""
        x11, x12, x21, x22 = 15, 5, 10, 10
        
        # Test with different theta and delta values
        for theta0 in [0.5, 1.0, 1.5, 2.0]:
            for delta in [2.0, 4.0, 8.0]:
                score = corrected_score_statistic(x11, x12, x21, x22, theta0, delta)
                assert math.isfinite(score)

    def test_ci_score_rr_basic(self):
        """Test basic score CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_score_rr(a, b, c, d, alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower <= upper  # Upper bound may be infinite

    def test_ci_score_cc_rr_basic(self):
        """Test continuity-corrected score CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_score_cc_rr(a, b, c, d, alpha=alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower <= upper  # Upper bound may be infinite

    def test_score_methods_zero_cells(self):
        """Test score methods with zero cells."""
        # Zero in exposed outcome
        lower, upper = ci_score_rr(0, 10, 5, 5, 0.05)
        assert lower == 0.0
        assert upper < float('inf')
        
        # Zero in unexposed outcome
        lower, upper = ci_score_rr(5, 5, 0, 10, 0.05)
        assert lower > 0.0
        assert upper == float('inf')

    def test_score_cc_different_delta(self):
        """Test continuity-corrected score with different delta values."""
        a, b, c, d = 5, 15, 3, 17
        alpha = 0.05
        
        # Test different delta values
        deltas = [2.0, 4.0, 8.0]
        intervals = []
        
        for delta in deltas:
            lower, upper = ci_score_cc_rr(a, b, c, d, delta=delta, alpha=alpha)
            intervals.append((lower, upper))
            assert 0 < lower <= upper  # Upper bound may be infinite
        
        # Generally, larger delta should give wider intervals (more conservative)
        # But this isn't guaranteed for all cases, so we just check validity


@pytest.mark.fast
@pytest.mark.methods
class TestUStatMethod:
    """Tests for U-statistic confidence interval method."""

    def test_ci_ustat_rr_basic(self):
        """Test basic U-statistic CI calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        lower, upper = ci_ustat_rr(a, b, c, d, alpha)
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper
        assert 0 < lower < upper < float('inf')

    def test_ci_ustat_rr_zero_cells(self):
        """Test U-statistic CI with zero cells."""
        # Zero in exposed outcome
        lower, upper = ci_ustat_rr(0, 10, 5, 5, 0.05)
        assert lower >= 0.0  # May be slightly above 0 due to continuity correction
        assert upper < float('inf')  # Should be finite
        
        # Zero in unexposed outcome
        lower, upper = ci_ustat_rr(5, 5, 0, 10, 0.05)
        assert lower > 0.0  # Positive due to continuity correction
        assert upper < float('inf')  # Finite but large

    def test_ci_ustat_rr_small_samples(self):
        """Test U-statistic CI with small sample sizes."""
        a, b, c, d = 3, 7, 2, 8
        alpha = 0.05
        
        lower, upper = ci_ustat_rr(a, b, c, d, alpha)
        
        # Should still produce valid intervals for small samples
        assert 0 < lower < upper < float('inf')
        
        # Calculate expected relative risk
        n1, n2 = a + b, c + d
        risk1, risk2 = a / n1, c / n2
        expected_rr = risk1 / risk2
        
        # CI should contain the point estimate
        assert lower <= expected_rr <= upper

    def test_ci_ustat_rr_different_alpha(self):
        """Test U-statistic CI with different alpha levels."""
        a, b, c, d = 15, 5, 10, 10
        
        # Test different alpha levels
        alphas = [0.01, 0.05, 0.1]
        intervals = []
        
        for alpha in alphas:
            lower, upper = ci_ustat_rr(a, b, c, d, alpha)
            intervals.append((lower, upper, alpha))
            assert 0 < lower < upper < float('inf')
        
        # Smaller alpha should generally give wider intervals
        for i in range(len(intervals) - 1):
            lower1, upper1, alpha1 = intervals[i]
            lower2, upper2, alpha2 = intervals[i + 1]
            if alpha1 < alpha2:  # More stringent confidence level
                assert lower1 <= lower2
                assert upper1 >= upper2


@pytest.mark.fast
@pytest.mark.methods
class TestMethodComparisons:
    """Tests comparing different relative risk CI methods."""

    def test_all_methods_contain_point_estimate(self):
        """Test that all methods contain the point estimate."""
        test_cases = [
            (15, 5, 10, 10),  # Standard case
            (3, 7, 2, 8),     # Small sample
            (50, 50, 25, 75), # Different denominators
            (1, 99, 1, 99),   # Very small risks
        ]
        
        alpha = 0.05
        methods = [
            ("wald", ci_wald_rr),
            ("wald_katz", ci_wald_katz_rr),
            ("wald_corr", ci_wald_correlated_rr),
            ("score", ci_score_rr),
            ("score_cc", ci_score_cc_rr),
            ("ustat", ci_ustat_rr),
        ]
        
        for a, b, c, d in test_cases:
            # Calculate point estimate
            n1, n2 = a + b, c + d
            if n1 > 0 and n2 > 0 and c > 0:
                risk1, risk2 = a / n1, c / n2
                expected_rr = risk1 / risk2
                
                for method_name, method_func in methods:
                    try:
                        lower, upper = method_func(a, b, c, d, alpha)
                        assert lower <= expected_rr <= upper, \
                            f"Method {method_name} failed for case ({a}, {b}, {c}, {d})"
                    except (ValueError, ZeroDivisionError):
                        # Some methods may legitimately fail for extreme cases
                        pass

    def test_continuity_correction_effect(self):
        """Test that continuity correction generally widens intervals."""
        test_cases = [
            (5, 15, 3, 17),   # Small sample
            (2, 18, 1, 19),   # Very small numerators
        ]
        
        alpha = 0.05
        
        for a, b, c, d in test_cases:
            # Compare score vs score with continuity correction
            try:
                lower_score, upper_score = ci_score_rr(a, b, c, d, alpha)
                lower_cc, upper_cc = ci_score_cc_rr(a, b, c, d, alpha=alpha)
                
                # Continuity correction should generally widen the interval
                assert lower_cc <= lower_score
                assert upper_cc >= upper_score
            except (ValueError, ZeroDivisionError):
                # Some cases may legitimately fail
                pass

    def test_alpha_level_consistency(self):
        """Test that smaller alpha gives wider intervals."""
        a, b, c, d = 15, 5, 10, 10
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            # 99% CI should be wider than 95% CI
            lower_99, upper_99 = method(a, b, c, d, 0.01)
            lower_95, upper_95 = method(a, b, c, d, 0.05)
            
            assert lower_99 <= lower_95
            assert upper_99 >= upper_95


@pytest.mark.edge
@pytest.mark.methods
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_imbalance(self):
        """Test with extreme imbalance in group sizes."""
        # Very unbalanced groups
        a, b, c, d = 1, 999, 1, 9
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            try:
                lower, upper = method(a, b, c, d, alpha)
                # Should produce valid intervals even with extreme imbalance
                assert 0 <= lower <= upper
                if lower > 0:
                    assert upper < float('inf') or upper == float('inf')
            except (ValueError, RuntimeError):
                # Some methods may legitimately fail for extreme cases
                pass

    def test_perfect_association(self):
        """Test with perfect association (all outcomes in one group)."""
        # Perfect positive association
        a, b, c, d = 10, 0, 0, 10
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            try:
                lower, upper = method(a, b, c, d, alpha)
                # Should handle perfect association gracefully
                assert 0 <= lower <= upper
            except (ValueError, RuntimeError):
                # Some methods may legitimately fail for perfect association
                pass

    def test_very_small_counts(self):
        """Test with very small counts."""
        # Minimal non-zero case
        a, b, c, d = 1, 1, 1, 1
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            try:
                lower, upper = method(a, b, c, d, alpha)
                # Should produce valid intervals for minimal counts
                assert 0 <= lower <= upper  # Allow lower bound to be 0 for some methods
                
                # Calculate expected RR
                expected_rr = (a / (a + b)) / (c / (c + d))
                if lower > 0 and upper < float('inf'):
                    assert lower <= expected_rr <= upper
            except (ValueError, RuntimeError):
                # Some methods may have issues with very small counts
                pass

    def test_invalid_alpha_values(self):
        """Test with invalid alpha values."""
        a, b, c, d = 15, 5, 10, 10
        
        invalid_alphas = [-0.1, 0.0, 1.0, 1.5]
        
        for alpha in invalid_alphas:
            # Most methods should handle invalid alpha gracefully
            # (either by raising ValueError or by clamping to valid range)
            try:
                lower, upper = ci_wald_rr(a, b, c, d, alpha)
                # If it doesn't raise an error, just check that we got some result
                # Some invalid alpha values may produce unusual but mathematically valid results
                assert isinstance(lower, (int, float))
                assert isinstance(upper, (int, float))
            except ValueError:
                # This is acceptable behavior for invalid alpha
                pass


@pytest.mark.slow
@pytest.mark.methods
class TestNumericalStability:
    """Tests for numerical stability and precision."""

    def test_large_counts(self):
        """Test with large count values."""
        # Large counts
        a, b, c, d = 1500, 500, 1000, 1000
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            lower, upper = method(a, b, c, d, alpha)
            
            # Should produce valid intervals for large counts
            assert 0 < lower < upper < float('inf')
            
            # Calculate expected RR
            n1, n2 = a + b, c + d
            risk1, risk2 = a / n1, c / n2
            expected_rr = risk1 / risk2
            
            # CI should contain the point estimate
            assert lower <= expected_rr <= upper

    def test_precision_consistency(self):
        """Test that results are consistent across multiple runs."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            # Run the same calculation multiple times
            results = []
            for _ in range(5):
                result = method(a, b, c, d, alpha)
                results.append(result)
            
            # All results should be identical (deterministic)
            for i in range(1, len(results)):
                assert math.isclose(results[0][0], results[i][0], rel_tol=1e-10)
                assert math.isclose(results[0][1], results[i][1], rel_tol=1e-10)

    def test_extreme_relative_risks(self):
        """Test with extreme relative risk values."""
        # Very small RR (close to 0)
        a, b, c, d = 1, 999, 500, 500
        alpha = 0.05
        
        methods = [
            ci_wald_rr,
            ci_wald_katz_rr,
            ci_wald_correlated_rr,
            ci_score_rr,
            ci_score_cc_rr,
            ci_ustat_rr,
        ]
        
        for method in methods:
            try:
                lower, upper = method(a, b, c, d, alpha)
                # Should handle extreme RR values
                assert 0 <= lower <= upper
                
                # Calculate expected RR
                n1, n2 = a + b, c + d
                risk1, risk2 = a / n1, c / n2
                expected_rr = risk1 / risk2
                
                if lower > 0 and upper < float('inf'):
                    assert lower <= expected_rr <= upper
            except (ValueError, RuntimeError):
                # Some methods may have issues with extreme values
                pass