"""
Unit tests for exactcis.utils.estimates module.

Tests centralized statistical estimates, standard errors, and confidence
interval utilities for both odds ratios and relative risks.
"""

import math
import pytest
from typing import Tuple

from exactcis.utils.estimates import (
    # Point estimates
    compute_odds_ratio, compute_log_odds_ratio, 
    compute_relative_risk, compute_log_relative_risk, compute_risk_difference,
    
    # Component calculations
    compute_risks, compute_odds,
    
    # Standard errors
    se_log_or_wald, se_log_rr_wald, se_log_rr_katz, se_log_rr_correlated,
    
    # Combined calculations
    compute_log_or_with_se, compute_log_rr_with_se,
    
    # CI utilities
    ci_from_log_estimate_se, ci_transform_log_to_original,
    
    # Diagnostics
    estimate_precision_diagnostics
)


class TestPointEstimates:
    """Tests for point estimate calculations."""
    
    def test_compute_odds_ratio_normal(self):
        """Test odds ratio computation with normal inputs."""
        a, b, c, d = 12, 5, 8, 10
        
        # Without correction
        or_estimate = compute_odds_ratio(a, b, c, d, correction="none")
        expected = (12 * 10) / (5 * 8)  # 120/40 = 3.0
        assert abs(or_estimate - expected) < 1e-10
        
        # With Haldane correction
        or_haldane = compute_odds_ratio(a, b, c, d, correction="haldane")
        expected_hal = (12.5 * 10.5) / (5.5 * 8.5)  # 131.25/46.75 ≈ 2.81
        assert abs(or_haldane - expected_hal) < 1e-10
    
    def test_compute_odds_ratio_zero_cells(self):
        """Test odds ratio with zero cells."""
        # Zero in 'a' cell
        assert compute_odds_ratio(0, 5, 8, 10) == 0.0
        
        # Zero in 'b' cell  
        assert compute_odds_ratio(12, 0, 8, 10) == float('inf')
        
        # With Haldane correction
        or_corrected = compute_odds_ratio(0, 5, 8, 10, correction="haldane")
        expected = (0.5 * 10.5) / (5.5 * 8.5)
        assert abs(or_corrected - expected) < 1e-10
    
    def test_compute_log_odds_ratio(self):
        """Test log odds ratio computation."""
        a, b, c, d = 12, 5, 8, 10
        
        log_or = compute_log_odds_ratio(a, b, c, d)
        expected = math.log((12 * 10) / (5 * 8))
        assert abs(log_or - expected) < 1e-10
        
        # With zero cell (should handle gracefully)
        log_or_zero = compute_log_odds_ratio(0, 5, 8, 10)
        assert log_or_zero < -20  # Very negative (log of small number)
    
    def test_compute_relative_risk_normal(self):
        """Test relative risk computation with normal inputs."""
        a, b, c, d = 20, 30, 15, 35
        
        rr = compute_relative_risk(a, b, c, d, correction="none")
        # risk1 = 20/50 = 0.4, risk2 = 15/50 = 0.3, RR = 0.4/0.3 = 4/3
        expected = (20/50) / (15/50)
        assert abs(rr - expected) < 1e-10
    
    def test_compute_relative_risk_zero_cells(self):
        """Test relative risk with zero cells."""
        # Zero in 'a' cell
        rr = compute_relative_risk(0, 17, 8, 10, correction="none")
        expected = (0/17) / (8/18)  # 0 / (8/18) = 0
        assert rr == 0.0
        
        # Zero in 'c' cell
        rr_zero_c = compute_relative_risk(12, 5, 0, 18, correction="none")
        assert rr_zero_c == float('inf')
        
        # With Katz correction for zero in 'a'
        rr_katz = compute_relative_risk(0, 17, 8, 10, correction="katz")
        # a becomes 0.5, so risk1 = 0.5/17.5, risk2 = 8/18
        risk1_katz = 0.5 / 17.5
        risk2 = 8 / 18
        expected_katz = risk1_katz / risk2
        assert abs(rr_katz - expected_katz) < 1e-10
    
    def test_compute_risk_difference(self):
        """Test risk difference computation."""
        a, b, c, d = 20, 30, 15, 35
        
        rd = compute_risk_difference(a, b, c, d)
        risk1 = 20 / 50
        risk2 = 15 / 50  
        expected = risk1 - risk2  # 0.4 - 0.3 = 0.1
        assert abs(rd - expected) < 1e-10


class TestComponentCalculations:
    """Tests for risk and odds component calculations."""
    
    def test_compute_risks(self):
        """Test risk computation for both groups."""
        a, b, c, d = 12, 8, 6, 14
        
        risk1, risk2 = compute_risks(a, b, c, d)
        
        expected_risk1 = 12 / 20  # a/(a+b) = 12/20 = 0.6
        expected_risk2 = 6 / 20   # c/(c+d) = 6/20 = 0.3
        
        assert abs(risk1 - expected_risk1) < 1e-10
        assert abs(risk2 - expected_risk2) < 1e-10
    
    def test_compute_odds(self):
        """Test odds computation for both groups."""
        a, b, c, d = 12, 8, 6, 14
        
        odds1, odds2 = compute_odds(a, b, c, d)
        
        expected_odds1 = 12 / 8   # a/b = 1.5
        expected_odds2 = 6 / 14   # c/d = 6/14 = 3/7
        
        assert abs(odds1 - expected_odds1) < 1e-10
        assert abs(odds2 - expected_odds2) < 1e-10
    
    def test_compute_odds_zero_denominator(self):
        """Test odds computation with zero denominators."""
        # Zero in 'b' cell
        odds1, odds2 = compute_odds(12, 0, 6, 14)
        assert odds1 == 0.0  # Safe division default
        assert abs(odds2 - (6/14)) < 1e-10


class TestStandardErrors:
    """Tests for standard error calculations."""
    
    def test_se_log_or_wald(self):
        """Test Wald standard error for log odds ratio."""
        a, b, c, d = 12, 5, 8, 10
        
        # Without Haldane correction - wald_variance_or applies Haldane when haldane_corrected=False
        se = se_log_or_wald(a, b, c, d, haldane=False) 
        expected_false = math.sqrt(1/12.5 + 1/5.5 + 1/8.5 + 1/10.5)  # Auto Haldane applied
        assert abs(se - expected_false) < 1e-10
        
        # With Haldane=True - wald_variance_or assumes correction already applied, uses raw counts
        se_hal = se_log_or_wald(a, b, c, d, haldane=True)
        expected_true = math.sqrt(1/12 + 1/5 + 1/8 + 1/10)  # No correction
        assert abs(se_hal - expected_true) < 1e-10
    
    def test_se_log_rr_wald(self):
        """Test Wald standard error for log relative risk."""
        a, b, c, d = 12, 8, 6, 14
        
        # Test without correction
        se_no_corr = se_log_rr_wald(a, b, c, d, correction="none")
        from exactcis.utils.mathops import wald_variance_rr
        expected_var_no_corr = wald_variance_rr(a, b, c, d, method="standard")
        expected_se_no_corr = math.sqrt(expected_var_no_corr)
        assert abs(se_no_corr - expected_se_no_corr) < 1e-10
        
        # Test with continuity correction  
        se_with_corr = se_log_rr_wald(a, b, c, d, correction="continuity")
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        expected_var_corr = wald_variance_rr(a_c, b_c, c_c, d_c, method="standard")
        expected_se_corr = math.sqrt(expected_var_corr)
        assert abs(se_with_corr - expected_se_corr) < 1e-10
    
    def test_se_log_rr_katz(self):
        """Test Katz-adjusted standard error for log relative risk.""" 
        a, b, c, d = 0, 17, 8, 10  # Zero in 'a' cell
        
        se = se_log_rr_katz(a, b, c, d)
        # Katz method should adjust 'a' to 0.5
        a_adj = 0.5
        n1 = a_adj + b  # 17.5
        n2 = c + d      # 18
        expected = math.sqrt(b/(a_adj * n1) + d/(c * n2))
        assert abs(se - expected) < 1e-10
    
    def test_se_log_rr_correlated(self):
        """Test correlated standard error for matched pairs."""
        a, b, c, d = 12, 5, 8, 10
        
        se = se_log_rr_correlated(a, b, c, d)
        # With continuity correction
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        # Correlated variance: (b+d)/(a+c)^2
        expected = math.sqrt((b_c + d_c) / ((a_c + c_c) ** 2))
        assert abs(se - expected) < 1e-10


class TestCombinedCalculations:
    """Tests for combined estimate + standard error calculations."""
    
    def test_compute_log_or_with_se_haldane(self):
        """Test combined log OR and SE calculation with Haldane."""
        a, b, c, d = 12, 5, 8, 10
        
        log_or, se = compute_log_or_with_se(a, b, c, d, method="wald_haldane")
        
        # Check that results match individual calculations
        expected_log_or = compute_log_odds_ratio(a, b, c, d, correction="haldane")
        expected_se = se_log_or_wald(a, b, c, d, haldane=True)
        
        assert abs(log_or - expected_log_or) < 1e-10
        assert abs(se - expected_se) < 1e-10
    
    def test_compute_log_rr_with_se_standard(self):
        """Test combined log RR and SE calculation for standard method."""
        a, b, c, d = 20, 30, 15, 35
        
        log_rr, se = compute_log_rr_with_se(a, b, c, d, method="wald_standard")
        
        # Check that results match individual calculations  
        expected_log_rr = compute_log_relative_risk(a, b, c, d, correction="continuity")
        expected_se = se_log_rr_wald(a, b, c, d, correction="continuity")
        
        assert abs(log_rr - expected_log_rr) < 1e-10
        assert abs(se - expected_se) < 1e-10
    
    def test_compute_log_rr_with_se_katz(self):
        """Test combined log RR and SE calculation for Katz method."""
        a, b, c, d = 0, 17, 8, 10
        
        log_rr, se = compute_log_rr_with_se(a, b, c, d, method="wald_katz")
        
        expected_log_rr = compute_log_relative_risk(a, b, c, d, correction="katz")
        expected_se = se_log_rr_katz(a, b, c, d)
        
        assert abs(log_rr - expected_log_rr) < 1e-10
        assert abs(se - expected_se) < 1e-10
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method names raise errors."""
        a, b, c, d = 12, 5, 8, 10
        
        with pytest.raises(ValueError, match="Unknown method"):
            compute_log_or_with_se(a, b, c, d, method="invalid")
        
        with pytest.raises(ValueError, match="Unknown method"):
            compute_log_rr_with_se(a, b, c, d, method="invalid")


class TestConfidenceIntervalUtilities:
    """Tests for confidence interval utility functions."""
    
    def test_ci_from_log_estimate_se_normal(self):
        """Test CI calculation from log estimate and SE."""
        log_estimate = math.log(2.0)  # log(2)
        se = 0.5
        alpha = 0.05
        
        lower, upper = ci_from_log_estimate_se(log_estimate, se, alpha, "normal")
        
        # Should use normal distribution
        from scipy import stats
        z = stats.norm.ppf(0.975)  # 95% CI
        expected_lower = math.exp(log_estimate - z * se)
        expected_upper = math.exp(log_estimate + z * se)
        
        assert abs(lower - expected_lower) < 1e-10
        assert abs(upper - expected_upper) < 1e-10
    
    def test_ci_from_log_estimate_se_infinite_se(self):
        """Test CI with infinite standard error."""
        lower, upper = ci_from_log_estimate_se(0.0, float('inf'), 0.05)
        
        assert lower == 0.0
        assert upper == float('inf')
    
    def test_ci_transform_log_to_original(self):
        """Test transformation from log scale to original scale."""
        log_lower = math.log(1.5)
        log_upper = math.log(3.5)
        
        lower, upper = ci_transform_log_to_original(log_lower, log_upper)
        
        assert abs(lower - 1.5) < 1e-10
        assert abs(upper - 3.5) < 1e-10
    
    def test_ci_invalid_distribution(self):
        """Test error handling for invalid distribution."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            ci_from_log_estimate_se(0.0, 1.0, 0.05, "invalid")


class TestDiagnostics:
    """Tests for estimate precision diagnostic utilities."""
    
    def test_diagnostics_normal_table(self):
        """Test diagnostics for normal-sized table."""
        a, b, c, d = 20, 30, 15, 35  # Total n=100
        
        diag = estimate_precision_diagnostics(a, b, c, d)
        
        assert diag["total_sample_size"] == 100
        assert diag["group_sizes"] == (50, 50)
        assert not diag["has_zero_cells"]
        assert not diag["small_sample"]
        assert diag["asymptotic_adequate"]
        assert not diag["sparse_table"]
    
    def test_diagnostics_sparse_table(self):
        """Test diagnostics for sparse table with zeros.""" 
        a, b, c, d = 0, 5, 2, 8  # Zero in 'a', small counts
        
        diag = estimate_precision_diagnostics(a, b, c, d)
        
        assert diag["total_sample_size"] == 15
        assert diag["has_zero_cells"]
        assert diag["small_sample"]
        assert not diag["asymptotic_adequate"]
        assert diag["sparse_table"]
        
        # Should have method recommendations
        assert len(diag["method_recommendations"]) > 0
        assert any("exact" in rec for rec in diag["method_recommendations"])
    
    def test_diagnostics_imbalanced_groups(self):
        """Test diagnostics for extremely imbalanced group sizes."""
        a, b, c, d = 50, 5, 2, 1  # Very imbalanced: 55 vs 3
        
        diag = estimate_precision_diagnostics(a, b, c, d)
        
        assert diag["extreme_imbalance"]
        assert "imbalance" in " ".join(diag["method_recommendations"]).lower()


class TestEdgeCases:
    """Tests for edge case handling."""
    
    def test_all_zero_cells(self):
        """Test behavior with all zero cells (degenerate case)."""
        # This should be caught by validation - but validation may allow zero margins
        # Test that it returns a reasonable result (with Haldane correction)
        or_result = compute_odds_ratio(0, 0, 0, 0, correction="haldane")
        # With Haldane: (0.5 * 0.5) / (0.5 * 0.5) = 1.0
        assert abs(or_result - 1.0) < 1e-10
    
    def test_zero_margin_handling(self):
        """Test handling of zero margins."""
        # Zero margin in first row - validation should allow this with allow_zero_margins=True
        a, b, c, d = 0, 0, 5, 10
        
        # Should handle gracefully - since b=0 (no unexposed without outcome), OR should be infinite
        # This is epidemiologically correct: if no unexposed group lacks the outcome, OR is infinite
        or_result = compute_odds_ratio(a, b, c, d)
        assert or_result == float('inf')
    
    def test_very_small_counts(self):
        """Test behavior with very small non-zero counts."""
        a, b, c, d = 0.1, 0.1, 0.1, 0.1
        
        or_est = compute_odds_ratio(a, b, c, d, correction="none")
        assert or_est == 1.0  # (0.1 * 0.1) / (0.1 * 0.1) = 1
    
    def test_large_counts_numerical_stability(self):
        """Test numerical stability with large counts."""
        a, b, c, d = 1e6, 2e6, 5e5, 1e7
        
        or_est = compute_odds_ratio(a, b, c, d)
        se = se_log_or_wald(a, b, c, d, haldane=True)
        
        # Should produce finite, reasonable results
        assert math.isfinite(or_est)
        assert math.isfinite(se)
        assert or_est > 0
        assert se > 0


class TestIntegration:
    """Integration tests combining multiple estimate functions."""
    
    def test_complete_or_analysis_workflow(self):
        """Test complete workflow for odds ratio analysis."""
        a, b, c, d = 15, 10, 8, 20
        
        # Get diagnostics
        diag = estimate_precision_diagnostics(a, b, c, d)
        
        # Compute estimate and CI
        log_or, se = compute_log_or_with_se(a, b, c, d, method="wald_haldane")
        lower, upper = ci_from_log_estimate_se(log_or, se, alpha=0.05)
        
        # Check reasonableness
        assert math.isfinite(log_or)
        assert math.isfinite(se)
        assert 0 < lower < upper
        assert math.isfinite(upper)
        
        # Point estimate should be within CI
        or_point = math.exp(log_or)
        assert lower <= or_point <= upper
    
    def test_complete_rr_analysis_workflow(self):
        """Test complete workflow for relative risk analysis."""
        a, b, c, d = 25, 75, 15, 85
        
        # Get diagnostics
        diag = estimate_precision_diagnostics(a, b, c, d)
        
        # Compute estimate and CI
        log_rr, se = compute_log_rr_with_se(a, b, c, d, method="wald_standard")
        lower, upper = ci_from_log_estimate_se(log_rr, se, alpha=0.05)
        
        # Check reasonableness
        assert math.isfinite(log_rr)
        assert math.isfinite(se)
        assert 0 < lower < upper
        
        # Point estimate should be within CI
        rr_point = math.exp(log_rr)
        assert lower <= rr_point <= upper
    
    def test_zero_cell_workflow_comparison(self):
        """Test that different zero-cell methods give different but reasonable results."""
        a, b, c, d = 0, 15, 8, 12  # Zero in 'a'
        
        # Standard method (should handle via continuity correction)
        log_rr_std, se_std = compute_log_rr_with_se(a, b, c, d, method="wald_standard")
        
        # Katz method (should handle via Katz adjustment)
        log_rr_katz, se_katz = compute_log_rr_with_se(a, b, c, d, method="wald_katz")
        
        # Both should be finite and reasonable
        assert math.isfinite(log_rr_std)
        assert math.isfinite(se_std)
        assert math.isfinite(log_rr_katz)
        assert math.isfinite(se_katz)
        
        # They should be different (different correction methods)
        assert abs(log_rr_std - log_rr_katz) > 1e-6  # Meaningfully different
        assert abs(se_std - se_katz) > 1e-6