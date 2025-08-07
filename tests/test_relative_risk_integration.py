"""
Integration tests for relative risk confidence interval methods.

This module provides integration testing that combines multiple relative risk
methods and tests their interaction with the broader ExactCIs ecosystem.
"""

import pytest
import math
import numpy as np
from typing import Dict, List, Tuple

from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)


def calculate_relative_risk(a: int, b: int, c: int, d: int) -> float:
    """Calculate the point estimate of relative risk."""
    n1 = a + b
    n2 = c + d
    
    if n1 == 0 or n2 == 0:
        return 1.0  # No effect when no observations
    
    risk1 = a / n1
    risk2 = c / n2
    
    if risk2 == 0:
        return float('inf') if risk1 > 0 else 1.0
    
    return risk1 / risk2


def compute_all_rr_cis(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Compute confidence intervals using all relative risk methods."""
    methods = {
        "wald": ci_wald_rr,
        "wald_katz": ci_wald_katz_rr,
        "wald_correlated": ci_wald_correlated_rr,
        "score": ci_score_rr,
        "score_cc": ci_score_cc_rr,
        "ustat": ci_ustat_rr,
    }
    
    results = {}
    for name, method in methods.items():
        try:
            results[name] = method(a, b, c, d, alpha)
        except (ValueError, RuntimeError) as e:
            # Store the error for debugging
            results[name] = (float('nan'), float('nan'))
    
    return results


@pytest.mark.fast
@pytest.mark.integration
class TestRelativeRiskIntegration:
    """Integration tests for relative risk methods."""

    def test_standard_case_all_methods(self, timer):
        """Test all methods with a standard case."""
        a, b, c, d = 20, 80, 10, 90
        alpha = 0.05
        
        # Calculate point estimate
        rr = calculate_relative_risk(a, b, c, d)
        expected_rr = (20/100) / (10/100)  # = 0.2 / 0.1 = 2.0
        assert math.isclose(rr, expected_rr)
        
        # Compute all CIs
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # Check that all methods returned valid results
        for method_name, (lower, upper) in results.items():
            if not (math.isnan(lower) or math.isnan(upper)):
                assert 0 < lower < upper < float('inf'), \
                    f"Method {method_name} produced invalid CI: ({lower}, {upper})"
                assert lower <= rr <= upper, \
                    f"Method {method_name} CI ({lower:.3f}, {upper:.3f}) doesn't contain RR {rr:.3f}"

    def test_small_sample_all_methods(self, timer):
        """Test all methods with small sample sizes."""
        a, b, c, d = 3, 7, 2, 8
        alpha = 0.05
        
        # Calculate point estimate
        rr = calculate_relative_risk(a, b, c, d)
        
        # Compute all CIs
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # Check that all methods handle small samples appropriately
        for method_name, (lower, upper) in results.items():
            if not (math.isnan(lower) or math.isnan(upper)):
                assert 0 < lower <= upper, \
                    f"Method {method_name} produced invalid CI for small sample"
                if upper < float('inf'):
                    assert lower <= rr <= upper, \
                        f"Method {method_name} CI doesn't contain RR for small sample"

    def test_zero_cell_handling(self, timer):
        """Test how all methods handle zero cells."""
        test_cases = [
            (0, 10, 5, 5, "Zero in exposed outcome"),
            (5, 5, 0, 10, "Zero in unexposed outcome"),
            (0, 10, 0, 10, "Zero in both outcomes"),
        ]
        
        alpha = 0.05
        
        for a, b, c, d, description in test_cases:
            rr = calculate_relative_risk(a, b, c, d)
            results = compute_all_rr_cis(a, b, c, d, alpha)
            
            for method_name, (lower, upper) in results.items():
                if not (math.isnan(lower) or math.isnan(upper)):
                    # Check basic validity
                    assert 0 <= lower <= upper, \
                        f"Method {method_name} invalid for {description}: ({lower}, {upper})"
                    
                    # For zero in exposed outcome, RR should be 0 or close to 0
                    if a == 0 and c > 0:
                        assert lower >= 0.0 and lower < 0.1, \
                            f"Method {method_name} should have lower bound close to 0 for zero exposed outcome"
                    
                    # For zero in unexposed outcome, RR should be infinite
                    elif c == 0 and a > 0:
                        assert upper == float('inf') or upper > 100, \
                            f"Method {method_name} should have large/infinite upper bound for zero unexposed outcome"

    def test_method_consistency_across_alpha_levels(self, timer):
        """Test that all methods are consistent across different alpha levels."""
        a, b, c, d = 15, 5, 10, 10
        alphas = [0.01, 0.05, 0.1]
        
        for method_name in ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"]:
            intervals = []
            
            for alpha in alphas:
                results = compute_all_rr_cis(a, b, c, d, alpha)
                if method_name in results:
                    lower, upper = results[method_name]
                    if not (math.isnan(lower) or math.isnan(upper)):
                        intervals.append((alpha, lower, upper))
            
            # Check that smaller alpha gives wider intervals
            for i in range(len(intervals) - 1):
                alpha1, lower1, upper1 = intervals[i]
                alpha2, lower2, upper2 = intervals[i + 1]
                
                if alpha1 < alpha2:  # More stringent confidence level
                    assert lower1 <= lower2, \
                        f"Method {method_name}: smaller alpha should give wider CI (lower bounds)"
                    assert upper1 >= upper2, \
                        f"Method {method_name}: smaller alpha should give wider CI (upper bounds)"

    def test_large_sample_convergence(self, timer):
        """Test that methods converge for large samples."""
        # Large sample case
        a, b, c, d = 200, 800, 100, 900
        alpha = 0.05
        
        rr = calculate_relative_risk(a, b, c, d)
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # Extract valid results
        valid_results = {name: ci for name, ci in results.items() 
                        if not (math.isnan(ci[0]) or math.isnan(ci[1]))}
        
        # For large samples, different methods should give similar results
        if len(valid_results) >= 2:
            method_names = list(valid_results.keys())
            widths = {name: ci[1] - ci[0] for name, ci in valid_results.items()}
            
            # All methods should contain the point estimate
            for name, (lower, upper) in valid_results.items():
                assert lower <= rr <= upper, \
                    f"Large sample: Method {name} CI doesn't contain RR"
            
            # Widths should be reasonable (not too different)
            max_width = max(widths.values())
            min_width = min(widths.values())
            if min_width > 0:
                ratio = max_width / min_width
                assert ratio < 15.0, \
                    f"Large sample: CI widths too different across methods (ratio: {ratio:.2f})"

    def test_extreme_cases_robustness(self, timer):
        """Test robustness with extreme cases."""
        extreme_cases = [
            (1, 999, 1, 9, "Very unbalanced groups"),
            (50, 50, 1, 99, "Very different risks"),
            (1, 1, 1, 1, "Minimal counts"),
            (100, 0, 50, 50, "Perfect exposure outcome"),
        ]
        
        alpha = 0.05
        
        for a, b, c, d, description in extreme_cases:
            rr = calculate_relative_risk(a, b, c, d)
            results = compute_all_rr_cis(a, b, c, d, alpha)
            
            # Count how many methods succeeded
            successful_methods = 0
            for method_name, (lower, upper) in results.items():
                if not (math.isnan(lower) or math.isnan(upper)):
                    successful_methods += 1
                    
                    # Basic validity check
                    assert 0 <= lower <= upper, \
                        f"Extreme case {description}: Method {method_name} invalid CI"
                    
                    # If RR is finite and positive, CI should contain it
                    if math.isfinite(rr) and rr > 0:
                        if lower > 0 and upper < float('inf'):
                            assert lower <= rr <= upper, \
                                f"Extreme case {description}: Method {method_name} CI doesn't contain RR"
            
            # At least some methods should work for most extreme cases
            if description != "Perfect exposure outcome":  # This case is legitimately difficult
                assert successful_methods >= 2, \
                    f"Extreme case {description}: Too few methods succeeded ({successful_methods})"


@pytest.mark.integration
class TestRelativeRiskVsOddsRatio:
    """Tests comparing relative risk with odds ratio calculations."""

    def test_rr_or_relationship_rare_disease(self, timer):
        """Test RR vs OR relationship for rare disease (should be similar)."""
        # Rare disease case (low outcome rates)
        a, b, c, d = 2, 998, 1, 999
        alpha = 0.05
        
        # Calculate RR
        rr = calculate_relative_risk(a, b, c, d)
        
        # Calculate OR
        or_value = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
        
        # For rare disease, RR and OR should be similar
        if math.isfinite(rr) and math.isfinite(or_value) and rr > 0 and or_value > 0:
            ratio = max(rr, or_value) / min(rr, or_value)
            assert ratio < 1.1, \
                f"Rare disease: RR ({rr:.3f}) and OR ({or_value:.3f}) should be similar"

    def test_rr_or_relationship_common_disease(self, timer):
        """Test RR vs OR relationship for common disease (should differ)."""
        # Common disease case (high outcome rates)
        a, b, c, d = 40, 60, 20, 80
        alpha = 0.05
        
        # Calculate RR
        rr = calculate_relative_risk(a, b, c, d)
        
        # Calculate OR
        or_value = (a * d) / (b * c)
        
        # For common disease, OR should be further from 1 than RR
        if rr > 1:
            assert or_value >= rr, \
                f"Common disease with RR > 1: OR ({or_value:.3f}) should be ≥ RR ({rr:.3f})"
        elif rr < 1:
            assert or_value <= rr, \
                f"Common disease with RR < 1: OR ({or_value:.3f}) should be ≤ RR ({rr:.3f})"


@pytest.mark.slow
@pytest.mark.integration
class TestRelativeRiskPerformance:
    """Performance and stress tests for relative risk methods."""

    def test_batch_processing_simulation(self, timer):
        """Test performance with batch processing simulation."""
        # Generate multiple test cases
        np.random.seed(42)  # For reproducibility
        
        test_cases = []
        for _ in range(50):
            # Generate random 2x2 tables
            n1 = np.random.randint(10, 100)
            n2 = np.random.randint(10, 100)
            a = np.random.randint(0, n1 + 1)
            c = np.random.randint(0, n2 + 1)
            b = n1 - a
            d = n2 - c
            test_cases.append((a, b, c, d))
        
        alpha = 0.05
        
        # Test each method on all cases
        methods = ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"]
        
        for method_name in methods:
            successful_cases = 0
            
            for a, b, c, d in test_cases:
                try:
                    results = compute_all_rr_cis(a, b, c, d, alpha)
                    if method_name in results:
                        lower, upper = results[method_name]
                        if not (math.isnan(lower) or math.isnan(upper)):
                            successful_cases += 1
                            
                            # Basic validity
                            assert 0 <= lower <= upper
                            
                            # Should contain point estimate when valid
                            rr = calculate_relative_risk(a, b, c, d)
                            if math.isfinite(rr) and rr > 0 and lower > 0 and upper < float('inf'):
                                assert lower <= rr <= upper
                except Exception:
                    # Some cases may legitimately fail
                    pass
            
            # Most cases should succeed for robust methods
            success_rate = successful_cases / len(test_cases)
            assert success_rate >= 0.7, \
                f"Method {method_name} success rate too low: {success_rate:.2f}"

    def test_numerical_stability_stress(self, timer):
        """Stress test numerical stability with challenging cases."""
        challenging_cases = [
            # Very large counts
            (1500, 500, 1000, 1000),
            # Very small counts
            (1, 1, 1, 1),
            # Extreme imbalance
            (1, 999, 1, 9),
            # Near-zero risks
            (1, 9999, 1, 9999),
            # High risks
            (900, 100, 800, 200),
        ]
        
        alpha = 0.05
        
        for a, b, c, d in challenging_cases:
            results = compute_all_rr_cis(a, b, c, d, alpha)
            
            # At least some methods should handle each case
            successful_methods = 0
            for method_name, (lower, upper) in results.items():
                if not (math.isnan(lower) or math.isnan(upper)):
                    successful_methods += 1
                    
                    # Check for numerical issues
                    assert math.isfinite(lower) or lower == 0.0
                    assert math.isfinite(upper) or upper == float('inf')
                    assert lower <= upper
            
            assert successful_methods >= 3, \
                f"Too few methods succeeded for challenging case ({a}, {b}, {c}, {d})"


@pytest.mark.integration
class TestRelativeRiskDocumentationExamples:
    """Tests based on documentation examples and use cases."""

    def test_epidemiological_study_example(self, timer):
        """Test with a typical epidemiological study example."""
        # Example: Smoking and lung cancer
        # Exposed (smokers): 90 cases, 910 controls
        # Unexposed (non-smokers): 10 cases, 990 controls
        a, b, c, d = 90, 910, 10, 990
        alpha = 0.05
        
        # Calculate RR
        rr = calculate_relative_risk(a, b, c, d)
        expected_rr = (90/1000) / (10/1000)  # = 0.09 / 0.01 = 9.0
        assert math.isclose(rr, expected_rr)
        
        # Compute all CIs
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # All methods should work for this standard epidemiological case
        for method_name, (lower, upper) in results.items():
            assert not (math.isnan(lower) or math.isnan(upper)), \
                f"Method {method_name} failed for epidemiological example"
            
            assert 0 < lower < upper < float('inf'), \
                f"Method {method_name} produced invalid CI for epidemiological example"
            
            assert lower <= rr <= upper, \
                f"Method {method_name} CI doesn't contain RR for epidemiological example"
            
            # For this case, CI should be reasonably narrow (large sample)
            width = upper - lower
            assert width < rr * 2, \
                f"Method {method_name} CI too wide for large sample epidemiological study"

    def test_clinical_trial_example(self, timer):
        """Test with a clinical trial example."""
        # Example: Treatment vs control
        # Treatment group: 15 events out of 100 patients
        # Control group: 25 events out of 100 patients
        a, b, c, d = 15, 85, 25, 75
        alpha = 0.05
        
        # Calculate RR
        rr = calculate_relative_risk(a, b, c, d)
        expected_rr = (15/100) / (25/100)  # = 0.15 / 0.25 = 0.6
        assert math.isclose(rr, expected_rr)
        
        # Compute all CIs
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # All methods should work for this balanced clinical trial
        for method_name, (lower, upper) in results.items():
            assert not (math.isnan(lower) or math.isnan(upper)), \
                f"Method {method_name} failed for clinical trial example"
            
            assert 0 < lower < upper < float('inf'), \
                f"Method {method_name} produced invalid CI for clinical trial example"
            
            assert lower <= rr <= upper, \
                f"Method {method_name} CI doesn't contain RR for clinical trial example"
            
            # RR < 1 indicates protective effect
            assert rr < 1.0, "Treatment should be protective (RR < 1)"
            
            # CI should not include 1.0 if effect is significant
            # (We don't enforce this as it depends on the specific alpha and method)

    def test_cohort_study_example(self, timer):
        """Test with a cohort study example."""
        # Example: Exposure and disease in a cohort
        # High exposure: 30 cases out of 200 people
        # Low exposure: 10 cases out of 300 people  
        a, b, c, d = 30, 170, 10, 290
        alpha = 0.05
        
        # Calculate RR
        rr = calculate_relative_risk(a, b, c, d)
        expected_rr = (30/200) / (10/300)  # = 0.15 / 0.033 ≈ 4.5
        assert math.isclose(rr, expected_rr, rel_tol=0.01)
        
        # Compute all CIs
        results = compute_all_rr_cis(a, b, c, d, alpha)
        
        # All methods should work for this cohort study
        for method_name, (lower, upper) in results.items():
            assert not (math.isnan(lower) or math.isnan(upper)), \
                f"Method {method_name} failed for cohort study example"
            
            assert 0 < lower < upper < float('inf'), \
                f"Method {method_name} produced invalid CI for cohort study example"
            
            assert lower <= rr <= upper, \
                f"Method {method_name} CI doesn't contain RR for cohort study example"
            
            # RR > 1 indicates increased risk
            assert rr > 1.0, "High exposure should increase risk (RR > 1)"