"""Unit tests for relative risk confidence interval methods."""

import pytest
import math
import numpy as np
from scipy import stats
from typing import Tuple, Dict

# Import the module to test
from relative_risk import (
    calculate_relative_risk,
    batch_calculate_relative_risks,
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr,
    compute_all_rr_cis,
    ci_score_rr_batch,
    ci_score_cc_rr_batch,
    ci_ustat_rr_batch,
    ci_wald_rr_batch,
    ci_wald_katz_rr_batch,
    ci_wald_correlated_rr_batch,
    compute_all_rr_cis_batch
)


@pytest.mark.fast
class TestRelativeRiskPointEstimates:
    """Tests for relative risk point estimate calculations."""

    def test_basic_calculation(self):
        """Test basic relative risk calculation."""
        # RR = (a/(a+b)) / (c/(c+d))
        a, b, c, d = 15, 5, 10, 10
        expected_rr = (15/20) / (10/20) # = 0.75 / 0.5 = 1.5

        result = calculate_relative_risk(a, b, c, d)
        assert math.isclose(result, expected_rr)

    def test_zero_cells(self):
        """Test relative risk calculation with zero cells."""
        # Zero in exposed outcome cell
        assert calculate_relative_risk(0, 10, 5, 5) == 0.0

        # Zero in unexposed outcome cell (RR = infinity)
        assert calculate_relative_risk(5, 5, 0, 10) == float('inf')

        # Zero in both outcome cells (RR = 1.0, no effect)
        assert calculate_relative_risk(0, 10, 0, 10) == 1.0

        # Empty table
        assert calculate_relative_risk(0, 0, 0, 0) == 1.0

    def test_batch_calculation(self):
        """Test batch calculation of relative risks."""
        tables = [
            (15, 5, 10, 10),  # RR = 1.5
            (0, 10, 5, 5),    # RR = 0.0
            (5, 5, 0, 10),    # RR = inf
        ]

        expected = [1.5, 0.0, float('inf')]
        results = batch_calculate_relative_risks(tables)

        assert len(results) == len(expected)
        for res, exp in zip(results, expected):
            if math.isfinite(exp):
                assert math.isclose(res, exp)
            else:
                assert res == exp


@pytest.mark.fast
class TestWaldConfidenceIntervals:
    """Tests for Wald confidence interval methods."""

    def test_wald_ci(self):
        """Test basic Wald confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        lower, upper = ci_wald_rr(a, b, c, d, alpha)

        # Verify the point estimate is within the interval
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper

        # Verify interval width is reasonable
        assert 0 < lower < upper < float('inf')

    def test_wald_katz_ci(self):
        """Test Katz-adjusted Wald confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        lower, upper = ci_wald_katz_rr(a, b, c, d, alpha)

        # Verify the point estimate is within the interval
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper

        # Verify interval width is reasonable
        assert 0 < lower < upper < float('inf')

    def test_wald_correlated_ci(self):
        """Test correlation-adjusted Wald confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        lower, upper = ci_wald_correlated_rr(a, b, c, d, alpha)

        # Verify the point estimate is within the interval
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper

        # Verify interval width is reasonable
        assert 0 < lower < upper < float('inf')

    def test_zero_cells(self):
        """Test Wald methods with zero cells."""
        # Zero in exposed outcome cell
        lower, upper = ci_wald_rr(0, 10, 5, 5, 0.05)
        assert lower == 0.0 or math.isclose(lower, 0.0, abs_tol=1e-10)
        assert upper < float('inf')

        # Zero in unexposed outcome cell
        lower, upper = ci_wald_rr(5, 5, 0, 10, 0.05)
        assert lower > 0.0
        assert upper == float('inf')


@pytest.mark.fast
class TestScoreConfidenceIntervals:
    """Tests for score-based confidence interval methods."""

    def test_score_ci(self):
        """Test basic score confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        lower, upper = ci_score_rr(a, b, c, d, alpha)

        # Verify the point estimate is within the interval
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper

        # Verify interval width is reasonable
        assert 0 < lower < upper < float('inf')

    def test_score_cc_ci(self):
        """Test continuity-corrected score confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        # Test with default delta (4.0)
        lower, upper = ci_score_cc_rr(a, b, c, d, alpha=alpha)
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper
        assert 0 < lower < upper < float('inf')

        # Test with different delta values
        for delta in [2.0, 4.0, 8.0]:
            lower, upper = ci_score_cc_rr(a, b, c, d, delta=delta, alpha=alpha)
            assert lower <= rr <= upper
            assert 0 < lower < upper < float('inf')

    def test_small_samples(self):
        """Test score methods with small sample sizes."""
        a, b, c, d = 3, 7, 2, 8
        alpha = 0.05

        # Standard score method
        lower1, upper1 = ci_score_rr(a, b, c, d, alpha)

        # Continuity-corrected score method
        lower2, upper2 = ci_score_cc_rr(a, b, c, d, alpha=alpha)

        # Continuity correction should generally give wider intervals for small samples
        assert lower2 <= lower1
        assert upper2 >= upper1

    def test_zero_cells(self):
        """Test score methods with zero cells."""
        # Zero in exposed outcome cell
        lower, upper = ci_score_rr(0, 10, 5, 5, 0.05)
        assert lower == 0.0
        assert upper < float('inf')

        # Zero in unexposed outcome cell
        lower, upper = ci_score_rr(5, 5, 0, 10, 0.05)
        assert lower > 0.0
        assert upper == float('inf')


@pytest.mark.fast
class TestUStatConfidenceIntervals:
    """Tests for U-statistic confidence interval method."""

    def test_ustat_ci(self):
        """Test U-statistic confidence interval calculation."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        lower, upper = ci_ustat_rr(a, b, c, d, alpha)

        # Verify the point estimate is within the interval
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper

        # Verify interval width is reasonable
        assert 0 < lower < upper < float('inf')

    def test_small_samples(self):
        """Test U-statistic method with small sample sizes."""
        a, b, c, d = 3, 7, 2, 8
        alpha = 0.05

        lower, upper = ci_ustat_rr(a, b, c, d, alpha)

        # For small samples, U-statistic should give valid intervals
        rr = calculate_relative_risk(a, b, c, d)
        assert lower <= rr <= upper
        assert 0 < lower < upper < float('inf')


@pytest.mark.fast
class TestConvenienceFunction:
    """Tests for the convenience function that computes all CI methods."""

    def test_compute_all_cis(self):
        """Test computing confidence intervals using all methods."""
        a, b, c, d = 15, 5, 10, 10
        alpha = 0.05

        results = compute_all_rr_cis(a, b, c, d, alpha)

        # Check that all expected methods are included
        expected_methods = [
            "point_estimate", "wald", "wald_katz", "wald_correlated", 
            "score", "score_cc", "ustat"
        ]
        for method in expected_methods:
            assert method in results

        # Check that point estimate is correct
        rr = calculate_relative_risk(a, b, c, d)
        assert math.isclose(results["point_estimate"], rr)

        # Check that all intervals contain the point estimate
        for method in expected_methods[1:]:  # Skip point_estimate
            lower, upper = results[method]
            assert lower <= rr <= upper

    def test_edge_cases(self):
        """Test convenience function with edge cases."""
        # Zero cells
        results = compute_all_rr_cis(0, 10, 5, 5, 0.05)
        assert results["point_estimate"] == 0.0

        # All methods should return valid intervals
        for method in ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"]:
            lower, upper = results[method]
            assert 0.0 <= lower <= upper  # Lower bound could be 0


@pytest.mark.fast
class TestBatchProcessing:
    """Tests for batch processing functions."""

    def test_batch_functions(self):
        """Test all batch processing functions."""
        tables = [
            (15, 5, 10, 10),  # Standard case
            (3, 7, 2, 8),     # Small sample
            (0, 10, 5, 5),    # Zero in exposed outcome
        ]
        alpha = 0.05

        # Test each batch function
        batch_funcs = [
            ci_wald_rr_batch,
            ci_wald_katz_rr_batch,
            ci_wald_correlated_rr_batch,
            ci_score_rr_batch,
            ci_score_cc_rr_batch,
            ci_ustat_rr_batch,
        ]

        for func in batch_funcs:
            results = func(tables, alpha)

            # Check that we got the right number of results
            assert len(results) == len(tables)

            # Check that each result is a valid confidence interval
            for (a, b, c, d), (lower, upper) in zip(tables, results):
                rr = calculate_relative_risk(a, b, c, d)
                if math.isfinite(rr) and rr > 0:
                    assert lower <= rr <= upper
                elif rr == 0:
                    assert lower == 0.0 or math.isclose(lower, 0.0, abs_tol=1e-10)
                # Note: If rr is inf, we don't check that it's in the interval

    def test_compute_all_batch(self):
        """Test batch computation of all confidence interval methods."""
        tables = [
            (15, 5, 10, 10),  # Standard case
            (3, 7, 2, 8),     # Small sample
        ]
        alpha = 0.05

        results = compute_all_rr_cis_batch(tables, alpha)

        # Check that we got the right number of results
        assert len(results) == len(tables)

        # Check that each result contains all methods
        expected_methods = [
            "point_estimate", "wald", "wald_katz", "wald_correlated", 
            "score", "score_cc", "ustat"
        ]

        for result in results:
            for method in expected_methods:
                assert method in result


@pytest.mark.slow
class TestNumericalComparisons:
    """Numerical validation of confidence interval methods."""

    def test_method_comparisons(self):
        """Compare different confidence interval methods for consistency."""
        # Define test cases with known properties
        test_cases = [
            # (a, b, c, d, description)
            (20, 80, 10, 90, "Standard case"),
            (5, 95, 2, 98, "Small numerators"),
            (50, 50, 25, 75, "Different denominators"),
            (1, 99, 1, 99, "Equal risks, very small"),
        ]

        alpha = 0.05
        for a, b, c, d, desc in test_cases:
            # Compute relative risk
            rr = calculate_relative_risk(a, b, c, d)

            # Compute CIs using all methods
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
                results[name] = method(a, b, c, d, alpha)

            # Check that all intervals contain the point estimate
            for name, (lower, upper) in results.items():
                assert lower <= rr <= upper, f"Method {name} failed for case: {desc}"

            # Typically, score methods should be narrower than Wald for larger samples
            # and U-statistic methods might be wider for small samples
            if a + b + c + d >= 200:  # Large sample
                assert results["score"][1] - results["score"][0] <= results["wald"][1] - results["wald"][0]

            # Continuity correction should widen the interval compared to uncorrected score
            assert results["score_cc"][0] <= results["score"][0]
            assert results["score_cc"][1] >= results["score"][1]

    def test_edge_case_handling(self):
        """Test handling of edge cases across methods."""
        # Edge cases
        edge_cases = [
            # (a, b, c, d, description)
            (0, 100, 10, 90, "Zero in exposed outcome"),
            (10, 90, 0, 100, "Zero in unexposed outcome"),
            (0, 100, 0, 100, "Zero in both outcomes"),
            (100, 0, 10, 90, "Perfect exposure outcome"),
            (10, 90, 100, 0, "Perfect unexposure outcome"),
        ]

        alpha = 0.05
        for a, b, c, d, desc in edge_cases:
            # Compute relative risk
            rr = calculate_relative_risk(a, b, c, d)

            # Compute CIs using all methods
            all_cis = compute_all_rr_cis(a, b, c, d, alpha)

            # Check specific properties based on the edge case
            if a == 0 and c > 0:  # Zero in exposed outcome
                for method in ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"]:
                    lower, upper = all_cis[method]
                    assert lower == 0.0 or math.isclose(lower, 0.0, abs_tol=1e-10)
                    assert upper < float('inf')

            elif c == 0 and a > 0:  # Zero in unexposed outcome
                for method in ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"]:
                    lower, upper = all_cis[method]
                    assert lower > 0.0
                    assert upper == float('inf') or math.isnan(upper)
