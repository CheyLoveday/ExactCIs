"""
Unit tests for exactcis.utils.inversion module.

Tests the confidence interval inversion algorithms on synthetic test functions
with known properties, ensuring correct behavior for finite and infinite bounds,
two-sided intervals, and edge cases.
"""

import math
import pytest
from typing import Callable

from exactcis.utils.inversion import (
    handle_infinite_bounds,
    invert_bound,
    invert_two_sided_ci,
    validate_ci_result,
    invert_score_test_rr,
    score_test_wrapper,
    InversionDiagnostics
)


class TestHandleInfiniteBounds:
    """Tests for handle_infinite_bounds function."""
    
    def test_finite_bounds_unchanged(self):
        """Test that finite bounds are unchanged when not at domain limits."""
        test_func = lambda x: 0.1  # Constant function
        bounds = (1.0, 10.0)
        domain = (0.001, 1000.0)
        
        result = handle_infinite_bounds(bounds, test_func, alpha=0.05, domain=domain)
        
        assert result == bounds
    
    def test_lower_infinite_bound_detection(self):
        """Test detection of legitimate infinite lower bound."""
        # Function that stays above alpha near lower domain boundary
        def test_func(x):
            return 0.1  # Always above alpha=0.05
        
        bounds = (1e-12, 10.0)  # Lower bound at domain limit
        domain = (1e-12, 1000.0)
        
        lower, upper = handle_infinite_bounds(bounds, test_func, alpha=0.05, domain=domain)
        
        assert lower == 0.0  # Infinite lower bound
        assert upper == 10.0  # Unchanged
    
    def test_upper_infinite_bound_detection(self):
        """Test detection of legitimate infinite upper bound."""
        # Function that stays above alpha near upper domain boundary
        def test_func(x):
            return 0.1  # Always above alpha=0.05
        
        bounds = (1.0, 1000.0)  # Upper bound at domain limit
        domain = (1e-3, 1000.0)
        
        lower, upper = handle_infinite_bounds(bounds, test_func, alpha=0.05, domain=domain)
        
        assert lower == 1.0  # Unchanged
        assert math.isinf(upper)  # Infinite upper bound
    
    def test_no_infinite_when_function_drops(self):
        """Test that bounds stay finite when function drops below alpha."""
        # Function that drops below alpha at boundaries
        def test_func(x):
            return 0.01  # Below alpha=0.05
        
        bounds = (1e-12, 1000.0)  # Both at domain limits
        domain = (1e-12, 1000.0)
        
        result = handle_infinite_bounds(bounds, test_func, alpha=0.05, domain=domain)
        
        assert result == bounds  # Should remain finite


class TestInvertBound:
    """Tests for invert_bound function."""
    
    def simple_decreasing_test(self, x: float) -> float:
        """Simple decreasing test function with known crossing."""
        return math.exp(-x) * 2  # Crosses 0.05 at x = ln(40) ≈ 3.69
    
    def simple_increasing_test(self, x: float) -> float:
        """Simple increasing test function."""
        return 1 - math.exp(-x/2)  # Asymptotically approaches 1
    
    def test_lower_bound_inversion(self):
        """Test finding lower confidence bound."""
        bound, diag = invert_bound(
            self.simple_decreasing_test, is_lower=True, alpha=0.05,
            theta_hat=1.0, domain=(0.1, 20.0)
        )
        
        # Should find bound near ln(40) ≈ 3.69
        expected = math.log(40)
        assert abs(bound - expected) < 0.01
        assert diag.converged
        assert diag.function_calls > 0
    
    def test_upper_bound_inversion(self):
        """Test finding upper confidence bound."""
        bound, diag = invert_bound(
            self.simple_increasing_test, is_lower=False, alpha=0.95,
            theta_hat=5.0, domain=(0.1, 50.0)
        )
        
        # For this function, should find finite upper bound
        assert math.isfinite(bound)
        assert diag.function_calls > 0
    
    def test_infinite_lower_bound(self):
        """Test detection of infinite lower bound."""
        # Function always above alpha
        always_high = lambda x: 0.1
        
        bound, diag = invert_bound(
            always_high, is_lower=True, alpha=0.05,
            theta_hat=1.0, domain=(1e-6, 10.0)
        )
        
        assert bound == 0.0 or bound <= 1e-6  # Should be infinite or at domain boundary
        assert not diag.converged  # No sign change found
    
    def test_infinite_upper_bound(self):
        """Test detection of infinite upper bound."""
        # Function always above alpha
        always_high = lambda x: 0.1
        
        bound, diag = invert_bound(
            always_high, is_lower=False, alpha=0.05,
            theta_hat=1.0, domain=(1e-3, 1e6)
        )
        
        # Should hit domain boundary or be infinite
        assert math.isinf(bound) or bound >= 1e6 - 1  # Allow for small numerical error
        assert not diag.converged  # No sign change found
    
    def test_function_evaluation_error(self):
        """Test handling of function evaluation errors."""
        def problematic_func(x):
            if x < 1.0:
                raise ValueError("Invalid evaluation")
            return x - 2
        
        bound, diag = invert_bound(
            problematic_func, is_lower=True, alpha=0.0,
            theta_hat=2.0, domain=(1.5, 10.0)  # Start in valid domain
        )
        
        # Should handle error gracefully and find the root
        assert math.isfinite(bound)
        assert abs(bound - 2.0) < 0.1  # Should find root near x=2


class TestInvertTwoSidedCi:
    """Tests for invert_two_sided_ci function."""
    
    def symmetric_test_func(self, x: float) -> float:
        """Symmetric test function with known CI."""
        # Function that has minimum at x=5, crosses alpha=0.05 at x=3 and x=7
        return 0.01 + 0.04 * ((x - 5) ** 2) / 4
    
    def test_symmetric_ci_inversion(self):
        """Test inversion of symmetric confidence interval."""
        (lower, upper), diag = invert_two_sided_ci(
            self.symmetric_test_func, alpha=0.05, theta_hat=5.0,
            domain=(0.1, 20.0)
        )
        
        # Should find bounds around x=3 and x=7
        assert abs(lower - 3.0) < 0.1
        assert abs(upper - 7.0) < 0.1
        assert lower < upper
        assert diag.total_function_calls > 0
        assert diag.lower_bound_diag is not None
        assert diag.upper_bound_diag is not None
    
    def test_asymmetric_ci_inversion(self):
        """Test inversion of asymmetric confidence interval."""
        # Asymmetric function
        def asymmetric_func(x):
            return 0.01 + 0.1 / (1 + math.exp(-(x - 3)))
        
        (lower, upper), diag = invert_two_sided_ci(
            asymmetric_func, alpha=0.05, theta_hat=2.0,
            domain=(0.1, 10.0)
        )
        
        assert lower < upper
        assert math.isfinite(lower) and math.isfinite(upper)
    
    def test_infinite_bounds_handling(self):
        """Test handling of infinite bounds in two-sided inversion."""
        # Function that never crosses alpha in domain
        always_high = lambda x: 0.1  # Always above alpha=0.05
        
        (lower, upper), diag = invert_two_sided_ci(
            always_high, alpha=0.05, theta_hat=1.0,
            domain=(1e-6, 1e6), handle_infinite=True
        )
        
        # Should detect infinite bounds
        assert lower == 0.0 or not math.isfinite(lower)
        assert math.isinf(upper) or upper >= 1e6
        assert diag.infinite_lower or diag.infinite_upper
    
    def test_no_infinite_handling(self):
        """Test two-sided inversion without infinite bounds handling."""
        always_high = lambda x: 0.1
        
        (lower, upper), diag = invert_two_sided_ci(
            always_high, alpha=0.05, theta_hat=1.0,
            domain=(1e-3, 1e3), handle_infinite=False
        )
        
        # Should return domain boundaries
        assert abs(lower - 1e-3) < 1e-6 or abs(upper - 1e3) < 1e-3
        assert not diag.infinite_lower
        assert not diag.infinite_upper
    
    def test_default_theta_hat(self):
        """Test two-sided inversion with default theta_hat."""
        (lower, upper), diag = invert_two_sided_ci(
            self.symmetric_test_func, alpha=0.05,
            theta_hat=None,  # Should use geometric mean
            domain=(1.0, 100.0)
        )
        
        assert lower < upper
        assert diag.total_function_calls > 0


class TestValidateCiResult:
    """Tests for validate_ci_result function."""
    
    def test_valid_finite_bounds(self):
        """Test validation of valid finite bounds."""
        bounds = (2.5, 7.5)
        
        result = validate_ci_result(bounds, alpha=0.05)
        
        assert result['valid_bounds'] is True
        assert result['lower_finite'] is True
        assert result['upper_finite'] is True
        assert result['properly_ordered'] is True
        assert len(result['messages']) == 0
    
    def test_invalid_bound_order(self):
        """Test validation with incorrectly ordered bounds."""
        bounds = (10.0, 5.0)  # Upper < Lower
        
        result = validate_ci_result(bounds, alpha=0.05)
        
        assert result['valid_bounds'] is False
        assert result['properly_ordered'] is False
        assert any("Lower bound exceeds upper bound" in msg for msg in result['messages'])
    
    def test_nan_bounds(self):
        """Test validation with NaN bounds."""
        bounds = (float('nan'), 5.0)
        
        result = validate_ci_result(bounds, alpha=0.05)
        
        assert result['valid_bounds'] is False
        assert any("NaN values" in msg for msg in result['messages'])
    
    def test_infinite_bounds_valid(self):
        """Test validation accepts infinite bounds when properly ordered."""
        bounds = (0.0, float('inf'))
        
        result = validate_ci_result(bounds, alpha=0.05)
        
        assert result['valid_bounds'] is True
        assert result['lower_finite'] is True  # 0.0 is finite
        assert result['upper_finite'] is False  # inf is not finite
        assert result['properly_ordered'] is True
    
    def test_validation_with_test_function(self):
        """Test validation using test function evaluation."""
        bounds = (3.0, 7.0)
        
        # Test function that returns alpha at the bounds
        def test_func(x):
            return 0.05  # Exact alpha
        
        result = validate_ci_result(bounds, alpha=0.05, test_func=test_func)
        
        assert result['valid_bounds'] is True
        assert len(result['messages']) == 0  # Should pass validation
    
    def test_validation_with_inaccurate_bounds(self):
        """Test validation with bounds that don't satisfy test function."""
        bounds = (3.0, 7.0)
        
        # Test function that returns different value at bounds
        def test_func(x):
            return 0.1  # Different from alpha=0.05
        
        result = validate_ci_result(bounds, alpha=0.05, test_func=test_func, tolerance=0.01)
        
        # Should generate warning messages
        assert len(result['messages']) > 0
        assert any("test: p=" in msg for msg in result['messages'])


class TestScoreTestIntegration:
    """Tests for score test specific inversion functions."""
    
    def test_score_test_wrapper(self):
        """Test basic score test wrapper functionality."""
        data = (10, 40, 15, 35)  # Sample 2x2 table
        
        result = score_test_wrapper(theta=1.0, data=data)
        
        assert math.isfinite(result)
        assert result >= 0.0  # Score statistic should be non-negative
    
    def test_score_test_wrapper_edge_cases(self):
        """Test score test wrapper with edge cases."""
        # Zero margin
        data_zero = (0, 0, 15, 35)
        result_zero = score_test_wrapper(theta=1.0, data=data_zero)
        assert math.isinf(result_zero)
        
        # Zero rates
        data_zero_rate = (0, 50, 15, 35)
        result_rate = score_test_wrapper(theta=1.0, data=data_zero_rate)
        assert math.isinf(result_rate)
    
    def test_invert_score_test_rr_basic(self):
        """Test basic score test inversion for relative risk."""
        data = (20, 30, 25, 25)  # Balanced table
        
        (lower, upper), diag = invert_score_test_rr(
            lambda theta, data, corr: score_test_wrapper(theta, data, corr),
            alpha=0.05, data=data
        )
        
        # Should return reasonable bounds
        assert lower < upper
        assert lower > 0.0  # RR must be positive
        assert math.isfinite(lower) and math.isfinite(upper)
        assert diag.total_function_calls > 0


class TestInversionDiagnostics:
    """Tests for InversionDiagnostics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of inversion diagnostics."""
        diag = InversionDiagnostics()
        
        assert diag.lower_bound_diag is None
        assert diag.upper_bound_diag is None
        assert diag.infinite_lower is False
        assert diag.infinite_upper is False
        assert diag.plateau_detected is False
        assert diag.total_function_calls == 0
        assert diag.inversion_method == "unknown"
    
    def test_custom_initialization(self):
        """Test custom initialization of inversion diagnostics."""
        from exactcis.utils.solvers import SolverDiagnostics
        
        lower_diag = SolverDiagnostics(iterations=5, converged=True)
        
        diag = InversionDiagnostics(
            lower_bound_diag=lower_diag,
            infinite_upper=True,
            total_function_calls=25,
            inversion_method="test_method"
        )
        
        assert diag.lower_bound_diag == lower_diag
        assert diag.infinite_upper is True
        assert diag.total_function_calls == 25
        assert diag.inversion_method == "test_method"


# Integration tests
class TestInversionIntegration:
    """Integration tests for complete inversion workflows."""
    
    def test_complete_inversion_workflow(self):
        """Test complete workflow from test function to validated CI."""
        # Create test function with known properties
        def chi_square_like(x):
            # Function that behaves like a chi-square test
            stat = ((x - 2) ** 2) / 2  # Parabola with minimum at x=2
            return stat
        
        # Invert to find CI where function equals critical value
        (lower, upper), diag = invert_two_sided_ci(
            chi_square_like, alpha=3.84,  # Chi-square critical value
            theta_hat=2.0, domain=(0.1, 10.0)
        )
        
        # Should find bounds around x = 2 ± sqrt(2*3.84) = 2 ± 2.77
        expected_lower = 2.0 - math.sqrt(2 * 3.84)
        expected_upper = 2.0 + math.sqrt(2 * 3.84)
        
        assert abs(lower - expected_lower) < 0.1
        assert abs(upper - expected_upper) < 0.1
        
        # Validate the result
        validation = validate_ci_result((lower, upper), alpha=3.84, 
                                      test_func=chi_square_like, tolerance=0.1)
        assert validation['valid_bounds'] is True
    
    def test_robust_error_handling(self):
        """Test robust handling of various error conditions."""
        def problematic_func(x):
            if x < 0.5:
                raise ValueError("Domain error")
            elif x > 20:
                return float('nan')
            else:
                return abs(x - 5) * 0.01
        
        # Should handle errors gracefully
        (lower, upper), diag = invert_two_sided_ci(
            problematic_func, alpha=0.05, theta_hat=5.0,
            domain=(0.1, 50.0)
        )
        
        # Should return some result even with problematic function
        assert math.isfinite(lower) and math.isfinite(upper)
        assert lower <= upper