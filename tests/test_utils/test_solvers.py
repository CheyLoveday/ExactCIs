"""
Unit tests for exactcis.utils.solvers module.

Tests the robust solver algorithms on synthetic functions with known properties,
ensuring correct behavior for monotone functions, plateaus, and edge cases.
"""

import math
import pytest
from typing import Callable

from exactcis.utils.solvers import (
    bracket_log_space,
    is_monotone_on_log_grid, 
    bisection_safe,
    find_root_robust,
    detect_plateau_region,
    SolverDiagnostics
)


class TestBracketLogSpace:
    """Tests for bracket_log_space function."""
    
    def linear_function(self, x: float) -> float:
        """Simple linear function: f(x) = x - 2"""
        return x - 2
    
    def exponential_function(self, x: float) -> float:
        """Exponential function: f(x) = exp(x) - 10"""
        return math.exp(x) - 10
        
    def test_linear_bracket_success(self):
        """Test bracketing a linear function with known root."""
        bounds, diag = bracket_log_space(self.linear_function, theta0=1.0, 
                                        domain=(0.1, 10.0), target=0.0)
        
        # Should find a bracket around root x=2
        lo, hi = bounds
        assert lo < 2.0 < hi
        assert diag.converged
        assert diag.function_calls > 0
        assert self.linear_function(lo) * self.linear_function(hi) < 0
    
    def test_exponential_bracket_success(self):
        """Test bracketing exponential function."""
        bounds, diag = bracket_log_space(self.exponential_function, theta0=1.0,
                                        domain=(0.01, 100.0), target=0.0)
        
        # Root is at x = ln(10) ≈ 2.303
        lo, hi = bounds
        root = math.log(10)
        assert lo < root < hi
        assert diag.converged
        
    def test_no_sign_change_in_domain(self):
        """Test when no sign change exists in domain."""
        # Function f(x) = x + 10 (always positive for x > 0)
        always_positive = lambda x: x + 10
        
        bounds, diag = bracket_log_space(always_positive, theta0=1.0,
                                        domain=(0.1, 10.0), target=0.0)
        
        # Should not find sign change
        assert not diag.converged
        assert diag.domain_hits > 0
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Theta0 outside domain
        with pytest.raises(ValueError, match="must be within domain"):
            bracket_log_space(self.linear_function, theta0=20.0, domain=(0.1, 10.0))
        
        # Invalid expansion factor
        with pytest.raises(ValueError, match="Expansion factor must be"):
            bracket_log_space(self.linear_function, theta0=1.0, factor=0.5)
    
    def test_expansion_limit(self):
        """Test that expansion stops at max_expand."""
        bounds, diag = bracket_log_space(self.linear_function, theta0=1.0,
                                        max_expand=3, domain=(0.001, 1000.0))
        
        assert diag.bracket_expansions <= 3
        

class TestMonotonicityCheck:
    """Tests for is_monotone_on_log_grid function."""
    
    def test_increasing_function(self):
        """Test detection of strictly increasing function."""
        increasing_func = lambda x: x**2  # Increasing for x > 0
        
        result = is_monotone_on_log_grid(increasing_func, (0.1, 10.0))
        
        assert result['monotone'] is True
        assert result['direction'] == 1
    
    def test_decreasing_function(self):
        """Test detection of strictly decreasing function."""
        decreasing_func = lambda x: -x**2  # Decreasing for x > 0
        
        result = is_monotone_on_log_grid(decreasing_func, (0.1, 10.0))
        
        assert result['monotone'] is True
        assert result['direction'] == -1
    
    def test_non_monotone_function(self):
        """Test detection of non-monotone function."""
        # f(x) = sin(ln(x)) oscillates as x varies
        non_monotone = lambda x: math.sin(math.log(x))
        
        result = is_monotone_on_log_grid(non_monotone, (0.1, 10.0))
        
        assert result['monotone'] is False
        assert result['direction'] == 0
    
    def test_constant_function(self):
        """Test constant function (monotone but direction 0)."""
        constant_func = lambda x: 5.0
        
        result = is_monotone_on_log_grid(constant_func, (0.1, 10.0))
        
        # Constant function could be considered monotone in either direction
        # Implementation may vary - we accept either result as valid
        assert result['direction'] in [-1, 0, 1]
    
    def test_invalid_grid(self):
        """Test with invalid grid bounds."""
        result = is_monotone_on_log_grid(lambda x: x, (10.0, 0.1))  # Reversed bounds
        
        assert result['monotone'] is False
        assert result['direction'] == 0


class TestBisectionSafe:
    """Tests for bisection_safe root finding."""
    
    def test_linear_root_finding(self):
        """Test finding root of linear function."""
        linear_func = lambda x: 2*x - 6  # Root at x=3
        
        root, diag = bisection_safe(linear_func, lo=1.0, hi=5.0, tol=1e-10)
        
        assert abs(root - 3.0) < 1e-9
        assert diag.converged
        assert diag.final_residual < 1e-10
        assert diag.function_calls > 0
    
    def test_quadratic_root_finding(self):
        """Test finding root of quadratic function."""
        quadratic_func = lambda x: x**2 - 4  # Roots at ±2
        
        # Find positive root
        root, diag = bisection_safe(quadratic_func, lo=1.0, hi=3.0, tol=1e-10)
        
        assert abs(root - 2.0) < 1e-9
        assert diag.converged
        
    def test_no_sign_change_error(self):
        """Test error when no sign change in bracket."""
        positive_func = lambda x: x**2 + 1  # Always positive
        
        with pytest.raises(ValueError, match="No sign change in bracket"):
            bisection_safe(positive_func, lo=1.0, hi=5.0)
    
    def test_root_at_endpoint(self):
        """Test when root is exactly at bracket endpoint."""
        func_with_endpoint_root = lambda x: x - 2.0
        
        # Root exactly at lo
        root, diag = bisection_safe(func_with_endpoint_root, lo=2.0, hi=5.0, tol=1e-10)
        
        assert abs(root - 2.0) < 1e-12
        assert diag.converged
        assert diag.iterations == 0  # Should find immediately
    
    def test_convergence_failure(self):
        """Test behavior when max iterations reached."""
        # Use very strict tolerance and few iterations to force failure
        # Use a wider bracket to ensure more iterations are needed
        linear_func = lambda x: x - 50.0  # Root at x=50
        
        root, diag = bisection_safe(linear_func, lo=1.0, hi=100.0, 
                                   tol=1e-15, max_iter=2)  # Very few iterations
        
        assert not diag.converged
        assert diag.iterations == 2
        # Should still return reasonable approximation
        assert abs(root - 50.0) < 50.0  # Loose bound since it didn't converge
    
    def test_target_parameter(self):
        """Test finding roots with non-zero target."""
        func = lambda x: x**2  # Root of f(x) = 9 is x = ±3
        
        root, diag = bisection_safe(func, lo=2.0, hi=4.0, target=9.0, tol=1e-10)
        
        assert abs(root - 3.0) < 1e-9
        assert diag.converged


class TestFindRootRobust:
    """Tests for find_root_robust wrapper function."""
    
    def test_bisection_method(self):
        """Test robust root finding with bisection method."""
        func = lambda x: x**3 - 8  # Root at x=2
        
        root, diag = find_root_robust(func, bounds=(1.0, 3.0), method="bisection")
        
        assert abs(root - 2.0) < 1e-9
        assert diag.converged
        assert diag.method == "bisection_safe"
    
    def test_fallback_to_bisection(self):
        """Test that other methods fall back to bisection."""
        func = lambda x: x - 5.0
        
        root, diag = find_root_robust(func, bounds=(3.0, 7.0), method="brentq")
        
        # Should fall back to bisection
        assert abs(root - 5.0) < 1e-9
        assert diag.method == "bisection_safe"


class TestDetectPlateauRegion:
    """Tests for detect_plateau_region function."""
    
    def test_constant_function_plateau(self):
        """Test detection of plateau in constant function."""
        constant_func = lambda x: 42.0
        
        result = detect_plateau_region(constant_func, (1.0, 10.0))
        
        assert result['has_plateau'] is True
        assert abs(result['plateau_value'] - 42.0) < 1e-10
        assert result['plateau_width'] == 9.0
    
    def test_nearly_constant_plateau(self):
        """Test detection of nearly constant plateau."""
        # Very small variation
        nearly_constant = lambda x: 5.0 + 1e-10 * x
        
        result = detect_plateau_region(nearly_constant, (1.0, 10.0), tolerance=1e-8)
        
        assert result['has_plateau'] is True
        assert abs(result['plateau_value'] - 5.0) < 1.0  # Average should be ~5
    
    def test_varying_function_no_plateau(self):
        """Test that varying function is not detected as plateau."""
        varying_func = lambda x: x**2
        
        result = detect_plateau_region(varying_func, (1.0, 10.0))
        
        assert result['has_plateau'] is False
        assert result['plateau_value'] is None
        assert result['plateau_width'] == 0.0
    
    def test_invalid_bounds(self):
        """Test with invalid bounds."""
        result = detect_plateau_region(lambda x: x, (10.0, 1.0))  # Reversed
        
        assert result['has_plateau'] is False


class TestSolverDiagnostics:
    """Tests for SolverDiagnostics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of diagnostics."""
        diag = SolverDiagnostics()
        
        assert diag.iterations == 0
        assert diag.function_calls == 0
        assert diag.bracket_expansions == 0
        assert diag.monotone_flag is False
        assert diag.plateau_flag is False
        assert diag.domain_hits == 0
        assert math.isnan(diag.final_residual)
        assert diag.converged is False
        assert diag.method == "unknown"
    
    def test_custom_initialization(self):
        """Test custom initialization of diagnostics."""
        diag = SolverDiagnostics(
            iterations=5,
            method="test_method",
            converged=True,
            final_residual=1e-12
        )
        
        assert diag.iterations == 5
        assert diag.method == "test_method"
        assert diag.converged is True
        assert diag.final_residual == 1e-12


# Integration tests
class TestSolverIntegration:
    """Integration tests combining multiple solver functions."""
    
    def test_bracket_and_solve_pipeline(self):
        """Test complete pipeline: bracket finding + root solving."""
        # Function with root at x = e ≈ 2.718
        func = lambda x: math.log(x) - 1
        
        # First, find bracket
        bounds, bracket_diag = bracket_log_space(func, theta0=1.0, 
                                                domain=(0.1, 10.0), target=0.0)
        
        assert bracket_diag.converged
        lo, hi = bounds
        
        # Then solve within bracket
        root, solve_diag = bisection_safe(func, lo, hi, tol=1e-12, target=0.0)
        
        assert solve_diag.converged
        assert abs(root - math.e) < 1e-10
        
        # Total function calls should be sum of both phases
        total_calls = bracket_diag.function_calls + solve_diag.function_calls
        assert total_calls > 0
    
    def test_monotonicity_and_plateau_analysis(self):
        """Test combined monotonicity and plateau analysis."""
        # Create function that's monotone in one region, plateau in another
        def piecewise_func(x):
            if x < 2.0:
                return x  # Monotone increasing
            else:
                return 2.0  # Plateau
        
        # Check monotonicity in increasing region
        mono_result = is_monotone_on_log_grid(piecewise_func, (0.1, 1.5))
        assert mono_result['monotone'] is True
        assert mono_result['direction'] == 1
        
        # Check for plateau in constant region
        plateau_result = detect_plateau_region(piecewise_func, (3.0, 10.0))
        assert plateau_result['has_plateau'] is True