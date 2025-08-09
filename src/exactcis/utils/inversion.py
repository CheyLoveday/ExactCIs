"""
Standardized confidence interval test inversion for ExactCIs.

This module provides centralized test inversion patterns used by CI methods,
ensuring consistent handling of infinite bounds, two-sided intervals, and
convergence diagnostics.
"""

import math
import logging
from typing import Tuple, Callable, Optional, Dict, Any, Union
from dataclasses import dataclass

from exactcis.constants import (
    DEFAULT_ALPHA, TOL, ROOT_TOL, SMALL_POS, LARGE_POS,
    STATISTICAL_TOL, MIN_PROB, MAX_PROB
)
from exactcis.utils.solvers import (
    bracket_log_space, bisection_safe, find_root_robust,
    detect_plateau_region, SolverDiagnostics
)

logger = logging.getLogger(__name__)


@dataclass
class InversionDiagnostics:
    """Diagnostic information from CI inversion algorithms."""
    lower_bound_diag: Optional[SolverDiagnostics] = None
    upper_bound_diag: Optional[SolverDiagnostics] = None
    infinite_lower: bool = False
    infinite_upper: bool = False
    plateau_detected: bool = False
    total_function_calls: int = 0
    inversion_method: str = "unknown"


def handle_infinite_bounds(bounds: Tuple[float, float], 
                          test_func: Callable[[float], float],
                          alpha: float,
                          domain: Tuple[float, float] = (SMALL_POS, LARGE_POS)
) -> Tuple[float, float]:
    """
    Handle potentially infinite CI bounds by checking domain boundaries.
    
    When test inversion doesn't find a sign change within the domain,
    this determines whether to report finite domain bounds or legitimate
    infinite bounds based on monotonicity and plateau analysis.
    
    Args:
        bounds: (lower, upper) bounds from inversion attempt
        test_func: Test function used for inversion
        alpha: Significance level
        domain: (min_theta, max_theta) search domain
        
    Returns:
        Tuple of (final_lower, final_upper) with appropriate infinite handling
    """
    lower, upper = bounds
    
    # Check if bounds hit domain limits
    hit_lower_domain = abs(lower - domain[0]) < TOL
    hit_upper_domain = abs(upper - domain[1]) < TOL
    
    # If we hit domain boundaries, check for legitimate infinite bounds
    final_lower = lower
    final_upper = upper
    
    if hit_lower_domain:
        # Check if function is monotone decreasing towards domain boundary
        try:
            boundary_value = test_func(domain[0])
            mid_value = test_func(math.sqrt(domain[0] * domain[1]))  # Geometric mean
            
            if boundary_value > alpha and mid_value > alpha:
                # Function stays above alpha threshold - legitimate infinite lower bound
                final_lower = 0.0
                logger.debug("Lower bound: legitimate infinite bound detected")
        except Exception:
            pass  # Keep finite bound on evaluation error
    
    if hit_upper_domain:
        # Check if function is monotone increasing towards domain boundary  
        try:
            boundary_value = test_func(domain[1])
            mid_value = test_func(math.sqrt(domain[0] * domain[1]))  # Geometric mean
            
            if boundary_value > alpha and mid_value > alpha:
                # Function stays above alpha threshold - legitimate infinite upper bound
                final_upper = float('inf')
                logger.debug("Upper bound: legitimate infinite bound detected")
        except Exception:
            pass  # Keep finite bound on evaluation error
    
    return final_lower, final_upper


def invert_bound(test_func: Callable[[float], float],
                is_lower: bool,
                alpha: float,
                theta_hat: float,
                domain: Tuple[float, float] = (SMALL_POS, LARGE_POS),
                tol: float = ROOT_TOL,
                seed_factor: float = 2.0,
                max_expand: int = 60) -> Tuple[float, SolverDiagnostics]:
    """
    Invert test statistic to find one-sided confidence bound.
    
    This is the core inversion function that finds the theta value where
    test_func(theta) = alpha, representing a confidence bound.
    
    Args:
        test_func: Test function that returns p-values
        is_lower: If True, find lower bound; if False, find upper bound
        alpha: Significance level (target p-value)
        theta_hat: Point estimate for seeding bracket search
        domain: (min_theta, max_theta) search domain
        tol: Root finding tolerance
        seed_factor: Factor for expanding search brackets
        max_expand: Maximum bracket expansions
        
    Returns:
        Tuple of (bound, diagnostics)
    """
    diag = SolverDiagnostics(method=f"invert_bound_{'lower' if is_lower else 'upper'}")
    
    # Create target function: test_func(theta) - alpha = 0
    target_func = lambda theta: test_func(theta) - alpha
    
    try:
        # Find bracketing interval
        bounds, bracket_diag = bracket_log_space(
            target_func, theta_hat, domain, 
            factor=seed_factor, max_expand=max_expand, target=0.0
        )
        
        diag.function_calls += bracket_diag.function_calls
        diag.bracket_expansions = bracket_diag.bracket_expansions
        diag.domain_hits = bracket_diag.domain_hits
        
        if not bracket_diag.converged:
            # No sign change found - check for infinite bounds
            if is_lower and bounds[0] <= domain[0] + TOL:
                logger.debug("Lower bound: no sign change at domain boundary")
                bound = 0.0  # Infinite lower bound
            elif not is_lower and bounds[1] >= domain[1] - TOL:
                logger.debug("Upper bound: no sign change at domain boundary") 
                bound = float('inf')  # Infinite upper bound
            else:
                # Return the appropriate boundary
                bound = bounds[0] if is_lower else bounds[1]  
                
            diag.converged = False
            return bound, diag
        
        # Solve within bracket
        root, solve_diag = bisection_safe(target_func, bounds[0], bounds[1], tol=tol)
        
        diag.function_calls += solve_diag.function_calls
        diag.iterations = solve_diag.iterations
        diag.final_residual = solve_diag.final_residual
        diag.converged = solve_diag.converged
        
        return root, diag
        
    except Exception as e:
        logger.warning(f"Inversion failed for {'lower' if is_lower else 'upper'} bound: {e}")
        diag.converged = False
        
        # Return domain boundary as fallback
        return domain[0] if is_lower else domain[1], diag


def invert_two_sided_ci(test_func: Callable[[float], float],
                       alpha: float = DEFAULT_ALPHA,
                       theta_hat: Optional[float] = None,
                       domain: Tuple[float, float] = (SMALL_POS, LARGE_POS),
                       tol: float = ROOT_TOL,
                       handle_infinite: bool = True) -> Tuple[Tuple[float, float], InversionDiagnostics]:
    """
    Invert test statistic to find two-sided confidence interval.
    
    This is the main interface for CI inversion, handling both bounds
    and providing comprehensive diagnostics.
    
    Args:
        test_func: Test function that returns p-values
        alpha: Significance level
        theta_hat: Point estimate for seeding (if None, use geometric mean of domain)
        domain: (min_theta, max_theta) search domain
        tol: Root finding tolerance
        handle_infinite: Whether to detect and handle infinite bounds
        
    Returns:
        Tuple of ((lower, upper), diagnostics)
    """
    if theta_hat is None:
        theta_hat = math.sqrt(domain[0] * domain[1])  # Geometric mean
    
    inversion_diag = InversionDiagnostics(inversion_method="two_sided_inversion")
    
    # Find lower bound
    lower_bound, lower_diag = invert_bound(
        test_func, is_lower=True, alpha=alpha, 
        theta_hat=theta_hat, domain=domain, tol=tol
    )
    inversion_diag.lower_bound_diag = lower_diag
    
    # Find upper bound  
    upper_bound, upper_diag = invert_bound(
        test_func, is_lower=False, alpha=alpha,
        theta_hat=theta_hat, domain=domain, tol=tol
    )
    inversion_diag.upper_bound_diag = upper_diag
    
    # Handle infinite bounds if requested
    if handle_infinite:
        bounds_handled = handle_infinite_bounds(
            (lower_bound, upper_bound), test_func, alpha, domain
        )
        lower_bound, upper_bound = bounds_handled
        
        inversion_diag.infinite_lower = lower_bound == 0.0
        inversion_diag.infinite_upper = math.isinf(upper_bound)
    
    # Check for plateau regions (useful for diagnostics)
    try:
        plateau_result = detect_plateau_region(test_func, (lower_bound, upper_bound))
        inversion_diag.plateau_detected = plateau_result['has_plateau']
    except Exception:
        pass
    
    # Total function calls
    inversion_diag.total_function_calls = (
        lower_diag.function_calls + upper_diag.function_calls
    )
    
    return (lower_bound, upper_bound), inversion_diag


def validate_ci_result(ci_bounds: Tuple[float, float],
                      alpha: float,
                      test_func: Optional[Callable[[float], float]] = None,
                      tolerance: float = STATISTICAL_TOL) -> Dict[str, Any]:
    """
    Validate confidence interval result for correctness.
    
    Args:
        ci_bounds: (lower, upper) confidence bounds
        alpha: Significance level used
        test_func: Optional test function for validation
        tolerance: Tolerance for validation checks
        
    Returns:
        Dict with validation results and flags
    """
    lower, upper = ci_bounds
    
    validation = {
        'valid_bounds': True,
        'lower_finite': math.isfinite(lower),
        'upper_finite': math.isfinite(upper), 
        'properly_ordered': lower <= upper,
        'messages': []
    }
    
    # Check bound ordering
    if not (lower <= upper):
        validation['valid_bounds'] = False
        validation['messages'].append("Lower bound exceeds upper bound")
    
    # Check for invalid values
    if math.isnan(lower) or math.isnan(upper):
        validation['valid_bounds'] = False
        validation['messages'].append("NaN values in bounds")
    
    # Optional test function validation
    if test_func is not None and validation['valid_bounds']:
        try:
            if math.isfinite(lower):
                p_lower = test_func(lower)
                if abs(p_lower - alpha) > tolerance:
                    validation['messages'].append(
                        f"Lower bound test: p={p_lower:.4f}, expected≈{alpha:.4f}"
                    )
            
            if math.isfinite(upper):
                p_upper = test_func(upper)
                if abs(p_upper - alpha) > tolerance:
                    validation['messages'].append(
                        f"Upper bound test: p={p_upper:.4f}, expected≈{alpha:.4f}"
                    )
        except Exception as e:
            validation['messages'].append(f"Test function evaluation error: {e}")
    
    return validation


# Specialized inversion functions for different method types

def invert_score_test_rr(score_func: Callable[[float, Tuple], float],
                        alpha: float,
                        data: Tuple[int, int, int, int],
                        correction_factor: float = 4.0,
                        domain: Tuple[float, float] = (SMALL_POS, LARGE_POS)
) -> Tuple[Tuple[float, float], InversionDiagnostics]:
    """
    Specialized inversion for relative risk score tests.
    
    This wraps the score test computation in the standardized inversion
    framework, handling the specific calling convention for RR methods.
    
    Args:
        score_func: Score test function that takes (theta, data)
        alpha: Significance level
        data: (a, b, c, d) contingency table data
        correction_factor: Correction factor for score test
        domain: Search domain for theta
        
    Returns:
        Tuple of ((lower, upper), diagnostics)
    """
    # Critical value for two-sided test
    z_crit = 1.96  # Approximate for alpha=0.05; could be made exact
    z_crit_squared = z_crit ** 2
    
    # Wrap score function to return p-value-like quantity
    def test_func(theta):
        try:
            score_stat = score_func(theta, data, correction_factor)
            # Convert score statistic to p-value equivalent
            # For score tests, we typically have score_stat ~ chi^2(1)
            # We want to find where score_stat = z_crit^2
            return abs(score_stat - z_crit_squared)
        except Exception:
            return float('inf')  # Invalid evaluation
    
    # Use standard inversion
    return invert_two_sided_ci(test_func, alpha=0.0, theta_hat=1.0, domain=domain)


def score_test_wrapper(theta: float, 
                      data: Tuple[int, int, int, int],
                      correction_factor: float = 4.0) -> float:
    """
    Wrapper for score test computation compatible with inversion framework.
    
    This provides a standardized interface for score tests that can be
    used with the general inversion functions.
    
    Args:
        theta: Relative risk parameter
        data: (a, b, c, d) contingency table
        correction_factor: Correction factor for continuity correction
        
    Returns:
        Score test statistic value
        
    Note:
        This is a placeholder - actual implementation will be added when
        refactoring the RR score methods.
    """
    a, b, c, d = data
    
    # Placeholder implementation - will be replaced with actual score computation
    # from the existing RR methods during refactoring
    n1, n2 = a + b, c + d
    if n1 == 0 or n2 == 0:
        return float('inf')
        
    r1, r2 = a / n1, c / n2
    if r1 == 0 or r2 == 0:
        return float('inf')
        
    # Very simplified score-like calculation for testing
    observed_rr = r1 / r2
    log_diff = math.log(observed_rr) - math.log(theta)
    
    return log_diff ** 2  # Placeholder statistic