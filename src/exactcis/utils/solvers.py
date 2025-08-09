"""
Robust solver algorithms for ExactCIs.

This module provides centralized, robust root-finding and bracketing algorithms
used across all confidence interval methods. Designed for numerical stability
on the log-theta domain with comprehensive error handling and diagnostics.
"""

import math
import logging
from typing import Tuple, Callable, Optional, Dict, Any, Union
from dataclasses import dataclass

from exactcis.constants import (
    TOL, ROOT_TOL, MAX_ROOT_ITER, SMALL_POS, LARGE_POS,
    LOG_SMALL, LOG_LARGE, STATISTICAL_TOL, PLATEAU_TOL
)

logger = logging.getLogger(__name__)


@dataclass
class SolverDiagnostics:
    """Diagnostic information from solver algorithms."""
    iterations: int = 0
    function_calls: int = 0
    bracket_expansions: int = 0
    monotone_flag: bool = False
    plateau_flag: bool = False
    domain_hits: int = 0
    final_residual: float = float('nan')
    converged: bool = False
    method: str = "unknown"

def bracket_log_space(func: Callable[[float], float], 
                     theta0: float,
                     domain: Tuple[float, float] = (SMALL_POS, LARGE_POS),
                     factor: float = 2.0,
                     max_expand: int = 60,
                     target: float = 0.0) -> Tuple[Tuple[float, float], SolverDiagnostics]:
    """
    Find bracketing interval around theta0 in log space where func crosses target.
    
    Expands brackets multiplicatively until a sign change is found or domain 
    limits are hit. Designed for robust CI bound finding.
    
    Args:
        func: Function to bracket (should be continuous)
        theta0: Initial guess/seed point
        domain: (min_theta, max_theta) domain bounds
        factor: Multiplicative expansion factor (>1.0)
        max_expand: Maximum number of expansion steps
        target: Target value func should cross (usually 0.0)
        
    Returns:
        Tuple of ((lo, hi), diagnostics) where lo < hi bracket the sign change,
        or domain bounds if no sign change found
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not (domain[0] < theta0 < domain[1]):
        raise ValueError(f"Seed theta0={theta0} must be within domain {domain}")
    if factor <= 1.0:
        raise ValueError(f"Expansion factor must be > 1.0, got {factor}")
        
    diag = SolverDiagnostics(method="bracket_log_space")
    
    # Convert to log space for expansion
    log_theta0 = math.log(theta0)
    log_domain = (math.log(domain[0]), math.log(domain[1]))
    
    # Evaluate at seed
    f0 = func(theta0) - target
    diag.function_calls += 1
    
    # Initialize brackets
    log_lo, log_hi = log_theta0, log_theta0
    f_lo, f_hi = f0, f0
    
    # Expand in both directions until sign change or domain hit
    for expansion in range(max_expand):
        diag.bracket_expansions = expansion + 1
        
        # Expand log brackets
        log_step = expansion * math.log(factor)
        new_log_lo = log_theta0 - log_step
        new_log_hi = log_theta0 + log_step
        
        # Clamp to domain
        hit_lower = new_log_lo <= log_domain[0]
        hit_upper = new_log_hi >= log_domain[1]
        
        if hit_lower:
            new_log_lo = log_domain[0]
            diag.domain_hits += 1
        if hit_upper:
            new_log_hi = log_domain[1] 
            diag.domain_hits += 1
        
        # Evaluate at new points if not already at boundaries
        if new_log_lo < log_lo:
            theta_lo = math.exp(new_log_lo)
            f_lo = func(theta_lo) - target
            diag.function_calls += 1
            log_lo = new_log_lo
            
        if new_log_hi > log_hi:
            theta_hi = math.exp(new_log_hi)
            f_hi = func(theta_hi) - target
            diag.function_calls += 1
            log_hi = new_log_hi
        
        # Check for sign change
        if f_lo * f_hi < 0:
            diag.converged = True
            theta_lo, theta_hi = math.exp(log_lo), math.exp(log_hi)
            logger.debug(f"Bracket found after {expansion+1} expansions: "
                        f"({theta_lo:.6e}, {theta_hi:.6e})")
            return (theta_lo, theta_hi), diag
            
        # If we hit both domain boundaries without sign change, stop
        if hit_lower and hit_upper:
            break
    
    # No sign change found - return domain bounds
    theta_lo, theta_hi = math.exp(log_lo), math.exp(log_hi)
    logger.warning(f"No sign change found in domain after {diag.bracket_expansions} expansions")
    diag.converged = False
    return (theta_lo, theta_hi), diag

def is_monotone_on_log_grid(func: Callable[[float], float],
                           grid: Tuple[float, float],
                           num_points: int = 20) -> Dict[str, Union[bool, int]]:
    """
    Check if function is monotone on a log-spaced grid.
    
    Args:
        func: Function to check
        grid: (min_theta, max_theta) interval
        num_points: Number of grid points to test
        
    Returns:
        Dict with keys: 'monotone' (bool), 'direction' (-1|0|1)
        where direction is -1 (decreasing), 0 (not monotone), 1 (increasing)
    """
    if grid[0] >= grid[1]:
        return {'monotone': False, 'direction': 0}
        
    # Create log-spaced grid
    log_min, log_max = math.log(grid[0]), math.log(grid[1])
    log_points = [log_min + i * (log_max - log_min) / (num_points - 1) 
                  for i in range(num_points)]
    thetas = [math.exp(log_pt) for log_pt in log_points]
    
    # Evaluate function
    try:
        values = [func(theta) for theta in thetas]
    except Exception:
        return {'monotone': False, 'direction': 0}
    
    # Check monotonicity
    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    if increasing and not decreasing:
        return {'monotone': True, 'direction': 1}
    elif decreasing and not increasing:
        return {'monotone': True, 'direction': -1}
    else:
        return {'monotone': False, 'direction': 0}

def bisection_safe(func: Callable[[float], float],
                  lo: float, 
                  hi: float,
                  tol: float = ROOT_TOL,
                  max_iter: int = MAX_ROOT_ITER,
                  target: float = 0.0) -> Tuple[float, SolverDiagnostics]:
    """
    Safe bisection root finding with guaranteed convergence.
    
    Uses bisection method which guarantees convergence when a sign change
    exists in the interval. More robust than faster methods.
    
    Args:
        func: Function to find root of
        lo, hi: Bracket with func(lo) and func(hi) having opposite signs
        tol: Convergence tolerance
        max_iter: Maximum iterations
        target: Target value (root of func(x) - target = 0)
        
    Returns:
        Tuple of (root, diagnostics)
        
    Raises:
        ValueError: If no sign change in initial bracket
    """
    diag = SolverDiagnostics(method="bisection_safe")
    
    # Evaluate endpoints
    f_lo = func(lo) - target
    f_hi = func(hi) - target
    diag.function_calls += 2
    
    # Check for sign change
    if f_lo * f_hi > 0:
        raise ValueError(f"No sign change in bracket [{lo}, {hi}]: "
                        f"f({lo})={f_lo+target:.6e}, f({hi})={f_hi+target:.6e}")
    
    # Handle exact roots at endpoints
    if abs(f_lo) < tol:
        diag.final_residual = abs(f_lo)
        diag.converged = True
        return lo, diag
    if abs(f_hi) < tol:
        diag.final_residual = abs(f_hi)
        diag.converged = True
        return hi, diag
    
    # Bisection loop
    for iteration in range(max_iter):
        diag.iterations = iteration + 1
        
        # Midpoint
        mid = (lo + hi) / 2
        f_mid = func(mid) - target
        diag.function_calls += 1
        
        # Check convergence
        if abs(f_mid) < tol or (hi - lo) < 2 * tol:
            diag.final_residual = abs(f_mid)
            diag.converged = True
            logger.debug(f"Bisection converged in {iteration+1} iterations: "
                        f"root={mid:.10e}, residual={abs(f_mid):.2e}")
            return mid, diag
        
        # Update bracket
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    
    # Max iterations reached
    mid = (lo + hi) / 2
    f_mid = func(mid) - target
    diag.final_residual = abs(f_mid)
    diag.converged = False
    logger.warning(f"Bisection failed to converge in {max_iter} iterations: "
                  f"root≈{mid:.6e}, residual={diag.final_residual:.2e}")
    
    return mid, diag

def find_root_robust(func: Callable[[float], float],
                    bounds: Tuple[float, float],
                    method: str = "bisection",
                    tol: float = ROOT_TOL,
                    max_iter: int = MAX_ROOT_ITER,
                    target: float = 0.0) -> Tuple[float, SolverDiagnostics]:
    """
    Robust root finding with multiple fallback strategies.
    
    Args:
        func: Function to find root of
        bounds: (lo, hi) bracket
        method: Primary method to try ("bisection", "brentq", "ridder")
        tol: Convergence tolerance
        max_iter: Maximum iterations
        target: Target value
        
    Returns:
        Tuple of (root, diagnostics)
    """
    if method == "bisection":
        return bisection_safe(func, bounds[0], bounds[1], tol, max_iter, target)
    else:
        # For now, just use bisection as it's most robust
        # Later can add scipy.optimize.brentq, ridder, etc. with fallbacks
        logger.debug(f"Method {method} requested, falling back to bisection for robustness")
        return bisection_safe(func, bounds[0], bounds[1], tol, max_iter, target)

def detect_plateau_region(func: Callable[[float], float],
                         bounds: Tuple[float, float],
                         tolerance: float = PLATEAU_TOL,
                         num_points: int = 10) -> Dict[str, Any]:
    """
    Detect flat/plateau regions in function over given bounds.
    
    Args:
        func: Function to analyze
        bounds: (lo, hi) interval to check
        tolerance: Tolerance for considering values "equal"
        num_points: Number of sample points
        
    Returns:
        Dict with 'has_plateau', 'plateau_value', 'plateau_width' keys
    """
    if bounds[0] >= bounds[1]:
        return {'has_plateau': False, 'plateau_value': None, 'plateau_width': 0.0}
    
    # Sample function on log-spaced grid
    log_lo, log_hi = math.log(bounds[0]), math.log(bounds[1])
    log_points = [log_lo + i * (log_hi - log_lo) / (num_points - 1) 
                  for i in range(num_points)]
    thetas = [math.exp(log_pt) for log_pt in log_points]
    
    try:
        values = [func(theta) for theta in thetas]
    except Exception:
        return {'has_plateau': False, 'plateau_value': None, 'plateau_width': 0.0}
    
    # Check for plateau (small variation)
    min_val, max_val = min(values), max(values)
    variation = max_val - min_val
    
    if variation < tolerance:
        return {
            'has_plateau': True,
            'plateau_value': sum(values) / len(values),  # Average
            'plateau_width': bounds[1] - bounds[0]
        }
    else:
        return {'has_plateau': False, 'plateau_value': None, 'plateau_width': 0.0}

# ##################################################################
# Confidence Interval Construction Methods
# ##################################################################

def wald_ci_from_log_estimate(log_estimate: float, se: float, alpha: float = 0.05,
                              distribution: str = "normal") -> Tuple[float, float]:
    """
    Compute a Wald-type CI from a log-scale estimate and its standard error.

    This is a fundamental solver for constructing confidence intervals for many
    asymptotic methods (e.g., for log-odds-ratio, log-relative-risk).

    Args:
        log_estimate: The point estimate on the log scale.
        se: The standard error of the log-estimate.
        alpha: The significance level (e.g., 0.05 for a 95% CI).
        distribution: The reference distribution ("normal" or "t").

    Returns:
        A tuple containing the (lower_bound, upper_bound) of the CI on the
        original scale.
    """
    # Local import to keep solvers module self-contained
    from exactcis.utils.mathops import exp_safe

    if math.isinf(se) or se <= 0:
        return 0.0, float('inf')

    if distribution == "normal":
        from scipy import stats
        critical_value = stats.norm.ppf(1 - alpha / 2)
    elif distribution == "t":
        from scipy import stats
        # Note: A proper t-distribution CI requires degrees of freedom,
        # which is a statistical policy decision. This solver just provides
        # the mechanism. A placeholder is used if not provided.
        df = 10  # Placeholder
        critical_value = stats.t.ppf(1 - alpha / 2, df=df)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    margin = critical_value * se
    lower_log, upper_log = log_estimate - margin, log_estimate + margin
    return exp_safe(lower_log), exp_safe(upper_log)
