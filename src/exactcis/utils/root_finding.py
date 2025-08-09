"""
Pure functions for numerical root finding operations.

TODO: REVIEW FOR REMOVAL - Potential duplication with solvers.py
This module contains root finding functionality that may overlap with
the more comprehensive solver algorithms in utils/solvers.py.
Consider consolidating into a single root finding module.
"""

import math
import logging
from typing import Optional, Callable, Tuple
from .data_models import RootFindingConfig, RootFindingResult

logger = logging.getLogger(__name__)


def validate_search_bounds(lo: float, hi: float) -> bool:
    """
    Validate search interval bounds.
    
    Args:
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If bounds are invalid
    """
    if lo <= 0 or hi <= 0:
        raise ValueError("Search interval bounds for find_root_log must be positive")
    if lo >= hi:
        raise ValueError("Lower bound must be less than upper bound")
    return True


def convert_bounds_to_log_space(lo: float, hi: float) -> Tuple[float, float]:
    """
    Convert bounds to log space for numerical stability.
    
    Args:
        lo: Lower bound in normal space
        hi: Upper bound in normal space
        
    Returns:
        Tuple of (log_lo, log_hi)
    """
    return math.log(lo), math.log(hi)


def evaluate_function_at_bounds(
    f: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Tuple[float, float, bool]:
    """
    Evaluate function at log-space bounds.
    
    Args:
        f: Function to evaluate
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        timeout_checker: Optional timeout checking function
        
    Returns:
        Tuple of (f_lo, f_hi, success)
    """
    if timeout_checker and timeout_checker():
        return 0.0, 0.0, False
    
    try:
        f_lo = f(math.exp(log_lo))
        if timeout_checker and timeout_checker():
            return 0.0, 0.0, False
        if f_lo is None:
            return 0.0, 0.0, False
        
        f_hi = f(math.exp(log_hi))
        if timeout_checker and timeout_checker():
            return 0.0, 0.0, False
        if f_hi is None:
            return 0.0, 0.0, False
        
        return f_lo, f_hi, True
        
    except Exception as e:
        logger.warning(f"Error evaluating function at bounds: {e}")
        return 0.0, 0.0, False


def check_for_immediate_roots(
    f_lo: float, 
    f_hi: float, 
    log_lo: float, 
    log_hi: float
) -> Optional[float]:
    """
    Check if either bound is already a root.
    
    Args:
        f_lo: Function value at lower bound
        f_hi: Function value at upper bound
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        
    Returns:
        Root value if found, None otherwise
    """
    if f_lo == 0:
        return log_lo
    if f_hi == 0:
        return log_hi
    return None


def expand_upper_bound(
    f: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    f_lo: float,
    timeout_checker: Optional[Callable[[], bool]] = None,
    max_attempts: int = 30
) -> Tuple[float, float, bool]:
    """
    Expand upper bound to bracket the root.
    
    Args:
        f: Function to evaluate
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        f_lo: Function value at lower bound
        timeout_checker: Optional timeout checking function
        max_attempts: Maximum expansion attempts
        
    Returns:
        Tuple of (new_log_hi, f_hi, success)
    """
    original_log_hi = log_hi
    
    for i in range(max_attempts):
        log_hi = original_log_hi + math.log(10) * (i + 1)
        
        if timeout_checker and timeout_checker():
            return original_log_hi, 0.0, False
        
        f_hi_new = f(math.exp(log_hi))
        if f_hi_new is None or (timeout_checker and timeout_checker()):
            return original_log_hi, 0.0, False
        
        if f_lo * f_hi_new <= 0:
            logger.info(f"Bracketed root by expanding hi to {math.exp(log_hi):.2e}")
            return log_hi, f_hi_new, True
    
    return original_log_hi, 0.0, False


def expand_lower_bound(
    f: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    f_hi: float,
    timeout_checker: Optional[Callable[[], bool]] = None,
    max_attempts: int = 30
) -> Tuple[float, float, bool]:
    """
    Expand lower bound to bracket the root.
    
    Args:
        f: Function to evaluate
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        f_hi: Function value at upper bound
        timeout_checker: Optional timeout checking function
        max_attempts: Maximum expansion attempts
        
    Returns:
        Tuple of (new_log_lo, f_lo, success)
    """
    original_log_lo = log_lo
    
    for i in range(max_attempts):
        log_lo = original_log_lo - math.log(10) * (i + 1)
        
        if timeout_checker and timeout_checker():
            return original_log_lo, 0.0, False
        
        f_lo_new = f(math.exp(log_lo))
        if f_lo_new is None or (timeout_checker and timeout_checker()):
            return original_log_lo, 0.0, False
        
        if f_lo_new * f_hi <= 0:
            logger.info(f"Bracketed root by expanding lo to {math.exp(log_lo):.2e}")
            return log_lo, f_lo_new, True
    
    return original_log_lo, 0.0, False


def attempt_bracket_expansion(
    f: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    f_lo: float,
    f_hi: float,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> Tuple[float, float, float, float, bool]:
    """
    Attempt to expand brackets to contain the root.
    
    Args:
        f: Function to evaluate
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        f_lo: Function value at lower bound
        f_hi: Function value at upper bound
        timeout_checker: Optional timeout checking function
        
    Returns:
        Tuple of (new_log_lo, new_log_hi, new_f_lo, new_f_hi, success)
    """
    if f_lo * f_hi <= 0:
        return log_lo, log_hi, f_lo, f_hi, True
    
    logger.warning(
        f"Initial interval [{math.exp(log_lo):.2e}, {math.exp(log_hi):.2e}] "
        f"(f_lo={f_lo:.2e}, f_hi={f_hi:.2e}) does not bracket root. Attempting to expand."
    )
    
    # Try expanding upper bound first
    new_log_hi, new_f_hi, upper_success = expand_upper_bound(
        f, log_lo, log_hi, f_lo, timeout_checker
    )
    
    if upper_success:
        return log_lo, new_log_hi, f_lo, new_f_hi, True
    
    # Try expanding lower bound
    new_log_lo, new_f_lo, lower_success = expand_lower_bound(
        f, log_lo, log_hi, f_hi, timeout_checker
    )
    
    if lower_success:
        return new_log_lo, log_hi, new_f_lo, f_hi, True
    
    # Failed to bracket
    return log_lo, log_hi, f_lo, f_hi, False


def bisection_method_log_space(
    f: Callable[[float], float],
    log_lo: float,
    log_hi: float,
    f_lo: float,
    config: RootFindingConfig,
    timeout_checker: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> RootFindingResult:
    """
    Perform bisection method in log space.
    
    Args:
        f: Function to find root for
        log_lo: Lower bound in log space
        log_hi: Upper bound in log space
        f_lo: Function value at lower bound
        config: Root finding configuration
        timeout_checker: Optional timeout checking function
        progress_callback: Optional progress callback
        
    Returns:
        Root finding result
    """
    iter_num = 0
    
    while iter_num < config.maxiter:
        if timeout_checker and timeout_checker():
            logger.warning("Timeout occurred during bisection")
            return RootFindingResult(
                value=None,
                iterations=iter_num,
                converged=False,
                method="timeout_bisection"
            )
        
        # Check convergence based on interval width in log space
        if (log_hi - log_lo) < config.tol:
            log_mid = 0.5 * (log_lo + log_hi)
            return RootFindingResult(
                value=log_mid,
                iterations=iter_num,
                converged=True,
                method="converged_bisection"
            )
        
        log_mid = 0.5 * (log_lo + log_hi)
        mid_val = math.exp(log_mid)
        f_mid = f(mid_val)
        
        if f_mid is None:
            logger.warning("Function returned None during bisection")
            return RootFindingResult(
                value=None,
                iterations=iter_num,
                converged=False,
                method="function_error_bisection"
            )
        
        if progress_callback:
            progress_callback(100 * iter_num / config.maxiter)
        
        if f_mid == 0:
            return RootFindingResult(
                value=log_mid,
                iterations=iter_num,
                converged=True,
                method="exact_root_bisection"
            )
        elif f_mid * f_lo < 0:
            log_hi = log_mid
        else:
            log_lo = log_mid
            f_lo = f_mid
        
        iter_num += 1
    
    # Max iterations reached
    log_mid = (log_lo + log_hi) / 2.0
    logger.warning(f"Max iterations ({config.maxiter}) reached. Returning {log_mid:.4e}")
    
    return RootFindingResult(
        value=log_mid,
        iterations=iter_num,
        converged=False,
        method="max_iter_bisection"
    )


def find_root_log_functional(
    f: Callable[[float], float],
    config: RootFindingConfig,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout_checker: Optional[Callable[[], bool]] = None
) -> RootFindingResult:
    """
    Find root using bisection method in log space with functional approach.
    
    This refactored function uses pure functions and clear separation of concerns
    to improve maintainability and testability.
    
    Args:
        f: Function to find root for
        config: Root finding configuration
        progress_callback: Optional progress callback
        timeout_checker: Optional timeout checking function
        
    Returns:
        Root finding result
    """
    try:
        # Validate inputs
        validate_search_bounds(config.lo, config.hi)
        
        # Convert to log space
        log_lo, log_hi = convert_bounds_to_log_space(config.lo, config.hi)
        
        # Evaluate function at bounds
        f_lo, f_hi, eval_success = evaluate_function_at_bounds(
            f, log_lo, log_hi, timeout_checker
        )
        
        if not eval_success:
            return RootFindingResult(
                value=None,
                iterations=0,
                converged=False,
                method="evaluation_failed"
            )
        
        # Check for immediate roots
        immediate_root = check_for_immediate_roots(f_lo, f_hi, log_lo, log_hi)
        if immediate_root is not None:
            return RootFindingResult(
                value=immediate_root,
                iterations=0,
                converged=True,
                method="immediate_root"
            )
        
        # Attempt to bracket the root
        log_lo, log_hi, f_lo, f_hi, bracket_success = attempt_bracket_expansion(
            f, log_lo, log_hi, f_lo, f_hi, timeout_checker
        )
        
        if not bracket_success:
            logger.error(
                f"Cannot bracket root: f({math.exp(log_lo):.2e}) = {f_lo:.2e}, "
                f"f({math.exp(log_hi):.2e}) = {f_hi:.2e}"
            )
            raise RuntimeError("Cannot bracket root in find_root_log")
        
        # Perform bisection method
        return bisection_method_log_space(
            f, log_lo, log_hi, f_lo, config, timeout_checker, progress_callback
        )
        
    except Exception as e:
        logger.error(f"Error in find_root_log_functional: {e}")
        return RootFindingResult(
            value=None,
            iterations=0,
            converged=False,
            method="error"
        )


# Backward compatibility function
def find_root_log_impl_functional(
    f: Callable[[float], float], 
    lo: float = 1e-8, 
    hi: float = 1.0,
    tol: float = 1e-8, 
    maxiter: int = 60,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout_checker: Optional[Callable[[], bool]] = None,
    **kwargs
) -> Optional[float]:
    """
    Functional implementation of find_root_log.
    
    This is a drop-in replacement for the original find_root_log function
    that uses pure functions and follows functional programming principles.
    
    Args:
        f: Function to find root for
        lo: Lower bound for search
        hi: Upper bound for search
        tol: Tolerance for convergence
        maxiter: Maximum number of iterations
        progress_callback: Optional progress callback
        timeout_checker: Optional timeout checking function
        **kwargs: Additional keyword arguments for compatibility
        
    Returns:
        Log of the approximate root, or None if timeout/failure
    """
    # Handle backward compatibility with xtol parameter
    if 'xtol' in kwargs:
        tol = kwargs['xtol']
    
    config = RootFindingConfig(
        lo=lo, hi=hi, tol=tol, maxiter=maxiter,
        timeout=kwargs.get('timeout')
    )
    
    result = find_root_log_functional(f, config, progress_callback, timeout_checker)
    # Return None instead of raising RuntimeError when result.value is None
    # This handles timeout cases and bracketing failures gracefully
    return result.value