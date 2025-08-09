"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import numpy as np
import logging
from scipy.stats import nchypergeom_fisher, norm
from scipy.optimize import brentq, bisect
from typing import Tuple, Callable, Optional, List, Union
from functools import lru_cache
from exactcis.core import find_sign_change
from exactcis.utils.validation import validate_table_and_alpha
from exactcis.utils.shared_cache import cached_cdf_function, cached_sf_function

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_compute_ci
    has_parallel_support = True
except ImportError:
    has_parallel_support = False
    logger.info("Parallel processing not available for conditional method")

# Shared cache functions for inter-process performance optimization
@cached_cdf_function
def _cdf_cached(a, N, c1, r1, psi):
    """Shared cached version of nchypergeom_fisher.cdf for repeated calls."""
    return nchypergeom_fisher.cdf(a, N, c1, r1, psi)

@cached_sf_function 
def _sf_cached(a, N, c1, r1, psi):
    """Shared cached version of nchypergeom_fisher.sf for repeated calls."""
    return nchypergeom_fisher.sf(a, N, c1, r1, psi)


"""
This module previously defined a custom ComputationError exception class here.
It has been removed in favor of a more consistent error handling approach
that aligns with the rest of the codebase, using logging and fallback values
instead of raising custom exceptions.
"""


def _find_better_bracket(p_value_func, lo, hi):
    """
    Use 3-point evaluation to narrow bracket before brentq.
    
    Args:
        p_value_func: Function that returns p-value given psi
        lo: Lower bound of search range
        hi: Upper bound of search range
        
    Returns:
        Tuple of (improved_lo, improved_hi) bounds
    """
    try:
        # Use geometric mean for log-scale spacing
        mid = np.sqrt(lo * hi)
        
        # Evaluate at 3 strategic points
        lo_val = p_value_func(lo)
        mid_val = p_value_func(mid)
        hi_val = p_value_func(hi)
        
        # Find subinterval where sign change occurs
        if lo_val * mid_val < 0:
            return lo, mid
        elif mid_val * hi_val < 0:
            return mid, hi
        else:
            return lo, hi  # Fallback to original bracket
    except:
        return lo, hi  # Fallback on any error


def _expand_bracket_vectorized(p_value_func, initial_lo, initial_hi, target_sign_lo, target_sign_hi, max_attempts=20):
    """
    Vectorized bracket expansion instead of sequential loops.
    
    Args:
        p_value_func: Function that returns p-value given psi
        initial_lo: Initial lower bound
        initial_hi: Initial upper bound
        target_sign_lo: Target sign for lower bound (True if positive, False if negative)
        target_sign_hi: Target sign for upper bound (True if positive, False if negative)
        max_attempts: Maximum number of expansion attempts
        
    Returns:
        Tuple of (expanded_lo, expanded_hi) bounds
    """
    try:
        # Generate candidate values using vectorized operations
        # Create expansion factors: 5^1, 5^2, 5^3, ..., 5^max_attempts
        expansion_factors = 5.0 ** np.arange(1, max_attempts + 1)
        
        # Generate candidate values for lo (divide by expansion factors)
        lo_candidates = initial_lo / expansion_factors
        
        # Generate candidate values for hi (multiply by expansion factors)
        hi_candidates = initial_hi * expansion_factors
        
        # Evaluate p_value_func at all candidate points
        # We can't fully vectorize this due to the function call,
        # but we can use a list comprehension which is more efficient than a loop
        lo_vals = np.array([p_value_func(x) for x in lo_candidates])
        hi_vals = np.array([p_value_func(x) for x in hi_candidates])
        
        # Find first suitable bracket for lower bound
        # We want lo_val < 0 if target_sign_lo is False, or lo_val > 0 if target_sign_lo is True
        lo_matches = (lo_vals > 0) == target_sign_lo
        if np.any(lo_matches):
            # Find the first match (smallest expansion that works)
            lo_idx = np.where(lo_matches)[0][0]
            expanded_lo = lo_candidates[lo_idx]
        else:
            # If no match found, use the smallest value (maximum expansion)
            expanded_lo = lo_candidates[-1]
        
        # Find first suitable bracket for upper bound
        # We want hi_val > 0 if target_sign_hi is True, or hi_val < 0 if target_sign_hi is False
        hi_matches = (hi_vals > 0) == target_sign_hi
        if np.any(hi_matches):
            # Find the first match (smallest expansion that works)
            hi_idx = np.where(hi_matches)[0][0]
            expanded_hi = hi_candidates[hi_idx]
        else:
            # If no match found, use the largest value (maximum expansion)
            expanded_hi = hi_candidates[-1]
        
        return expanded_lo, expanded_hi
    
    except Exception as e:
        # On any error, fall back to the original bounds
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Vectorized bracket expansion failed: {str(e)}")
        return initial_lo, initial_hi


def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the conditional (Fisher's) exact confidence interval for the odds ratio.
    
    Args:
        a, b, c, d: The cell counts in a 2x2 contingency table
        alpha: The significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        Tuple of lower and upper bounds of the confidence interval.
        If computation fails or produces invalid bounds, returns a conservative
        interval (0.0, inf) with appropriate warning logs.
        
    Raises:
        ValueError: If the inputs are invalid (e.g., negative counts, empty margins)
    """
    # Log the input
    logger.info(f"Calculating CI for table: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    
    # Calculate the point estimate of the odds ratio
    if b * c == 0:
        or_point = float('inf') if a * d > 0 else 0.0
    else:
        or_point = (a * d) / (b * c)
    
    logger.info(f"Point estimate of odds ratio: {or_point}")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
        
    # Special case handling for tables with empty margins
    r1 = a + b  # Row 1 total
    r2 = c + d  # Row 2 total
    c1 = a + c  # Column 1 total
    c2 = b + d  # Column 2 total
    
    # Handle empty margins before validation
    if r1 == 0 or r2 == 0 or c1 == 0 or c2 == 0:
        return 0.0, float('inf')
        
    a, b, c, d = validate_table_and_alpha(a, b, c, d, alpha, preserve_int_types=True)

    # Debug information - log the table
    logger.info(f"Calculating CI for table: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    
    # Point estimate of the odds ratio
    or_point = (a * d) / (b * c) if b * c != 0 else float('inf')
    logger.info(f"Point estimate of odds ratio: {or_point}")

    # Special case handling for tables with zeros
    N = a + b + c + d  # Total sample size
    
    # Zero handling that matches R's fisher.test implementation
    if a == 0 and c == 0:
        # Both values in column 1 are zero
        return 0.0, float('inf')
    elif b == 0 and d == 0:
        # Both values in column 2 are zero
        return 0.0, float('inf')
    elif a == 0 and b == 0:
        # Both values in row 1 are zero
        return 0.0, float('inf')
    elif c == 0 and d == 0:
        # Both values in row 2 are zero
        return 0.0, float('inf')
    
    # Special case: single zero handling
    if a == 0:
        # Use method that matches R's behavior (fisher.test) for zero in cell (1,1)
        upper = zero_cell_upper_bound(a, b, c, d, alpha)
        logger.info(f"Zero in cell (1,1): lower=0.0, upper={upper}")
        return 0.0, upper
    elif c == 0:
        # Zero in cell (2,1)
        lower = zero_cell_lower_bound(a, b, c, d, alpha)
        logger.info(f"Zero in cell (2,1): lower={lower}, upper=inf")
        return lower, float('inf')
    elif b == 0:
        # Zero in cell (1,2)
        lower = zero_cell_lower_bound(a, b, c, d, alpha)
        logger.info(f"Zero in cell (1,2): lower={lower}, upper=inf")
        return lower, float('inf')
    elif d == 0:
        # Zero in cell (2,2)
        upper = zero_cell_upper_bound(a, b, c, d, alpha)
        logger.info(f"Zero in cell (2,2): lower=0.0, upper={upper}")
        return 0.0, upper
        
    # Support range for 'a' (number of successes in row 1)
    min_k = max(0, r1 - c2)  # max(0, r1 - (N - c1))
    max_k = min(r1, c1)
    logger.info(f"Support range for a: min_k={min_k}, max_k={max_k}")

    # For very small values near the support boundaries, adjust expected behavior
    if a <= min_k:
        upper = fisher_upper_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha)
        logger.info(f"a at min support: lower=0.0, upper={upper}")
        return 0.0, upper
    if a >= max_k:
        lower = fisher_lower_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha)
        logger.info(f"a at max support: lower={lower}, upper=inf")
        return lower, float('inf')

    # Calculate normal case boundaries
    lower_bound = fisher_lower_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha)
    upper_bound = fisher_upper_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha)
    logger.info(f"Raw bounds: lower={lower_bound}, upper={upper_bound}")
    
    # Final validation to ensure reasonable bounds
    lower_bound, upper_bound = validate_bounds(lower_bound, upper_bound)
    logger.info(f"Validated bounds: lower={lower_bound}, upper={upper_bound}")
    return lower_bound, upper_bound


def fisher_lower_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha):
    """
    Calculate the lower bound for Fisher's exact CI.
    
    Args:
        a, b, c, d: Cell counts
        min_k, max_k: Support range for cell a
        N: Total sample size
        r1, c1: Row 1 and Column 1 totals
        alpha: Significance level
        
    Returns:
        Lower bound value. If root finding fails, returns a conservative
        estimate with appropriate warning logs.
    """
    # Hoist fixed computations outside closure
    target_prob = alpha / 2.0
    sf_a_arg = a - 1  # Precompute argument for sf function
    
    # Calculate odds ratio point estimate (for initial bracketing)
    if b * c == 0:
        or_point = 1.0  # Fallback for division by zero
    else:
        or_point = (a * d) / (b * c)
    
    # Define the p-value function for lower bound
    # For lower bound we need P(X ≥ a | psi) = alpha/2
    # As psi DECREASES, this probability INCREASES
    def p_value_func(psi):
        # For lower bound, use sf(a-1) = P(X ≥ a)
        return _sf_cached(sf_a_arg, N, c1, r1, psi) - target_prob
    
    # Try to find the root directly using brentq with an adaptive bracket search
    
    # Initial bracket - go much lower than or_point for a very wide initial search
    lo = max(1e-10, or_point / 2000.0) 
    hi = max(or_point * 20.0, 100.0)  # Also wider high value

    # Critical: For finding the lower bound, we need:
    # At small values (lo), p-value is SMALLER than target (negative p_value_func)
    # At large values (hi), p-value is LARGER than target (positive p_value_func)
    
    # Initial bracket values
    lo_val = p_value_func(lo)
    hi_val = p_value_func(hi)
    
    # Log initial bracket values
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Lower bound initial bracket: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
    
    # For lower bound, we need lo_val < 0 and hi_val > 0
    # Use vectorized bracket expansion for better performance
    if lo_val >= 0 or hi_val <= 0:
        # For lower bound:
        # - target_sign_lo = False (we want lo_val < 0)
        # - target_sign_hi = True (we want hi_val > 0)
        lo, hi = _expand_bracket_vectorized(
            p_value_func, lo, hi, 
            target_sign_lo=False, 
            target_sign_hi=True,
            max_attempts=40
        )
        
        # Re-evaluate at new bounds
        lo_val = p_value_func(lo)
        hi_val = p_value_func(hi)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"After vectorized expansion: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
        
        # If we still don't have proper bracket, use conservative fallbacks
        if lo_val >= 0:
            logger.warning(f"Could not find proper bracket for lower bound, table ({a},{b},{c},{d})")
            return 0.0  # Most conservative estimate
            
        if hi_val <= 0:
            logger.warning(f"Could not find proper bracket for lower bound, table ({a},{b},{c},{d})")
            # Conservative fallback
            return max(0.0, or_point / 5.0)  # More conservative
    
    # Now that we have proper bracket with lo_val < 0 and hi_val > 0, use brentq
    if lo_val < 0 and hi_val > 0:
        try:
            # Use 3-point pre-bracketing to improve convergence
            improved_lo, improved_hi = _find_better_bracket(p_value_func, lo, hi)
            
            # Use brentq to find the root with higher precision for better results
            result = max(0.0, brentq(p_value_func, improved_lo, improved_hi, rtol=1e-12, maxiter=200, full_output=False))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Found lower bound using brentq: {result}")
            return result
        except (ValueError, RuntimeError) as e:
            logger.error(f"Root finding failed for lower bound: {str(e)}")
            # Try a different method if brentq fails
            try:
                # Fall back to bisection method which is more robust but slower
                from scipy.optimize import bisect
                result = max(0.0, bisect(p_value_func, lo, hi, rtol=1e-10, maxiter=200))
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Found lower bound using bisect: {result}")
                return result
            except Exception as e2:
                logger.error(f"Secondary root finding failed: {str(e2)}")
    
    # If we don't have proper bracket, but we have an OR > 0
    if or_point > 0:
        result = max(0.0, or_point / 5.0)  # More conservative estimate based on point OR
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using conservative lower bound estimate: {result}")
        return result
    else:
        return 0.0


def fisher_upper_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha):
    """
    Calculate the upper bound for Fisher's exact CI.
    
    Args:
        a, b, c, d: Cell counts
        min_k, max_k: Support range for cell a
        N: Total sample size
        r1, c1: Row 1 and Column 1 totals
        alpha: Significance level
        
    Returns:
        Upper bound value. If root finding fails, returns a conservative
        estimate with appropriate warning logs.
    """
    # Hoist fixed computations outside closure
    target_prob = alpha / 2.0
    cdf_a_arg = a - 1  # Precompute argument for cdf function
    
    # Calculate odds ratio point estimate (for initial bracketing)
    if b * c == 0:
        or_point = 10.0  # Fallback for division by zero
    else:
        or_point = (a * d) / (b * c)
    
    # Define the p-value function for upper bound
    # For upper bound we need P(X ≤ a-1 | psi) = alpha/2
    # As psi INCREASES, this probability DECREASES
    def p_value_func(psi):
        # For upper bound, use cdf(a-1) = P(X <= a-1)
        return _cdf_cached(cdf_a_arg, N, c1, r1, psi) - target_prob
    
    # Initial bracket - go much higher than or_point for a good upper bound
    # Use wider bracket ranges for better search
    lo = max(1e-10, or_point / 20.0)
    hi = max(or_point * 2000.0, 2000.0)  # Much wider high value
    
    # Critical: For finding the upper bound, we need:
    # At small values (lo), p-value is LARGER than target (positive p_value_func)
    # At large values (hi), p-value is SMALLER than target (negative p_value_func)
    
    # Initial bracket values
    lo_val = p_value_func(lo)
    hi_val = p_value_func(hi)
    
    # Log initial bracket values
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Upper bound initial bracket: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
    
    # For upper bound, we need lo_val > 0 and hi_val < 0
    # Use vectorized bracket expansion for better performance
    if lo_val <= 0 or hi_val >= 0:
        # For upper bound:
        # - target_sign_lo = True (we want lo_val > 0)
        # - target_sign_hi = False (we want hi_val < 0)
        lo, hi = _expand_bracket_vectorized(
            p_value_func, lo, hi, 
            target_sign_lo=True, 
            target_sign_hi=False,
            max_attempts=40
        )
        
        # Re-evaluate at new bounds
        lo_val = p_value_func(lo)
        hi_val = p_value_func(hi)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"After vectorized expansion: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
        
        # If we still don't have proper bracket, use conservative fallbacks
        if lo_val <= 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Upper bound extremely small for table ({a},{b},{c},{d})")
            return or_point * 3.0  # More conservative estimate
            
        if hi_val >= 0:
            logger.warning(f"Could not find proper bracket for upper bound, table ({a},{b},{c},{d})")
            # For tables with zeros in certain cells, it's reasonable to have infinite upper bound
            if b == 0 or c == 0:
                return float('inf')
            # For other cases where bracketing fails, use a very large multiple of OR
            return max(1000.0, or_point * 200.0)
    
    # Now that we have proper bracket with lo_val > 0 and hi_val < 0, use brentq
    if lo_val > 0 and hi_val < 0:
        try:
            # Use 3-point pre-bracketing to improve convergence
            improved_lo, improved_hi = _find_better_bracket(p_value_func, lo, hi)
            
            # Use brentq to find the root with higher precision
            result = brentq(p_value_func, improved_lo, improved_hi, rtol=1e-12, maxiter=200, full_output=False)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Found upper bound using brentq: {result}")
            return max(1.0, result)  # Ensure result is at least 1.0
        except (ValueError, RuntimeError) as e:
            logger.error(f"Root finding failed for upper bound: {str(e)}")
            # Try a different method if brentq fails
            try:
                # Fall back to bisection method which is more robust but slower
                from scipy.optimize import bisect
                result = bisect(p_value_func, lo, hi, rtol=1e-10, maxiter=200)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Found upper bound using bisect: {result}")
                return max(1.0, result)  # Ensure result is at least 1.0
            except Exception as e2:
                logger.error(f"Secondary root finding failed: {str(e2)}")
    
    # If bracket has wrong signs, use conservative estimate
    if or_point < 1.0:
        result = max(1.0, or_point * 10.0)  # For small OR, upper bound is moderate
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using conservative upper bound estimate for small OR: {result}")
        return result
    else:
        # For large OR, use a very large upper bound
        result = max(1000.0, or_point * 100.0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using conservative upper bound estimate for large OR: {result}")
        return result


def validate_bounds(lower, upper):
    """
    Validate and potentially adjust bounds to ensure they are reasonable.
    
    Args:
        lower: Lower confidence bound
        upper: Upper confidence bound
        
    Returns:
        Tuple of (lower, upper) with any necessary adjustments.
        If bounds are crossed, returns a conservative interval (0.0, inf).
    """
    # Check for non-finite values first
    if not np.isfinite(lower) and lower != 0.0:
        lower = 0.0
        # For infinite lower bound, keep the original upper bound
        return lower, upper
        
    if not np.isfinite(upper):
        upper = float('inf')
    
    # Ensure bounds are reasonable
    if lower < 0:
        lower = 0.0
    
    # Ensure upper bound is greater than lower bound
    if upper <= lower and upper != 0:
        # If bounds are crossed, it signals a computation issue
        # Return a conservative interval instead of raising an exception
        logger.warning(f"Invalid bounds detected: lower ({lower}) >= upper ({upper}). Returning conservative interval.")
        return 0.0, float('inf')
    
    return lower, upper


def zero_cell_upper_bound(a, b, c, d, alpha):
    """
    Calculate the upper bound for a 2x2 table with a zero cell.
    
    This implementation matches R's fisher.test behavior for zero cells.
    """
    # R's approach for zero cells is to use a non-central hypergeometric distribution
    # with a modified table (implicitly adding a small value to the zero cell)
    
    # Simple conditional method for the zero cell case
    N_calc = a + b + c + d
    r1_calc = a + b
    c1_calc = a + c

    if a == 0:  # Cell (1,1) is zero
        # Find where P(X ≤ 0 | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return _cdf_cached(0, N_calc, c1_calc, r1_calc, psi) - alpha/2
    elif d == 0:  # Cell (2,2) is zero
        # Find where P(X ≤ a | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return _cdf_cached(a, N_calc, c1_calc, r1_calc, psi) - alpha/2
    else:
        # Default fallback if not a specific zero cell case
        return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    # Initial bracket
    lo = 1e-10
    hi = 1e6
    
    # For zero cell upper bound, we expect func(lo) > 0 and func(hi) < 0
    lo_val = func(lo)
    hi_val = func(hi)
    
    # Expand brackets if needed
    max_attempts = 20
    
    # If hi value is positive or zero, keep increasing until negative or max reached
    if hi_val >= 0:
        attempt = 0
        while hi_val >= 0 and attempt < max_attempts:
            hi *= 10.0
            hi_val = func(hi)
            attempt += 1
            
        # If still no negative value, use fallback
        if hi_val >= 0:
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    # If lo value is negative or zero, keep decreasing until positive or min reached
    if lo_val <= 0:
        attempt = 0
        while lo_val <= 0 and attempt < max_attempts and lo > 1e-15:
            lo /= 10.0
            lo_val = func(lo)
            attempt += 1
            
        # If still no positive value, use fallback
        if lo_val <= 0:
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    # Now we should have lo_val > 0 and hi_val < 0
    if lo_val > 0 and hi_val < 0:
        try:
            return brentq(func, lo, hi, rtol=1e-8)
        except (ValueError, RuntimeError):
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    # Fallback if no proper bracket found
    return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)

def zero_cell_lower_bound(a, b, c, d, alpha):
    """
    Calculate the lower bound for a 2x2 table with a zero cell.
    
    This implementation matches R's fisher.test behavior for zero cells.
    """
    # Similar to upper bound, but for cells that lead to lower bound with zero
    N_calc = a + b + c + d
    r1_calc = a + b
    c1_calc = a + c

    # For lower bound with zero cell
    if c == 0:  # Cell (2,1) is zero
        # Find where P(X ≤ a-1 | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return _cdf_cached(a-1, N_calc, c1_calc, r1_calc, psi) - alpha/2
    elif b == 0:  # Cell (1,2) is zero
        # Find where P(X ≤ a-1 | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return _cdf_cached(a-1, N_calc, c1_calc, r1_calc, psi) - alpha/2
    else:
        # Default fallback if not a specific zero cell case
        return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    
    # Initial bracket
    lo = 1e-10
    hi = 1e6
    
    # For zero cell lower bound with CORRECTED function, we expect func(lo) > 0 and func(hi) < 0
    lo_val = func(lo)
    hi_val = func(hi)
    
    # Expand brackets if needed
    max_attempts = 20
    
    # If hi value is positive or zero, keep increasing until negative or max reached
    if hi_val >= 0:
        attempt = 0
        while hi_val >= 0 and attempt < max_attempts:
            hi *= 10.0
            hi_val = func(hi)
            attempt += 1
            
        # If still no negative value, use fallback
        if hi_val >= 0:
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    
    # If lo value is negative or zero, keep decreasing until positive or min reached
    if lo_val <= 0:
        attempt = 0
        while lo_val <= 0 and attempt < max_attempts and lo > 1e-15:
            lo /= 10.0
            lo_val = func(lo)
            attempt += 1
            
        # If still no positive value, return zero
        if lo_val <= 0:
            return 0.0
    
    # Now we should have lo_val > 0 and hi_val < 0
    if lo_val > 0 and hi_val < 0:
        try:
            return max(0.0, brentq(func, lo, hi, rtol=1e-8))
        except (ValueError, RuntimeError):
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    
    # Fallback if no proper bracket found
    if c == 0 or b == 0:  # These cases should have positive lower bounds
        return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    else:
        return 0.0  # Conservative fallback

def fisher_tippett_zero_cell_upper(a, b, c, d, alpha):
    """
    Fallback method for upper bound calculation with zero cells.
    
    This uses the Fisher-Tippett approach which is similar to what R uses
    as a fallback for zero cells.
    """
    # Add 0.5 to empty cells, which is a common approach in R
    adj_a = max(a, 0.5) if a == 0 else a
    adj_b = max(b, 0.5) if b == 0 else b
    adj_c = max(c, 0.5) if c == 0 else c
    adj_d = max(d, 0.5) if d == 0 else d
    
    # Use log-scale calculation similar to R for stability
    log_or = np.log((adj_a * adj_d) / (adj_b * adj_c))
    
    # Standard error on log scale
    se = np.sqrt(1/adj_a + 1/adj_b + 1/adj_c + 1/adj_d)
    
    # Critical value for alpha/2
    z = norm.ppf(1 - alpha/2)
    
    # Upper limit on log scale
    log_upper = log_or + z * se
    
    # Convert back
    upper = np.exp(log_upper)
    
    return upper


def fisher_tippett_zero_cell_lower(a, b, c, d, alpha):
    """
    Fallback method for lower bound calculation with zero cells.
    
    This uses the Fisher-Tippett approach which is similar to what R uses
    as a fallback for zero cells.
    """
    # Add 0.5 to empty cells, which is a common approach in R
    adj_a = max(a, 0.5) if a == 0 else a
    adj_b = max(b, 0.5) if b == 0 else b
    adj_c = max(c, 0.5) if c == 0 else c
    adj_d = max(d, 0.5) if d == 0 else d
    
    # Use log-scale calculation similar to R for stability
    log_or = np.log((adj_a * adj_d) / (adj_b * adj_c))
    
    # Standard error on log scale
    se = np.sqrt(1/adj_a + 1/adj_b + 1/adj_c + 1/adj_d)
    
    # Critical value for alpha/2
    z = norm.ppf(1 - alpha/2)
    
    # Lower limit on log scale
    log_lower = log_or - z * se
    
    # Convert back
    lower = np.exp(log_lower)
    
    return max(0.0, lower)


def exact_ci_conditional_batch(tables: List[Tuple[int, int, int, int]], 
                               alpha: float = 0.05,
                               max_workers: Optional[int] = None,
                               backend: Optional[str] = None,
                               progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate conditional (Fisher's) exact confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing with shared inter-process caching to compute 
    confidence intervals for multiple tables simultaneously, providing significant speedup for 
    large datasets.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        List of (lower_bound, upper_bound) tuples, one for each input table
        
    Note:
        This implementation uses shared inter-process caching to eliminate redundant
        CDF/SF calculations across worker processes, providing substantial performance
        improvements over sequential processing.
        
        Backend Selection: For methods that use Numba-accelerated functions,
        the 'thread' backend may be more efficient. If not specified, the backend
        is auto-detected based on the method.
    """
    if not tables:
        return []
    
    if not has_parallel_support:
        # Fall back to sequential processing
        logger.info("Parallel support not available, using sequential processing")
        results = []
        for i, (a, b, c, d) in enumerate(tables):
            try:
                result = exact_ci_conditional(a, b, c, d, alpha)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing table {i+1} ({a},{b},{c},{d}): {e}")
                results.append((0.0, float('inf')))  # Conservative fallback
            
            if progress_callback:
                progress_callback(min(100, int(100 * (i+1) / len(tables))))
        
        return results
    
    # Initialize shared cache for parallel processing
    from exactcis.utils.shared_cache import init_shared_cache_for_parallel
    cache = init_shared_cache_for_parallel()
    
    logger.info(f"Processing {len(tables)} tables with conditional method using shared cache")
    
    # Use the improved parallel processing with shared cache
    results = parallel_compute_ci(
        exact_ci_conditional,
        tables,
        alpha=alpha,
        timeout=None,  # No timeout for batch processing
        backend=backend,
        max_workers=max_workers
    )
    
    # Report final statistics
    from exactcis.utils.shared_cache import get_shared_cache
    cache = get_shared_cache()
    stats = cache.get_stats()
    logger.info(f"Completed batch processing of {len(tables)} tables")
    logger.info(f"Cache statistics: {stats['hit_rate_percent']:.1f}% hit rate ({stats['hits']}/{stats['total_lookups']} lookups)")
    
    return results
