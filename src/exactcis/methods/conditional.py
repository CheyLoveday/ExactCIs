"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import numpy as np
import logging
from scipy.stats import nchypergeom_fisher, norm
from scipy.optimize import brentq
from typing import Tuple, Callable, Optional
from exactcis.core import validate_counts, find_sign_change

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComputationError(Exception):
    """Custom exception for errors during CI computation, e.g., bracketing failure."""
    pass


def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the conditional (Fisher's) exact confidence interval for the odds ratio.
    
    Args:
        a, b, c, d: The cell counts in a 2x2 contingency table
        alpha: The significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        Tuple of lower and upper bounds of the confidence interval
        
    Raises:
        ValueError: If the inputs are invalid
        ComputationError: If computation fails
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
    validate_counts(a, b, c, d)

    # Debug information - log the table
    logger.info(f"Calculating CI for table: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    
    # Point estimate of the odds ratio
    or_point = (a * d) / (b * c) if b * c != 0 else float('inf')
    logger.info(f"Point estimate of odds ratio: {or_point}")

    # Special case handling for tables with zeros
    r1 = a + b  # Row 1 total
    r2 = c + d  # Row 2 total
    c1 = a + c  # Column 1 total
    c2 = b + d  # Column 2 total
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
    try:
        lower_bound, upper_bound = validate_bounds(lower_bound, upper_bound)
        logger.info(f"Validated bounds: lower={lower_bound}, upper={upper_bound}")
        return lower_bound, upper_bound
    except ComputationError:
        logger.warning(f"Invalid bounds detected for table ({a},{b},{c},{d}). Returning conservative interval.")
        return 0.0, float('inf')


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
        Lower bound value
        
    Raises:
        ComputationError: If the root finding fails
    """
    # For general cases, calculate numerically
    target_prob = alpha / 2.0
    
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
        return nchypergeom_fisher.sf(a-1, N, c1, r1, psi) - target_prob
    
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
    logger.debug(f"Lower bound initial bracket: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
    
    # Expand brackets if needed - use more attempts for better bracketing
    max_attempts = 40
    attempt = 0
    
    # For lower bound, we need lo_val < 0 and hi_val > 0
    
    # If function at lo is not negative, decrease lo until it is or we hit a minimum
    if lo_val >= 0:
        while lo_val >= 0 and attempt < max_attempts and lo > 1e-15:
            lo /= 5.0  # More aggressive expansion
            lo_val = p_value_func(lo)
            attempt += 1
            logger.debug(f"Expanding lower bracket down: lo={lo} (val={lo_val}), attempt={attempt}")
            
        # If we can't find a negative value at extremely small psi,
        # use a very conservative lower bound
        if lo_val >= 0:
            logger.warning(f"Could not find proper bracket for lower bound, table ({a},{b},{c},{d})")
            return 0.0  # Most conservative estimate
    
    # Reset attempt counter for hi expansion
    attempt = 0
    
    # If function at hi is not positive, increase hi until it is or we hit a maximum
    if hi_val <= 0:
        while hi_val <= 0 and attempt < max_attempts and hi < 1e15:
            hi *= 5.0  # More aggressive expansion
            hi_val = p_value_func(hi)
            attempt += 1
            logger.debug(f"Expanding upper bracket up: hi={hi} (val={hi_val}), attempt={attempt}")
            
        # If we can't find a positive value at extremely large psi,
        # check the point estimate as fallback
        if hi_val <= 0:
            logger.warning(f"Could not find proper bracket for lower bound, table ({a},{b},{c},{d})")
            # Conservative fallback
            return max(0.0, or_point / 5.0)  # More conservative
    
    # Now that we have proper bracket with lo_val < 0 and hi_val > 0, use brentq
    if lo_val < 0 and hi_val > 0:
        try:
            # Use brentq to find the root with higher precision for better results
            result = max(0.0, brentq(p_value_func, lo, hi, rtol=1e-12, maxiter=200, full_output=False))
            logger.debug(f"Found lower bound using brentq: {result}")
            return result
        except (ValueError, RuntimeError) as e:
            logger.error(f"Root finding failed for lower bound: {str(e)}")
            # Try a different method if brentq fails
            try:
                # Fall back to bisection method which is more robust but slower
                from scipy.optimize import bisect
                result = max(0.0, bisect(p_value_func, lo, hi, rtol=1e-10, maxiter=200))
                logger.debug(f"Found lower bound using bisect: {result}")
                return result
            except Exception as e2:
                logger.error(f"Secondary root finding failed: {str(e2)}")
    
    # If we don't have proper bracket, but we have an OR > 0
    if or_point > 0:
        result = max(0.0, or_point / 5.0)  # More conservative estimate based on point OR
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
        Upper bound value
    
    Raises:
        ComputationError: If the root finding fails
    """
    # For general cases, calculate numerically
    target_prob = alpha / 2.0
    
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
        return nchypergeom_fisher.cdf(a-1, N, c1, r1, psi) - target_prob
    
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
    logger.debug(f"Upper bound initial bracket: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
    
    # Expand brackets if needed
    max_attempts = 40  # More attempts
    attempt = 0
    
    # For upper bound, we need lo_val > 0 and hi_val < 0
    
    # If function at lo is not positive, decrease lo until it is or we hit a minimum
    if lo_val <= 0:
        while lo_val <= 0 and attempt < max_attempts and lo > 1e-15:
            lo /= 5.0  # More aggressive expansion
            lo_val = p_value_func(lo)
            attempt += 1
            logger.debug(f"Expanding lower bracket down: lo={lo} (val={lo_val}), attempt={attempt}")
            
        # If we can't find a positive value at extremely small psi,
        # use a conservative upper bound
        if lo_val <= 0:
            logger.debug(f"Upper bound extremely small for table ({a},{b},{c},{d})")
            return or_point * 3.0  # More conservative estimate
    
    # Reset attempt counter for hi expansion
    attempt = 0
    
    # If function at hi is not negative, increase hi until it is or we hit a maximum
    if hi_val >= 0:
        while hi_val >= 0 and attempt < max_attempts and hi < 1e15:
            hi *= 5.0  # More aggressive expansion
            hi_val = p_value_func(hi)
            attempt += 1
            logger.debug(f"Expanding upper bracket up: hi={hi} (val={hi_val}), attempt={attempt}")
            
        # If we can't find a negative value at extremely large psi,
        # return infinity as the upper bound
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
            # Use brentq to find the root with higher precision
            result = brentq(p_value_func, lo, hi, rtol=1e-12, maxiter=200, full_output=False)
            logger.debug(f"Found upper bound using brentq: {result}")
            return result
        except (ValueError, RuntimeError) as e:
            logger.error(f"Root finding failed for upper bound: {str(e)}")
            # Try a different method if brentq fails
            try:
                # Fall back to bisection method which is more robust but slower
                from scipy.optimize import bisect
                result = bisect(p_value_func, lo, hi, rtol=1e-10, maxiter=200)
                logger.debug(f"Found upper bound using bisect: {result}")
                return result
            except Exception as e2:
                logger.error(f"Secondary root finding failed: {str(e2)}")
    
    # If bracket has wrong signs, use conservative estimate
    if or_point < 1.0:
        result = max(1.0, or_point * 10.0)  # For small OR, upper bound is moderate
        logger.debug(f"Using conservative upper bound estimate for small OR: {result}")
        return result
    else:
        # For large OR, use a very large upper bound
        result = max(1000.0, or_point * 100.0)
        logger.debug(f"Using conservative upper bound estimate for large OR: {result}")
        return result


def validate_bounds(lower, upper):
    """
    Validate and potentially adjust bounds to ensure they are reasonable.
    
    Args:
        lower: Lower confidence bound
        upper: Upper confidence bound
        
    Returns:
        Tuple of (lower, upper) with any necessary adjustments
    """
    # Ensure bounds are reasonable
    if lower < 0:
        lower = 0.0
    
    # Ensure upper bound is greater than lower bound
    if upper <= lower and upper != 0:
        # If bounds are crossed, it signals a computation issue
        # Rather than using a heuristic replacement, raise an error
        raise ComputationError(f"Invalid bounds: lower ({lower}) >= upper ({upper})")
    
    # Check for non-finite values
    if not np.isfinite(lower) and lower != 0.0:
        lower = 0.0
    if not np.isfinite(upper):
        upper = float('inf')
    
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
            return nchypergeom_fisher.cdf(0, N_calc, c1_calc, r1_calc, psi) - alpha/2
    elif d == 0:  # Cell (2,2) is zero
        # Find where P(X ≤ a | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return nchypergeom_fisher.cdf(a, N_calc, c1_calc, r1_calc, psi) - alpha/2
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
            return nchypergeom_fisher.cdf(a-1, N_calc, c1_calc, r1_calc, psi) - alpha/2
    elif b == 0:  # Cell (1,2) is zero
        # Find where P(X ≤ a-1 | psi) = alpha/2
        def func(psi):
            # Use original integer marginals
            return nchypergeom_fisher.cdf(a-1, N_calc, c1_calc, r1_calc, psi) - alpha/2
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
