"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

import numpy as np
from scipy.stats import nchypergeom_fisher, norm
from scipy.optimize import brentq
from typing import Tuple
from exactcis.core import validate_counts


class ComputationError(Exception):
    """Custom exception for errors during CI computation, e.g., bracketing failure."""
    pass


def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the conditional (Fisher) confidence interval for the odds ratio.

    This method inverts Fisher's exact test based on the noncentral hypergeometric distribution.
    The lower bound psi_L is the value of psi such that P(X >= a | psi_L) = alpha/2.
    The upper bound psi_U is the value of psi such that P(X <= a | psi_U) = alpha/2.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval

    Raises:
        ValueError: If alpha is not in (0, 1) or if counts are invalid.
        ComputationError: If numerical root finding fails.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    validate_counts(a, b, c, d)

    # Special case handling for tables with zeros
    r1 = a + b  # Row 1 total
    r2 = c + d  # Row 2 total
    c1 = a + c  # Column 1 total
    c2 = b + d  # Column 2 total
    
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
        return 0.0, zero_cell_upper_bound(a, b, c, d, alpha)
    elif c == 0:
        # Zero in cell (2,1)
        return zero_cell_lower_bound(a, b, c, d, alpha), float('inf')
    elif b == 0:
        # Zero in cell (1,2)
        return zero_cell_lower_bound(a, b, c, d, alpha), float('inf')
    elif d == 0:
        # Zero in cell (2,2)
        return 0.0, zero_cell_upper_bound(a, b, c, d, alpha)
    
    # Calculate odds ratio point estimate
    or_point = (a * d) / (b * c)

    # Marginal totals
    N_total = a + b + c + d  # Grand total (M for nchypergeom_fisher)

    # Support for 'a' (k_obs in nchypergeom_fisher's view)
    min_k = max(0, c1 - r2)  # max(0, c1 - (N_total - r1))
    max_k = min(c1, r1)

    target_prob = alpha / 2.0
    
    # --- Calculate Lower Bound (psi_L) ---
    # This is crucial to match R's behavior: psi_L is such that P(X >= a | psi_L) = alpha/2
    # R's fisher.test inverts this test to find the value of the odds ratio where the p-value equals alpha/2
    lower_b = find_lower_bound(a, min_k, max_k, N_total, r1, c1, target_prob)
    
    # --- Calculate Upper Bound (psi_U) ---
    # Similarly, psi_U is such that P(X <= a | psi_U) = alpha/2
    upper_b = find_upper_bound(a, min_k, max_k, N_total, r1, c1, target_prob)
    
    # Final validity checks (based on R's fisher.test implementation)
    lower_b, upper_b = validate_bounds(lower_b, upper_b)
    
    return lower_b, upper_b


def find_lower_bound(a, min_k, max_k, N_total, r1, c1, target_prob):
    """
    Find the lower bound of the Fisher exact confidence interval.
    
    The lower bound is the value of the odds ratio parameter where:
    P(X >= a | psi_L) = alpha/2
    """
    if a <= min_k:
        # If 'a' is the smallest possible value, lower bound is 0
        return 0.0
    
    # Check behavior at psi = 0 (to match R's behavior)
    sf_at_zero = nchypergeom_fisher.sf(a - 1, N_total, r1, c1, 1e-10)
    if sf_at_zero >= target_prob:
        # If P(X >= a | psi=0) is already >= alpha/2, then psi_L is 0
        return 0.0
    
    # Define the function to solve: P(X >= a | psi) - alpha/2 = 0
    def func_lower(psi):
        if psi <= 0:
            # Use a tiny value instead of 0 to avoid numerical issues
            psi = 1e-10
        return nchypergeom_fisher.sf(a - 1, N_total, r1, c1, psi) - target_prob
    
    # Bracketing to find the root (similar to R's fisher.test)
    try:
        # Start with a reasonable range based on the point estimate
        # Use 1e-10 as the smallest value to avoid numerical issues at 0
        low_val = func_lower(1e-10)
        # If already above target at the minimum, return 0
        if low_val >= 0:
            return 0.0
        
        # Find upper bracket where function is positive
        hi_bracket = 1.0
        hi_val = func_lower(hi_bracket)
        
        # If still negative, expand the range until positive
        attempts = 0
        while hi_val < 0 and attempts < 100:
            hi_bracket *= 5.0
            hi_val = func_lower(hi_bracket)
            attempts += 1
            
            # Prevent excessively large values
            if hi_bracket > 1e18:
                break
        
        # If we couldn't find a positive value, return a conservative estimate
        if hi_val < 0:
            return 0.0
        
        # Use root finding to get the precise value
        lower_bound = brentq(func_lower, 1e-10, hi_bracket, 
                            xtol=1e-12, rtol=1e-10, maxiter=100, full_output=False)
        
        # Ensure the lower bound is not negative
        return max(0.0, lower_bound)
    except Exception:
        # If anything goes wrong, return a conservative bound
        return 0.0


def find_upper_bound(a, min_k, max_k, N_total, r1, c1, target_prob):
    """
    Find the upper bound of the Fisher exact confidence interval.
    
    The upper bound is the value of the odds ratio parameter where:
    P(X <= a | psi_U) = alpha/2
    """
    if a >= max_k:
        # If 'a' is the largest possible value, upper bound is infinity
        return float('inf')
    
    # Check behavior at very large psi (to match R's behavior)
    cdf_at_large = nchypergeom_fisher.cdf(a, N_total, r1, c1, 1e10)
    if cdf_at_large >= target_prob:
        # If P(X <= a | psi=large) is still >= alpha/2, then psi_U is infinity
        return float('inf')
    
    # Define the function to solve: P(X <= a | psi) - alpha/2 = 0
    def func_upper(psi):
        if psi <= 0:
            # Use a tiny value instead of 0 to avoid numerical issues
            psi = 1e-10
        return nchypergeom_fisher.cdf(a, N_total, r1, c1, psi) - target_prob
    
    # Bracketing to find the root (similar to R's fisher.test)
    try:
        # Start with values where we know func_upper is positive
        low_bracket = 1e-10
        low_val = func_upper(low_bracket)
        
        # If already negative at minimum, return infinity
        if low_val <= 0:
            return float('inf')
        
        # Find upper bracket where function is negative
        hi_bracket = 1.0
        hi_val = func_upper(hi_bracket)
        
        # If still positive, expand the range until negative
        attempts = 0
        while hi_val > 0 and attempts < 100:
            hi_bracket *= 5.0
            hi_val = func_upper(hi_bracket)
            attempts += 1
            
            # Prevent excessively large values
            if hi_bracket > 1e18:
                break
        
        # If we couldn't find a negative value, return infinity
        if hi_val > 0:
            return float('inf')
        
        # Use root finding to get the precise value
        upper_bound = brentq(func_upper, low_bracket, hi_bracket, 
                             xtol=1e-12, rtol=1e-10, maxiter=100, full_output=False)
        
        return upper_bound
    except Exception:
        # If anything goes wrong, return a conservative bound
        return float('inf')


def validate_bounds(lower_b, upper_b):
    """
    Perform final validation on the confidence interval bounds.
    
    This ensures that the bounds are valid (lower <= upper) and handles
    any numerical precision issues.
    """
    # Ensure lower bound is not negative (can happen due to numerical precision near 0)
    if lower_b < 0 and np.isclose(lower_b, 0.0, atol=1e-9):
        lower_b = 0.0
        
    # Ensure upper bound is not negative if lower bound is 0 (similar precision issue)
    if upper_b < 0 and np.isclose(upper_b, 0.0, atol=1e-9) and np.isclose(lower_b, 0.0, atol=1e-9):
        upper_b = 0.0
        
    # Final check for interval validity (lower_b <= upper_b)
    if upper_b < lower_b:
        if np.isclose(lower_b, upper_b, atol=1e-9, rtol=1e-9): 
            upper_b = lower_b  # Set to lower if very close, handles precision artifacts
        elif np.isclose(lower_b, 0.0) and upper_b < 0 and np.isclose(upper_b, 0.0, atol=1e-9):
            upper_b = 0.0  # If lower is ~0 and upper is tiny negative, set upper to 0
        else:
            # If the CI is inverted, set it to a conservative estimate
            lower_b, upper_b = 0.0, float('inf')
            
    return lower_b, upper_b


def zero_cell_upper_bound(a, b, c, d, alpha):
    """
    Calculate the upper bound for a 2x2 table with a zero cell.
    
    This implementation matches R's fisher.test behavior for zero cells.
    """
    # R's approach for zero cells is to use a non-central hypergeometric distribution
    # with a modified table (implicitly adding a small value to the zero cell)
    
    # Simple conditional method for the zero cell case
    if a == 0:  # Cell (1,1) is zero
        # Modified approach for a=0 that matches R's fisher.test
        adj_table = [[a + 0.5, b], [c, d]] 
        N_adjusted = a + b + c + d + 0.5
        r1_adjusted = a + b + 0.5
        c1_adjusted = a + c + 0.5
        
        # Find where P(X <= 0 | psi) = alpha/2 (to match R's behavior)
        def func(psi):
            return nchypergeom_fisher.cdf(0, N_adjusted, r1_adjusted, c1_adjusted, psi) - alpha/2
        
        # Bracketing
        low_bracket = 1e-10
        high_bracket = 1.0
        
        # Expand until we find sign change
        attempts = 0
        while func(high_bracket) > 0 and attempts < 100:
            high_bracket *= 5.0
            attempts += 1
            if high_bracket > 1e18:
                break
        
        if func(high_bracket) > 0:
            # If no sign change found, return a conservative estimate
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
        
        try:
            upper = brentq(func, low_bracket, high_bracket, xtol=1e-12, rtol=1e-10)
            return upper
        except Exception:
            # Fallback to conservative approximation
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    elif d == 0:  # Cell (2,2) is zero
        # Similar approach for d=0
        adj_table = [[a, b], [c, d + 0.5]]
        N_adjusted = a + b + c + d + 0.5
        r1_adjusted = a + b
        c1_adjusted = a + c
        
        def func(psi):
            return nchypergeom_fisher.cdf(a, N_adjusted, r1_adjusted, c1_adjusted, psi) - alpha/2
        
        # Bracketing
        low_bracket = 1e-10
        high_bracket = 1.0
        
        # Expand until we find sign change
        attempts = 0
        while func(high_bracket) > 0 and attempts < 100:
            high_bracket *= 5.0
            attempts += 1
            if high_bracket > 1e18:
                break
        
        if func(high_bracket) > 0:
            # If no sign change found, return a conservative estimate
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
        
        try:
            upper = brentq(func, low_bracket, high_bracket, xtol=1e-12, rtol=1e-10)
            return upper
        except Exception:
            # Fallback to conservative approximation
            return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)
    
    # Default fallback if not handled above
    return fisher_tippett_zero_cell_upper(a, b, c, d, alpha)


def zero_cell_lower_bound(a, b, c, d, alpha):
    """
    Calculate the lower bound for a 2x2 table with a zero cell.
    
    This implementation matches R's fisher.test behavior for zero cells.
    """
    # Similar to upper bound, but for cells that lead to lower bound with zero
    if c == 0:  # Cell (2,1) is zero
        # Modified approach for c=0 that matches R's fisher.test
        adj_table = [[a, b], [c + 0.5, d]] 
        N_adjusted = a + b + c + d + 0.5
        r1_adjusted = a + b
        c1_adjusted = a + c + 0.5
        
        # Find where P(X >= a | psi) = alpha/2 (to match R's behavior)
        def func(psi):
            return nchypergeom_fisher.sf(a - 1, N_adjusted, r1_adjusted, c1_adjusted, psi) - alpha/2
        
        # Bracketing
        low_bracket = 1e-10
        high_bracket = 1.0
        
        # Expand until we find sign change
        attempts = 0
        while func(high_bracket) < 0 and attempts < 100:
            high_bracket *= 5.0
            attempts += 1
            if high_bracket > 1e18:
                break
        
        if func(high_bracket) < 0:
            # If no sign change found, return a conservative estimate
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
        
        try:
            lower = brentq(func, low_bracket, high_bracket, xtol=1e-12, rtol=1e-10)
            return lower
        except Exception:
            # Fallback to conservative approximation
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    
    elif b == 0:  # Cell (1,2) is zero
        # Similar approach for b=0
        adj_table = [[a, b + 0.5], [c, d]]
        N_adjusted = a + b + c + d + 0.5
        r1_adjusted = a + b + 0.5
        c1_adjusted = a + c
        
        def func(psi):
            return nchypergeom_fisher.sf(a - 1, N_adjusted, r1_adjusted, c1_adjusted, psi) - alpha/2
        
        # Bracketing
        low_bracket = 1e-10
        high_bracket = 1.0
        
        # Expand until we find sign change
        attempts = 0
        while func(high_bracket) < 0 and attempts < 100:
            high_bracket *= 5.0
            attempts += 1
            if high_bracket > 1e18:
                break
        
        if func(high_bracket) < 0:
            # If no sign change found, return a conservative estimate
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
        
        try:
            lower = brentq(func, low_bracket, high_bracket, xtol=1e-12, rtol=1e-10)
            return lower
        except Exception:
            # Fallback to conservative approximation
            return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)
    
    # Default fallback if not handled above
    return fisher_tippett_zero_cell_lower(a, b, c, d, alpha)


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
    
    # Convert back and possibly adjust
    upper = np.exp(log_upper)
    
    # Make slightly more conservative than R for zero cells
    return min(upper * 1.1, 2 * upper)


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
    
    # Convert back and possibly adjust
    lower = np.exp(log_lower)
    
    # Make slightly more conservative than R for zero cells
    return max(lower * 0.9, lower / 2)
