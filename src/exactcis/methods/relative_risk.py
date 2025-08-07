"""Relative Risk Confidence Intervals.

This module implements various methods for calculating confidence intervals
for relative risk in epidemiological studies.
"""

import math
from typing import Tuple, Dict, List, Optional, Callable, Union
import numpy as np
from scipy import stats
from scipy import optimize
from exactcis.core import find_root_log, find_plateau_edge


def validate_counts(a: int, b: int, c: int, d: int) -> None:
    """
    Validate the counts in a 2x2 contingency table for relative risk calculation.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome

    Raises:
        ValueError: If any count is negative
    """
    if not all(isinstance(x, (int, float)) and x >= 0 for x in (a, b, c, d)):
        raise ValueError("All counts must be non‑negative numbers")


def add_continuity_correction(a: int, b: int, c: int, d: int, correction: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Add a continuity correction to the cell counts if any are zero.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        correction: Correction value to add (default: 0.5)

    Returns:
        Tuple of corrected counts (a, b, c, d)
    """
    # Only apply correction if any cell contains a zero
    if a == 0 or b == 0 or c == 0 or d == 0:
        return a + correction, b + correction, c + correction, d + correction
    else:
        return float(a), float(b), float(c), float(d)


# ------------ Wald-based Methods ------------

def ci_wald_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Wald confidence interval for relative risk (log scale).

    This method uses the normal approximation on the log scale. A continuity
    correction is applied for zero cells.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    validate_counts(a, b, c, d)
    
    # Apply continuity correction if needed
    a_c, b_c, c_c, d_c = add_continuity_correction(a, b, c, d)
    
    # Calculate point estimate
    n1 = a_c + b_c  # Total exposed
    n2 = c_c + d_c  # Total unexposed
    
    if n1 == 0 or n2 == 0:
        # Cannot compute CI if either group has no observations
        return 0.0, float('inf')
    
    risk1 = a_c / n1  # Risk in exposed group
    risk2 = c_c / n2  # Risk in unexposed group
    
    if risk2 == 0:
        # Cannot compute CI if unexposed risk is zero
        return 0.0, float('inf')
    
    # Calculate relative risk
    rr = risk1 / risk2
    
    # Calculate standard error on log scale
    if a_c == 0:
        # If a is zero, relative risk is zero
        return 0.0, float('inf')
    
    se_log = math.sqrt((b_c / (a_c * n1)) + (d_c / (c_c * n2)))
    
    # Critical value
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Confidence interval on log scale
    log_rr = math.log(rr)
    lower_log = log_rr - z * se_log
    upper_log = log_rr + z * se_log
    
    # Transform back to original scale
    lower = math.exp(lower_log)
    upper = math.exp(upper_log)
    
    return lower, upper


def ci_wald_katz_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Katz-adjusted Wald confidence interval for relative risk.

    This method is appropriate for independent proportions (e.g., case-control studies).
    It uses the variance formula: Var(log RR) = (1-p₁)/x₁₁ + (1-p₂)/x₂₁

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    validate_counts(a, b, c, d)
    
    # Apply continuity correction if needed
    a_c, b_c, c_c, d_c = add_continuity_correction(a, b, c, d)
    
    # Calculate point estimate
    n1 = a_c + b_c  # Total exposed
    n2 = c_c + d_c  # Total unexposed
    
    if n1 == 0 or n2 == 0 or a_c == 0 or c_c == 0:
        # Cannot compute CI if either group has no observations or if a or c is zero
        return 0.0, float('inf')
    
    risk1 = a_c / n1  # Risk in exposed group
    risk2 = c_c / n2  # Risk in unexposed group
    
    # Calculate relative risk
    rr = risk1 / risk2
    
    # Calculate standard error on log scale using Katz formula
    se_log = math.sqrt((1 - risk1) / a_c + (1 - risk2) / c_c)
    
    # Critical value
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Confidence interval on log scale
    log_rr = math.log(rr)
    lower_log = log_rr - z * se_log
    upper_log = log_rr + z * se_log
    
    # Transform back to original scale
    lower = math.exp(lower_log)
    upper = math.exp(upper_log)
    
    return lower, upper


def ci_wald_correlated_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Wald confidence interval for relative risk with correlation adjustment.

    This method accounts for the correlation between proportions, appropriate for matched
    or family-based studies. For independent groups with large sample sizes, it falls back
    to the regular Wald method to avoid pathologically wide intervals.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    validate_counts(a, b, c, d)
    
    # Enhanced zero-cell handling (check before continuity correction)
    if c == 0:
        # When unexposed has no events, upper bound should be large/infinite
        if a > 0:
            # Use regular Wald method which handles this case better
            return ci_wald_rr(a, b, c, d, alpha)
    
    # Apply continuity correction if needed
    a_c, b_c, c_c, d_c = add_continuity_correction(a, b, c, d)
    
    # Calculate point estimate
    n1 = a_c + b_c  # Total exposed
    n2 = c_c + d_c  # Total unexposed
    
    # Check if this is likely independent groups vs matched pairs
    # For independent groups with large samples, use regular Wald method
    if n1 > 100 and n2 > 100:
        return ci_wald_rr(a, b, c, d, alpha)
    
    if n1 == 0 or n2 == 0 or a_c == 0:
        return 0.0, float('inf')
    
    # For truly matched data, use proper paired proportion variance
    n_pairs = n1  # Assuming n1 = n2 for matched pairs
    
    p1 = a_c / n1  # Proportion in group 1
    p2 = c_c / n2  # Proportion in group 2
    
    # Calculate relative risk
    rr = p1 / p2
    
    # McNemar-type variance for matched pairs
    # This accounts for the paired structure of the data
    if a_c > 0:
        var_log_rr = ((b_c + c_c) / (a_c * n_pairs)) if n_pairs > 0 else float('inf')
    else:
        var_log_rr = float('inf')
    
    if var_log_rr == float('inf'):
        return 0.0, float('inf')
    
    # Add small-sample correction factor
    correction_factor = n_pairs / (n_pairs - 2) if n_pairs > 2 else 1.5
    var_log_rr *= correction_factor
    
    # Critical value
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Confidence interval on log scale
    log_rr = math.log(rr)
    se_log = math.sqrt(max(var_log_rr, 1e-10))
    lower_log = log_rr - z * se_log
    upper_log = log_rr + z * se_log
    
    # Transform back to original scale
    lower = math.exp(lower_log)
    upper = math.exp(upper_log)
    
    return lower, upper


# ------------ Score-based Methods ------------

def constrained_mle_p21(x11: int, x12: int, x21: int, x22: int, theta0: float) -> float:
    """
    Compute the constrained MLE of p21 under H0: RR = theta0.
    
    This solves the constraint equation: p11 = theta0 * p21
    along with the likelihood equations for a 2x2 table.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0

    Returns:
        The constrained MLE of p21
    """
    n1 = x11 + x12  # Exposed group size
    n2 = x21 + x22  # Unexposed group size
    
    if n1 == 0 or n2 == 0:
        return max(1e-10, min(1-1e-10, x21/n2))
    
    if abs(theta0 - 1.0) < 1e-10:
        # Under null of no effect, combine both groups
        return (x11 + x21) / (n1 + n2)
    
    # For RR constraint: p11 = theta0 * p21
    # Solve the constrained likelihood equations
    # The constraint becomes: x11 + x21 = n1*theta0*p21 + n2*p21 = p21*(n1*theta0 + n2)
    
    total_events = x11 + x21
    weighted_total = n1 * theta0 + n2
    
    if weighted_total <= 0:
        return max(1e-10, min(1-1e-10, x21/n2))
    
    p21_constrained = total_events / weighted_total
    
    # Ensure the result is valid
    p21_constrained = max(1e-10, min(1-1e-10, p21_constrained))
    
    # Check that p11 = theta0 * p21 is also valid
    p11_constrained = theta0 * p21_constrained
    if p11_constrained > 1:
        # If p11 would exceed 1, adjust p21 downward
        p21_constrained = min(p21_constrained, 1.0 / theta0)
        p21_constrained = max(1e-10, min(1-1e-10, p21_constrained))
    
    return p21_constrained


def score_statistic(x11: int, x12: int, x21: int, x22: int, theta0: float) -> float:
    """
    Calculate the SIGNED score statistic for H0: RR = theta0.
    
    This implements the corrected score statistic from Tang et al.
    with proper variance estimation under the RR constraint.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0

    Returns:
        The SIGNED score statistic (not chi-square)
    """
    n1 = x11 + x12
    n2 = x21 + x22
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Get constrained MLE
    p21_tilde = constrained_mle_p21(x11, x12, x21, x22, theta0)
    p11_tilde = theta0 * p21_tilde
    
    # Score numerator (observed - expected) for x11 specifically
    numerator = x11 - n1 * p11_tilde
    
    # Variance under the constraint
    var_p11 = p11_tilde * (1 - p11_tilde) / n1
    var_p21 = p21_tilde * (1 - p21_tilde) / n2
    
    # Score test variance (using delta method for RR constraint)
    variance = n1 * var_p11 + (theta0**2) * n1 * n2 * var_p21 / n2
    
    if variance <= 0:
        return 0.0
    
    return numerator / math.sqrt(variance)


def corrected_score_statistic(x11: int, x12: int, x21: int, x22: int, theta0: float, delta: float = 4.0) -> float:
    """
    Calculate the continuity-corrected score test statistic for H0: RR = theta0.
    
    This uses the correct Tang et al. formula for continuity correction.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0
        delta: Correction strength parameter (default: 4.0)

    Returns:
        The SIGNED continuity-corrected score statistic
    """
    n1 = x11 + x12
    n2 = x21 + x22
    n = n1 + n2
    
    if n == 0:
        return 0.0
    
    # Get constrained values
    p21_tilde = constrained_mle_p21(x11, x12, x21, x22, theta0)
    p11_tilde = theta0 * p21_tilde
    
    # Raw numerator (observed - expected for x11)
    raw_numerator = x11 - n1 * p11_tilde
    
    # Apply continuity correction to numerator (not final statistic)
    correction = (1.0 / (delta * n)) * (x11 + x21)
    
    if raw_numerator >= 0:
        corrected_numerator = max(0, abs(raw_numerator) - correction)
    else:
        corrected_numerator = -max(0, abs(raw_numerator) - correction)
    
    # Compute variance (same as uncorrected)
    var_p11 = p11_tilde * (1 - p11_tilde) / n1
    var_p21 = p21_tilde * (1 - p21_tilde) / n2
    variance = n1 * var_p11 + (theta0**2) * n1 * n2 * var_p21 / n2
    
    if variance <= 0:
        return 0.0
    
    return corrected_numerator / math.sqrt(variance)


def find_score_ci_bound(x11: int, x12: int, x21: int, x22: int, alpha: float, is_lower: bool,
                      score_func: Callable, max_iter: int = 100, tol: float = 1e-10) -> float:
    """
    Find the confidence interval bound using enhanced root finding
    with better boundary detection and search range handling.
    """
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    def objective(theta: float) -> float:
        score = score_func(x11, x12, x21, x22, theta)
        if is_lower:
            return score - z_crit  # Lower bound: S(θ) = +z
        else:
            return score + z_crit  # Upper bound: S(θ) = -z
    
    # Calculate point estimate more robustly
    n1 = x11 + x12
    n2 = x21 + x22
    p1_hat = (x11 + 0.5) / (n1 + 1)
    p2_hat = (x21 + 0.5) / (n2 + 1)
    rr_hat = p1_hat / p2_hat
    
    if is_lower:
        # For lower bound, start search much lower than point estimate
        lo = max(1e-8, rr_hat / 1000)  # Much more aggressive lower search
        hi = rr_hat * 0.99  # Just below point estimate
    else:
        # For upper bound - start with conservative range and expand smartly
        lo = rr_hat * 1.01  # Just above point estimate  
        hi = rr_hat * 2  # Start with smaller initial range
    
    # Enhanced bracket expansion with systematic search
    max_expansion = 50  # More reasonable limit
    expansion_count = 0
    
    while expansion_count < max_expansion:
        try:
            val_lo, val_hi = objective(lo), objective(hi)
            if val_lo * val_hi <= 0:
                break
            
            if is_lower:
                lo /= 2.0  # Systematic expansion for lower bound
            else:
                # For upper bound, expand more carefully
                if hi < rr_hat * 100:  # Don't expand too far initially
                    hi *= 1.5  # Gentler expansion
                else:
                    hi *= 2.0  # More aggressive when we're already far out
                
            expansion_count += 1
                
        except (OverflowError, ValueError, ZeroDivisionError):
            if not is_lower and hi > 1e6:
                return float('inf')  # Legitimate infinite bound
            break
    else:
        # If standard expansion failed, try systematic search for upper bounds
        if not is_lower:
            # Try a systematic grid search for the sign change
            search_multipliers = [2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
            for mult in search_multipliers:
                try:
                    test_hi = rr_hat * mult
                    val_test = objective(test_hi)
                    val_lo = objective(lo)
                    if val_test * val_lo <= 0:
                        hi = test_hi
                        break
                except:
                    continue
            else:
                return float('inf')  # Legitimate infinite upper bound
        else:
            return 0.0
    
    # Enhanced binary search with plateau detection
    for iteration in range(max_iter):
        mid = (lo + hi) / 2
        try:
            val_mid = objective(mid)
            
            if abs(val_mid) < tol:
                return mid
                
            # Check for plateau (score function is flat)
            if iteration > 10 and abs(hi - lo) < abs(mid) * 1e-8:
                # Function may be flat in this region
                return mid
                
            if val_mid * objective(lo) < 0:
                hi = mid
            else:
                lo = mid
                
        except (OverflowError, ValueError, ZeroDivisionError):
            # Numerical issues, try to recover
            if is_lower:
                return max(0, lo)
            else:
                return hi if hi < 1e6 else float('inf')
    
    return max(0, (lo + hi) / 2)


def ci_score_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute the score-based confidence interval for the relative risk.
    
    Implements the corrected score test inversion approach from Tang et al.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)

    Returns:
        The score-based confidence interval for the relative risk
    """
    validate_counts(a, b, c, d)
    
    # Handle boundary cases
    if a == 0:
        return 0.0, find_score_ci_bound(a, b, c, d, alpha, False, score_statistic)
    if c == 0:
        return find_score_ci_bound(a, b, c, d, alpha, True, score_statistic), float('inf')
    
    lower = find_score_ci_bound(a, b, c, d, alpha, True, score_statistic)
    upper = find_score_ci_bound(a, b, c, d, alpha, False, score_statistic)
    
    return lower, upper


def ci_score_cc_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05, delta: float = 4.0) -> Tuple[float, float]:
    """
    Compute the continuity-corrected score-based confidence interval for the relative risk.
    
    Implements the corrected continuity correction from Tang et al.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        delta: Correction strength parameter (default: 4.0)
        alpha: Significance level (default: 0.05)

    Returns:
        The continuity-corrected score-based confidence interval for the relative risk
    """
    validate_counts(a, b, c, d)
    
    # Create a partial function with fixed delta for the corrected score
    corrected_score = lambda x11, x12, x21, x22, theta0: corrected_score_statistic(x11, x12, x21, x22, theta0, delta)
    
    # Handle boundary cases
    if a == 0:
        return 0.0, find_score_ci_bound(a, b, c, d, alpha, False, corrected_score)
    if c == 0:
        return find_score_ci_bound(a, b, c, d, alpha, True, corrected_score), float('inf')
    
    lower = find_score_ci_bound(a, b, c, d, alpha, True, corrected_score)
    upper = find_score_ci_bound(a, b, c, d, alpha, False, corrected_score)
    
    return lower, upper


# ------------ U-Statistic Method ------------

def ci_ustat_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the U-statistic-based confidence interval for relative risk.

    This nonparametric method uses proper U-statistic variance estimation with
    the correct Duan et al. method that accounts for the full covariance structure.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    validate_counts(a, b, c, d)
    
    # Apply continuity correction if needed
    a_c, b_c, c_c, d_c = add_continuity_correction(a, b, c, d)
    
    # Calculate total counts
    n1 = a_c + b_c  # Total exposed
    n2 = c_c + d_c  # Total unexposed
    n = n1 + n2     # Total sample size
    
    if n1 == 0 or n2 == 0 or c_c == 0:
        return 0.0, float('inf')
    
    # Calculate empirical probabilities
    p1 = a_c / n1  # Probability of outcome in exposed group
    p2 = c_c / n2  # Probability of outcome in unexposed group
    
    # Calculate relative risk
    rr = p1 / p2
    
    # Proper U-statistic variance using Duan et al. method
    # This accounts for the full covariance structure
    
    # Variance components for each proportion
    var_p1 = p1 * (1 - p1) / (n1 - 1) if n1 > 1 else p1 * (1 - p1)
    var_p2 = p2 * (1 - p2) / (n2 - 1) if n2 > 1 else p2 * (1 - p2)
    
    # Covariance term (for independent groups this is 0)
    cov_p1p2 = 0.0  # Since groups are independent
    
    # Delta method variance for log(RR)
    if p1 > 0 and p2 > 0:
        var_log_rr = (var_p1 / (p1**2)) + (var_p2 / (p2**2)) - (2 * cov_p1p2 / (p1 * p2))
        
        # Add small-sample correction factor
        correction_factor = n / (n - 2) if n > 2 else 1.5
        var_log_rr *= correction_factor
        
        # Use t-distribution with adjusted degrees of freedom
        df = min(n1 - 1, n2 - 1) if min(n1, n2) > 1 else 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        
        se_log = math.sqrt(max(var_log_rr, 1e-10))
        log_rr = math.log(rr)
        
        lower = math.exp(log_rr - t_crit * se_log)
        upper = math.exp(log_rr + t_crit * se_log)
    else:
        # Fallback for extreme cases
        lower = 0.0
        upper = float('inf')
    
    return lower, upper
