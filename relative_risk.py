"""Relative Risk Confidence Intervals.

This module implements various methods for calculating confidence intervals
for relative risk in epidemiological studies.
"""

import math
from typing import Tuple, Dict, List, Optional, Callable, Union
import numpy as np
from scipy import stats
from scipy import optimize


def calculate_relative_risk(a: int, b: int, c: int, d: int) -> float:
    """
    Calculate the point estimate of relative risk for a 2x2 contingency table.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome

    Returns:
        The relative risk estimate
    """
    n1 = a + b  # Total exposed
    n2 = c + d  # Total unexposed

    if n1 == 0 and n2 == 0:
        return 1.0  # No data
    if n1 == 0:
        return float('inf') if c > 0 else 1.0  # Risk1 undefined, if Risk2 >0 then RR=inf
    if n2 == 0:
        return 0.0 if a > 0 else 1.0  # Risk2 undefined, if Risk1 >0 then RR=0

    risk1 = a / n1  # Risk in exposed group
    risk2 = c / n2  # Risk in unexposed group

    if risk2 == 0:
        return float('inf') if risk1 > 0 else 1.0  # If risk2 is 0, RR is inf if risk1 > 0, else 1 (0/0)

    return risk1 / risk2


def batch_calculate_relative_risks(tables: List[Tuple[int, int, int, int]]) -> List[float]:
    """
    Calculate relative risks for multiple 2x2 tables in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables

    Returns:
        List of relative risk values
    """
    return [calculate_relative_risk(a, b, c, d) for a, b, c, d in tables]


def validate_counts(a: int, b: int, c: int, d: int) -> None:
    """
    Validate that the counts are non-negative integers.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)

    Raises:
        ValueError: If any count is negative or not an integer
    """
    if not all(isinstance(x, int) for x in (a, b, c, d)):
        raise ValueError("All counts must be integers")
    if any(x < 0 for x in (a, b, c, d)):
        raise ValueError("All counts must be non-negative")


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
    if 0 in (a, b, c, d):
        return (a + correction, b + correction, c + correction, d + correction)
    return (a, b, c, d)


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
    if 0 in (a, c):
        a_corr, b_corr, c_corr, d_corr = add_continuity_correction(a, b, c, d)
    else:
        a_corr, b_corr, c_corr, d_corr = a, b, c, d

    # Calculate proportions
    n1 = a_corr + b_corr
    n2 = c_corr + d_corr
    p1 = a_corr / n1
    p2 = c_corr / n2

    # Calculate relative risk
    rr = p1 / p2

    # Calculate standard error of log(RR)
    se_log_rr = math.sqrt((1 - p1) / a_corr + (1 - p2) / c_corr)

    # Calculate confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    log_rr = math.log(rr)

    lower = math.exp(log_rr - z * se_log_rr)
    upper = math.exp(log_rr + z * se_log_rr)

    return (lower, upper)


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
    if 0 in (a, c):
        a_corr, b_corr, c_corr, d_corr = add_continuity_correction(a, b, c, d)
    else:
        a_corr, b_corr, c_corr, d_corr = a, b, c, d

    # Calculate proportions
    n1 = a_corr + b_corr
    n2 = c_corr + d_corr
    p1 = a_corr / n1
    p2 = c_corr / n2

    # Calculate relative risk
    rr = p1 / p2

    # Calculate Katz-adjusted variance of log(RR)
    var_log_rr = (1 - p1) / a_corr + (1 - p2) / c_corr
    se_log_rr = math.sqrt(var_log_rr)

    # Calculate confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    log_rr = math.log(rr)

    lower = math.exp(log_rr - z * se_log_rr)
    upper = math.exp(log_rr + z * se_log_rr)

    return (lower, upper)


def ci_wald_correlated_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Wald confidence interval for relative risk with correlation adjustment.

    This method accounts for the correlation between proportions, appropriate for matched
    or family-based studies. It uses the variance formula that includes the covariance term.

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
    if 0 in (a, b, c, d):
        a_corr, b_corr, c_corr, d_corr = add_continuity_correction(a, b, c, d)
    else:
        a_corr, b_corr, c_corr, d_corr = a, b, c, d

    # Calculate totals and proportions
    n = a_corr + b_corr + c_corr + d_corr
    n1 = a_corr + b_corr
    n2 = c_corr + d_corr
    p1 = a_corr / n1
    p2 = c_corr / n2

    # Calculate relative risk
    rr = p1 / p2

    # Calculate variances
    var_p1 = p1 * (1 - p1) / n1
    var_p2 = p2 * (1 - p2) / n2

    # Calculate covariance term - critical for correlated data
    cov_p1p2 = (a_corr * d_corr - b_corr * c_corr) / (n * (n - 1))

    # Calculate variance of log(RR) with covariance
    var_log_rr = var_p1 / (p1 ** 2) + var_p2 / (p2 ** 2) - 2 * cov_p1p2 / (p1 * p2)
    se_log_rr = math.sqrt(var_log_rr)

    # Calculate confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    log_rr = math.log(rr)

    lower = math.exp(log_rr - z * se_log_rr)
    upper = math.exp(log_rr + z * se_log_rr)

    return (lower, upper)


# ------------ Score-based Methods ------------

def constrained_mle_p21(x11: int, x12: int, x21: int, x22: int, theta0: float) -> float:
    """
    Compute the constrained MLE of p21 under H0: RR = theta0.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0

    Returns:
        The constrained MLE of p21
    """
    n = x11 + x12 + x21 + x22
    n1 = x11 + x12
    n2 = x21 + x22

    # Handle special cases
    if theta0 == 0:
        return 0 if x21 == 0 else 1
    if theta0 == float('inf'):
        return 0

    # Coefficients for quadratic equation: a*p²+ b*p + c = 0
    a_coef = n * (1 + theta0)
    b_coef = -((x11 + x21) + n * theta0 + (x11 + x12) * (1 + theta0))
    c_coef = (x11 + x12) * (x11 + x21) / n

    # Solve quadratic equation
    discriminant = b_coef ** 2 - 4 * a_coef * c_coef
    if discriminant < 0:
        raise ValueError("Constrained MLE calculation failed: discriminant < 0")

    # We need the solution between 0 and 1
    p1 = (-b_coef + math.sqrt(discriminant)) / (2 * a_coef)
    p2 = (-b_coef - math.sqrt(discriminant)) / (2 * a_coef)

    # Choose the valid solution (between 0 and 1)
    if 0 <= p1 <= 1:
        return p1
    if 0 <= p2 <= 1:
        return p2

    # If neither solution is valid, choose the closest to [0,1]
    return min(max(p1, 0), 1) if abs(p1 - 0.5) < abs(p2 - 0.5) else min(max(p2, 0), 1)


def score_statistic(x11: int, x12: int, x21: int, x22: int, theta0: float) -> float:
    """
    Calculate the score test statistic for H0: RR = theta0.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0

    Returns:
        The score test statistic
    """
    n = x11 + x12 + x21 + x22

    # Calculate constrained MLE of p21
    p21_tilde = constrained_mle_p21(x11, x12, x21, x22, theta0)

    # Calculate score statistic numerator
    numerator = (x11 + x12) - (x11 + x21) * theta0

    # Calculate score statistic denominator
    denominator = math.sqrt(n * (1 + theta0) * p21_tilde + (x11 + x12 + x21) * (theta0 - 1))

    # Return score statistic
    if denominator == 0:
        return float('inf') if numerator != 0 else 0
    return numerator / denominator


def corrected_score_statistic(x11: int, x12: int, x21: int, x22: int, theta0: float, delta: float = 4.0) -> float:
    """
    Calculate the continuity-corrected score test statistic for H0: RR = theta0.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        theta0: The relative risk value under H0
        delta: Correction strength parameter (default: 4.0)

    Returns:
        The continuity-corrected score test statistic
    """
    n = x11 + x12 + x21 + x22

    # Calculate constrained MLE of p21
    p21_tilde = constrained_mle_p21(x11, x12, x21, x22, theta0)

    # Calculate score statistic numerator with continuity correction
    numerator_base = (x11 + x12) - (x11 + x21) * theta0
    correction = (x11 + x21) / (delta * n)
    numerator = abs(numerator_base) - correction
    numerator = max(numerator, 0) * (1 if numerator_base >= 0 else -1)

    # Calculate score statistic denominator
    denominator = math.sqrt(n * (1 + theta0) * p21_tilde + (x11 + x12 + x21) * (theta0 - 1))

    # Return corrected score statistic
    if denominator == 0:
        return float('inf') if numerator != 0 else 0
    return numerator / denominator


def find_score_ci_bound(x11: int, x12: int, x21: int, x22: int, alpha: float, is_lower: bool, 
                       score_func: Callable, max_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Find the confidence interval bound using root-finding on the score statistic.

    Args:
        x11: Count in cell (1,1)
        x12: Count in cell (1,2)
        x21: Count in cell (2,1)
        x22: Count in cell (2,2)
        alpha: Significance level
        is_lower: If True, find lower bound; if False, find upper bound
        score_func: The score function to use (either score_statistic or corrected_score_statistic)
        max_iter: Maximum number of iterations for root-finding
        tol: Tolerance for root-finding

    Returns:
        The confidence interval bound
    """
    # Critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Define the function to find the root of
    def f(theta):
        return score_func(x11, x12, x21, x22, theta) - (-z_crit if is_lower else z_crit)

    # Calculate point estimate
    n1 = x11 + x12
    n2 = x21 + x22

    # Apply continuity correction if needed
    if 0 in (x11, x21):
        x11_corr, x12_corr, x21_corr, x22_corr = add_continuity_correction(x11, x12, x21, x22)
    else:
        x11_corr, x12_corr, x21_corr, x22_corr = x11, x12, x21, x22

    p1 = x11_corr / (x11_corr + x12_corr) if (x11_corr + x12_corr) > 0 else 0
    p2 = x21_corr / (x21_corr + x22_corr) if (x21_corr + x22_corr) > 0 else 0

    if p2 == 0:
        return 0 if is_lower else float('inf')

    rr_point = p1 / p2

    # Adjust bounds based on point estimate and cell counts
    if is_lower:
        if x11 == 0 or n2 == 0:
            return 0
        bracket_min = max(0.001, rr_point / 100)
        bracket_max = rr_point
    else:
        if x21 == 0 or n1 == 0:
            return float('inf')
        bracket_min = rr_point
        bracket_max = min(1000, rr_point * 100)

    # Try to find the root in the bracket
    try:
        result = optimize.brentq(f, bracket_min, bracket_max, maxiter=max_iter, rtol=tol)
        return result
    except ValueError:
        # If root-finding fails, use a more robust but slower approach
        try:
            return optimize.minimize_scalar(lambda x: abs(f(x)), 
                                          bounds=[bracket_min, bracket_max], 
                                          method='bounded').x
        except:
            # If all fails, return conservative bounds
            return 0 if is_lower else float('inf')


def ci_score_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the asymptotic score confidence interval for relative risk.

    This method inverts the score test statistic and is generally recommended for
    moderate to large sample sizes.

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

    # Apply continuity correction if needed for extreme zero cells
    if (a == 0 and b == 0) or (c == 0 and d == 0):
        return (0.0, float('inf'))

    # Find lower and upper bounds
    lower = find_score_ci_bound(a, b, c, d, alpha, True, score_statistic)
    upper = find_score_ci_bound(a, b, c, d, alpha, False, score_statistic)

    return (lower, upper)


def ci_score_cc_rr(a: int, b: int, c: int, d: int, delta: float = 4.0, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the continuity-corrected score confidence interval for relative risk.

    This method applies a continuity correction to the score statistic, improving coverage
    for small sample sizes. The delta parameter controls the strength of the correction.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        delta: Correction strength parameter (default: 4.0)
            - delta = 2: High correction (ASCC-H)
            - delta = 4: Medium correction (ASCC-M) - recommended
            - delta = 8: Low correction (ASCC-L)
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    validate_counts(a, b, c, d)

    # Apply continuity correction if needed for extreme zero cells
    if (a == 0 and b == 0) or (c == 0 and d == 0):
        return (0.0, float('inf'))

    # Create a score function with the specified delta
    score_func = lambda x11, x12, x21, x22, theta0: corrected_score_statistic(x11, x12, x21, x22, theta0, delta)

    # Find lower and upper bounds
    lower = find_score_ci_bound(a, b, c, d, alpha, True, score_func)
    upper = find_score_ci_bound(a, b, c, d, alpha, False, score_func)

    return (lower, upper)


# ------------ U-Statistic Method ------------

def ci_ustat_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the U-statistic-based confidence interval for relative risk.

    This nonparametric method uses a rank-based approach and is particularly robust
    for small sample sizes and when distributional assumptions are questionable.
    It uses the t-distribution with n-1 degrees of freedom for improved coverage.

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
    if 0 in (a, b, c, d):
        a_corr, b_corr, c_corr, d_corr = add_continuity_correction(a, b, c, d)
    else:
        a_corr, b_corr, c_corr, d_corr = a, b, c, d

    # Calculate totals and proportions
    n = a_corr + b_corr + c_corr + d_corr
    n1 = a_corr + b_corr
    n2 = c_corr + d_corr
    p1 = a_corr / n1 if n1 > 0 else 0
    p2 = c_corr / n2 if n2 > 0 else 0

    # Early return for edge cases
    if p1 == 0 and p2 == 0:
        return (0.0, float('inf'))
    if p2 == 0:
        return (0.0, float('inf'))

    # Calculate relative risk
    rr = p1 / p2

    # Create 2x2 matrix for variance calculation
    counts_matrix = np.array([[a_corr, b_corr], [c_corr, d_corr]])
    row_sums = np.sum(counts_matrix, axis=1)
    col_sums = np.sum(counts_matrix, axis=0)

    # Calculate U-statistic variance for log(RR)
    # This is a simplified implementation of the complex variance expression
    v11 = (n * a_corr - row_sums[0] * col_sums[0]) / (n * (n - 1))
    v22 = (n * c_corr - row_sums[1] * col_sums[0]) / (n * (n - 1))

    # Calculate variance of log(RR) using rank-based estimation
    var_log_rr = v11 / (p1 ** 2) + v22 / (p2 ** 2)
    se_log_rr = math.sqrt(var_log_rr)

    # Use t-distribution with n-1 degrees of freedom for better small-sample coverage
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    log_rr = math.log(rr)

    lower = math.exp(log_rr - t_crit * se_log_rr)
    upper = math.exp(log_rr + t_crit * se_log_rr)

    return (lower, upper)


# ------------ Convenience Function ------------

def compute_all_rr_cis(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for the relative risk using all available methods.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary mapping method names to confidence intervals (lower_bound, upper_bound)
    """
    validate_counts(a, b, c, d)

    results = {}

    # Calculate point estimate
    rr_point = calculate_relative_risk(a, b, c, d)
    results["point_estimate"] = rr_point

    # Calculate confidence intervals using all methods
    try:
        results["wald"] = ci_wald_rr(a, b, c, d, alpha)
    except Exception as e:
        results["wald"] = (float('nan'), float('nan'))

    try:
        results["wald_katz"] = ci_wald_katz_rr(a, b, c, d, alpha)
    except Exception as e:
        results["wald_katz"] = (float('nan'), float('nan'))

    try:
        results["wald_correlated"] = ci_wald_correlated_rr(a, b, c, d, alpha)
    except Exception as e:
        results["wald_correlated"] = (float('nan'), float('nan'))

    try:
        results["score"] = ci_score_rr(a, b, c, d, alpha)
    except Exception as e:
        results["score"] = (float('nan'), float('nan'))

    try:
        results["score_cc"] = ci_score_cc_rr(a, b, c, d, alpha)
    except Exception as e:
        results["score_cc"] = (float('nan'), float('nan'))

    try:
        results["ustat"] = ci_ustat_rr(a, b, c, d, alpha)
    except Exception as e:
        results["ustat"] = (float('nan'), float('nan'))

    return results


# ------------ Batch Processing Functions ------------

def ci_method_batch(method_func: Callable, tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                   max_workers: Optional[int] = None, backend: Optional[str] = None,
                   progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Generic batch processor for confidence interval methods.

    Args:
        method_func: The confidence interval method function to use
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    # Simple sequential implementation - can be enhanced with parallel processing
    results = []
    total = len(tables)

    for i, (a, b, c, d) in enumerate(tables):
        try:
            ci = method_func(a, b, c, d, alpha)
        except Exception:
            ci = (float('nan'), float('nan'))

        results.append(ci)

        if progress_callback and i % max(1, total // 100) == 0:
            progress_callback(i / total)

    if progress_callback:
        progress_callback(1.0)

    return results


def ci_score_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                     max_workers: Optional[int] = None, backend: Optional[str] = None,
                     progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate asymptotic score confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    return ci_method_batch(ci_score_rr, tables, alpha, max_workers, backend, progress_callback)


def ci_score_cc_rr_batch(tables: List[Tuple[int, int, int, int]], delta: float = 4.0, alpha: float = 0.05,
                        max_workers: Optional[int] = None, backend: Optional[str] = None,
                        progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate continuity-corrected score confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        delta: Correction strength parameter (default: 4.0)
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    # Create a method function with fixed delta
    method_func = lambda a, b, c, d, alpha: ci_score_cc_rr(a, b, c, d, delta, alpha)
    return ci_method_batch(method_func, tables, alpha, max_workers, backend, progress_callback)


def ci_ustat_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                     max_workers: Optional[int] = None, backend: Optional[str] = None,
                     progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate U-statistic-based confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    return ci_method_batch(ci_ustat_rr, tables, alpha, max_workers, backend, progress_callback)


def ci_wald_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                    max_workers: Optional[int] = None, backend: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Wald confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    return ci_method_batch(ci_wald_rr, tables, alpha, max_workers, backend, progress_callback)


def ci_wald_katz_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                         max_workers: Optional[int] = None, backend: Optional[str] = None,
                         progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Katz-adjusted Wald confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    return ci_method_batch(ci_wald_katz_rr, tables, alpha, max_workers, backend, progress_callback)


def ci_wald_correlated_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                               max_workers: Optional[int] = None, backend: Optional[str] = None,
                               progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate correlation-adjusted Wald confidence intervals for relative risk in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of confidence intervals (lower_bound, upper_bound)
    """
    return ci_method_batch(ci_wald_correlated_rr, tables, alpha, max_workers, backend, progress_callback)


def compute_all_rr_cis_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05,
                            max_workers: Optional[int] = None, backend: Optional[str] = None,
                            progress_callback: Optional[Callable] = None) -> List[Dict[str, Tuple[float, float]]]:
    """
    Compute confidence intervals for relative risk using all methods in batch.

    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        max_workers: Maximum number of worker processes/threads
        backend: Parallelization backend ('threading', 'multiprocessing', or None)
        progress_callback: Callback function for progress reporting

    Returns:
        List of dictionaries mapping method names to confidence intervals
    """
    # Simple sequential implementation - can be enhanced with parallel processing
    results = []
    total = len(tables)

    for i, (a, b, c, d) in enumerate(tables):
        try:
            result = compute_all_rr_cis(a, b, c, d, alpha)
        except Exception:
            # Create empty result with NaN values
            result = {
                "point_estimate": float('nan'),
                "wald": (float('nan'), float('nan')),
                "wald_katz": (float('nan'), float('nan')),
                "wald_correlated": (float('nan'), float('nan')),
                "score": (float('nan'), float('nan')),
                "score_cc": (float('nan'), float('nan')),
                "ustat": (float('nan'), float('nan'))
            }

        results.append(result)

        if progress_callback and i % max(1, total // 100) == 0:
            progress_callback(i / total)

    if progress_callback:
        progress_callback(1.0)

    return results
