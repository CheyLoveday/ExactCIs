"""
Core functionality for ExactCIs package.

This module provides the fundamental calculations and utilities used by the
various confidence interval methods, including validation, probability mass
function calculations, and root-finding algorithms.
"""

import math
import logging
from functools import lru_cache
from typing import Tuple, Callable, List, Union, Optional, Dict, Set, NamedTuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_counts(a: Union[int, float], b: Union[int, float], 
                   c: Union[int, float], d: Union[int, float]) -> None:
    """
    Validate the counts in a 2x2 contingency table.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)

    Raises:
        ValueError: If any count is negative, or if any margin is zero
    """
    if not all(isinstance(x, (int, float)) and x >= 0 for x in (a, b, c, d)):
        raise ValueError("All counts must be non‑negative numbers")
    if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        raise ValueError("Cannot compute odds ratio with empty margins")


def apply_haldane_correction(a: Union[int, float], b: Union[int, float], 
                            c: Union[int, float], d: Union[int, float]) -> Tuple[float, float, float, float]:
    """
    Apply Haldane's correction to a 2x2 contingency table.
    
    This adds 0.5 to each cell if any cell contains a zero, to prevent issues with
    division by zero in odds ratio calculations.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        
    Returns:
        Tuple (a, b, c, d) with Haldane's correction applied
    """
    # Only apply correction if any cell contains a zero
    if a == 0 or b == 0 or c == 0 or d == 0:
        logger.info("Applying Haldane's correction (adding 0.5 to each cell)")
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        return float(a), float(b), float(c), float(d)


def logsumexp(log_terms: List[float]) -> float:
    """
    Compute log(sum(exp(log_terms))) in a numerically stable way.
    
    This function avoids overflow/underflow when summing exponentials of large
    negative or positive values by using the identity:
    log(sum(exp(a_i))) = M + log(sum(exp(a_i - M)))
    where M is the maximum value in a_i.
    
    Args:
        log_terms: List of values in log space to sum
        
    Returns:
        log(sum(exp(log_terms)))
    """
    if not log_terms:
        return float('-inf')
    
    # Filter out -inf values which would be exp(-inf) = 0 in the sum
    filtered_terms = [x for x in log_terms if x != float('-inf')]
    if not filtered_terms:
        return float('-inf')
    
    log_max = max(filtered_terms)
    if log_max == float('-inf'): # All terms were -inf after filtering
        return float('-inf')
    
    # Use a numerically stable approach to sum exponentials
    # exp(700) is close to the maximum representable value
    return log_max + math.log(sum(
        math.exp(min(x - log_max, 700)) for x in filtered_terms
    ))


def log_binom_coeff(n: Union[int, float], k: Union[int, float]) -> float:
    """
    Calculate log of binomial coefficient in a numerically stable way.
    
    For large values, this uses lgamma instead of direct factorial calculation.
    Supports non-integer values through the generalized binomial coefficient
    using the gamma function.
    
    Args:
        n: Upper value in binomial coefficient
        k: Lower value in binomial coefficient
        
    Returns:
        log(n choose k)
    """
    if k < 0 or k > n:
        return float('-inf')  # Invalid combinations
    if k == 0 or k == n:
        return 0.0  # log(1) = 0
    
    # For small integer values, use direct calculation
    if isinstance(n, int) and isinstance(k, int) and n < 20:
        return math.log(math.comb(n, k))
    
    # For large or non-integer values, use lgamma for better numerical stability
    # log(n choose k) = log(n!) - log(k!) - log((n-k)!)
    # For non-integers, we use the gamma function: Γ(n+1) = n!
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_nchg_pmf(k: int, n1: int, n2: int, m1: int, theta: float) -> float:
    """
    Calculate log of noncentral hypergeometric PMF at point k.
    
    This is used for the Fisher exact CI calculation in log space.
    
    Args:
        k: Value at which to evaluate the PMF
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes in first margin
        theta: Odds ratio parameter
        
    Returns:
        log(P(X = k))
    """
    # Check if k is in the support
    supp = support(n1, n2, m1)
    if k not in supp.x:
        return float('-inf')
        
    # Special case for theta = 0
    if theta <= 0:
        return 0.0 if k == supp.min_val else float('-inf')
    
    # Calculate in log space to avoid overflow
    log_theta = math.log(theta)
    log_comb_n1_k = log_binom_coeff(n1, k)
    log_comb_n2_m1_k = log_binom_coeff(n2, m1 - k)
    
    # Unnormalized log probability
    log_p_unnorm = log_comb_n1_k + log_comb_n2_m1_k + k * log_theta
    
    # Calculate normalizing constant in log space
    log_norm_terms = []
    for i in supp.x:
        log_comb_n1_i = log_binom_coeff(n1, i)
        log_comb_n2_m1_i = log_binom_coeff(n2, m1 - i)
        log_norm_terms.append(log_comb_n1_i + log_comb_n2_m1_i + i * log_theta)
    
    log_norm = logsumexp(log_norm_terms)
    
    # Return normalized log probability
    return log_p_unnorm - log_norm


def log_nchg_cdf(k: int, n1: int, n2: int, m1: int, theta: float) -> float:
    """
    Calculate log of noncentral hypergeometric CDF at point k: P(X ≤ k).
    
    Args:
        k: Value at which to evaluate the CDF
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes in first margin
        theta: Odds ratio parameter
        
    Returns:
        log(P(X ≤ k))
    """
    supp = support(n1, n2, m1)
    if k < supp.min_val:
        return float('-inf')  # log(0)
    if k >= supp.max_val:
        return 0.0  # log(1)
        
    # Calculate log probabilities for each value in support up to k
    log_probs = []
    for i in supp.x:
        if i <= k:
            log_probs.append(log_nchg_pmf(i, n1, n2, m1, theta))
    
    # Sum probabilities in log space
    return logsumexp(log_probs)


def log_nchg_sf(k: int, n1: int, n2: int, m1: int, theta: float) -> float:
    """
    Calculate log of noncentral hypergeometric survival function at k: P(X > k).
    
    Args:
        k: Value at which to evaluate the survival function
        n1: Size of first group
        n2: Size of second group
        m1: Number of successes in first margin
        theta: Odds ratio parameter
        
    Returns:
        log(P(X > k))
    """
    supp = support(n1, n2, m1)
    if k >= supp.max_val:
        return float('-inf')  # log(0)
    if k < supp.min_val:
        return 0.0  # log(1)
        
    # Calculate log probabilities for each value in support above k
    log_probs = []
    for i in supp.x:
        if i > k:
            log_probs.append(log_nchg_pmf(i, n1, n2, m1, theta))
    
    # Sum probabilities in log space
    return logsumexp(log_probs)


def nchg_pdf(k_values: np.ndarray, n1: int, n2: int, m1: int, theta: float) -> np.ndarray:
    """
    Calculate noncentral hypergeometric PDF (actually PMF) for multiple k values.
    P(X=k | n1, n2, m1, theta) for each k in k_values.

    Args:
        k_values: Array of integers (support values for cell 'a') at which to evaluate the PMF.
        n1: Size of first group (row 1 total).
        n2: Size of second group (row 2 total).
        m1: Number of successes in the first column (column 1 total, for cell 'a').
        theta: Odds ratio parameter.

    Returns:
        np.ndarray of probabilities corresponding to each k in k_values.
    """
    # The existing pmf function in core.py uses 'm' for the column total argument.
    # Here, m1 is used in the signature for consistency with blaker.py context.
    probabilities = [pmf(int(k), n1, n2, m1, theta) for k in k_values]
    return np.array(probabilities)


SupportData = NamedTuple('SupportData', [
    ('x', np.ndarray),      # Array of support values for cell 'a'
    ('min_val', int),       # Minimum value in the support (k_min)
    ('max_val', int),       # Maximum value in the support (k_max)
    ('offset', int)         # Offset to map 'a' to 0-based index in x (offset = -k_min)
])


@lru_cache(maxsize=None)
def support(n1: Union[int, float], n2: Union[int, float], m1: Union[int, float]) -> SupportData:
    """
    Calculate the support of the noncentral hypergeometric distribution.

    Args:
        n1: Size of first group (row 1 total)
        n2: Size of second group (row 2 total)
        m1: Number of successes in the first column (column 1 total, for cell 'a')

    Returns:
        SupportData object containing the support array, min/max values, and offset.
    """
    # Ensure inputs are integers for range calculations
    n1_int, n2_int, m1_int = int(n1), int(n2), int(m1)

    k_min = max(0, m1_int - n2_int)
    k_max = min(m1_int, n1_int)
    
    if k_min > k_max: # This can happen if margins are inconsistent, e.g. m1 > n1+n2 or similar
        # An empty support. This should ideally be caught by validate_counts if margins make no sense.
        # However, if validate_counts passes but k_min > k_max, it implies no possible values for 'a'.
        logger.warning(f"Support calculation for n1={n1_int}, n2={n2_int}, m1={m1_int} resulted in k_min ({k_min}) > k_max ({k_max}). Returning empty support structure.")
        support_array = np.array([], dtype=int)
        # For min_val, max_val, offset in empty case, conventions might vary.
        # Let's use 0 for offset, and perhaps -1 for min/max to indicate invalid/empty.
        # Or, more practically for downstream code, ensure k_min <= k_max logic is robust.
        # Using k_min (which is > k_max) for both might be a signal.
        return SupportData(x=support_array, min_val=k_min, max_val=k_max, offset=-k_min if k_min <= k_max else 0)

    support_array = np.array(range(k_min, k_max + 1), dtype=int)
    # Offset: if 'a' is a value in support, its index in support_array is a - k_min.
    # So, for blaker's idx_a = s.offset + a, we need offset = -k_min.
    calculated_offset = -k_min
    
    return SupportData(x=support_array, min_val=k_min, max_val=k_max, offset=calculated_offset)


def pmf_weights(n1: Union[int, float], n2: Union[int, float], m: Union[int, float], theta: float) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    Calculate the weights for the probability mass function of the noncentral hypergeometric distribution.

    Args:
        n1: Size of first group
        n2: Size of second group
        m: Number of successes
        theta: Odds ratio parameter

    Returns:
        Tuple containing (support, probabilities)
    """
    supp = support(n1, n2, m)
    
    # Special cases handling for theta
    if theta <= 0:
        # For theta = 0, all probability mass is at the minimum value
        w = [1.0 if k == supp.min_val else 0.0 for k in supp.x]
        return supp.x, tuple(w)
    elif theta >= 1e6 or np.isinf(theta):
        # For extremely large theta, all probability mass is at the maximum value
        # This ensures numerical stability and correct behavior for edge cases
        max_val = supp.max_val
        w = [1.0 if k == max_val else 0.0 for k in supp.x]
        
        # Debug logging for the specific problematic case
        if n1 == 5 and n2 == 10 and m == 7:
            logger.info(f"[DEBUG_PMF_WEIGHTS] n1=5, n2=10, m=7, theta={theta:.2e}")
            logger.info(f"[DEBUG_PMF_WEIGHTS] Using special large theta handler, max_val={max_val}")
            logger.info(f"[DEBUG_PMF_WEIGHTS] Resulting pmf: {w}")
        
        return supp.x, tuple(w)
    
    logt = math.log(theta)

    # Check for large values that might cause numerical issues
    # Ensure n1, n2, m are treated as numbers for comparison, int() for large value check if they are floats.
    check_n1, check_n2, check_m = (float(val) for val in (n1, n2, m))

    if check_n1 > 100 or check_n2 > 100 or check_m > 100:
        logger.warning(f"Large values detected in pmf_weights: n1={n1}, n2={n2}, m={m}")
        # Use Stirling's approximation for very large factorials
        # or return a simplified distribution for extremely large values
        # if check_n1 > 1000 or check_n2 > 1000 or check_m > 1000:
        #     logger.warning("Extremely large values, using simplified distribution (NOW DISABLED)")
        #     # For extremely large values, return a simplified distribution
        #     # centered around the expected value
        #     expected_k = check_m * check_n1 / (check_n1 + check_n2)
        #     w = [math.exp(-0.5 * ((k - expected_k) / (0.1 * check_n1))**2) for k in supp.x]
        #     # Normalize
        #     sum_w = sum(w)
        #     if sum_w == 0: # Avoid division by zero if all weights are tiny
        #         w = [1.0/len(supp.x)] * len(supp.x) if supp.x else []
        #     else:
        #         w = [wi / sum_w for wi in w]
        #     return supp.x, tuple(w)

    # DEBUG LOGGING FOR SPECIFIC CASE
    is_debug_case_pmf = (n1 == 5 and n2 == 10 and m == 7)
    if is_debug_case_pmf:
        logger.info(f"[DEBUG_PMF_WEIGHTS] n1=5,n2=10,m=7,theta={theta:.2e}")
        logger.info(f"[DEBUG_PMF_WEIGHTS] supp={supp.x}")

    # Calculate log-probabilities with safeguards against overflow
    logs = []
    for k in supp.x:  # k from support() is guaranteed to be int
        try:
            # Use log-space calculations to avoid overflow
            # Use log_binom_coeff which handles float inputs for n1, n2, m
            log_comb_n1_k = log_binom_coeff(n1, k)
            log_comb_n2_m_k = log_binom_coeff(n2, m - k)
            log_term = log_comb_n1_k + log_comb_n2_m_k + k * logt
            logs.append(log_term)
            
            if is_debug_case_pmf:
                logger.info(f"[DEBUG_PMF_WEIGHTS] k={k}: log_comb_n1_k={log_comb_n1_k:.2e}, " 
                           f"log_comb_n2_m_k={log_comb_n2_m_k:.2e}, k*logt={k*logt:.2e}, " 
                           f"log_term={log_term:.2e}")
        except (OverflowError, ValueError) as e:
            logger.warning(f"Numerical error in pmf_weights for k={k}: {e}")
            # Assign a very small probability to this value
            logs.append(float('-inf'))

    # Filter out -inf values
    valid_logs = [l for l in logs if l != float('-inf')]
    if not valid_logs:
        logger.warning("No valid log probabilities, using uniform distribution")
        return supp.x, tuple([1.0/len(supp.x)] * len(supp.x))

    # Use logsumexp for numerical stability
    M = max(valid_logs)
    
    if is_debug_case_pmf:
        # Find index of max value in valid_logs using numpy
        max_idx = np.argmax(valid_logs)
        logger.info(f"[DEBUG_PMF_WEIGHTS] max_log={M:.2e}, index of max={max_idx}")
    
    # Use a numerically stable approach to sum exponentials
    exp_terms = [math.exp(min(x - M, 700)) for x in valid_logs]
    log_sum = M + math.log(sum(exp_terms))
    
    if is_debug_case_pmf:
        logger.info(f"[DEBUG_PMF_WEIGHTS] exp_terms={exp_terms}, log_sum={log_sum:.2e}")

    # Normalize in log-space with protection against underflow
    w = []
    for i, l in enumerate(logs):
        if l == float('-inf'):
            w.append(0.0)
        else:
            # Protect against underflow
            exp_term = min(l - log_sum, 700)  # exp(700) is close to the maximum representable value
            weight = math.exp(exp_term)
            w.append(weight)
            
            if is_debug_case_pmf:
                logger.info(f"[DEBUG_PMF_WEIGHTS] k={supp.x[i]}, log diff={l-log_sum:.2e}, weight={weight:.2e}")

    # Safety check: Ensure maximum probability is at k_max for large theta
    if theta > 1e3 and not np.isinf(theta):  # For large but not infinite theta
        # Find indices using numpy methods instead of .index()
        max_k_idx = np.where(supp.x == supp.max_val)[0][0]
        max_prob_idx = np.argmax(w)
        
        if max_prob_idx != max_k_idx:
            logger.warning(f"Unexpected probability distribution for large theta={theta:.2e}: "
                          f"max prob at k={supp.x[max_prob_idx]} instead of k_max={supp.max_val}")
            # Consider correcting the distribution here if needed

    if is_debug_case_pmf:
        logger.info(f"[DEBUG_PMF_WEIGHTS] Raw weights: {w}")
        logger.info(f"[DEBUG_PMF_WEIGHTS] Total weight: {sum(w):.2e}")

    # Normalize weights to sum to 1
    total_weight = sum(w)
    normalized_w = [wi / total_weight for wi in w] if total_weight > 0 else [0.0] * len(w)

    if is_debug_case_pmf:
        logger.info(f"[DEBUG_PMF_WEIGHTS] Normalized pmf_vals: {normalized_w}")

    return supp.x, tuple(normalized_w)


def pmf(k: int, n1: int, n2: int, m: int, theta: float) -> float:
    """
    Calculate the probability mass function of the noncentral hypergeometric distribution.

    Args:
        k: Value at which to evaluate the PMF
        n1: Size of first group
        n2: Size of second group
        m: Number of successes
        theta: Odds ratio parameter

    Returns:
        Probability mass at k
    """
    supp, pmf_vals = pmf_weights(n1, n2, m, theta)
    # Use dict for O(1) lookup instead of O(n) list search
    pmf_dict = dict(zip(supp, pmf_vals))
    return pmf_dict[k]


def find_root(f: Callable[[float], float], lo: float = 1e-8, hi: float = 1.0,
              tol: float = 1e-8, maxiter: int = 60) -> float:
    """
    Find the root of a function using bisection method.

    Args:
        f: Function for which to find the root
        lo: Lower bound for the search
        hi: Upper bound for the search
        tol: Tolerance for convergence
        maxiter: Maximum number of iterations

    Returns:
        Approximate root of the function

    Raises:
        RuntimeError: If the root cannot be bracketed
    """
    f_lo, f_hi = f(lo), f(hi)
    while f_lo * f_hi > 0:
        hi *= 2
        f_hi = f(hi)
        if hi > 1e16:
            raise RuntimeError("Failed to bracket root")
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol * max(1, hi):
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def find_root_log(f: Callable[[float], float], lo: float = 1e-8, hi: float = 1.0,
                tol: float = 1e-8, maxiter: int = 60,
                progress_callback: Optional[Callable[[float], None]] = None,
                timeout_checker: Optional[Callable[[], bool]] = None,
                **kwargs) -> Optional[float]:
    """
    Find the root of a function using bisection method in log space.
    
    This is more stable for functions with wide ranges of inputs,
    particularly when dealing with confidence intervals for odds ratios.
    
    Args:
        f: Function to evaluate
        lo: Lower bound for the search
        hi: Upper bound for the search
        tol: Tolerance for convergence
        maxiter: Maximum iterations
        progress_callback: Optional callback function to report progress (0-100)
        timeout_checker: Optional function that returns True if a timeout has occurred
        **kwargs: Additional keyword arguments for compatibility (e.g., xtol is mapped to tol)
        
    Returns:
        Log of the approximate root of the function, or None if timeout occurred or root not bracketed.
        The caller should exponentiate the result if a root in normal space is needed.
        
    Raises:
        RuntimeError: If the root cannot be bracketed after attempts to expand search.
    """
    # For backward compatibility: if xtol is provided, use it instead of tol
    if 'xtol' in kwargs:
        tol = kwargs['xtol']
    
    if lo <= 0 or hi <= 0:
        logger.error(f"find_root_log: Search interval bounds must be positive. Got lo={lo}, hi={hi}")
        # This indicates a fundamental issue, as log-space search requires positive bounds.
        # Depending on context, might want to return 0 or raise error. Let's raise for now.
        raise ValueError("Search interval bounds for find_root_log must be positive.")

    # Convert to log space
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    
    f_lo = f(math.exp(log_lo))
    if timeout_checker and timeout_checker(): return None
    if f_lo is None: return None # Function itself might indicate timeout

    f_hi = f(math.exp(log_hi))
    if timeout_checker and timeout_checker(): return None
    if f_hi is None: return None
    
    if f_lo == 0: return log_lo
    if f_hi == 0: return log_hi

    # Attempt to bracket the root if not already bracketed
    if f_lo * f_hi > 0:
        logger.warning(f"find_root_log: Initial interval [{lo:.2e}, {hi:.2e}] (f_lo={f_lo:.2e}, f_hi={f_hi:.2e}) does not bracket root. Attempting to expand.")
        # Try expanding the interval. Max 10 attempts each side.
        # Expand upper bound first
        original_log_hi = log_hi
        for _ in range(10):
            log_hi = original_log_hi + math.log(2) * (_ + 1) # Expand by factors of 2
            f_hi_new = f(math.exp(log_hi))
            if timeout_checker and timeout_checker(): return None
            if f_hi_new is None: return None
            if f_lo * f_hi_new <= 0:
                f_hi = f_hi_new
                logger.info(f"find_root_log: Bracketed root by expanding hi to {math.exp(log_hi):.2e}")
                break
        else: # if loop completed without break
            log_hi = original_log_hi # Reset if upper expansion failed
            f_hi = f(math.exp(log_hi)) # Reset f_hi
            # Expand lower bound if upper expansion failed
            original_log_lo = log_lo
            for _ in range(10):
                log_lo = original_log_lo - math.log(2) * (_ + 1) # Shrink by factors of 2
                f_lo_new = f(math.exp(log_lo))
                if timeout_checker and timeout_checker(): return None
                if f_lo_new is None: return None
                if f_lo_new * f_hi <= 0:
                    f_lo = f_lo_new
                    logger.info(f"find_root_log: Bracketed root by expanding lo to {math.exp(log_lo):.2e}")
                    break
            else:
                logger.error(f"find_root_log: Could not bracket root for f after expanding search. Final interval [{math.exp(log_lo):.2e}, {math.exp(log_hi):.2e}], f_values [{f_lo:.2e}, {f_hi:.2e}]")
                raise RuntimeError(f"Cannot bracket root in find_root_log: f({math.exp(log_lo):.2e}) = {f_lo:.2e}, f({math.exp(log_hi):.2e}) = {f_hi:.2e}")

    # Bisection method in log space
    iter_num = 0
    while iter_num < maxiter:
        if timeout_checker and timeout_checker():
            logger.warning("Timeout occurred during find_root_log bisection")
            return None
            
        log_mid = 0.5 * (log_lo + log_hi)
        # Check for convergence based on interval width in log space
        if (log_hi - log_lo) < tol:
            return log_mid

        mid_val = math.exp(log_mid)
        f_mid = f(mid_val)
        
        if f_mid is None: # Function might indicate timeout
             logger.warning("Timeout (f_mid is None) occurred during find_root_log bisection")
             return None
        
        if progress_callback:
            progress_callback(100 * iter_num / maxiter)
        
        if f_mid == 0:
            return log_mid
        elif f_mid * f_lo < 0:
            log_hi = log_mid
            # f_hi = f_mid # Not strictly necessary for bisection, but if f_mid stored, use it
        else:
            log_lo = log_mid
            f_lo = f_mid # Update f_lo to the value at the new log_lo
        
        iter_num += 1
    
    logger.warning(f"find_root_log: Max iterations ({maxiter}) reached. Returning current log_mid: {(log_lo + log_hi) / 2.0:.4e}")
    return (log_lo + log_hi) / 2.0 # Return best estimate if max_iter reached


def find_plateau_edge(f: Callable[[float], float], lo: float, hi: float, target: float = 0, 
                    xtol: float = 1e-6, increasing: bool = True, max_iter: int = 100,
                    progress_callback: Optional[Callable[[float], None]] = None,
                    timeout_checker: Optional[Callable[[], bool]] = None) -> Optional[Tuple[float, int]]:
    """
    Find the edge of a plateau where f(θ) ≈ target.
    
    This is useful for p-value functions that can be flat over ranges of theta,
    such as in Fisher's or Blaker's methods, where we want the smallest theta
    with f(theta) ≈ target.
    """
    # Check if low and high bounds already meet the condition
    f_lo = f(lo)
    
    # Check for timeout after first evaluation
    if timeout_checker and timeout_checker():
        logger.info(f"Timeout reached in find_plateau_edge")
        return None
        
    # Handle timeout from function
    if f_lo is None:
        return None
        
    f_hi = f(hi)
    
    # Check for timeout after second evaluation
    if timeout_checker and timeout_checker():
        logger.info(f"Timeout reached in find_plateau_edge")
        return None
        
    # Handle timeout from function
    if f_hi is None:
        return None
    
    # If finding the smallest theta where f(theta) ≥ target
    if increasing:
        # If lo already meets the condition, return it
        if f_lo >= target:
            # Check for timeout before returning
            if timeout_checker and timeout_checker():
                logger.info(f"Timeout reached in find_plateau_edge")
                return None
            return (lo, 0)
        # If hi doesn't meet the condition, can't find a valid theta
        if f_hi < target:
            # Check for timeout before returning
            if timeout_checker and timeout_checker():
                logger.info(f"Timeout reached in find_plateau_edge")
                return None
            # Return hi as the best we can do
            return (hi, 0)
    # If finding the smallest theta where f(theta) ≥ target (decreasing function)
    else:
        # If lo already meets the condition, return it
        if f_lo >= target:
            # Check for timeout before returning
            if timeout_checker and timeout_checker():
                logger.info(f"Timeout reached in find_plateau_edge")
                return None
            return (lo, 0)
        # If hi doesn't meet the condition, can't find a valid theta
        if f_hi < target:
            # Check for timeout before returning
            if timeout_checker and timeout_checker():
                logger.info(f"Timeout reached in find_plateau_edge")
                return None
            # Return hi as the best we can do
            return (hi, 0)
    
    # Initialize the bounds for binary search
    lower, upper = lo, hi
    
    # Track iterations
    iter_count = 0
    
    # Binary search for the plateau edge
    while iter_count < max_iter and upper - lower > xtol:
        # Check for timeout
        if timeout_checker and timeout_checker():
            logger.info(f"Timeout reached in find_plateau_edge")
            return None
            
        mid = (lower + upper) / 2
        f_mid = f(mid)
        
        # Handle timeout from function
        if f_mid is None:
            return None
        
        # Report progress if callback provided
        if progress_callback:
            progress = 100 * iter_count / max_iter
            progress_callback(progress, iter_count, (lower, upper))
        
        # Update bounds based on the function value and search direction
        if f_mid >= target:  # We're in the region where the condition is met
            if increasing:
                # If looking for smallest theta where f(theta) ≥ target,
                # this is our upper bound
                upper = mid
            else:
                # If looking for largest theta where f(theta) ≥ target,
                # this is our lower bound
                lower = mid
        else:  # We're in the region where the condition is not met
            if increasing:
                # If looking for smallest theta where f(theta) ≥ target,
                # this is our lower bound
                lower = mid
            else:
                # If looking for largest theta where f(theta) ≥ target,
                # this is our upper bound
                upper = mid
        
        iter_count += 1
    
    # Check for timeout one last time before returning final result
    if timeout_checker and timeout_checker():
        logger.info(f"Timeout reached in find_plateau_edge before returning final result")
        return None
        
    # Return the appropriate edge of the plateau
    if increasing:
        return (upper, iter_count)
    else:
        return (lower, iter_count)


def find_smallest_theta(
    func: Callable[[float], float],
    alpha: float, # This is the target p-value we are comparing against (e.g., 0.05)
    lo: float = 1e-8,
    hi: float = 1.0,
    two_sided: bool = True, # If True, func is a two-sided p-value. If False, func is one-sided.
                           # Blaker calls with two_sided=False, meaning func(theta) is directly compared to alpha.
    increasing: bool = False, # Is func(theta) generally increasing or decreasing with theta? 
                              # This helps find_plateau_edge determine the correct side.
                              # For Blaker's p-value (a 'bump'), this is context-dependent for the CI's lower/upper tail.
                              # However, find_plateau_edge side='left' is used, aiming for smallest theta.
    xtol: float = 1e-7, # Tolerance for root finding
    max_iter: int = 100,
    progress_callback: Optional[Callable[[float], None]] = None, # Pass to find_root_log
    timeout_checker: Optional[Callable[[], bool]] = None       # Pass to find_root_log and find_plateau_edge
):
    """
    Finds the smallest theta such that func(theta) is close to alpha.
    This function is designed to find the edge of an acceptance region for a test statistic
    or p-value function `func(theta)` relative to a significance level `alpha`.
    It first tries to find a root of `func(theta) - alpha = 0` using `find_root_log`.
    If `func(theta)` is flat around `alpha` (e.g., a p-value plateau),
    `find_root_log` might return a point in the middle of the plateau.
    In such cases, `find_plateau_edge` is used to find the actual edge of this plateau.
    """
    target_alpha = alpha # Blaker calls with two_sided=False, so alpha is the direct target.

    func_name = func.__name__ if hasattr(func, '__name__') else 'lambda_func'
    logger.info(f"find_smallest_theta: Initial call with func={func_name}, alpha(input)={alpha:.4e}, target_alpha={target_alpha:.4e}, lo={lo:.2e}, hi={hi:.2e}, two_sided={two_sided}, increasing={increasing}")

    g = lambda t: func(t) - target_alpha

    current_lo, current_hi = lo, hi
    root = None
    try:
        g_lo = g(current_lo)
        g_hi = g(current_hi)
        logger.info(f"find_smallest_theta: Boundary values for g(theta)=func(theta)-target_alpha: g({current_lo:.2e}) = {g_lo:.4e}, g({current_hi:.2e}) = {g_hi:.4e}")
        if np.isnan(g_lo) or np.isnan(g_hi):
            logger.warning(f"find_smallest_theta: NaN detected in g boundary values. g_lo={g_lo}, g_hi={g_hi}. Aborting root search.")
            return None
        if np.sign(g_lo) == np.sign(g_hi) and not np.isclose(g_lo, 0) and not np.isclose(g_hi, 0):
            logger.warning(f"find_smallest_theta: Root not bracketed for g. g({current_lo:.2e})={g_lo:.4e}, g({current_hi:.2e})={g_hi:.4e}. Will attempt find_plateau_edge.")
    except (OverflowError, ValueError) as e:
        logger.error(f"find_smallest_theta: Error evaluating g at boundaries ({current_lo:.2e}, {current_hi:.2e}): {e}")
        return None

    try:
        logger.info(f"find_smallest_theta: Attempting find_root_log for g with lo={current_lo:.2e}, hi={current_hi:.2e}")
        # Note: find_root_log needs a function that crosses zero. g is already func(t) - target_alpha.
        log_root_g = find_root_log(g, lo=current_lo, hi=current_hi, tol=xtol / 10, maxiter=max_iter // 2, progress_callback=progress_callback, timeout_checker=timeout_checker)
        logger.info(f"find_smallest_theta: find_root_log returned: {log_root_g}")
        if log_root_g is not None:
            root = math.exp(log_root_g) # Exponentiate the result from find_root_log
            val_at_root = func(root)
            logger.info(f"find_smallest_theta: func(root={root:.4e}) from find_root_log = {val_at_root:.4e} (target_alpha={target_alpha:.4e}, diff={(val_at_root - target_alpha):.2e})")
            if abs(val_at_root - target_alpha) < 0.01: # Threshold for a 'good' root - further relaxed to accept good approximations
                if not (np.isclose(root, current_lo, atol=xtol) or np.isclose(root, current_hi, atol=xtol)):
                    logger.info(f"find_smallest_theta: Root {root:.4e} found by find_root_log is close to target and not at boundary. Returning this root.")
                    return root
                else:
                    logger.info(f"find_smallest_theta: Root {root:.4e} found by find_root_log is at boundary. Proceeding to plateau search.")
            else:
                 logger.info(f"find_smallest_theta: Root {root:.4e} from find_root_log is not sufficiently close to target_alpha (diff={(val_at_root - target_alpha):.2e}). Proceeding to plateau_edge.")
        else:
            logger.info(f"find_smallest_theta: find_root_log did not find a root.")
    except RuntimeError as e:
        logger.warning(f"find_smallest_theta: find_root_log failed: {e}. Proceeding to find_plateau_edge.")
        root = None
    except (OverflowError, ValueError) as e:
        logger.error(f"find_smallest_theta: Error during or after find_root_log for g({current_lo:.2e}, {current_hi:.2e}): {e}")
        root = None

    plateau_side = 'left' # For find_smallest_theta, we generally want the minimal theta.
    logger.info(f"find_smallest_theta: Calling find_plateau_edge for func={func_name} with target={target_alpha:.4e}, lo={lo:.2e}, hi={hi:.2e}, increasing={increasing}")
    try:
        result_theta_tuple = find_plateau_edge(
            f=func, # Pass original func, not g
            lo=lo, # Use original search bounds for plateau search
            hi=hi,
            target=target_alpha,
            xtol=xtol, # plateau_edge uses xtol for its convergence
            increasing=increasing, 
            max_iter=max_iter,
            timeout_checker=timeout_checker # Pass timeout_checker here
        )
        logger.info(f"find_smallest_theta: find_plateau_edge returned: {result_theta_tuple}")
        if result_theta_tuple is not None:
            result_theta, _ = result_theta_tuple
            val_at_plateau_edge = func(result_theta)
            logger.info(f"find_smallest_theta: func(plateau_edge_theta={result_theta:.4e}) = {val_at_plateau_edge:.4e} (target_alpha={target_alpha:.4e}, diff={(val_at_plateau_edge - target_alpha):.2e})")
            # Prefer find_root_log result if it was good and not at boundary, unless plateau_edge is significantly better or closer.
            if root is not None and abs(func(root) - target_alpha) < 1e-9 and not (np.isclose(root, lo, atol=xtol) or np.isclose(root, hi, atol=xtol)):
                if abs(val_at_plateau_edge - target_alpha) < abs(func(root) - target_alpha) - 1e-10: # Plateau edge is notably better
                    logger.info(f"find_smallest_theta: find_plateau_edge result {result_theta:.4e} is preferred over find_root_log result {root:.4e}. Using plateau edge.")
                    return result_theta
                else:
                    logger.info(f"find_smallest_theta: find_root_log result {root:.4e} was good and preferred or similar to plateau_edge. Using root.")
                    return root
            return result_theta # Default to plateau edge if no good root or root was at boundary
        else:
            if root is not None and abs(func(root) - target_alpha) < xtol * 100: # Looser check for root if plateau failed
                logger.warning(f"find_smallest_theta: find_plateau_edge failed, but find_root_log found {root:.4e}. Returning this as fallback.")
                return root
            logger.warning(f"find_smallest_theta: find_plateau_edge also failed. No suitable theta found.")
            return None
    except Exception as e:
        logger.error(f"find_smallest_theta: Error during find_plateau_edge: {e}", exc_info=True)
        if root is not None and abs(func(root) - target_alpha) < xtol * 100:
            logger.warning(f"find_smallest_theta: Exception in find_plateau_edge. Falling back to find_root_log result {root:.4e}.")
            return root
        return None


def find_sign_change(f: Callable[[float], float], lo: float, hi: float, 
                    tol: float = 1e-6, max_iter: int = 100,
                    progress_callback: Optional[Callable[[float], None]] = None,
                    timeout_checker: Optional[Callable[[], bool]] = None,
                    min_interval_size: float = 1e-15) -> Optional[float]:
    """
    Find a point where the function changes sign using bisection.
    """
    f_lo = f(lo)
    f_hi = f(hi)

    if f_lo is None or f_hi is None: # Check if f returned None (e.g. timeout from sub-function)
        logger.warning("find_sign_change: f(lo) or f(hi) returned None. Cannot proceed.")
        return None

    if np.isnan(f_lo) or np.isnan(f_hi):
        logger.warning(f"find_sign_change: NaN detected at boundaries: f(lo)={f_lo}, f(hi)={f_hi}")
        return None

    if np.sign(f_lo) == np.sign(f_hi):
        logger.debug(f"find_sign_change: No sign change in interval [{lo}, {hi}]. f(lo)={f_lo}, f(hi)={f_hi}")
        return None
    
    if abs(f_lo) < tol: return lo
    if abs(f_hi) < tol: return hi
    
    for i in range(max_iter):
        if timeout_checker and timeout_checker():
            logger.warning("Timeout occurred during sign change search")
            return None
            
        if progress_callback: progress_callback((i / max_iter) * 100)
            
        mid = (lo + hi) / 2.0
        f_mid = f(mid)

        if f_mid is None: # Timeout from f(mid)
            logger.warning("find_sign_change: f(mid) returned None. Cannot proceed.")
            return None
        if np.isnan(f_mid):
            logger.warning(f"find_sign_change: NaN detected at mid={mid}, f(mid)={f_mid}")
            # Decide how to handle: e.g., try to shrink interval away from NaN if possible, or fail
            return None # Or try to be smarter, but for now, fail
            
        if abs(f_mid) < tol:
            return mid
            
        if np.sign(f_lo) != np.sign(f_mid):
            hi = mid
            # f_hi = f_mid # Not strictly needed as we don't use f_hi beyond initial check
        else:
            lo = mid
            f_lo = f_mid # f_lo needs to be updated for the next iteration's sign check
            
        if (hi - lo) < tol * abs(hi + lo) or (hi - lo) < min_interval_size: # Relative and absolute tolerance
            return (lo + hi) / 2.0
            
    logger.debug(f"find_sign_change: Max iterations reached. Returning best guess: {(lo + hi) / 2.0}")
    return (lo + hi) / 2.0 # Return best estimate if max_iter reached


def calculate_odds_ratio(a: Union[int, float], b: Union[int, float], 
                         c: Union[int, float], d: Union[int, float]) -> float:
    """
    Calculate the odds ratio for a 2x2 contingency table.
    Conventionally, 0/0 is treated as 1.0, and x/0 (for x > 0) as infinity.
    Haldane-Anscombe correction is typically applied *before* this function
    when calculating confidence intervals if zeros are present.
    """
    numerator = a * d
    denominator = b * c

    if denominator == 0:
        if numerator == 0:
            # Case 0/0 (e.g., a=0,c=0 or b=0,d=0 or a=0,b=0 or c=0,d=0 or all zero)
            return 1.0  # Convention for indeterminate 0/0 form
        else:
            # Case x/0 where x > 0
            return float('inf')
    else:
        # Denominator is non-zero, standard calculation
        # This will correctly return 0.0 if numerator is 0 and denominator is non-zero.
        return numerator / denominator


def estimate_point_or(a: Union[int, float], b: Union[int, float], 
                      c: Union[int, float], d: Union[int, float], 
                      correction_type: str = None) -> float:
    """
    Calculate a point estimate of the odds ratio with optional corrections.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        correction_type: Type of correction to apply ('haldane', 'laplace', None)
        
    Returns:
        Point estimate of the odds ratio with specified correction
    """
    # Apply corrections if specified
    if correction_type == 'haldane':
        # Haldane's correction: add 0.5 to all cells if any cell is 0
        a_corr, b_corr, c_corr, d_corr = apply_haldane_correction(a, b, c, d)
        return calculate_odds_ratio(a_corr, b_corr, c_corr, d_corr)
    elif correction_type == 'laplace':
        # Laplace's correction: add 1 to all cells
        return calculate_odds_ratio(a + 1, b + 1, c + 1, d + 1)
    else:
        # No correction
        return calculate_odds_ratio(a, b, c, d)


def calculate_relative_risk(a: Union[int, float], b: Union[int, float], 
                           c: Union[int, float], d: Union[int, float]) -> float:
    """
    Calculate the relative risk (risk ratio) for a 2x2 contingency table.
    """
    n1 = a + b
    n2 = c + d
    if n1 == 0 and n2 == 0: return 1.0 # No data
    if n1 == 0: return float('inf') if c > 0 else 1.0 # Risk1 undefined, if Risk2 >0 then RR=inf
    if n2 == 0: return 0.0 if a > 0 else 1.0      # Risk2 undefined, if Risk1 >0 then RR=0

    risk1 = a / n1
    risk2 = c / n2

    if risk2 == 0:
        return float('inf') if risk1 > 0 else 1.0 # If risk2 is 0, RR is inf if risk1 > 0, else 1 (0/0)
    return risk1 / risk2


def create_2x2_table(a: Union[int, float], b: Union[int, float], 
                    c: Union[int, float], d: Union[int, float]) -> Dict[str, Dict[str, float]]:
    """
    Create a structured 2x2 contingency table from cell counts.
    """
    table = {
        "row1": {"col1": float(a), "col2": float(b), "total": float(a + b)},
        "row2": {"col1": float(c), "col2": float(d), "total": float(c + d)},
        "total": {"col1": float(a + c), "col2": float(b + d), "total": float(a + b + c + d)}
    }
    n = table["total"]["total"]
    if n > 0:
        table["props"] = {
            "row1": {"col1": a / n, "col2": b / n, "total": (a + b) / n},
            "row2": {"col1": c / n, "col2": d / n, "total": (c + d) / n},
            "total": {"col1": (a + c) / n, "col2": (b + d) / n, "total": 1.0}
        }
    else:
        table["props"] = {
            "row1": {"col1": 0.0, "col2": 0.0, "total": 0.0},
            "row2": {"col1": 0.0, "col2": 0.0, "total": 0.0},
            "total": {"col1": 0.0, "col2": 0.0, "total": 0.0}
        }
    table["stats"] = {
        "odds_ratio": calculate_odds_ratio(a, b, c, d),
        "relative_risk": calculate_relative_risk(a, b, c, d)
    }
    return table
