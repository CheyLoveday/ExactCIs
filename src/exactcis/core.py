"""
Core functionality for ExactCIs package.

This module provides the fundamental calculations and utilities used by the
various confidence interval methods, including validation, probability mass
function calculations, and root-finding algorithms.
"""

import math
import logging
from functools import lru_cache
from typing import Tuple, Callable, List, Union, Optional, Dict, Set

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
    
    This adds 0.5 to each cell count if any of the cells contains a zero.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        
    Returns:
        Tuple (a, b, c, d) with Haldane's correction applied if needed
    """
    # Check if any cell has a zero count
    if a == 0 or b == 0 or c == 0 or d == 0:
        logger.info("Applying Haldane's correction (adding 0.5 to each cell) due to zero counts")
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        return a, b, c, d


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
    if k not in supp:
        return float('-inf')
        
    # Special case for theta = 0
    if theta <= 0:
        return 0.0 if k == min(supp) else float('-inf')
    
    # Calculate in log space to avoid overflow
    log_theta = math.log(theta)
    log_comb_n1_k = log_binom_coeff(n1, k)
    log_comb_n2_m1_k = log_binom_coeff(n2, m1 - k)
    
    # Unnormalized log probability
    log_p_unnorm = log_comb_n1_k + log_comb_n2_m1_k + k * log_theta
    
    # Calculate normalizing constant in log space
    log_norm_terms = []
    for i in supp:
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
    if k < min(supp):
        return float('-inf')  # log(0)
    if k >= max(supp):
        return 0.0  # log(1)
        
    # Calculate log probabilities for each value in support up to k
    log_probs = []
    for i in supp:
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
    if k >= max(supp):
        return float('-inf')  # log(0)
    if k < min(supp):
        return 0.0  # log(1)
        
    # Calculate log probabilities for each value in support above k
    log_probs = []
    for i in supp:
        if i > k:
            log_probs.append(log_nchg_pmf(i, n1, n2, m1, theta))
    
    # Sum probabilities in log space
    return logsumexp(log_probs)


@lru_cache(maxsize=None)
def support(n1: int, n2: int, m: int) -> Tuple[int, ...]:
    """
    Calculate the support of the noncentral hypergeometric distribution.

    Args:
        n1: Size of first group
        n2: Size of second group
        m: Number of successes

    Returns:
        Tuple of possible values for the random variable
    """
    low = max(0, m - n2)
    high = min(m, n1)
    return tuple(range(low, high + 1))


def pmf_weights(n1: int, n2: int, m: int, theta: float) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
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
    if theta <= 0:
        w = [1.0 if k == supp[0] else 0.0 for k in supp]
    else:
        logt = math.log(theta)

        # Check for large values that might cause numerical issues
        if n1 > 100 or n2 > 100 or m > 100:
            logger.warning(f"Large values detected in pmf_weights: n1={n1}, n2={n2}, m={m}")
            # Use Stirling's approximation for very large factorials
            # or return a simplified distribution for extremely large values
            if n1 > 1000 or n2 > 1000 or m > 1000:
                logger.warning("Extremely large values, using simplified distribution")
                # For extremely large values, return a simplified distribution
                # centered around the expected value
                expected_k = m * n1 / (n1 + n2)
                w = [math.exp(-0.5 * ((k - expected_k) / (0.1 * n1))**2) for k in supp]
                # Normalize
                sum_w = sum(w)
                w = [wi / sum_w for wi in w]
                return supp, tuple(w)

        # Calculate log-probabilities with safeguards against overflow
        logs = []
        for k in supp:
            try:
                # Use log-space calculations to avoid overflow
                log_comb_n1_k = math.log(math.comb(n1, k))
                log_comb_n2_m_k = math.log(math.comb(n2, m - k))
                logs.append(log_comb_n1_k + log_comb_n2_m_k + k * logt)
            except (OverflowError, ValueError) as e:
                logger.warning(f"Numerical error in pmf_weights for k={k}: {e}")
                # Assign a very small probability to this value
                logs.append(float('-inf'))

        # Filter out -inf values
        valid_logs = [l for l in logs if l != float('-inf')]
        if not valid_logs:
            logger.warning("No valid log probabilities, using uniform distribution")
            return supp, tuple([1.0/len(supp)] * len(supp))

        # Use logsumexp for numerical stability
        M = max(valid_logs)
        # Use a numerically stable approach to sum exponentials
        log_sum = M + math.log(sum(math.exp(min(l - M, 700)) for l in valid_logs))

        # Normalize in log-space with protection against underflow
        w = []
        for l in logs:
            if l == float('-inf'):
                w.append(0.0)
            else:
                # Protect against underflow
                exp_term = min(l - log_sum, 700)  # exp(700) is close to the maximum representable value
                w.append(math.exp(exp_term))

    return supp, tuple(w)


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
                timeout_checker: Optional[Callable[[], bool]] = None) -> Optional[float]:
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
        
    Returns:
        Approximate root of the function, or None if timeout occurred
        
    Raises:
        RuntimeError: If the root cannot be bracketed
    """
    # Convert to log space
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    
    # Check that the interval brackets a root
    f_lo = f(math.exp(log_lo))
    
    # Check for timeout after first evaluation
    if timeout_checker and timeout_checker():
        return None
    
    f_hi = f(math.exp(log_hi))
    
    # Check for timeout after second evaluation
    if timeout_checker and timeout_checker():
        return None
    
    # Handle the case where f returns None (indicating timeout)
    if f_lo is None or f_hi is None:
        return None
    
    if f_lo * f_hi > 0:
        # If both values have the same sign, try to bracket root
        # Expand interval until it brackets a root or until we give up
        # Expand by factors of 2 first on the upper end
        orig_log_hi = log_hi
        for _ in range(10):  # Limit number of attempts
            log_hi = orig_log_hi * 1.5  # Try extending upper bound
            f_hi = f(math.exp(log_hi))
            
            # Check for timeout
            if timeout_checker and timeout_checker():
                return None
                
            # Handle timeout from function
            if f_hi is None:
                return None
                
            if f_lo * f_hi <= 0:  # Successfully bracketed
                break
                
        # If that didn't work, try extending lower bound
        if f_lo * f_hi > 0:
            orig_log_lo = log_lo
            for _ in range(10):  # Limit number of attempts
                log_lo = orig_log_lo * 0.5  # Try extending lower bound
                f_lo = f(math.exp(log_lo))
                
                # Check for timeout
                if timeout_checker and timeout_checker():
                    return None
                    
                # Handle timeout from function
                if f_lo is None:
                    return None
                    
                if f_lo * f_hi <= 0:  # Successfully bracketed
                    break
        
        # If still no bracket found, raise an error
        if f_lo * f_hi > 0:
            raise RuntimeError(f"Cannot bracket root: f({math.exp(log_lo)}) = {f_lo}, f({math.exp(log_hi)}) = {f_hi}")
    
    # Bisection method in log space
    log_mid = 0.5 * (log_lo + log_hi)
    
    # Track iteration number for progress reporting
    iter_num = 0
    
    while iter_num < maxiter and (log_hi - log_lo) > tol:
        # Check for timeout
        if timeout_checker and timeout_checker():
            return None
            
        log_mid = 0.5 * (log_lo + log_hi)
        mid = math.exp(log_mid)
        f_mid = f(mid)
        
        # Handle timeout from function
        if f_mid is None:
            return None
            
        # Report progress if callback provided
        if progress_callback:
            progress = 100 * iter_num / maxiter
            progress_callback(progress)
        
        # Update bounds
        if f_mid * f_lo > 0:
            log_lo = log_mid
            f_lo = f_mid
        else:
            log_hi = log_mid
            f_hi = f_mid
        
        iter_num += 1
    
    # Return final midpoint as our approximate root
    return log_mid


def find_plateau_edge(f: Callable[[float], float], lo: float, hi: float, target: float = 0, 
                    xtol: float = 1e-6, increasing: bool = True, max_iter: int = 100,
                    progress_callback: Optional[Callable[[float, int, Tuple[float, float]], None]] = None,
                    timeout_checker: Optional[Callable[[], bool]] = None) -> Optional[Tuple[float, int]]:
    """
    Find the edge of a plateau where f(θ) ≈ target.
    
    This is useful for p-value functions that can be flat over ranges of theta,
    such as in Fisher's or Blaker's methods, where we want the smallest theta
    with f(theta) ≈ target.
    
    Args:
        f: Function to evaluate
        lo: Lower bound for search
        hi: Upper bound for search
        target: Target value we're seeking
        xtol: Tolerance for convergence
        increasing: If True, find smallest theta where f(theta) ≥ target
                   If False, find largest theta where f(theta) ≥ target
        max_iter: Maximum iterations
        progress_callback: Optional callback to report progress (percent, iteration, bounds)
        timeout_checker: Optional function that returns True if a timeout has occurred
        
    Returns:
        Tuple of (result, iterations) where result is the theta value at the plateau edge,
        or None if timeout occurred
    """
    # Check if low and high bounds already meet the condition
    f_lo = f(lo)
    
    # Check for timeout after first evaluation
    if timeout_checker and timeout_checker():
        return None
        
    # Handle timeout from function
    if f_lo is None:
        return None
        
    f_hi = f(hi)
    
    # Check for timeout after second evaluation
    if timeout_checker and timeout_checker():
        return None
        
    # Handle timeout from function
    if f_hi is None:
        return None
    
    # If finding the smallest theta where f(theta) ≥ target
    if increasing:
        # If lo already meets the condition, return it
        if f_lo >= target:
            return (lo, 0)
        # If hi doesn't meet the condition, can't find a valid theta
        if f_hi < target:
            # Return hi as the best we can do
            return (hi, 0)
    # If finding the largest theta where f(theta) ≥ target
    else:
        # If hi already meets the condition, return it
        if f_hi >= target:
            return (hi, 0)
        # If lo doesn't meet the condition, can't find a valid theta
        if f_lo < target:
            # Return lo as the best we can do
            return (lo, 0)
    
    # Initialize the bounds for binary search
    lower, upper = lo, hi
    
    # Track iterations
    iter_count = 0
    
    # Binary search for the plateau edge
    while iter_count < max_iter and upper - lower > xtol:
        # Check for timeout
        if timeout_checker and timeout_checker():
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
    
    # Return the appropriate edge of the plateau
    if increasing:
        return (upper, iter_count)
    else:
        return (lower, iter_count)


def find_smallest_theta(f: Callable[[float], float], alpha: float, 
                        lo: float = 1e-8, hi: float = 1.0, 
                        two_sided: bool = True,
                        progress_callback: Optional[Callable[[float], None]] = None,
                        timeout_checker: Optional[Callable[[], bool]] = None) -> Optional[float]:
    """
    Find the smallest theta value that satisfies a given constraint.

    Args:
        f: Function to evaluate (should return a p-value)
        alpha: Significance level
        lo: Lower bound for the search
        hi: Upper bound for the search
        two_sided: Whether to compare p-values against alpha/2 (True) or alpha (False)
                  Methods should ensure their p-value calculation is consistent with this choice.
        progress_callback: Optional callback function to report progress (0-100)
        timeout_checker: Optional function that returns True if a timeout has occurred

    Returns:
        Smallest theta value that satisfies the constraint, or None if timeout occurred
    """
    target_alpha = alpha / 2 if two_sided else alpha
    
    logger.info(f"Finding smallest theta with alpha={alpha}, target_alpha={target_alpha}, two_sided={two_sided}, lo={lo}, hi={hi}")
    
    # Function for root finding
    def g(theta):
        # Check for timeout
        if timeout_checker and timeout_checker():
            return None
            
        # Get p-value
        p_val = f(theta)
        
        # Handle timeout from function
        if p_val is None:
            return None
            
        # For confidence interval, we need p-value ≥ target_alpha
        return p_val - target_alpha
    
    # Find initial theta where g(theta) = 0 (f(theta) = target_alpha)
    logger.info(f"Finding initial theta where f(theta) = target_alpha")
    
    # If large range, use log-space search
    log_search = (hi / lo) > 1000
    
    if log_search:
        logger.info(f"Working in log-space: log_lo={math.log(lo):.6f}, log_hi={math.log(hi):.6f}")
        logger.info(f"Calling find_root_log to find initial theta")
        
        # Find initial theta using log-space search
        try:
            log_theta0 = find_root_log(g, lo, hi, xtol=1e-2, 
                                    progress_callback=progress_callback,
                                    timeout_checker=timeout_checker)
            
            # Handle timeout
            if log_theta0 is None and timeout_checker is not None:
                logger.info(f"Timeout during root finding")
                return None
                
            theta0 = math.exp(log_theta0)
            logger.info(f"Initial theta found: theta0={theta0:.6f}")
        except Exception as e:
            logger.error(f"Error in root finding: {e}")
            return None
    else:
        logger.info(f"Working in linear-space: lo={lo:.6f}, hi={hi:.6f}")
        logger.info(f"Calling find_root to find initial theta")
        
        # Find initial theta using standard bisection
        try:
            theta0 = find_root(g, lo, hi, tol=1e-2)
            logger.info(f"Initial theta found: theta0={theta0:.6f}")
        except Exception as e:
            logger.error(f"Error in root finding: {e}")
            return None
    
    # Check for timeout
    if timeout_checker and timeout_checker():
        return None
    
    # Refine the interval
    lo_refined = 0.9 * theta0
    hi_refined = 1.1 * theta0
    logger.info(f"Starting refinement with lo={lo_refined:.6f}, hi={hi_refined:.6f}")
    
    # Find the plateau edge precisely
    edge_result = find_plateau_edge(g, lo_refined, hi_refined, target=0, 
                                xtol=1e-6, increasing=True, max_iter=100,
                                progress_callback=lambda p, i, bounds: 
                                    logger.info(f"Refinement progress: iteration {i}, current bounds=[{bounds[0]:.6f}, {bounds[1]:.6f}]") 
                                    if i % 5 == 0 else None,
                                timeout_checker=timeout_checker)
    
    # Handle timeout
    if edge_result is None and timeout_checker is not None:
        logger.info(f"Timeout during refinement")
        return None
        
    theta, iterations = edge_result
    logger.info(f"Refinement converged after {iterations} iterations")
    logger.info(f"Final result: theta={theta:.6f}")
    
    return theta
