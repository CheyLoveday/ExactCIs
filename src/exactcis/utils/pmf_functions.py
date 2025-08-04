"""
Pure functions for PMF (Probability Mass Function) calculations.
"""

import math
import logging
import time
import numpy as np
from typing import Union, Tuple, Dict
from functools import lru_cache
from .data_models import PMFWeightsConfig, PMFWeightsResult
from ..core import support, log_binom_coeff, SupportData

logger = logging.getLogger(__name__)

# Rate limiting for repetitive warnings
_warning_timestamps: Dict[str, float] = {}
_warning_cooldown = 60.0  # Seconds between similar warnings

def _rate_limited_warning(message: str, warning_key: str = None) -> None:
    """
    Issue a warning with rate limiting to prevent log spam.
    
    Args:
        message: Warning message to log
        warning_key: Unique key for this type of warning (defaults to message)
    """
    if warning_key is None:
        warning_key = message
    
    current_time = time.time()
    last_warning_time = _warning_timestamps.get(warning_key, 0)
    
    if current_time - last_warning_time > _warning_cooldown:
        logger.warning(message)
        _warning_timestamps[warning_key] = current_time


def handle_theta_zero_case(supp: SupportData) -> PMFWeightsResult:
    """
    Handle the special case where theta = 0.
    
    Args:
        supp: Support data structure
        
    Returns:
        PMF weights with all mass at minimum value
    """
    weights = [1.0 if k == supp.min_val else 0.0 for k in supp.x]
    return PMFWeightsResult(
        support=tuple(supp.x.tolist()),
        weights=tuple(weights),
        method="theta_zero"
    )


def handle_theta_large_case(supp: SupportData, theta: float) -> PMFWeightsResult:
    """
    Handle the special case where theta is very large or infinite.
    
    Args:
        supp: Support data structure
        theta: Large theta value
        
    Returns:
        PMF weights with all mass at maximum value
    """
    max_val = supp.max_val
    weights = [1.0 if k == max_val else 0.0 for k in supp.x]
    
    return PMFWeightsResult(
        support=tuple(supp.x.tolist()),
        weights=tuple(weights),
        method="theta_large"
    )


def validate_pmf_parameters(config: PMFWeightsConfig) -> bool:
    """
    Validate PMF calculation parameters.
    
    Args:
        config: PMF weights configuration
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    if any(val < 0 for val in [config.n1, config.n2, config.m]):
        raise ValueError("n1, n2, and m must be non-negative")
    
    if not math.isfinite(config.theta):
        if not (config.theta > 0 or np.isinf(config.theta)):
            raise ValueError("theta must be positive or positive infinity")
    elif config.theta < 0:
        raise ValueError("theta must be non-negative")
    
    return True


def check_for_large_values(config: PMFWeightsConfig) -> bool:
    """
    Check if parameters contain unusually large values that might cause issues.
    
    Increased threshold to 5000 to be more appropriate for epidemiological studies.
    
    Args:
        config: PMF weights configuration
        
    Returns:
        True if large values detected
    """
    return any(float(val) > 5000 for val in [config.n1, config.n2, config.m])


def calculate_log_binomial_terms(
    supp: SupportData, 
    config: PMFWeightsConfig
) -> Tuple[np.ndarray, bool]:
    """
    Calculate log binomial coefficient terms.
    
    Args:
        supp: Support data structure
        config: PMF weights configuration
        
    Returns:
        Tuple of (log_terms, success_flag)
    """
    try:
        k = supp.x
        
        with np.errstate(divide='ignore'):
            log_comb_n1_k = np.vectorize(log_binom_coeff)(config.n1, k)
            log_comb_n2_m_k = np.vectorize(log_binom_coeff)(config.n2, config.m - k)
        
        log_theta = math.log(config.theta)
        log_terms = log_comb_n1_k + log_comb_n2_m_k + k * log_theta
        
        return log_terms, True
        
    except (OverflowError, ValueError) as e:
        logger.warning(f"Numerical error in binomial term calculation: {e}")
        return np.full(len(supp.x), float('-inf')), False


def normalize_weights_numerically_stable(log_terms: np.ndarray) -> np.ndarray:
    """
    Normalize weights using numerically stable method.
    
    Args:
        log_terms: Log-space probability terms
        
    Returns:
        Normalized weights
    """
    # Filter out -inf values
    valid_mask = log_terms != float('-inf')
    
    if not np.any(valid_mask):
        logger.warning("No valid log probabilities, using uniform distribution")
        return np.full(len(log_terms), 1.0 / len(log_terms))
    
    valid_logs = log_terms[valid_mask]
    max_log = np.max(valid_logs)
    
    # Use numerically stable approach
    weights = np.zeros_like(log_terms)
    
    for i, log_val in enumerate(log_terms):
        if log_val == float('-inf'):
            weights[i] = 0.0
        else:
            # Protect against underflow
            exp_term = min(log_val - max_log, 700)
            weights[i] = math.exp(exp_term)
    
    # Normalize
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        weights = np.full(len(weights), 1.0 / len(weights))
    
    return weights


def validate_weight_distribution(
    weights: np.ndarray, 
    supp: SupportData, 
    config: PMFWeightsConfig
) -> np.ndarray:
    """
    Validate and potentially correct weight distribution.
    
    Args:
        weights: Computed weights
        supp: Support data structure
        config: PMF weights configuration
        
    Returns:
        Validated/corrected weights
    """
    # For large theta, maximum probability should be at k_max
    if config.theta > 1e3 and not np.isinf(config.theta):
        max_k_idx = np.where(supp.x == supp.max_val)[0][0]
        max_prob_idx = np.argmax(weights)
        
        if max_prob_idx != max_k_idx:
            logger.warning(
                f"Unexpected probability distribution for large theta={config.theta:.2e}: "
                f"max prob at k={supp.x[max_prob_idx]} instead of k_max={supp.max_val}"
            )
    
    return weights


def pmf_weights_normal_case(config: PMFWeightsConfig) -> PMFWeightsResult:
    """
    Calculate PMF weights for normal (non-edge-case) theta values.
    
    Args:
        config: PMF weights configuration
        
    Returns:
        PMF weights result
    """
    supp = support(config.n1, config.n2, config.m)
    
    # Check for potential numerical issues with rate limiting
    if check_for_large_values(config):
        _rate_limited_warning(
            f"Large values detected in pmf_weights: "
            f"n1={config.n1}, n2={config.n2}, m={config.m}",
            "large_values_pmf"
        )
    
    # Calculate log binomial terms
    log_terms, calculation_success = calculate_log_binomial_terms(supp, config)
    
    if not calculation_success:
        # Return uniform distribution as fallback
        uniform_weights = [1.0 / len(supp.x)] * len(supp.x)
        return PMFWeightsResult(
            support=tuple(supp.x.tolist()),
            weights=tuple(uniform_weights),
            method="uniform_fallback"
        )
    
    # Normalize weights
    weights = normalize_weights_numerically_stable(log_terms)
    
    # Validate distribution
    weights = validate_weight_distribution(weights, supp, config)
    
    return PMFWeightsResult(
        support=tuple(supp.x.tolist()),
        weights=tuple(weights.tolist()),
        method="normal_calculation"
    )


def pmf_weights_functional(config: PMFWeightsConfig) -> PMFWeightsResult:
    """
    Calculate PMF weights using functional approach with pure functions.
    
    Args:
        config: PMF weights configuration
        
    Returns:
        PMF weights result
    """
    # Validate parameters
    validate_pmf_parameters(config)
    
    # Get support structure
    supp = support(config.n1, config.n2, config.m)
    
    # Handle special cases using pure functions
    if config.theta <= 0:
        return handle_theta_zero_case(supp)
    
    if config.theta >= 1e6 or np.isinf(config.theta):
        return handle_theta_large_case(supp, config.theta)
    
    # Handle normal case
    return pmf_weights_normal_case(config)


# Backward compatibility function
def pmf_weights_impl_functional(
    n1: Union[int, float], 
    n2: Union[int, float], 
    m: Union[int, float], 
    theta: float
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    Functional implementation of PMF weights calculation.
    
    This is a drop-in replacement for the original _pmf_weights_impl function
    that uses pure functions and follows functional programming principles.
    
    Args:
        n1: Size of first group
        n2: Size of second group
        m: Number of successes
        theta: Odds ratio parameter
        
    Returns:
        Tuple containing (support, probabilities)
    """
    config = PMFWeightsConfig(n1=n1, n2=n2, m=m, theta=theta)
    result = pmf_weights_functional(config)
    return result.support, result.weights