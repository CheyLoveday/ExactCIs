"""
Fixed version of Barnard's unconditional exact confidence interval for odds ratio.

This module implements an improved version of Barnard's unconditional exact confidence interval
method that addresses numerical stability issues in the original implementation.
"""

import math
import logging
import time
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

# Third-party imports
import numpy as np
from scipy import stats

# Local imports
from ..core import calculate_odds_ratio
from .unconditional import exact_ci_unconditional, _log_pvalue_barnard as unconditional_log_pvalue
from exactcis.core import validate_counts, apply_haldane_correction
from ..utils.optimization import get_global_cache, derive_search_params

# Configure logging
logger = logging.getLogger(__name__)

def improved_ci_unconditional(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05, 
    fallback_method: str = "statsmodels",
    max_time: float = 120,
    optimization_params: Optional[Dict[str, Any]] = None
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """
    Calculate improved Barnard's unconditional exact confidence interval.
    
    This function implements a two-step approach:
    1. Get quick initial estimates to bound the search
    2. Run the unconditional method with these bounds for a guided, efficient search
    
    Args:
        a, b, c, d: Cell counts for 2x2 table
        alpha: Significance level (default 0.05)
        fallback_method: Method to use as fallback ('statsmodels' or 'fisher')
        max_time: Maximum time in seconds to attempt calculation
        optimization_params: Additional optimization parameters
        
    Returns:
        Tuple of ((lower, upper), metadata) where metadata contains information about the calculation
    """
    start_time = time.time()
    logger.info(f"Starting improved CI calculation for ({a},{b},{c},{d}) at alpha={alpha}")
    
    # Initialize result metadata
    metadata = {
        "original_method_used": False,
        "method_used": "unconditional_guided",
        "warnings": [],
        "calculation_time": 0,
        "fallback_used": False,
        "initial_bounds_method": None
    }
    
    # Check cache first
    cache = get_global_cache()
    cached_result = cache.get_exact(a, b, c, d, alpha)
    if cached_result is not None:
        result, cached_metadata = cached_result
        logger.info(f"Using cached CI: {result}")
        
        # Combine cached metadata with our metadata
        metadata.update(cached_metadata)
        metadata["calculation_time"] = 0  # Reset time since we're using cache
        
        return result, metadata
    
    # Step 1: Get quick initial estimates
    logger.info("Calculating initial bounds using quick methods")
    quick_ci, quick_method = get_quick_ci_estimate(a, b, c, d, alpha)
    logger.info(f"Initial bounds from {quick_method}: {quick_ci}")
    
    # Save initial bounds information
    metadata["initial_bounds_method"] = quick_method
    
    # Handle special cases where quick method might already be optimal
    if quick_method == "fisher_exact":
        # If Fisher's exact test already gave a good result for special cases
        if a == 0 or d == 0 or b == 0 or c == 0:
            logger.info("Using Fisher's exact result for table with zeros")
            metadata["method_used"] = "fisher_exact"
            elapsed = time.time() - start_time
            metadata["calculation_time"] = elapsed
            
            # Add to cache
            cache.add(a, b, c, d, alpha, quick_ci, metadata)
            
            return quick_ci, metadata
    
    # Step 2: Use quick estimates to guide unconditional method
    logger.info("Using initial bounds to guide unconditional method")
    theta_min, theta_max = quick_ci
    
    # Add safety margins to ensure we don't miss the true bounds
    if theta_min > 0:
        theta_min = max(1e-6, theta_min * 0.5)
    
    if theta_max < float('inf'):
        theta_max = min(1e6, theta_max * 2)
    
    # Look for similar tables in cache to further refine search params
    similar_entries = cache.get_similar(a, b, c, d, alpha)
    if similar_entries:
        search_params = derive_search_params(a, b, c, d, similar_entries)
        logger.info(f"Using parameters from similar tables: {search_params}")
        
        # Use optimization params from similar entries if available
        optimization_params = optimization_params or {}
        optimization_params.update(search_params)
    
    # Track how much time we've spent
    time_so_far = time.time() - start_time
    remaining_time = max_time - time_so_far
    
    # Only proceed with unconditional method if we have enough time
    if remaining_time > 5:  # At least 5 seconds remaining
        try:
            # Run unconditional method with guided search
            unconditional_ci = exact_ci_unconditional(
                a, b, c, d, alpha,
                theta_min=theta_min,
                theta_max=theta_max,
                optimization_params=optimization_params
            )
            
            # Check if result is informative
            if (unconditional_ci[0] > 0 or unconditional_ci[0] == 0 and (a == 0 or b == 0)) and \
               (unconditional_ci[1] < float('inf') or unconditional_ci[1] == float('inf') and (c == 0 or d == 0)):
                logger.info(f"Unconditional method with guided search produced CI: {unconditional_ci}")
                metadata["original_method_used"] = True
                
                # We're done!
                elapsed = time.time() - start_time
                metadata["calculation_time"] = elapsed
                
                # Add to cache
                cache.add(a, b, c, d, alpha, unconditional_ci, metadata)
                
                return unconditional_ci, metadata
            else:
                # Result is not informative
                logger.warning("Unconditional method produced uninformative CI bounds")
                metadata["warnings"].append("Unconditional method produced uninformative CI bounds")
        except Exception as e:
            logger.error(f"Error in unconditional method: {e}")
            metadata["warnings"].append(f"Error in unconditional calculation: {str(e)}")
    else:
        logger.warning(f"Not enough time for unconditional method, using quick result (remaining time: {remaining_time:.2f}s)")
        metadata["warnings"].append("Not enough time for unconditional calculation")
    
    # Step 3: If we get here, fallback to the quick method result
    logger.warning("Original method produced uninformative CI bounds, using alternative approach")
    metadata["warnings"].append("Original method produced uninformative CI bounds")
    metadata["fallback_used"] = True
    metadata["method_used"] = quick_method
    
    logger.info("Using quick method results as final CI")
    elapsed = time.time() - start_time
    metadata["calculation_time"] = elapsed
    
    # Add to cache before returning
    cache.add(a, b, c, d, alpha, quick_ci, metadata)
    
    return quick_ci, metadata

def get_quick_ci_estimate(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05
) -> Tuple[Tuple[float, float], str]:
    """
    Get a quick estimate of confidence interval bounds.
    
    This is used to guide the more detailed unconditional method.
    
    Args:
        a, b, c, d: Cell counts
        alpha: Significance level
        
    Returns:
        Tuple of (CI, method_used)
    """
    # Try statsmodels first
    try:
        ci = get_statsmodels_ci(a, b, c, d, alpha)
        return ci, "statsmodels"
    except Exception:
        pass
    
    # Fall back to Fisher's approximation
    try:
        ci = get_fisher_approx_ci(a, b, c, d, alpha)
        return ci, "fisher_approximation"
    except Exception:
        pass
    
    # Last resort - use a very wide estimate based on odds ratio
    table = [[a, b], [c, d]]
    try:
        odds_ratio, _ = stats.fisher_exact(table)
    except Exception:
        # If Fisher's exact test fails, calculate OR directly
        if b * c == 0:
            odds_ratio = float('inf') if a * d > 0 else 0.0
        else:
            odds_ratio = (a * d) / (b * c)
    
    # Very wide search space
    if odds_ratio < 1e-5:
        ci_low = 1e-10
        ci_high = 0.1
    elif odds_ratio > 1e5:
        ci_low = 10
        ci_high = 1e10
    else:
        ci_low = max(1e-10, odds_ratio * 0.01)
        ci_high = min(1e10, odds_ratio * 100)
    
    return (ci_low, ci_high), "wide_estimate"

def get_statsmodels_ci(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate confidence interval using statsmodels logistic regression.
    
    Args:
        a, b, c, d: Cell counts
        alpha: Significance level
        
    Returns:
        Tuple (lower_bound, upper_bound)
    """
    # Create table for statsmodels using pandas DataFrame
    # This approach avoids array dimension mismatch issues
    data = pd.DataFrame({
        'outcome': [1] * int(a + b + 0.5) + [0] * int(c + d + 0.5),
        'exposure': [1] * int(a + 0.5) + [0] * int(b + 0.5) + 
                    [1] * int(c + 0.5) + [0] * int(d + 0.5)
    })
    
    # Fit the logistic regression model with formula
    formula = 'outcome ~ exposure'
    model = sm.formula.logit(formula=formula, data=data)
    result = model.fit(disp=0)
    
    # Extract confidence interval
    coef = result.params['exposure']
    ci_low = math.exp(result.conf_int(alpha=alpha).loc['exposure'][0])
    ci_high = math.exp(result.conf_int(alpha=alpha).loc['exposure'][1])
    
    return ci_low, ci_high

def get_fisher_approx_ci(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate an approximate confidence interval based on Fisher's exact test.
    
    Args:
        a, b, c, d: Cell counts
        alpha: Significance level
        
    Returns:
        Tuple (lower_bound, upper_bound)
    """
    table = [[a, b], [c, d]]
    odds_ratio, p_value = stats.fisher_exact(table)
    
    # Use a simple approximation based on the observed odds ratio
    if all(x > 0 for x in [a, b, c, d]):
        # Calculate standard error on log scale
        log_odds = math.log(max(odds_ratio, 1e-10))
        stderr = stats.norm.ppf(1 - alpha/2) * math.sqrt(1/a + 1/b + 1/c + 1/d)
        
        # Calculate confidence interval
        ci_low = math.exp(log_odds - stderr)
        ci_high = math.exp(log_odds + stderr)
        
        # Make CI slightly wider for a more conservative estimate
        width = ci_high - ci_low
        ci_low = max(0.001, ci_low - width * 0.1)
        ci_high = ci_high + width * 0.1
    elif odds_ratio == 0:
        # Special case for odds ratio of 0
        ci_low = 0.0
        ci_high = 0.5  # Arbitrary but reasonable upper bound
    elif odds_ratio == float('inf'):
        # Special case for infinite odds ratio
        ci_low = 2.0  # Arbitrary but reasonable lower bound
        ci_high = float('inf')
    else:
        # General case for tables with zeros
        if odds_ratio < 1:
            ci_low = odds_ratio * 0.1
            ci_high = 1.0
        else:
            ci_low = 1.0
            ci_high = odds_ratio * 10
    
    return ci_low, ci_high

def fisher_approximation(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate confidence interval for odds ratio using Fisher's exact test approximation.
    
    This method calculates a fast approximation of the confidence interval using
    the normal approximation of the log odds ratio.
    
    Args:
        a, b, c, d: Cell counts in the 2x2 table
        alpha: Significance level (default: 0.05)
        
    Returns:
        Tuple of (lower, upper) confidence interval bounds
    """
    import logging
    import numpy as np
    from scipy import stats
    
    logger = logging.getLogger(__name__)
    
    # Handle zero cells with continuity correction
    a_adj = a if a > 0 else 0.5
    b_adj = b if b > 0 else 0.5
    c_adj = c if c > 0 else 0.5
    d_adj = d if d > 0 else 0.5
    
    # Calculate odds ratio and its log
    odds_ratio = (a_adj * d_adj) / (b_adj * c_adj)
    log_odds = np.log(odds_ratio)
    
    # Calculate standard error of log odds ratio
    se = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
    
    # Get critical value for the confidence level
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate confidence interval on log scale
    log_lower = log_odds - z * se
    log_upper = log_odds + z * se
    
    # Transform back to original scale
    lower = np.exp(log_lower)
    upper = np.exp(log_upper)
    
    logger.info(f"Fisher approximation CI calculated: ({lower}, {upper})")
    return lower, upper
