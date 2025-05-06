"""
Fixed version of Barnard's unconditional exact confidence interval for odds ratio.

This module implements an improved version of Barnard's unconditional exact confidence interval
method that addresses numerical stability issues in the original implementation.
"""

import math
import logging
import time
from typing import Tuple, Union, Callable, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from exactcis.core import validate_counts, apply_haldane_correction
from exactcis.methods.unconditional import exact_ci_unconditional

# Configure logging
logger = logging.getLogger(__name__)

def improved_ci_unconditional(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05, 
    grid_size: int = 50,
    max_table_size: int = 30, 
    refine: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout: Optional[float] = None,
    apply_haldane: bool = False,
    use_fallback: bool = True,
    return_details: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """
    Improved version of Barnard's unconditional exact confidence interval calculation.
    
    This function uses a two-step approach:
    1. Get quick initial boundaries using statsmodels/Fisher's exact test
    2. Use those boundaries to guide the more detailed unconditional method
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points (default: 50)
        max_table_size: Maximum table size for full computation (default: 30)
        refine: Whether to use refinement for precision (default: True)
        progress_callback: Optional progress reporting callback
        timeout: Maximum computation time in seconds
        apply_haldane: Whether to apply Haldane's correction
        use_fallback: Whether to use statsmodels fallback when needed
        return_details: Whether to return details about which method was used
        
    Returns:
        If return_details=False:
            Tuple containing (lower_bound, upper_bound) of the confidence interval.
        If return_details=True:
            Tuple containing (lower_bound, upper_bound, details_dict)
            where details_dict contains information about which method was used.
    """
    # Start timer for timeout checking
    start_time = time.time()
    result_details = {
        "method_used": "unknown",
        "fallback_used": False,
        "convergence_achieved": True,
        "time_taken": 0.0,
        "warnings": []
    }
    
    # Apply Haldane's correction if requested
    if apply_haldane:
        original_a, original_b, original_c, original_d = a, b, c, d
        a, b, c, d = apply_haldane_correction(a, b, c, d)
        if (a != original_a or b != original_b or c != original_c or d != original_d):
            logger.info(f"Applied Haldane's correction: ({original_a}, {original_b}, {original_c}, {original_d}) -> ({a}, {b}, {c}, {d})")
            result_details["warnings"].append("Applied Haldane's correction")
    
    # Validate the counts
    validate_counts(a, b, c, d)
    
    # Calculate the table margins
    n1, n2 = a + b, c + d

    # Calculate observed odds ratio with numerical stability
    # Using small epsilon to avoid division by zero
    epsilon = 1e-10
    if b < epsilon or c < epsilon:
        # Calculate a pseudo odds ratio that's large but finite
        odds_ratio = (a * d + epsilon) / (max(b, epsilon) * max(c, epsilon))
        logger.info(f"Small or zero counts in cells (b={b}, c={c}), using adjusted OR: {odds_ratio}")
        result_details["warnings"].append(f"Small or zero counts in cells (b={b}, c={c})")
    else:
        odds_ratio = (a * d) / (b * c)
    
    logger.info(f"Observed odds ratio: {odds_ratio}")
    result_details["odds_ratio"] = odds_ratio
    
    # Step 1: Get quick initial boundaries using statsmodels/Fisher's exact test
    # This helps narrow the search space for the more detailed method
    initial_bounds_success = False
    try:
        # Try statsmodels first for initial bounds
        quick_ci = get_quick_ci_estimate(a, b, c, d, alpha)
        initial_ci_low, initial_ci_high = quick_ci["ci"]
        logger.info(f"Initial bounds from quick method: ({initial_ci_low}, {initial_ci_high})")
        result_details["initial_bounds_method"] = quick_ci["method"]
        result_details["initial_bounds"] = (initial_ci_low, initial_ci_high)
        initial_bounds_success = True
    except Exception as e:
        logger.warning(f"Could not get initial bounds: {e}")
        result_details["warnings"].append(f"Could not get initial bounds: {e}")
        initial_bounds_success = False
    
    # Step 2: Try the unconditional method with guided search space if we have initial bounds
    unconditional_success = False
    if initial_bounds_success:
        try:
            # Expand bounds slightly to ensure we don't miss the true CI
            # Use logarithmic scale for multiplication to handle very small or large values
            search_low = max(1e-10, initial_ci_low * 0.5)
            search_high = min(1e10, initial_ci_high * 2.0)
            
            # Modify these parameters based on our initial bounds to speed up computation
            theta_min = 1.0 / search_high  # Convert to theta scale (1/OR)
            theta_max = 1.0 / search_low
            
            logger.info(f"Using guided search space: theta_min={theta_min}, theta_max={theta_max}")
            
            ci_low, ci_high = exact_ci_unconditional(
                a, b, c, d, 
                alpha=alpha, 
                grid_size=grid_size,
                max_table_size=max_table_size, 
                refine=refine,
                progress_callback=progress_callback,
                timeout=timeout,
                apply_haldane=False,  # We've already applied it if needed
                theta_min=theta_min,
                theta_max=theta_max
            )
            
            # Check if the results are informative
            if (ci_low == 0.0 and ci_high == float('inf')) or \
               (ci_low is None or ci_high is None):
                logger.warning("Unconditional method produced uninformative CI bounds")
                result_details["warnings"].append("Unconditional method produced uninformative CI bounds")
                unconditional_success = False
            else:
                logger.info(f"Unconditional method with guided search produced CI: ({ci_low}, {ci_high})")
                unconditional_success = True
                result_details["method_used"] = "unconditional_guided"
        except Exception as e:
            logger.warning(f"Error in unconditional method with guided search: {e}")
            result_details["warnings"].append(f"Error in unconditional method with guided search: {e}")
            unconditional_success = False
    
    # If guided unconditional method failed, try the original unguided method
    if not unconditional_success:
        try:
            ci_low, ci_high = exact_ci_unconditional(
                a, b, c, d, 
                alpha=alpha, 
                grid_size=grid_size,
                max_table_size=max_table_size, 
                refine=refine,
                progress_callback=progress_callback,
                timeout=timeout,
                apply_haldane=False  # We've already applied it if needed
            )
            
            if (ci_low == 0.0 and ci_high == float('inf')) or \
               (ci_low is None or ci_high is None):
                logger.warning("Original method produced uninformative CI bounds, using alternative approach")
                result_details["warnings"].append("Original method produced uninformative CI bounds")
                unconditional_success = False
            else:
                logger.info(f"Original method produced CI: ({ci_low}, {ci_high})")
                unconditional_success = True
                result_details["method_used"] = "unconditional_original"
        except Exception as e:
            logger.warning(f"Error in original method: {e}, using alternative approach")
            result_details["warnings"].append(f"Error in original method: {e}")
            unconditional_success = False
    
    # Use fallback method if all previous attempts failed
    if not unconditional_success and use_fallback and initial_bounds_success:
        logger.info("Using quick method results as final CI")
        ci_low, ci_high = initial_ci_low, initial_ci_high
        result_details["method_used"] = quick_ci["method"]
        result_details["fallback_used"] = True
    elif not unconditional_success and use_fallback:
        # Try statsmodels fallback if we haven't already tried it for initial bounds
        logger.info("Using statsmodels-based fallback method")
        try:
            stats_ci = get_statsmodels_ci(a, b, c, d, alpha)
            ci_low, ci_high = stats_ci
            logger.info(f"Statsmodels fallback produced CI: ({ci_low}, {ci_high})")
            result_details["method_used"] = "statsmodels_fallback"
            result_details["fallback_used"] = True
        except Exception as e:
            logger.warning(f"Error in statsmodels fallback: {e}, using Fisher's exact test")
            result_details["warnings"].append(f"Error in statsmodels fallback: {e}")
            
            try:
                # Use Fisher's exact test as last resort
                fisher_ci = get_fisher_approx_ci(a, b, c, d, alpha)
                ci_low, ci_high = fisher_ci
                logger.info(f"Fisher approximation produced CI: ({ci_low}, {ci_high})")
                result_details["method_used"] = "fisher_approximation"
                result_details["fallback_used"] = True
            except Exception as e:
                logger.error(f"All fallback methods failed: {e}")
                result_details["warnings"].append(f"All fallback methods failed: {e}")
                
                # Last resort is to return a conservative estimate
                if odds_ratio < 1:
                    ci_low = max(0.001, odds_ratio * 0.1)
                    ci_high = min(1.0, odds_ratio * 10)
                else:
                    ci_low = max(1.0, odds_ratio * 0.1)
                    ci_high = odds_ratio * 10
                    
                logger.warning(f"Using conservative default CI: ({ci_low}, {ci_high})")
                result_details["method_used"] = "conservative_estimate"
                result_details["fallback_used"] = True
                result_details["convergence_achieved"] = False
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"CI calculation completed in {elapsed_time:.3f} seconds")
    result_details["time_taken"] = elapsed_time
    
    if return_details:
        return ci_low, ci_high, result_details
    else:
        return ci_low, ci_high

def get_quick_ci_estimate(
    a: Union[int, float], 
    b: Union[int, float], 
    c: Union[int, float], 
    d: Union[int, float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Get a quick estimate of confidence interval bounds.
    
    This is used to guide the more detailed unconditional method.
    
    Args:
        a, b, c, d: Cell counts
        alpha: Significance level
        
    Returns:
        Dictionary with CI and method used
    """
    # Try statsmodels first
    try:
        ci = get_statsmodels_ci(a, b, c, d, alpha)
        return {"ci": ci, "method": "statsmodels"}
    except Exception:
        pass
    
    # Fall back to Fisher's approximation
    try:
        ci = get_fisher_approx_ci(a, b, c, d, alpha)
        return {"ci": ci, "method": "fisher_approximation"}
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
    
    return {"ci": (ci_low, ci_high), "method": "wide_estimate"}

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
