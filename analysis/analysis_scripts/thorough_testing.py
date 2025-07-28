"""
Thorough testing of the ExactCIs methods.

This script tests the improved confidence interval methods against a variety
of 2x2 tables to ensure they work correctly across different scenarios.
"""

import numpy as np
import pandas as pd
import logging
import time
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our methods
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.fixed_ci import fisher_approximation
from exactcis.utils.optimization import CICache

def calculate_normal_approximation_ci(a, b, c, d, alpha=0.05, measure="diff"):
    """Calculate confidence interval using normal approximation."""
    n1 = a + b
    n2 = c + d
    p1 = a / n1 if n1 > 0 else 0
    p2 = c / n2 if n2 > 0 else 0
    
    z = stats.norm.ppf(1 - alpha/2)
    
    if measure == "diff":
        # Difference of proportions
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        diff = p1 - p2
        lower = diff - z * se
        upper = diff + z * se
        return (lower, upper)
    
    elif measure == "rr":
        # Risk ratio
        if p2 == 0:
            return (np.nan, np.nan)
        rr = p1 / p2
        se_log_rr = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)) if p1 > 0 and p2 > 0 else np.nan
        if np.isnan(se_log_rr):
            return (np.nan, np.nan)
        log_rr = np.log(rr)
        lower = np.exp(log_rr - z * se_log_rr)
        upper = np.exp(log_rr + z * se_log_rr)
        return (lower, upper)
    
    elif measure == "or":
        # Odds ratio
        if b == 0 or c == 0 or a == 0 or d == 0:
            return (np.nan, np.nan)
        odds1 = a / b
        odds2 = c / d
        or_est = odds1 / odds2
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        log_or = np.log(or_est)
        lower = np.exp(log_or - z * se_log_or)
        upper = np.exp(log_or + z * se_log_or)
        return (lower, upper)
    
    else:
        raise ValueError(f"Unknown measure: {measure}")

def test_table(a, b, c, d, alpha=0.05, cache_instance=None, description=""):
    """Test a single 2x2 table with all methods."""
    print("=" * 80)
    print(f"Testing table: a={a}, b={b}, c={c}, d={d} - {description}")
    print("-" * 80)
    
    results = {}
    
    # Original Barnard's exact method
    start_time = time.time()
    try:
        results["original"] = exact_ci_unconditional(a, b, c, d, alpha=alpha)
        results["original_time"] = time.time() - start_time
    except Exception as e:
        results["original"] = (np.nan, np.nan)
        results["original_time"] = time.time() - start_time
        print(f"Original method failed: {str(e)}")
    
    # Improved Barnard's exact method
    start_time = time.time()
    try:
        results["improved"] = exact_ci_unconditional(a, b, c, d, alpha=alpha, adaptive_grid=True, use_cache=True, 
                                                       use_cache=True, 
                                                       cache_instance=cache_instance)
        results["improved_time"] = time.time() - start_time
    except Exception as e:
        results["improved"] = (np.nan, np.nan)
        results["improved_time"] = time.time() - start_time
        print(f"Improved method failed: {str(e)}")
    
    # Fisher approximation
    start_time = time.time()
    try:
        results["fisher"] = fisher_approximation(a, b, c, d, alpha=alpha)
        results["fisher_time"] = time.time() - start_time
    except Exception as e:
        results["fisher"] = (np.nan, np.nan)
        results["fisher_time"] = time.time() - start_time
        print(f"Fisher approximation failed: {str(e)}")
    
    # Normal approximation (odds ratio)
    try:
        results["normal_or"] = calculate_normal_approximation_ci(a, b, c, d, alpha=alpha, measure="or")
    except Exception as e:
        results["normal_or"] = (np.nan, np.nan)
        print(f"Normal approximation (OR) failed: {str(e)}")
    
    # Format results
    df = pd.DataFrame({
        "Method": ["Original Barnard's", "Improved Barnard's", "Fisher Approximation", "Normal Approximation (OR)"],
        "Lower Bound": [results["original"][0], results["improved"][0], 
                        results["fisher"][0], results["normal_or"][0]],
        "Upper Bound": [results["original"][1], results["improved"][1], 
                        results["fisher"][1], results["normal_or"][1]],
        "Time (s)": [results["original_time"], results["improved_time"], 
                      results["fisher_time"], np.nan]
    })
    
    print(df.to_string(index=False))
    print()
    
    # Compare methods
    if not np.isnan(results["original"][0]) and not np.isnan(results["improved"][0]):
        lower_diff = abs(results["original"][0] - results["improved"][0])
        upper_diff = abs(results["original"][1] - results["improved"][1])
        lower_rel_diff = 100 * lower_diff / abs(results["original"][0]) if results["original"][0] != 0 else np.inf
        upper_rel_diff = 100 * upper_diff / abs(results["original"][1]) if results["original"][1] != 0 else np.inf
        
        print("Comparison between Original and Improved:")
        print(f"  Lower bound absolute difference: {lower_diff:.6f}")
        print(f"  Upper bound absolute difference: {upper_diff:.6f}")
        print(f"  Lower bound relative difference: {lower_rel_diff:.2f}%")
        print(f"  Upper bound relative difference: {upper_rel_diff:.2f}%")
        
        if lower_rel_diff > 5 or upper_rel_diff > 5:
            print("  WARNING: Methods differ by more than 5%")
        else:
            print("  Methods are in good agreement")
    
    print()
    return results

def main():
    """Run thorough testing on a variety of 2x2 tables."""
    # Create a cache instance for the improved method
    cache = CICache()
    
    # Test cases
    test_cases = [
        # Standard case
        {"a": 7, "b": 3, "c": 2, "d": 8, "description": "Standard small table"},
        
        # Zero cells
        {"a": 0, "b": 5, "c": 6, "d": 12, "description": "Zero in cell a"},
        {"a": 5, "b": 0, "c": 6, "d": 12, "description": "Zero in cell b"},
        {"a": 5, "b": 5, "c": 0, "d": 12, "description": "Zero in cell c"},
        {"a": 5, "b": 5, "c": 6, "d": 0, "description": "Zero in cell d"},
        
        # Multiple zeros
        {"a": 0, "b": 0, "c": 6, "d": 12, "description": "Zeros in cells a,b"},
        {"a": 0, "b": 5, "c": 0, "d": 12, "description": "Zeros in cells a,c"},
        
        # Sparse tables
        {"a": 1, "b": 9, "c": 1, "d": 99, "description": "Sparse table 1"},
        {"a": 9, "b": 1, "c": 99, "d": 1, "description": "Sparse table 2"},
        
        # Medium tables
        {"a": 25, "b": 25, "c": 25, "d": 25, "description": "Balanced medium table"},
        {"a": 40, "b": 10, "c": 20, "d": 30, "description": "Unbalanced medium table"},
        
        # Large tables
        {"a": 100, "b": 50, "c": 60, "d": 120, "description": "Large table 1"},
        {"a": 500, "b": 500, "c": 300, "d": 700, "description": "Large table 2"},
        
        # Extreme proportions
        {"a": 99, "b": 1, "c": 50, "d": 50, "description": "Extreme proportion 1"},
        {"a": 1, "b": 99, "c": 50, "d": 50, "description": "Extreme proportion 2"},
        
        # Very different group sizes
        {"a": 5, "b": 5, "c": 90, "d": 10, "description": "Very different group sizes 1"},
        {"a": 90, "b": 10, "c": 5, "d": 5, "description": "Very different group sizes 2"},
    ]
    
    # Run tests
    for tc in test_cases:
        test_table(tc["a"], tc["b"], tc["c"], tc["d"], alpha=0.05, 
                  cache_instance=cache, description=tc["description"])
    
    # Test with different alpha levels
    print("\n" + "=" * 80)
    print("Testing with different alpha levels")
    print("-" * 80)
    
    a, b, c, d = 7, 3, 2, 8
    for alpha in [0.01, 0.05, 0.1]:
        test_table(a, b, c, d, alpha=alpha, cache_instance=cache, 
                  description=f"Standard table with alpha={alpha}")

if __name__ == "__main__":
    main()
