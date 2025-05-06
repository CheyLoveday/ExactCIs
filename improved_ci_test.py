"""
Test script to compare the improved confidence interval calculation with the original
implementation and statsmodels results.

This script uses the existing framework and imports rather than standalone functions.
"""
import time
import math
import logging
import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd

from exactcis.methods import (
    exact_ci_unconditional,
    improved_ci_unconditional
)
from exactcis.core import apply_haldane_correction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_scipy_ci(table, alpha=0.05):
    """Calculate odds ratio and CI using statsmodels."""
    a, b = table[0]
    c, d = table[1]
    
    # Calculate the observed odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0.0
    else:
        odds_ratio = (a * d) / (b * c)
    
    # Create a DataFrame for statsmodels
    data = pd.DataFrame({
        'outcome': [1] * int(a + b + 0.5) + [0] * int(c + d + 0.5),
        'exposure': [1] * int(a + 0.5) + [0] * int(b + 0.5) + 
                     [1] * int(c + 0.5) + [0] * int(d + 0.5)
    })
    
    # Fit the logistic regression model
    try:
        formula = 'outcome ~ exposure'
        model = sm.formula.logit(formula=formula, data=data)
        result = model.fit(disp=0)
        
        # Extract the odds ratio and confidence interval
        coef = result.params['exposure']
        odds_ratio = math.exp(coef)
        ci_low, ci_high = math.exp(result.conf_int(alpha=alpha).loc['exposure'][0]), math.exp(result.conf_int(alpha=alpha).loc['exposure'][1])
        
        return odds_ratio, (ci_low, ci_high), "statsmodels"
    except Exception as e:
        logger.warning(f"Statsmodels error: {e}, using Fisher's exact test")
        # Use Fisher's exact test as fallback
        odds_ratio, p_value = stats.fisher_exact(table)
        
        # Simple approximation for CI when Fisher's exact test is used
        if all(x > 0 for x in [a, b, c, d]):
            log_odds = math.log(odds_ratio)
            stderr = 1.96 / math.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_low = math.exp(log_odds - stderr)
            ci_high = math.exp(log_odds + stderr)
        else:
            ci_low, ci_high = 0, float('inf')
        
        return odds_ratio, (ci_low, ci_high), "fisher"

def run_test_for_table(table, alpha=0.05, apply_haldane=False):
    """Run tests for a specific 2x2 table with all methods."""
    a, b = table[0]
    c, d = table[1]
    
    print(f"\nTable: {table}")
    
    # Calculate the odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0.0
    else:
        odds_ratio = (a * d) / (b * c)
    
    print(f"Odds Ratio: {odds_ratio:.6f}")
    
    # Method 1: SciPy/statsmodels
    start_time = time.time()
    scipy_or, scipy_ci, scipy_method = calculate_scipy_ci(table, alpha)
    scipy_time = time.time() - start_time
    print(f"SciPy/statsmodels CI: ({scipy_ci[0]:.6f}, {scipy_ci[1]:.6f}), Method: {scipy_method}, Time: {scipy_time:.6f}s")
    
    # Method 2: Original ExactCIs
    print("\nCalculating with original ExactCIs...")
    start_time = time.time()
    try:
        original_low, original_high = exact_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane)
        original_time = time.time() - start_time
        original_method = "unconditional"
        print(f"Original ExactCIs CI: ({original_low:.6f}, {original_high if original_high != float('inf') else 'inf'}), Method: {original_method}, Time: {original_time:.6f}s")
    except Exception as e:
        original_time = time.time() - start_time
        print(f"Error with original ExactCIs: {e}")
        original_low, original_high, original_method = None, None, "error"
    
    # Method 3: Improved ExactCIs
    print("\nCalculating with improved ExactCIs...")
    start_time = time.time()
    try:
        improved_low, improved_high, details = improved_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane, return_details=True)
        improved_time = time.time() - start_time
        method_used = details.get("method_used", "unknown")
        fallback_used = details.get("fallback_used", False)
        warnings = details.get("warnings", [])
        
        method_info = f"{method_used}"
        if fallback_used:
            method_info += " (fallback)"
        
        print(f"Improved ExactCIs CI: ({improved_low:.6f}, {improved_high:.6f}), Method: {method_info}, Time: {improved_time:.6f}s")
        
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if "initial_bounds_method" in details:
            print(f"Initial bounds method: {details['initial_bounds_method']}")
            if "initial_bounds" in details:
                bounds = details["initial_bounds"]
                print(f"Initial bounds: ({bounds[0]:.6f}, {bounds[1]:.6f})")
    except Exception as e:
        improved_time = time.time() - start_time
        print(f"Error with improved ExactCIs: {e}")
        improved_low, improved_high, method_used = None, None, "error"
    
    # Compare the CI widths
    if all(x is not None for x in [scipy_ci[0], scipy_ci[1], improved_low, improved_high]):
        scipy_width = scipy_ci[1] - scipy_ci[0]
        improved_width = improved_high - improved_low
        if improved_width > 0 and scipy_width > 0:
            width_ratio = improved_width / scipy_width
            print(f"\nCI width ratio (Improved/SciPy): {width_ratio:.2f}")
    
    return {
        'table': table,
        'or': odds_ratio,
        'scipy': {'ci': scipy_ci, 'time': scipy_time, 'method': scipy_method},
        'original': {'ci': (original_low, original_high) if original_low is not None else None, 'time': original_time, 'method': original_method},
        'improved': {'ci': (improved_low, improved_high) if improved_low is not None else None, 'time': improved_time, 'method': method_used}
    }

def run_tests():
    """Run tests for various sample tables."""
    # Test tables from various examples
    tables = [
        # Basic test case from previous examples
        [[50, 500], [10, 1000]],
        
        # Small counts test case
        [[2, 1000], [10, 1000]],
        
        # Equal counts in one column
        [[10, 50], [10, 100]],
        
        # Zero in one cell
        [[0, 100], [10, 100]],
        
        # Standard example table
        [[6, 9], [3, 12]]
    ]
    
    results = []
    for table in tables:
        result = run_test_for_table(table)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Table':<20} {'Method':<20} {'OR':<10} {'CI':<30} {'Method Used':<20} {'Time (s)':<10}")
    print("-"*80)
    
    for result in results:
        table = result['table']
        table_str = f"[{table[0][0]},{table[0][1]}],[{table[1][0]},{table[1][1]}]"
        
        # Print the odds ratio once for the table
        print(f"{table_str:<20} {'OR':<20} {result['or']:<10.4f}")
        
        # SciPy row
        ci = result['scipy']['ci']
        ci_str = f"({ci[0]:.6f}, {ci[1] if ci[1] != float('inf') else 'inf'})"
        print(f"{'':<20} {'SciPy':<20} {'':<10} {ci_str:<30} {result['scipy']['method']:<20} {result['scipy']['time']:<10.4f}")
        
        # Original ExactCIs row
        if result['original']['ci'] is not None:
            ci = result['original']['ci']
            ci_str = f"({ci[0]:.6f}, {ci[1] if ci[1] != float('inf') else 'inf'})"
            print(f"{'':<20} {'Original':<20} {'':<10} {ci_str:<30} {result['original']['method']:<20} {result['original']['time']:<10.4f}")
        else:
            print(f"{'':<20} {'Original':<20} {'':<10} {'Error':<30} {result['original']['method']:<20} {result['original']['time']:<10.4f}")
        
        # Improved ExactCIs row
        if result['improved']['ci'] is not None:
            ci = result['improved']['ci']
            ci_str = f"({ci[0]:.6f}, {ci[1]:.6f})"
            print(f"{'':<20} {'Improved':<20} {'':<10} {ci_str:<30} {result['improved']['method']:<20} {result['improved']['time']:<10.4f}")
        else:
            print(f"{'':<20} {'Improved':<20} {'':<10} {'Error':<30} {result['improved']['method']:<20} {result['improved']['time']:<10.4f}")
        
        print("-"*80)

if __name__ == "__main__":
    run_tests()
