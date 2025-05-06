"""
Test script to verify confidence interval calculation with a single 2x2 table.
"""
import time
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm

from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.core import apply_haldane_correction

def calculate_scipy_ci(table, alpha=0.05):
    """Calculate odds ratio and CI using statsmodels."""
    # Create a DataFrame for statsmodels
    x1 = np.array([1] * table[0][0] + [0] * table[0][1])
    x2 = np.array([1] * table[1][0] + [0] * table[1][1])
    y = np.array([1] * (table[0][0] + table[0][1]) + [0] * (table[1][0] + table[1][1]))
    
    # Add the intercept
    X = np.column_stack((np.ones(len(y)), x1))
    
    # Fit the logistic regression model
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        
        # Extract the odds ratio and confidence interval
        odds_ratio = math.exp(result.params[1])
        ci_low, ci_high = math.exp(result.conf_int(alpha=alpha)[1][0]), math.exp(result.conf_int(alpha=alpha)[1][1])
        
        return odds_ratio, (ci_low, ci_high)
    except Exception as e:
        # Use Fisher's exact test as fallback
        odds_ratio, p_value = stats.fisher_exact(table)
        # Cannot get CI from fisher_exact directly, return a placeholder
        return odds_ratio, (0, float('inf'))

def calculate_exactcis_ci(table, alpha=0.05, apply_haldane=False):
    """Calculate odds ratio and CI using ExactCIs."""
    a, b = table[0]
    c, d = table[1]
    
    # Calculate the odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0
    else:
        odds_ratio = (a * d) / (b * c)
    
    # Calculate the confidence interval
    start_time = time.time()
    ci_low, ci_high = exact_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane)
    duration = time.time() - start_time
    
    return odds_ratio, (ci_low, ci_high), duration

def run_test():
    """Run test for a single 2x2 table."""
    # Test table: known to produce meaningful results with statsmodels
    # Using [50, 500], [10, 1000] from your previous tests
    table = [[50, 500], [10, 1000]]
    
    print(f"\nTable: {table}")
    
    # Calculate with SciPy/statsmodels
    scipy_or, scipy_ci = calculate_scipy_ci(table)
    print(f"SciPy OR: {scipy_or:.6f}, CI: {scipy_ci[0]:.6f} - {scipy_ci[1]:.6f}")
    
    # Calculate with ExactCIs (without Haldane correction)
    print("\nCalculating with ExactCIs (no Haldane correction)...")
    exactcis_or, exactcis_ci, duration = calculate_exactcis_ci(table)
    print(f"ExactCIs OR: {exactcis_or:.6f}, CI: {exactcis_ci[0]:.6f} - {exactcis_ci[1]:.6f}")
    print(f"Time taken: {duration:.6f} seconds")

if __name__ == "__main__":
    run_test()
