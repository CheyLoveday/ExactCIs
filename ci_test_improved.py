"""
Test script to compare the improved confidence interval calculation with original and statsmodels.
"""
import time
import math
import numpy as np
from scipy import stats
import statsmodels.api as sm

from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.fixed_ci import improved_ci_unconditional
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

def run_test_with_table(table, alpha=0.05, apply_haldane=False):
    """Run test for a specific 2x2 table with all methods."""
    a, b = table[0]
    c, d = table[1]
    
    print(f"\nTable: {table}")
    
    # Calculate with SciPy/statsmodels
    start_time = time.time()
    scipy_or, scipy_ci = calculate_scipy_ci(table, alpha)
    scipy_time = time.time() - start_time
    print(f"SciPy OR:     {scipy_or:.6f}, CI: ({scipy_ci[0]:.6f}, {scipy_ci[1]:.6f}), Time: {scipy_time:.6f}s")
    
    # Calculate with original ExactCIs
    print("\nCalculating with original ExactCIs...")
    start_time = time.time()
    # Calculate odds ratio
    if b * c == 0:
        exact_or = float('inf') if a * d > 0 else 0
    else:
        exact_or = (a * d) / (b * c)
    
    # Calculate the confidence interval
    ci_low, ci_high = exact_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane)
    exact_time = time.time() - start_time
    print(f"Original OR:  {exact_or:.6f}, CI: ({ci_low:.6f}, {ci_high if ci_high != float('inf') else 'inf'}), Time: {exact_time:.6f}s")
    
    # Calculate with improved ExactCIs
    print("\nCalculating with improved ExactCIs...")
    start_time = time.time()
    improved_low, improved_high = improved_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane)
    improved_time = time.time() - start_time
    print(f"Improved OR:  {exact_or:.6f}, CI: ({improved_low:.6f}, {improved_high:.6f}), Time: {improved_time:.6f}s")
    
    # Calculate ratio of SciPy CI width to improved CI width
    scipy_width = scipy_ci[1] - scipy_ci[0]
    improved_width = improved_high - improved_low
    if improved_width > 0 and scipy_width > 0:
        width_ratio = improved_width / scipy_width
        print(f"\nCI width ratio (Improved/SciPy): {width_ratio:.2f}")
    
    return {
        'table': table,
        'scipy': {'or': scipy_or, 'ci': scipy_ci, 'time': scipy_time},
        'original': {'or': exact_or, 'ci': (ci_low, ci_high), 'time': exact_time},
        'improved': {'or': exact_or, 'ci': (improved_low, improved_high), 'time': improved_time}
    }

def run_tests():
    """Run tests for all sample tables."""
    # Test tables from various examples
    tables = [
        # Basic test case where we had infinite bounds previously
        [[50, 500], [10, 1000]],
        
        # Small counts test case
        [[2, 1000], [10, 1000]],
        
        # Equal counts in one column
        [[10, 50], [10, 100]],
        
        # Zero in one cell
        [[0, 100], [10, 100]]
    ]
    
    results = []
    for table in tables:
        result = run_test_with_table(table)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Table':<15} {'Method':<10} {'OR':<10} {'CI':<25} {'Time (s)':<10}")
    print("-"*60)
    
    for result in results:
        table = result['table']
        table_str = f"[{table[0]},{table[1]}]"
        
        # SciPy row
        ci = result['scipy']['ci']
        ci_str = f"({ci[0]:.4f}, {ci[1] if ci[1] != float('inf') else 'inf'})"
        print(f"{table_str:<15} {'SciPy':<10} {result['scipy']['or']:<10.4f} {ci_str:<25} {result['scipy']['time']:<10.4f}")
        
        # Original row
        ci = result['original']['ci']
        ci_str = f"({ci[0]:.4f}, {ci[1] if ci[1] != float('inf') else 'inf'})"
        print(f"{'':<15} {'Original':<10} {result['original']['or']:<10.4f} {ci_str:<25} {result['original']['time']:<10.4f}")
        
        # Improved row
        ci = result['improved']['ci']
        ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
        print(f"{'':<15} {'Improved':<10} {result['improved']['or']:<10.4f} {ci_str:<25} {result['improved']['time']:<10.4f}")
        
        print("-"*60)

if __name__ == "__main__":
    run_tests()
