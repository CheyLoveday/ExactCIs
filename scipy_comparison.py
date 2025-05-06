"""
Compare ExactCIs confidence intervals with SciPy/statsmodels implementation
for integer counts without correction.
"""

import numpy as np
import logging
import time
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from exactcis.methods import exact_ci_unconditional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_odds_ratio(a, b, c, d):
    """Calculate the odds ratio for a 2x2 table."""
    try:
        return (a * d) / (b * c)
    except ZeroDivisionError:
        return float('inf') if a * d > 0 else 0.0

def format_float_or_inf(value):
    """Format a float value, handling infinity."""
    if np.isinf(value):
        return "inf"
    else:
        return f"{value:.6f}"

def analyze_with_scipy_statsmodels(a, b, c, d, alpha=0.05):
    """
    Calculate the odds ratio and confidence interval using SciPy's Fisher exact test
    and statsmodels for confidence intervals.
    """
    # Create the contingency table
    table = np.array([[a, b], [c, d]])
    
    # Calculate odds ratio manually
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    
    # Calculate Fisher's exact test with SciPy
    start_time = time.time()
    result = stats.fisher_exact(table, alternative='two-sided')
    
    # Create a dataframe for statsmodels to calculate CI
    import pandas as pd
    data = []
    # Group 1
    for _ in range(a):
        data.append({'group': 1, 'outcome': 1})
    for _ in range(b):
        data.append({'group': 1, 'outcome': 0})
    # Group 2
    for _ in range(c):
        data.append({'group': 0, 'outcome': 1})
    for _ in range(d):
        data.append({'group': 0, 'outcome': 0})
    
    df = pd.DataFrame(data)
    
    # Fit logistic regression model
    model = smf.logit('outcome ~ group', data=df).fit(disp=False)
    conf_int = model.conf_int(alpha=alpha)
    ci_lower = np.exp(conf_int.loc['group'][0])
    ci_upper = np.exp(conf_int.loc['group'][1])
    
    elapsed = time.time() - start_time
    
    # SciPy returns (odds_ratio, p_value)
    scipy_or = result[0]
    p_value = result[1]
    
    result_str = f"""
SciPy/Statsmodels Analysis:
---------------------
Contingency Table:
[{a}, {b}]
[{c}, {d}]

Odds Ratio: {format_float_or_inf(scipy_or)}
p-value: {p_value:.6f}
95% CI: ({format_float_or_inf(ci_lower)}, {format_float_or_inf(ci_upper)})
Computation Time: {elapsed:.6f} seconds
"""
    print(result_str)
    return scipy_or, p_value, ci_lower, ci_upper, elapsed

def analyze_with_exactcis(a, b, c, d, alpha=0.05):
    """
    Calculate confidence interval using ExactCIs.
    """
    # Parameters for CI calculation
    grid_size = 10  # Slightly larger grid for better accuracy
    refine = True   # Enable refinement for more precise results
    timeout = 30    # Longer timeout for accuracy
    
    # Calculate odds ratio manually
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    odds_ratio_str = format_float_or_inf(odds_ratio)
    
    # Calculate CI without Haldane correction
    start_time = time.time()
    try:
        lower, upper = exact_ci_unconditional(
            a, b, c, d, alpha=alpha,
            grid_size=grid_size, refine=refine,
            timeout=timeout, apply_haldane=False
        )
        elapsed = time.time() - start_time
        
        # Format the CI bounds
        lower_str = format_float_or_inf(lower)
        upper_str = format_float_or_inf(upper)
        ci_str = f"({lower_str}, {upper_str})"
    except Exception as e:
        elapsed = time.time() - start_time
        ci_str = f"Error: {str(e)}"
    
    result_str = f"""
ExactCIs Analysis:
---------------------
Contingency Table:
[{a}, {b}]
[{c}, {d}]

Odds Ratio: {odds_ratio_str}
95% CI: {ci_str}
Computation Time: {elapsed:.6f} seconds
"""
    print(result_str)
    return lower, upper, elapsed

def print_comparison_table(tables):
    """Print a side-by-side comparison of all results."""
    print("\n========== SIDE-BY-SIDE COMPARISON ==========\n")
    
    # Header
    print(f"{'Table':<15} {'Method':<15} {'Odds Ratio':<15} {'95% CI':<30} {'Time (s)':<10}")
    print("-" * 85)
    
    for i, (a, b, c, d) in enumerate(tables):
        table_name = f"Table {i+1}"
        
        # Run both analyses
        or_val = calculate_odds_ratio(a, b, c, d)
        
        # SciPy/Statsmodels results
        try:
            scipy_or, _, scipy_lower, scipy_upper, scipy_time = analyze_with_scipy_statsmodels(a, b, c, d)
            scipy_ci = f"({format_float_or_inf(scipy_lower)}, {format_float_or_inf(scipy_upper)})"
        except Exception as e:
            scipy_or = or_val
            scipy_ci = f"Error: {str(e)}"
            scipy_time = 0
        
        # ExactCIs results
        try:
            exactcis_lower, exactcis_upper, exactcis_time = analyze_with_exactcis(a, b, c, d)
            exactcis_ci = f"({format_float_or_inf(exactcis_lower)}, {format_float_or_inf(exactcis_upper)})"
        except Exception as e:
            exactcis_ci = f"Error: {str(e)}"
            exactcis_time = 0
        
        # Print the comparison row
        print(f"{table_name:<15} {'SciPy':<15} {format_float_or_inf(scipy_or):<15} {scipy_ci:<30} {scipy_time:.6f}")
        print(f"{'':<15} {'ExactCIs':<15} {format_float_or_inf(or_val):<15} {exactcis_ci:<30} {exactcis_time:.6f}")
        print("-" * 85)
    
    print("\n===============================================\n")


if __name__ == "__main__":
    print("\n===== COMPARISON OF SCIPY VS EXACTCIS =====\n")
    
    # Example tables with integer counts as requested by user
    tables = [
        (2, 1000, 10, 1000),   # Low event rates, OR = 0.2
        (20, 100, 10, 100),    # Higher event rates, OR = 2.0
        (50, 500, 10, 1000),   # Mixed rates, OR = 10.0
    ]
    
    # Print detailed comparison for each table
    for i, (a, b, c, d) in enumerate(tables):
        print(f"\n----- TABLE {i+1}: [{a}, {b}], [{c}, {d}] -----\n")
        print(f"Theoretical odds ratio: {format_float_or_inf(calculate_odds_ratio(a, b, c, d))}")
        
        # SciPy analysis
        print("\n" + "="*50 + "\n")
        analyze_with_scipy_statsmodels(a, b, c, d)
        
        # ExactCIs analysis
        print("\n" + "="*50 + "\n")
        analyze_with_exactcis(a, b, c, d)
        
        print("\n" + "="*50 + "\n")
    
    # Print a final comparison table
    print_comparison_table(tables)
    
    print("\n===== COMPARISON COMPLETE =====\n")
