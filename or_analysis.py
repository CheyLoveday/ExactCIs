"""
Analyze odds ratios and confidence intervals for 2x2 tables with odds ratios > 1.
"""

import logging
import time
import numpy as np
from exactcis.methods import exact_ci_unconditional
from exactcis.core import apply_haldane_correction

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

def calculate_haldane_corrected_or(a, b, c, d):
    """Calculate the odds ratio with Haldane's correction."""
    a_h, b_h, c_h, d_h = apply_haldane_correction(a, b, c, d)
    return (a_h * d_h) / (b_h * c_h)

def format_ci(lower, upper):
    """Format a confidence interval nicely."""
    if np.isinf(upper):
        upper_str = "∞"
    else:
        upper_str = f"{upper:.4f}"
    return f"({lower:.4f}, {upper_str})"

def analyze_table(a, b, c, d, description):
    """Analyze a 2x2 table, calculating OR and CIs."""
    # Original OR calculation
    try:
        or_value = calculate_odds_ratio(a, b, c, d)
        or_str = f"{or_value:.4f}" if not np.isinf(or_value) else "∞"
    except:
        or_str = "Undefined"
    
    # Haldane-corrected OR
    or_haldane = calculate_haldane_corrected_or(a, b, c, d)
    or_haldane_str = f"{or_haldane:.4f}" if not np.isinf(or_haldane) else "∞"
    
    # Parameters for CI calculation
    alpha = 0.05
    grid_size = 10  # Slightly larger grid for better accuracy
    refine = True   # Enable refinement for more precise results
    timeout = 30    # Longer timeout for accuracy
    
    # Calculate CI without Haldane correction
    try:
        lower1, upper1 = exact_ci_unconditional(
            a, b, c, d, alpha=alpha,
            grid_size=grid_size, refine=refine,
            timeout=timeout, apply_haldane=False
        )
        ci_str = format_ci(lower1, upper1)
    except Exception as e:
        ci_str = f"Error: {str(e)}"
    
    # Calculate CI with Haldane correction
    try:
        lower2, upper2 = exact_ci_unconditional(
            a, b, c, d, alpha=alpha,
            grid_size=grid_size, refine=refine,
            timeout=timeout, apply_haldane=True
        )
        ci_haldane_str = format_ci(lower2, upper2)
    except Exception as e:
        ci_haldane_str = f"Error: {str(e)}"
    
    # Format and print results
    result = f"""
Table: {description}
Counts: [a={a}, b={b}], [c={c}, d={d}]
------------------------------------------------------
                     | Odds Ratio | 95% CI
------------------------------------------------------
Without Haldane      | {or_str.ljust(10)} | {ci_str}
With Haldane         | {or_haldane_str.ljust(10)} | {ci_haldane_str}
------------------------------------------------------
"""
    print(result)
    return result

if __name__ == "__main__":
    print("\n===== ODDS RATIO ANALYSIS WITH CONFIDENCE INTERVALS =====\n")
    
    # Table 1: Case with zero count in cell c, OR = ∞
    analyze_table(10, 100, 0, 100, "Zero count case (OR = ∞)")
    
    # Table 2: Integer counts with OR = 10
    analyze_table(10, 1, 10, 100, "Integer counts (OR = 10)")
    
    # Table 3: Decimal counts with OR = 10
    analyze_table(10.5, 1.05, 10.5, 100.5, "Decimal counts (OR = 10)")
    
    print("\n===== ANALYSIS COMPLETE =====\n")
