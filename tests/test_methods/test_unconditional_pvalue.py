import sys
import os
import time
import math
import numpy as np
from typing import Tuple, List, Dict, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from exactcis.methods.unconditional import _log_pvalue_barnard, calculate_odds_ratio

def test_pvalue_calculation():
    """
    Test the p-value calculation for different theta values, including the odds ratio.
    """
    # Define the 2x2 table
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate the odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    print(f"Odds ratio: {odds_ratio:.6f}")
    
    # Define a range of theta values to test
    theta_values = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        odds_ratio,  # Include the odds ratio
        2.5,
        3.0,
        4.0,
        5.0
    ]
    
    # Calculate p-values for each theta
    results = []
    for theta in theta_values:
        start_time = time.time()
        log_pval = _log_pvalue_barnard(a, c, a+b, c+d, theta, grid_size=50)
        elapsed_time = time.time() - start_time
        
        p_val = math.exp(log_pval) if log_pval > float('-inf') else 0.0
        results.append((theta, p_val, elapsed_time))
    
    # Print the results
    print("\nTheta\tP-value\tTime (s)")
    print("-----\t-------\t--------")
    for theta, p_val, elapsed_time in results:
        print(f"{theta:.6f}\t{p_val:.6f}\t{elapsed_time:.2f}")
    
    # Find the maximum p-value and corresponding theta
    max_p_val = max(results, key=lambda x: x[1])
    print(f"\nMaximum p-value: {max_p_val[1]:.6f} at theta={max_p_val[0]:.6f}")
    
    # Check if the p-value at the odds ratio is high
    odds_ratio_p_val = next(p_val for theta, p_val, _ in results if abs(theta - odds_ratio) < 1e-6)
    print(f"P-value at odds ratio ({odds_ratio:.6f}): {odds_ratio_p_val:.6f}")
    
    # Calculate confidence interval using alpha=0.05
    alpha = 0.05
    in_ci = [(theta, p_val) for theta, p_val, _ in results if p_val >= alpha]
    
    if in_ci:
        lower_bound = min(in_ci, key=lambda x: x[0])[0]
        upper_bound = max(in_ci, key=lambda x: x[0])[0]
        print(f"\nConfidence interval (alpha={alpha}): [{lower_bound:.6f}, {upper_bound:.6f}]")
        print(f"Odds ratio ({odds_ratio:.6f}) in CI: {lower_bound <= odds_ratio <= upper_bound}")
    else:
        print(f"\nNo theta values in confidence interval (alpha={alpha})")
    
    return results

if __name__ == "__main__":
    print("Testing p-value calculation for different theta values")
    print("====================================================")
    
    results = test_pvalue_calculation()