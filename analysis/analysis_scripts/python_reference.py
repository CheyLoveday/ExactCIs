#!/usr/bin/env python
"""
Python script to calculate reference confidence intervals for 2x2 tables
using statsmodels and scipy packages.
"""

import numpy as np
import math
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2

def print_separator():
    print("\n" + "-" * 60 + "\n")


def compute_reference_cis(a, b, c, d, alpha=0.05):
    """Compute reference confidence intervals using established Python packages."""
    print(f"Test case: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    
    # Create the 2x2 table
    table = np.array([[a, b], [c, d]])
    print("Table:")
    print(table)
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
    print(f"\nOdds ratio: {odds_ratio:.4f}")
    
    # Create a Table2x2 object
    table_obj = Table2x2(table)
    
    # Fisher's exact test (conditional)
    try:
        result = table_obj.oddsratio_confint(alpha=alpha, method='fisher')
        print(f"\nFisher's exact (conditional):")
        print(f"CI: ({result[0]:.4f}, {result[1]:.4f})")
    except Exception as e:
        print(f"\nFisher's exact (conditional): ERROR - {str(e)}")

    # Calculate confidence interval using midp method if available
    try:
        result = table_obj.oddsratio_confint(alpha=alpha, method='midp')
        print(f"\nMid-P adjusted:")
        print(f"CI: ({result[0]:.4f}, {result[1]:.4f})")
    except Exception as e:
        print(f"\nMid-P adjusted: ERROR - {str(e)}")
        
        # Manual Mid-P calculation as fallback
        try:
            print("\nManual Mid-P calculation (fallback):")
            # Code adapted from statsmodels but with mid-p adjustment
            def midp_score(theta):
                # Similar to our exact_ci_midp function
                n1, n2, m1 = a + b, c + d, a + c
                probs = []
                support = []
                
                # Calculate the support
                min_val = max(0, m1 - n2)
                max_val = min(m1, n1)
                
                for k in range(min_val, max_val + 1):
                    # Hypergeometric pmf calculations
                    log_p = (
                        math.log(math.comb(n1, k)) + 
                        math.log(math.comb(n2, m1 - k)) + 
                        k * math.log(theta) - 
                        math.log(sum(
                            math.comb(n1, j) * math.comb(n2, m1 - j) * theta**j 
                            for j in range(min_val, max_val + 1)
                        ))
                    )
                    probs.append(math.exp(log_p))
                    support.append(k)
                
                idx = support.index(a)
                less = sum(p for i, p in enumerate(probs) if support[i] < a)
                eq = probs[idx]
                more = sum(p for i, p in enumerate(probs) if support[i] > a)
                
                # Mid-P adjustment: give half weight to observed value
                return 2 * min(less + 0.5 * eq, more + 0.5 * eq)
            
            # Find the smallest theta such that midp_score(theta) <= alpha
            def find_root(f, lo, hi, target, tol=1e-8, max_iter=100):
                for _ in range(max_iter):
                    mid = (lo + hi) / 2
                    if abs(f(mid) - target) < tol or hi - lo < tol:
                        return mid
                    if f(mid) > target:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2
            
            # Find lower and upper confidence limits
            lo = 0.0 if a == 0 else find_root(midp_score, 1e-8, 1.0, alpha)
            hi = float('inf') if a == a + b else find_root(midp_score, 1.0, 1e4, alpha)
            
            print(f"CI: ({lo:.4f}, {hi:.4f if hi != float('inf') else 'inf'})")
        except Exception as e:
            print(f"Manual Mid-P calculation failed: {str(e)}")

    # Wald with Haldane-Anscombe correction
    try:
        # Add 0.5 to each cell
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        or_adj = (a_adj * d_adj) / (b_adj * c_adj)
        se = math.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
        z = stats.norm.ppf(1 - alpha/2)
        lo = math.exp(math.log(or_adj) - z*se)
        hi = math.exp(math.log(or_adj) + z*se)
        
        print(f"\nWald with Haldane-Anscombe correction:")
        print(f"CI: ({lo:.4f}, {hi:.4f})")
    except Exception as e:
        print(f"\nWald with Haldane-Anscombe correction: ERROR - {str(e)}")


# Test cases
print_separator()
print("Example from README:")
compute_reference_cis(12, 5, 8, 10)

print_separator()
print("Small counts:")
compute_reference_cis(1, 1, 1, 1)

print_separator()
print("Zero in one cell:")
compute_reference_cis(0, 5, 8, 10)

print_separator()
print("Large imbalance:")
compute_reference_cis(50, 5, 2, 20)

if __name__ == "__main__":
    print("\nDone.")
