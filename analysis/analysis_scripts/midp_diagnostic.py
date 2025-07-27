#!/usr/bin/env python
"""
Diagnostic script to debug the mid-p confidence interval implementation.
"""

import math
from typing import Tuple, List

def exact_ci_midp(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate confidence interval for odds ratio using mid-p adjustment.
    
    Parameters:
    -----------
    a, b, c, d : int
        Cell counts in a 2x2 contingency table
    alpha : float, default 0.05
        Significance level
        
    Returns:
    --------
    Tuple[float, float]
        Lower and upper bounds of the confidence interval
    """
    # Validate inputs
    if not all(isinstance(x, int) and x >= 0 for x in [a, b, c, d]):
        raise ValueError("All counts must be non-negative integers")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
    # Save original values for special case handling
    orig_a, orig_b, orig_c, orig_d = a, b, c, d
    
    n1, n2 = a + b, c + d
    m1 = a + c
    
    # Find the support
    supp = list(range(max(0, m1 - n2), min(m1, n1) + 1))
    
    # Make sure 'a' is in the support
    if a not in supp:
        raise ValueError(f"Observed value {a} not in support {supp}")
    
    idx = supp.index(a)
    
    # Define the mid-p function for confidence interval calculation
    def log_nchg_pmf(k, n1, n2, m1, theta):
        """Log of non-central hypergeometric PMF."""
        return (math.log(math.comb(n1, k)) + 
                math.log(math.comb(n2, m1 - k)) + 
                k * math.log(theta) - 
                math.log(sum(
                    math.comb(n1, j) * math.comb(n2, m1 - j) * theta**j 
                    for j in range(max(0, m1 - n2), min(m1, n1) + 1)
                )))
    
    def midp(theta: float) -> float:
        """
        Calculate mid-p value for given odds ratio theta.
        
        Returns two-sided p-value with mid-p adjustment.
        """
        log_probs = [log_nchg_pmf(k, n1, n2, m1, theta) for k in supp]
        probs = [math.exp(lp) for lp in log_probs]
        
        less = sum(p for i, p in enumerate(probs) if supp[i] < a)
        eq = probs[idx]
        more = sum(p for i, p in enumerate(probs) if supp[i] > a)
        
        # Mid-p adjustment: give half weight to observed value
        return 2 * min(less + 0.5*eq, more + 0.5*eq)
    
    # Binary search for confidence limits
    def find_root(f, lo, hi, target, tol=1e-8, max_iter=100):
        """Find root of f(x) = target using binary search."""
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if abs(f(mid) - target) < tol or hi - lo < tol:
                return mid
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    
    # Handle special cases for zero cells
    if orig_a * orig_d == 0:
        lower_bound = 0.0
    else:
        try:
            lower_bound = find_root(midp, 1e-8, 1.0, alpha)
        except Exception as e:
            print(f"Warning: Error finding lower bound: {e}")
            lower_bound = 0.0
            
    if orig_b * orig_c == 0:
        upper_bound = float('inf')
    else:
        try:
            upper_bound = find_root(midp, 1.0, 1000.0, alpha)
        except Exception as e:
            print(f"Warning: Error finding upper bound: {e}")
            upper_bound = float('inf')
    
    # Numerical issues sometimes cause values very close to zero or infinity
    lower_bound = max(0, lower_bound)
    upper_bound = min(1e6, upper_bound) if upper_bound != float('inf') else float('inf')
    
    return lower_bound, upper_bound


def run_diagnostic():
    """Run diagnostic tests on the implementation."""
    print("\n===== Running Mid-P Diagnostic Tests =====\n")
    
    # Test cases
    test_cases = [
        (12, 5, 8, 10, 0.05),  # README example
        (1, 1, 1, 1, 0.05),    # Small counts
        (0, 5, 8, 10, 0.05),   # Zero in one cell
        (50, 5, 2, 20, 0.05)   # Large imbalance
    ]
    
    # Run tests and print results
    for a, b, c, d, alpha in test_cases:
        print(f"Test case: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
        
        # Calculate odds ratio
        odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
        print(f"Odds ratio: {odds_ratio:.4f}" if odds_ratio != float('inf') else "Odds ratio: inf")
        
        # Calculate mid-p CI
        try:
            lower, upper = exact_ci_midp(a, b, c, d, alpha)
            if upper == float('inf'):
                print(f"Mid-P CI: ({lower:.4f}, inf)")
            else:
                print(f"Mid-P CI: ({lower:.4f}, {upper:.4f})")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    run_diagnostic()
