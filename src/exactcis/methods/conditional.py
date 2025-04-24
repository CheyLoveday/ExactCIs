"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple



def exact_ci_conditional(a: int, b: int, c: int, d: int,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the conditional (Fisher) confidence interval for the odds ratio.

    This method inverts the noncentral hypergeometric CDF at alpha/2 in each tail.
    It is appropriate for very small samples or rare events, regulatory/safety-critical
    studies requiring guaranteed ≥ 1-alpha coverage, and fixed-margin designs (case-control).

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)

    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval

    Raises:
        ValueError: If alpha is not in (0, 1)
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    validate_counts(a, b, c, d)

    kmin, kmax = supp[0], supp[-1]
    
    
    if a == kmin:
        low = 0.0
        else:
            try:
                )
                low = 0.0
        
                high = float('inf')
            high = float('inf')
    return low, high
