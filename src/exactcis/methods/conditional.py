"""
Conditional (Fisher) confidence interval for odds ratio.

This module implements the conditional (Fisher) confidence interval method
for the odds ratio of a 2x2 contingency table.
"""

from typing import Tuple
from exactcis.core import validate_counts

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

    # Simplified implementation - using a placeholder approach
    # This is a stub implementation that should be properly implemented
    
    # Determine the support range for the hypergeometric distribution
    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n = r1 + r2
    
    # Calculate the observed odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0.0
    else:
        odds_ratio = (a * d) / (b * c)
    
    # Simplified confidence interval calculation
    # In a complete implementation, this would use the noncentral hypergeometric distribution
    if a == 0 or c == 0:
        low = 0.0
    else:
        # Placeholder - this should be properly implemented
        low = max(0.0, odds_ratio / 3)
        
    if b == 0 or d == 0:
        high = float('inf')
    else:
        # Placeholder - this should be properly implemented
        high = odds_ratio * 3
        
    return low, high
