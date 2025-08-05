"""
Implementation of the Clopper-Pearson exact confidence interval method.

The Clopper-Pearson method calculates exact confidence intervals for binomial proportions
using the binomial cumulative distribution function. It is sometimes called the "exact" method
because it uses the binomial distribution directly rather than approximations.

For a binomial proportion p with x successes in n trials, the Clopper-Pearson interval
is calculated as follows:

1. Lower bound: Find p_L such that P(X ≥ x | n, p_L) = α/2
   - This is equivalent to finding p_L such that the cumulative distribution function F(x-1; n, p_L) = 1 - α/2
   - If x = 0, the lower bound is 0

2. Upper bound: Find p_U such that P(X ≤ x | n, p_U) = α/2
   - This is equivalent to finding p_U such that the cumulative distribution function F(x; n, p_U) = α/2
   - If x = n, the upper bound is 1

Implementation Note:
This implementation uses the relationship between the binomial and beta distributions
to calculate the interval more efficiently. The lower bound is the α/2 quantile of the
beta(x, n-x+1) distribution, and the upper bound is the 1-α/2 quantile of the beta(x+1, n-x)
distribution. This is mathematically equivalent to the original definition but more
computationally efficient and numerically stable.

In the context of a 2x2 contingency table, we can calculate the confidence interval for
either the proportion in group 1 (p1 = a/(a+b)) or the proportion in group 2 (p2 = c/(c+d)).
"""

import logging
from typing import Tuple, List, Optional, Callable
import numpy as np
from scipy import stats

from exactcis.core import validate_counts

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import parallel utilities
try:
    from ..utils.parallel import parallel_compute_ci, get_optimal_workers
    has_parallel = True
    logger.info("Using parallel processing for Clopper-Pearson method")
except ImportError:
    has_parallel = False
    logger.info("Parallel processing not available")
    
    # Fallback function if parallel utilities are not available
    def get_optimal_workers():
        return 1


def _clopper_pearson_interval(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate the Clopper-Pearson confidence interval for a binomial proportion.
    
    This implementation uses the relationship between the Clopper-Pearson interval
    and the beta distribution:
    - Lower bound is the alpha/2 quantile of the beta(x, n-x+1) distribution
    - Upper bound is the 1-alpha/2 quantile of the beta(x+1, n-x) distribution
    
    This is mathematically equivalent to the original definition using the binomial
    distribution but is more computationally efficient and numerically stable.
    
    Args:
        x: Number of successes
        n: Number of trials
        alpha: Significance level (default: 0.05)
        
    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
    """
    # For Clopper-Pearson, we use the relationship with the beta distribution
    # Lower bound is the alpha/2 quantile of the beta(x, n-x+1) distribution
    # Upper bound is the 1-alpha/2 quantile of the beta(x+1, n-x) distribution
    
    # Handle edge cases
    if x == 0:
        lower = 0.0
    else:
        # Lower bound is the alpha/2 quantile of the beta(x, n-x+1) distribution
        lower = stats.beta.ppf(alpha/2, x, n-x+1)
    
    if x == n:
        upper = 1.0
    else:
        # Upper bound is the 1-alpha/2 quantile of the beta(x+1, n-x) distribution
        upper = stats.beta.ppf(1-alpha/2, x+1, n-x)
    
    return lower, upper


def exact_ci_clopper_pearson(a: int, b: int, c: int, d: int, alpha: float = 0.05, group: int = 1) -> Tuple[float, float]:
    """
    Calculate the Clopper-Pearson exact confidence interval for a binomial proportion.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        group: Which group to calculate the confidence interval for (1 or 2, default: 1)
            - group=1: Calculate CI for p1 = a/(a+b)
            - group=2: Calculate CI for p2 = c/(c+d)
            
    Returns:
        Tuple containing (lower_bound, upper_bound) of the confidence interval
        
    Raises:
        ValueError: If inputs are invalid (negative counts, empty margins, invalid alpha, or invalid group)
    """
    # Validate inputs
    validate_counts(a, b, c, d)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if group not in [1, 2]:
        raise ValueError("group must be 1 or 2")
    
    # Calculate row totals
    n1 = a + b
    n2 = c + d
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Clopper-Pearson exact_ci_clopper_pearson: Input ({a},{b},{c},{d}), alpha={alpha}, group={group}")
    
    # Calculate confidence interval based on the specified group
    if group == 1:
        if n1 == 0:
            raise ValueError("Cannot calculate confidence interval for group 1: n1 = 0")
        x = a  # Number of successes in group 1
        n = n1  # Number of trials in group 1
        p_hat = a / n1 if n1 > 0 else 0  # Point estimate for group 1
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Calculating CI for group 1: p1_hat = {p_hat:.4f}")
    else:  # group == 2
        if n2 == 0:
            raise ValueError("Cannot calculate confidence interval for group 2: n2 = 0")
        x = c  # Number of successes in group 2
        n = n2  # Number of trials in group 2
        p_hat = c / n2 if n2 > 0 else 0  # Point estimate for group 2
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Calculating CI for group 2: p2_hat = {p_hat:.4f}")
    
    # Calculate the Clopper-Pearson interval
    lower, upper = _clopper_pearson_interval(x, n, alpha)
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Clopper-Pearson CI: [{lower:.4f}, {upper:.4f}]")
    
    return lower, upper


def exact_ci_clopper_pearson_batch(tables: List[Tuple[int, int, int, int]], 
                                  alpha: float = 0.05,
                                  group: int = 1,
                                  max_workers: Optional[int] = None,
                                  backend: Optional[str] = None,
                                  progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """
    Calculate Clopper-Pearson exact confidence intervals for multiple 2x2 tables in parallel.
    
    This function leverages parallel processing to compute confidence intervals for
    multiple tables simultaneously, providing significant speedup for large datasets.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        alpha: Significance level (default: 0.05)
        group: Which group to calculate the confidence interval for (1 or 2, default: 1)
        max_workers: Maximum number of parallel workers (default: auto-detected)
        backend: Backend to use ('thread', 'process', or None for auto-detection)
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        List of (lower_bound, upper_bound) tuples, one for each input table
        
    Note:
        Error Handling: If computation fails for any individual table (due to
        numerical issues, invalid data, etc.), a conservative interval (0.0, 1.0)
        is returned for that table, allowing the batch processing to complete
        successfully.
    """
    if not tables:
        return []
    
    if not has_parallel:
        # Fall back to sequential processing
        if logger.isEnabledFor(logging.INFO):
            logger.info("Parallel support not available, using sequential processing")
        results = []
        for i, (a, b, c, d) in enumerate(tables):
            try:
                result = exact_ci_clopper_pearson(a, b, c, d, alpha, group)
                results.append(result)
            except Exception as e:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Error processing table {i+1} ({a},{b},{c},{d}): {e}")
                results.append((0.0, 1.0))  # Conservative fallback for proportions
            
            if progress_callback:
                progress_callback(min(100, int(100 * (i+1) / len(tables))))
        
        return results
    
    # Determine number of workers
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    max_workers = min(max_workers, len(tables))  # Don't use more workers than tables
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Processing {len(tables)} tables with Clopper-Pearson method using {max_workers} workers")
    
    # Use parallel processing with the specified backend
    results = parallel_compute_ci(
        lambda a, b, c, d, alpha=alpha: exact_ci_clopper_pearson(a, b, c, d, alpha, group),
        tables,
        alpha=alpha,
        timeout=None,  # No timeout for batch processing
        backend=backend,
        max_workers=max_workers,
        progress_callback=progress_callback
    )
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Completed batch processing of {len(tables)} tables with Clopper-Pearson method")
    
    return results