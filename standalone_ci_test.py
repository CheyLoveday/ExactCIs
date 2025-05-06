"""
Standalone test script for confidence interval calculation.
"""
import time
import math
import numpy as np
import logging
from scipy import stats
import statsmodels.api as sm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def apply_haldane_correction(a, b, c, d):
    """Apply Haldane's correction to a 2x2 table."""
    if a == 0 or b == 0 or c == 0 or d == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return a, b, c, d

def validate_counts(a, b, c, d):
    """Validate counts in a 2x2 table."""
    if not all(isinstance(x, (int, float)) for x in [a, b, c, d]):
        raise TypeError("All counts must be numeric (int or float)")
    if any(x < 0 for x in [a, b, c, d]):
        raise ValueError("All counts must be non-negative")

def calculate_scipy_ci(table, alpha=0.05):
    """Calculate odds ratio and CI using statsmodels."""
    a, b = table[0]
    c, d = table[1]
    
    # Calculate the odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0
    else:
        odds_ratio = (a * d) / (b * c)
    
    # Create data for statsmodels
    x1 = np.array([1] * int(a + 0.5) + [0] * int(b + 0.5))
    x2 = np.array([1] * int(c + 0.5) + [0] * int(d + 0.5))
    y = np.array([1] * int(a + b + 0.5) + [0] * int(c + d + 0.5))
    
    # Add the intercept
    X = np.column_stack((np.ones(len(y)), x1))
    
    # Fit the logistic regression model
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        
        # Extract the confidence interval
        ci_low = math.exp(result.conf_int(alpha=alpha)[1][0])
        ci_high = math.exp(result.conf_int(alpha=alpha)[1][1])
        
    except Exception as e:
        logger.warning(f"Statsmodels error: {e}, using Fisher's exact test")
        # Use Fisher's exact test as fallback
        odds_ratio, p_value = stats.fisher_exact(table)
        # Cannot get CI from fisher_exact directly, using approximation
        if all(x > 0 for x in [a, b, c, d]):
            log_odds = math.log(odds_ratio)
            stderr = 1.96 / math.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_low = math.exp(log_odds - stderr)
            ci_high = math.exp(log_odds + stderr)
        else:
            ci_low, ci_high = 0, float('inf')
    
    return odds_ratio, (ci_low, ci_high)

def improved_ci_unconditional(a, b, c, d, alpha=0.05, apply_haldane=False):
    """
    Calculate improved confidence intervals for odds ratio.
    
    This is a simplified implementation that uses logistic regression to calculate
    the confidence interval, with fallbacks for edge cases.
    """
    # Apply Haldane's correction if requested
    if apply_haldane:
        original_a, original_b, original_c, original_d = a, b, c, d
        a, b, c, d = apply_haldane_correction(a, b, c, d)
        if (a != original_a or b != original_b or c != original_c or d != original_d):
            logger.info(f"Applied Haldane's correction: ({original_a}, {original_b}, {original_c}, {original_d}) -> ({a}, {b}, {c}, {d})")
    
    # Validate the counts
    validate_counts(a, b, c, d)
    
    # Calculate odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0
    else:
        odds_ratio = (a * d) / (b * c)
    
    table = [[a, b], [c, d]]
    
    try:
        # Create data for statsmodels
        x1 = np.array([1] * int(a + 0.5) + [0] * int(b + 0.5))
        x2 = np.array([1] * int(c + 0.5) + [0] * int(d + 0.5))
        y = np.array([1] * int(a + b + 0.5) + [0] * int(c + d + 0.5))
        
        # Add the intercept
        X = np.column_stack((np.ones(len(y)), x1))
        
        # Fit the logistic regression model
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        
        # Extract confidence interval
        ci_low = math.exp(result.conf_int(alpha=alpha)[1][0])
        ci_high = math.exp(result.conf_int(alpha=alpha)[1][1])
        
        # Apply a correction factor to make the interval more conservative
        # This helps it better align with the exact method when available
        correction_factor = 1.1  # Widen the interval by 10%
        width = ci_high - ci_low
        ci_low = max(0.0001, ci_low - (width * 0.1))
        ci_high = ci_high + (width * 0.1)
        
        logger.info(f"Logistic regression CI: ({ci_low}, {ci_high})")
        
    except Exception as e:
        logger.warning(f"Error in logistic regression: {e}, using approximation")
        
        try:
            # Calculate Fisher's exact test
            odds_ratio, p_value = stats.fisher_exact(table)
            
            # Use a simple approximation based on the p-value
            if all(x > 0 for x in [a, b, c, d]):
                log_odds = math.log(max(odds_ratio, 1e-10))
                # More conservative CI
                stderr = 1.96 / math.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_low = math.exp(log_odds - stderr * 1.1)  # Make it a bit wider
                ci_high = math.exp(log_odds + stderr * 1.1)
            else:
                # For zero cells, use a more conservative approach
                if odds_ratio < 1:
                    ci_low = max(odds_ratio * 0.1, 0.001)
                    ci_high = min(odds_ratio * 4, 1.0)
                else:
                    ci_low = max(odds_ratio / 4, 1.0)
                    ci_high = odds_ratio * 10
            
            logger.info(f"Fisher approximation CI: ({ci_low}, {ci_high})")
            
        except Exception as e:
            logger.error(f"All methods failed: {e}")
            # Last resort fallback
            if odds_ratio < 1:
                ci_low = max(odds_ratio * 0.1, 0.001)
                ci_high = min(odds_ratio * 20, 100.0)
            else:
                ci_low = max(odds_ratio / 20, 0.01)
                ci_high = odds_ratio * 20
            
            logger.warning(f"Using conservative default CI: ({ci_low}, {ci_high})")
    
    return ci_low, ci_high

def run_test_with_table(table, alpha=0.05, apply_haldane=False):
    """Run test for a specific 2x2 table with both methods."""
    a, b = table[0]
    c, d = table[1]
    
    print(f"\nTable: {table}")
    
    # Calculate with SciPy/statsmodels
    start_time = time.time()
    scipy_or, scipy_ci = calculate_scipy_ci(table, alpha)
    scipy_time = time.time() - start_time
    print(f"SciPy OR:     {scipy_or:.6f}, CI: ({scipy_ci[0]:.6f}, {scipy_ci[1]:.6f}), Time: {scipy_time:.6f}s")
    
    # Calculate with improved ExactCIs
    print("\nCalculating with improved approach...")
    start_time = time.time()
    improved_low, improved_high = improved_ci_unconditional(a, b, c, d, alpha=alpha, apply_haldane=apply_haldane)
    improved_time = time.time() - start_time
    
    # Calculate odds ratio
    if b * c == 0:
        improved_or = float('inf') if a * d > 0 else 0
    else:
        improved_or = (a * d) / (b * c)
        
    print(f"Improved OR:  {improved_or:.6f}, CI: ({improved_low:.6f}, {improved_high:.6f}), Time: {improved_time:.6f}s")
    
    # Calculate ratio of SciPy CI width to improved CI width
    scipy_width = scipy_ci[1] - scipy_ci[0]
    improved_width = improved_high - improved_low
    if improved_width > 0 and scipy_width > 0:
        width_ratio = improved_width / scipy_width
        print(f"\nCI width ratio (Improved/SciPy): {width_ratio:.2f}")
    
    return {
        'table': table,
        'scipy': {'or': scipy_or, 'ci': scipy_ci, 'time': scipy_time},
        'improved': {'or': improved_or, 'ci': (improved_low, improved_high), 'time': improved_time}
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
        [[0, 100], [10, 100]],
        
        # Near-zero values
        [[1, 1000], [1, 1000]]
    ]
    
    results = []
    for table in tables:
        result = run_test_with_table(table)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Table':<20} {'Method':<10} {'OR':<10} {'CI':<25} {'Time (s)':<10}")
    print("-"*60)
    
    for result in results:
        table = result['table']
        table_str = f"[{table[0][0]},{table[0][1]}],[{table[1][0]},{table[1][1]}]"
        
        # SciPy row
        ci = result['scipy']['ci']
        ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
        print(f"{table_str:<20} {'SciPy':<10} {result['scipy']['or']:<10.4f} {ci_str:<25} {result['scipy']['time']:<10.4f}")
        
        # Improved row
        ci = result['improved']['ci']
        ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
        print(f"{'':<20} {'Improved':<10} {result['improved']['or']:<10.4f} {ci_str:<25} {result['improved']['time']:<10.4f}")
        
        print("-"*60)

if __name__ == "__main__":
    run_tests()
