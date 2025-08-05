import sys
import os
import time
import numpy as np
from typing import Tuple, List, Dict, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from exactcis.methods.unconditional import exact_ci_unconditional

def test_large_sample_case():
    """
    Test the unconditional method with a large sample case (50/1000 vs 25/1000).
    
    This case previously returned [0.767, 10.263] with the root-finding approach,
    but should return a more reasonable interval with the grid search approach.
    """
    # Define the 2x2 table
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate the confidence interval
    start_time = time.time()
    lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
    elapsed_time = time.time() - start_time
    
    # Print the results
    print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
    print(f"Confidence interval: [{lower:.3f}, {upper:.3f}] (width: {upper-lower:.3f})")
    print(f"Computation time: {elapsed_time:.2f} seconds")
    
    # Check if the result is reasonable
    # The interval should be finite and not too wide
    assert lower > 0, "Lower bound should be positive"
    assert upper < float('inf'), "Upper bound should be finite"
    assert upper - lower < 5, "Interval width should be reasonable (< 5)"
    
    # The interval should include the odds ratio
    odds_ratio = (a / b) / (c / d)
    print(f"Odds ratio: {odds_ratio:.3f}")
    assert lower <= odds_ratio <= upper, "Interval should include the odds ratio"
    
    return lower, upper, elapsed_time

def test_multiple_large_sample_cases():
    """
    Test the unconditional method with multiple large sample cases.
    """
    # Define several 2x2 tables with large sample sizes
    tables = [
        (50, 950, 25, 975),    # 50/1000 vs 25/1000
        (100, 900, 50, 950),   # 100/1000 vs 50/1000
        (200, 800, 100, 900),  # 200/1000 vs 100/1000
        (300, 700, 150, 850),  # 300/1000 vs 150/1000
        (400, 600, 200, 800)   # 400/1000 vs 200/1000
    ]
    
    results = []
    
    for a, b, c, d in tables:
        # Calculate the confidence interval
        start_time = time.time()
        lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
        elapsed_time = time.time() - start_time
        
        # Calculate the odds ratio
        odds_ratio = (a / b) / (c / d)
        
        # Print the results
        print(f"\nTest case: {a}/{a+b} vs {c}/{c+d}")
        print(f"Odds ratio: {odds_ratio:.3f}")
        print(f"Confidence interval: [{lower:.3f}, {upper:.3f}] (width: {upper-lower:.3f})")
        print(f"Computation time: {elapsed_time:.2f} seconds")
        
        # Check if the result is reasonable
        assert lower > 0, "Lower bound should be positive"
        assert upper < float('inf'), "Upper bound should be finite"
        assert lower <= odds_ratio <= upper, "Interval should include the odds ratio"
        
        results.append((a, b, c, d, lower, upper, elapsed_time))
    
    return results

if __name__ == "__main__":
    print("Testing unconditional method with grid search approach")
    print("======================================================")
    
    # Test the large sample case
    lower, upper, elapsed_time = test_large_sample_case()
    
    # Test multiple large sample cases
    results = test_multiple_large_sample_cases()
    
    print("\nSummary of results:")
    print("------------------")
    for a, b, c, d, lower, upper, elapsed_time in results:
        odds_ratio = (a / b) / (c / d)
        print(f"{a}/{a+b} vs {c}/{c+d}: OR={odds_ratio:.3f}, CI=[{lower:.3f}, {upper:.3f}], width={upper-lower:.3f}, time={elapsed_time:.2f}s")