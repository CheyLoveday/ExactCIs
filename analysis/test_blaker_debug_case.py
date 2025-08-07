"""
Test script to investigate potential issues with the Blaker method
for the specific case highlighted in debug logging.
"""

import logging
import numpy as np
from exactcis.methods import exact_ci_blaker
from exactcis.core import support, SupportData

# Set up logging to see debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exactcis")
logger.setLevel(logging.INFO)

def test_blaker_debug_case():
    """
    Test the specific case highlighted in debug logging:
    n1=5, n2=10, m1=7, theta>1e6
    
    This corresponds to a 2x2 table with:
    a=5, b=0, c=2, d=8
    """
    print("\nTesting Blaker method with debug case: a=5, b=0, c=2, d=8")
    print("This gives: n1=5, n2=10, m1=7")
    
    # Run the Blaker method
    try:
        lower, upper = exact_ci_blaker(5, 0, 2, 8, alpha=0.05)
        print(f"Confidence interval: [{lower:.6f}, {upper}]")
        
        # Check if the interval is valid
        if lower > 0 and upper > lower:
            print("✓ Valid interval: lower > 0 and upper > lower")
        else:
            print("✗ Invalid interval")
            
        if np.isfinite(upper):
            print(f"✓ Upper bound is finite: {upper:.6f}")
        else:
            print("✗ Upper bound is infinite")
            
        # Calculate point estimate
        from exactcis.core import calculate_odds_ratio
        point_estimate = calculate_odds_ratio(5, 0, 2, 8)
        print(f"Point estimate: {point_estimate}")
        
        if lower <= point_estimate <= upper:
            print("✓ Interval contains point estimate")
        else:
            print(f"✗ Interval [{lower:.6f}, {upper}] does not contain point estimate {point_estimate}")
            
    except Exception as e:
        print(f"Error: {e}")

def test_blaker_with_different_alpha():
    """Test the debug case with different alpha values to see if it affects the results."""
    print("\nTesting Blaker method with different alpha values:")
    
    for alpha in [0.01, 0.025, 0.05, 0.1]:
        try:
            lower, upper = exact_ci_blaker(5, 0, 2, 8, alpha=alpha)
            print(f"Alpha={alpha}: CI=[{lower:.6f}, {upper}]")
        except Exception as e:
            print(f"Alpha={alpha}: Error: {e}")

def test_similar_tables():
    """Test similar tables to see if they exhibit the same behavior."""
    print("\nTesting similar tables:")
    
    tables = [
        (5, 0, 2, 8),  # Original debug case
        (4, 1, 3, 7),  # Another table with n1=5, n2=10, m1=7
        (3, 2, 4, 6),  # Another table with n1=5, n2=10, m1=7
        (2, 3, 5, 5),  # Another table with n1=5, n2=10, m1=7
    ]
    
    for a, b, c, d in tables:
        try:
            lower, upper = exact_ci_blaker(a, b, c, d, alpha=0.05)
            print(f"Table ({a},{b},{c},{d}): CI=[{lower:.6f}, {upper}]")
        except Exception as e:
            print(f"Table ({a},{b},{c},{d}): Error: {e}")

def test_blaker_p_value_directly():
    """Test the blaker_p_value function directly for the debug case."""
    print("\nTesting blaker_p_value function directly:")
    
    from exactcis.methods.blaker import blaker_p_value
    
    a, b, c, d = 5, 0, 2, 8
    n1, n2 = a + b, c + d
    m1 = a + c
    
    s = support(n1, n2, m1)
    
    # Test with different theta values
    for theta in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
        try:
            p_val = blaker_p_value(a, n1, n2, m1, theta, s)
            print(f"Theta={theta:.1e}: p-value={p_val:.6f}")
        except Exception as e:
            print(f"Theta={theta:.1e}: Error: {e}")

if __name__ == "__main__":
    test_blaker_debug_case()
    test_blaker_with_different_alpha()
    test_similar_tables()
    test_blaker_p_value_directly()