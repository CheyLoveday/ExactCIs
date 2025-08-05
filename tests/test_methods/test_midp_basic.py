"""
Test script for the new Mid-P implementation.

This script tests the new grid search implementation of the Mid-P method
with the example case (50/1000 vs 25/1000) that previously failed.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the Mid-P method
from exactcis.methods.midp import exact_ci_midp

def test_midp_large_sample():
    """Test the Mid-P method with a large sample size."""
    # Example case: 50/1000 vs 25/1000
    a = 50
    b = 950
    c = 25
    d = 975
    
    # Calculate the confidence interval
    logger.info(f"Testing Mid-P method with a={a}, b={b}, c={c}, d={d}")
    
    # Calculate the odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate the confidence interval with default parameters
    ci_default = exact_ci_midp(a, b, c, d)
    logger.info(f"Mid-P CI (default parameters): {ci_default}")
    
    # Calculate the confidence interval with increased grid size
    ci_fine = exact_ci_midp(a, b, c, d, grid_size=500)
    logger.info(f"Mid-P CI (fine grid): {ci_fine}")
    
    # Calculate the confidence interval with extended range
    ci_extended = exact_ci_midp(a, b, c, d, theta_min=0.0001, theta_max=10000)
    logger.info(f"Mid-P CI (extended range): {ci_extended}")
    
    # Check if the confidence interval includes the odds ratio
    for ci, name in [(ci_default, "default"), (ci_fine, "fine grid"), (ci_extended, "extended range")]:
        lower, upper = ci
        if lower <= odds_ratio <= upper:
            logger.info(f"✅ {name.capitalize()} CI includes the odds ratio")
        else:
            logger.error(f"❌ {name.capitalize()} CI does not include the odds ratio")
        
        # Check if the confidence interval is finite
        if upper < float('inf'):
            logger.info(f"✅ {name.capitalize()} CI has a finite upper bound")
        else:
            logger.error(f"❌ {name.capitalize()} CI has an infinite upper bound")
        
        # Calculate the width of the confidence interval
        width = upper - lower if upper < float('inf') else float('inf')
        logger.info(f"{name.capitalize()} CI width: {width:.4f}")

def test_midp_batch():
    """Test the Mid-P batch processing function."""
    # Create a list of tables
    tables = [
        (50, 950, 25, 975),  # Large sample
        (10, 90, 5, 95),     # Medium sample
        (2, 8, 1, 9)         # Small sample
    ]
    
    # Import the batch function
    from exactcis.methods.midp import exact_ci_midp_batch
    
    # Calculate confidence intervals for all tables
    logger.info("Testing Mid-P batch processing")
    results = exact_ci_midp_batch(tables)
    
    # Print the results
    for i, ((a, b, c, d), (lower, upper)) in enumerate(zip(tables, results)):
        odds_ratio = (a * d) / (b * c)
        logger.info(f"Table {i+1}: a={a}, b={b}, c={c}, d={d}")
        logger.info(f"Odds ratio: {odds_ratio:.4f}")
        logger.info(f"Mid-P CI: ({lower:.4f}, {upper:.4f})")
        
        # Check if the confidence interval includes the odds ratio
        if lower <= odds_ratio <= upper:
            logger.info(f"✅ CI includes the odds ratio")
        else:
            logger.error(f"❌ CI does not include the odds ratio")
        
        # Check if the confidence interval is finite
        if upper < float('inf'):
            logger.info(f"✅ CI has a finite upper bound")
        else:
            logger.error(f"❌ CI has an infinite upper bound")
        
        # Calculate the width of the confidence interval
        width = upper - lower if upper < float('inf') else float('inf')
        logger.info(f"CI width: {width:.4f}")
        logger.info("---")

if __name__ == "__main__":
    logger.info("Testing the new Mid-P implementation")
    
    # Test the Mid-P method with a large sample size
    test_midp_large_sample()
    
    # Test the Mid-P batch processing function
    test_midp_batch()
    
    logger.info("Testing completed")