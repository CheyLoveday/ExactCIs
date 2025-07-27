"""
Run sample 2x2 tables with the exact_ci_unconditional method
to demonstrate the functionality of Haldane correction and decimal support.
"""

import logging
import time
from exactcis.methods import exact_ci_unconditional
from exactcis.core import apply_haldane_correction

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_sample_table(a, b, c, d, description):
    """Run a sample table with and without Haldane correction."""
    logger.info(f"Running sample table: {description}")
    logger.info(f"Counts: a={a}, b={b}, c={c}, d={d}")
    
    # Parameters for the CI calculation
    alpha = 0.05
    grid_size = 10  # Using a slightly larger grid for better accuracy
    refine = False  # Disable refinement for faster results
    timeout = 30  # Longer timeout for accuracy
    
    # Without Haldane correction
    try:
        start_time = time.time()
        lower1, upper1 = exact_ci_unconditional(
            a, b, c, d, alpha=alpha,
            grid_size=grid_size, refine=refine,
            timeout=timeout, apply_haldane=False
        )
        time1 = time.time() - start_time
        logger.info(f"Without Haldane: CI=({lower1:.6f}, {upper1:.6f}), time={time1:.2f}s")
    except Exception as e:
        logger.error(f"Error without Haldane: {str(e)}")
        lower1, upper1 = None, None
    
    # With Haldane correction
    try:
        start_time = time.time()
        lower2, upper2 = exact_ci_unconditional(
            a, b, c, d, alpha=alpha,
            grid_size=grid_size, refine=refine,
            timeout=timeout, apply_haldane=True
        )
        time2 = time.time() - start_time
        logger.info(f"With Haldane: CI=({lower2:.6f}, {upper2:.6f}), time={time2:.2f}s")
    except Exception as e:
        logger.error(f"Error with Haldane: {str(e)}")
        lower2, upper2 = None, None
    
    # Print a comparison if both methods worked
    if lower1 is not None and lower2 is not None:
        diff_lower = abs(lower2 - lower1)
        diff_upper = abs(upper2 - upper1) if upper1 != float('inf') and upper2 != float('inf') else "N/A"
        logger.info(f"Differences: Lower={diff_lower:.6f}, Upper={diff_upper}")
    
    logger.info("=" * 50)


# Run the sample tables
if __name__ == "__main__":
    # Sample 1: Regular table with decent counts
    run_sample_table(1, 100, 10, 100, "Regular table with decent counts")
    
    # Sample 2: Table with a zero - requires Haldane correction
    run_sample_table(0, 100, 1, 100, "Table with a zero count")
    
    # Sample 3: Table with decimal values
    run_sample_table(0.5, 100.5, 10.5, 100.5, "Table with decimal values")
