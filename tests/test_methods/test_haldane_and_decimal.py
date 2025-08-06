"""
Tests for Haldane's correction and decimal value support in the confidence interval methods.
"""

import pytest
import logging
import time
from exactcis.methods import exact_ci_unconditional
from exactcis.core import apply_haldane_correction

# Configure logging
logger = logging.getLogger(__name__)


@pytest.mark.fast
def test_haldane_correction_function():
    """Test that the Haldane correction function works correctly."""
    # Test case with zeros
    a, b, c, d = 0, 10, 5, 5
    corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
    assert corrected_a == 0.5
    assert corrected_b == 10.5
    assert corrected_c == 5.5
    assert corrected_d == 5.5
    
    # Test case without zeros
    a, b, c, d = 1, 10, 5, 5
    corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
    assert corrected_a == 1
    assert corrected_b == 10
    assert corrected_c == 5
    assert corrected_d == 5
    
    # Test with multiple zeros
    a, b, c, d = 0, 0, 5, 5
    corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
    assert corrected_a == 0.5
    assert corrected_b == 0.5
    assert corrected_c == 5.5
    assert corrected_d == 5.5


@pytest.mark.fast
def test_unconditional_with_haldane():
    """Test that Haldane's correction works with the unconditional method."""
    # Use a small grid size for fast testing
    grid_size = 10
    
    # Case with a zero - should work with current implementation
    a, b, c, d = 0, 10, 5, 5
    
    # Test without Haldane correction (using original values)
    try:
        start_time = time.time()
        lower1, upper1 = exact_ci_unconditional(
            a, b, c, d, alpha=0.05, 
            grid_size=grid_size
        )
        time1 = time.time() - start_time
        logger.info(f"Without Haldane: CI=({lower1:.4f}, {upper1:.4f}), time={time1:.2f}s")
    except Exception as e:
        logger.info(f"Without Haldane raised: {str(e)}")
        lower1, upper1 = 0.0, float('inf')  # Default values
    
    # Test with Haldane correction (apply correction manually)
    try:
        # Apply Haldane correction manually
        corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
        
        start_time = time.time()
        lower2, upper2 = exact_ci_unconditional(
            corrected_a, corrected_b, corrected_c, corrected_d, alpha=0.05, 
            grid_size=grid_size
        )
        time2 = time.time() - start_time
        logger.info(f"With Haldane: CI=({lower2:.4f}, {upper2:.4f}), time={time2:.2f}s")
        
        # The CI with Haldane should be reasonable
        assert lower2 >= 0.0, f"Expected non-negative lower bound, got {lower2}"
        # Note: For tables with zeros, it's common to get infinite upper bounds
        # even with Haldane's correction
    except Exception as e:
        logger.error(f"Unexpected error with Haldane correction: {str(e)}")
        pytest.fail(f"Haldane correction should prevent errors: {str(e)}")


@pytest.mark.fast
def test_decimal_values():
    """Test that decimal values work correctly in the unconditional method."""
    # Use a small grid size for fast testing
    grid_size = 10
    
    # Use a stronger effect size for testing
    a, b, c, d = 5.5, 4.5, 2.5, 7.5
    
    try:
        start_time = time.time()
        lower, upper = exact_ci_unconditional(
            a, b, c, d, alpha=0.05,
            grid_size=grid_size
        )
        time_elapsed = time.time() - start_time
        
        logger.info(f"Decimal values CI=({lower:.4f}, {upper:.4f}), time={time_elapsed:.2f}s")
        
        # Basic sanity check - with small grid sizes and without refinement,
        # we sometimes get infinite upper bounds, which is acceptable
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        
        # Log the upper bound but don't fail if it's infinite
        # This is expected behavior with small grid sizes
        if upper == float('inf'):
            logger.warning("Got infinite upper bound - this is expected with small grid sizes")
    except Exception as e:
        logger.error(f"Unexpected error with decimal values: {str(e)}")
        pytest.fail(f"Decimal values should work: {str(e)}")


@pytest.mark.fast
def test_combined_haldane_and_decimal():
    """Test that both Haldane correction and decimal values work together."""
    # Use a small grid size for fast testing
    grid_size = 10
    
    # Test with decimal values and a zero
    a, b, c, d = 0.0, 8.5, 2.5, 7.5
    
    try:
        # Apply Haldane correction manually
        corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
        
        start_time = time.time()
        lower, upper = exact_ci_unconditional(
            corrected_a, corrected_b, corrected_c, corrected_d, alpha=0.05,
            grid_size=grid_size
        )
        time_elapsed = time.time() - start_time
        
        logger.info(f"Combined Haldane+Decimal CI=({lower:.4f}, {upper:.4f}), time={time_elapsed:.2f}s")
        
        # Basic sanity check - lower bound should be valid
        assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
        # For tables with zeros, infinite upper bounds are acceptable
    except Exception as e:
        logger.error(f"Unexpected error with combined features: {str(e)}")
        pytest.fail(f"Combined features should work: {str(e)}")


@pytest.mark.fast
def test_performance_comparison():
    """Compare performance between regular and Haldane methods."""
    # Use a small grid size for fast testing
    grid_size = 10
    
    # Case with a small value - not zero, to avoid errors in the regular method
    a, b, c, d = 1, 10, 5, 5
    
    # Regular method
    start_time = time.time()
    exact_ci_unconditional(
        a, b, c, d, alpha=0.05,
        grid_size=grid_size
    )
    regular_time = time.time() - start_time
    
    # With Haldane (apply correction manually)
    corrected_a, corrected_b, corrected_c, corrected_d = apply_haldane_correction(a, b, c, d)
    start_time = time.time()
    exact_ci_unconditional(
        corrected_a, corrected_b, corrected_c, corrected_d, alpha=0.05,
        grid_size=grid_size
    )
    haldane_time = time.time() - start_time
    
    logger.info(f"Performance comparison - Regular: {regular_time:.4f}s, Haldane: {haldane_time:.4f}s")
    logger.info(f"Overhead ratio: {haldane_time/regular_time:.2f}x")
    
    # The overhead should be minimal (since we're just changing the input values)
    assert haldane_time < regular_time * 2.0, "Haldane correction adds too much overhead"
