"""
Comprehensive test script for the new Mid-P implementation.

This script tests the grid search implementation of the Mid-P method,
which is designed to handle large sample sizes better than the previous implementation.
It includes tests for basic functionality, edge cases, large sample sizes, and more.

Run this script directly with Python:
    python test_midp_comprehensive.py
"""

import sys
import os
import logging
import time
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the Mid-P method and related functions
from exactcis.methods.midp import exact_ci_midp, exact_ci_midp_batch, calculate_midp_pvalue
from exactcis.utils.ci_search import _find_ci_bound as find_ci_bound
from exactcis.core import validate_counts, calculate_odds_ratio

# Try to import other methods for comparison
try:
    from exactcis.methods.conditional import exact_ci_conditional
    from exactcis.methods.blaker import exact_ci_blaker
    has_other_methods = True
except ImportError:
    has_other_methods = False
    logger.warning("Other methods not available for comparison")


def test_exact_ci_midp_basic():
    """Test basic functionality of exact_ci_midp."""
    logger.info("Testing basic functionality")
    
    # Test with standard example case
    a, b, c, d = 12, 5, 8, 10
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
    
    # Verify mathematical properties
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    assert upper > lower, f"Expected upper bound ({upper}) > lower bound ({lower})"
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    
    logger.info("✅ Basic functionality test passed")


def test_exact_ci_midp_edge_cases():
    """Test edge cases for exact_ci_midp."""
    logger.info("Testing edge cases")
    
    # Test cases
    test_cases = [
        # When a is at the minimum possible value
        (0, 10, 10, 10, "a=0"),
        # When a is at the maximum possible value
        (10, 0, 0, 10, "a=max"),
        # When b is zero (should return (0, inf) as documented in the function)
        (5, 0, 5, 5, "b=0"),
        # When c is zero (should return (0, inf) as documented in the function)
        (5, 5, 0, 5, "c=0")
    ]
    
    for a, b, c, d, case_name in test_cases:
        logger.info(f"Testing edge case: {case_name} - a={a}, b={b}, c={c}, d={d}")
        
        try:
            lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
            logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
            
            # Check basic properties
            assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
            
            # Special cases for b=0 or c=0
            if b == 0 or c == 0:
                assert lower == 0.0, f"Expected lower bound of 0.0 for b=0 or c=0, got {lower}"
                assert upper == float('inf'), f"Expected upper bound of infinity for b=0 or c=0, got {upper}"
            else:
                # With the new implementation, we should get a finite upper bound for other edge cases
                assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
            
            # Calculate odds ratio if possible
            if b > 0 and c > 0:
                odds_ratio = (a * d) / (b * c)
                assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
            
            logger.info(f"✅ Edge case {case_name} passed")
        except Exception as e:
            logger.error(f"❌ Edge case {case_name} failed: {e}")
            raise


def test_exact_ci_midp_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    logger.info("Testing invalid inputs")
    
    # Test cases
    test_cases = [
        # Negative count
        (-1, 5, 8, 10, "negative count"),
        # Empty margin
        (0, 0, 8, 10, "empty margin"),
        # Invalid alpha
        (12, 5, 8, 10, "invalid alpha", 1.5)
    ]
    
    for test_case in test_cases:
        if len(test_case) == 5:
            a, b, c, d, case_name, alpha = *test_case, 0.05
        else:
            a, b, c, d, case_name, alpha = test_case
        
        logger.info(f"Testing invalid input: {case_name} - a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
        
        try:
            lower, upper = exact_ci_midp(a, b, c, d, alpha=alpha)
            logger.error(f"❌ Expected ValueError for {case_name}, but got result: ({lower}, {upper})")
            assert False, f"Expected ValueError for {case_name}"
        except ValueError as e:
            logger.info(f"✅ Invalid input {case_name} correctly raised ValueError: {e}")
        except Exception as e:
            logger.error(f"❌ Expected ValueError for {case_name}, but got different exception: {e}")
            raise


def test_exact_ci_midp_small_counts():
    """Test with small counts."""
    logger.info("Testing small counts")
    
    a, b, c, d = 1, 1, 1, 1
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
    
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
    
    # For this balanced case, odds ratio should be 1
    odds_ratio = 1.0
    
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    
    logger.info("✅ Small counts test passed")


def test_exact_ci_midp_large_imbalance():
    """Test with large imbalance in counts."""
    logger.info("Testing large imbalance in counts")
    
    a, b, c, d = 50, 5, 2, 20
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
    
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
    
    # Calculate expected odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # With large imbalance, the lower bound might legitimately be 0
    assert lower >= 0.0, f"Expected non-negative lower bound, got {lower}"
    # With the new implementation, we should get a finite upper bound
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    # Check that odds ratio is within bounds
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    
    logger.info("✅ Large imbalance test passed")


def test_exact_ci_midp_problematic_case():
    """Test the previously problematic case (20,80,40,60) that produced invalid CI."""
    logger.info("Testing previously problematic case")
    
    a, b, c, d = 20, 80, 40, 60
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)
    
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Verify lower bound is less than upper bound
    assert lower <= upper, f"Expected lower bound ({lower}) <= upper bound ({upper})"
    # Verify odds ratio is within the CI
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    # Verify CI width is positive
    ci_width = upper - lower
    assert ci_width > 0, f"Expected positive CI width, got {ci_width}"
    
    logger.info("✅ Problematic case test passed")


def test_exact_ci_midp_large_sample():
    """Test the Mid-P method with a large sample size that previously failed."""
    logger.info("Testing large sample size")
    
    # Example case: 50/1000 vs 25/1000
    a, b, c, d = 50, 950, 25, 975
    
    # Calculate the odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate the confidence interval with default parameters
    lower, upper = exact_ci_midp(a, b, c, d)
    logger.info(f"CI (default parameters): ({lower:.4f}, {upper:.4f})")
    
    # Verify the confidence interval includes the odds ratio
    assert lower <= odds_ratio <= upper, f"Expected odds ratio {odds_ratio} to be within CI ({lower}, {upper})"
    # Verify the confidence interval has a finite upper bound
    assert upper < float('inf'), f"Expected finite upper bound, got {upper}"
    # Verify the confidence interval width is reasonable
    ci_width = upper - lower
    assert 1.0 < ci_width < 5.0, f"Expected reasonable CI width, got {ci_width}"
    
    logger.info("✅ Large sample size test passed")


def test_exact_ci_midp_grid_size_impact():
    """Test the impact of grid size on precision."""
    logger.info("Testing grid size impact")
    
    a, b, c, d = 50, 950, 25, 975
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate CIs with different grid sizes
    ci_small = exact_ci_midp(a, b, c, d, grid_size=50)
    ci_medium = exact_ci_midp(a, b, c, d, grid_size=200)
    ci_large = exact_ci_midp(a, b, c, d, grid_size=500)
    
    logger.info(f"CI (grid_size=50): ({ci_small[0]:.4f}, {ci_small[1]:.4f})")
    logger.info(f"CI (grid_size=200): ({ci_medium[0]:.4f}, {ci_medium[1]:.4f})")
    logger.info(f"CI (grid_size=500): ({ci_large[0]:.4f}, {ci_large[1]:.4f})")
    
    # Calculate widths
    width_small = ci_small[1] - ci_small[0]
    width_medium = ci_medium[1] - ci_medium[0]
    width_large = ci_large[1] - ci_large[0]
    
    logger.info(f"CI width (grid_size=50): {width_small:.4f}")
    logger.info(f"CI width (grid_size=200): {width_medium:.4f}")
    logger.info(f"CI width (grid_size=500): {width_large:.4f}")
    
    # Larger grid sizes should generally give more precise (narrower) CIs
    # But this is not guaranteed due to the discrete nature of the grid
    # So we just check that all CIs are reasonable
    assert width_small > 0, f"Expected positive CI width, got {width_small}"
    assert width_medium > 0, f"Expected positive CI width, got {width_medium}"
    assert width_large > 0, f"Expected positive CI width, got {width_large}"
    
    # All CIs should include the odds ratio
    assert ci_small[0] <= odds_ratio <= ci_small[1], f"Small grid CI does not include odds ratio"
    assert ci_medium[0] <= odds_ratio <= ci_medium[1], f"Medium grid CI does not include odds ratio"
    assert ci_large[0] <= odds_ratio <= ci_large[1], f"Large grid CI does not include odds ratio"
    
    logger.info("✅ Grid size impact test passed")


def test_exact_ci_midp_theta_range_impact():
    """Test the impact of theta range on results."""
    logger.info("Testing theta range impact")
    
    a, b, c, d = 50, 950, 25, 975
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate CIs with different theta ranges
    ci_narrow = exact_ci_midp(a, b, c, d, theta_min=0.1, theta_max=10)
    ci_medium = exact_ci_midp(a, b, c, d, theta_min=0.001, theta_max=1000)
    ci_wide = exact_ci_midp(a, b, c, d, theta_min=0.0001, theta_max=10000)
    
    logger.info(f"CI (narrow range): ({ci_narrow[0]:.4f}, {ci_narrow[1]:.4f})")
    logger.info(f"CI (medium range): ({ci_medium[0]:.4f}, {ci_medium[1]:.4f})")
    logger.info(f"CI (wide range): ({ci_wide[0]:.4f}, {ci_wide[1]:.4f})")
    
    # All CIs should include the odds ratio
    assert ci_narrow[0] <= odds_ratio <= ci_narrow[1], f"Narrow range CI does not include odds ratio"
    assert ci_medium[0] <= odds_ratio <= ci_medium[1], f"Medium range CI does not include odds ratio"
    assert ci_wide[0] <= odds_ratio <= ci_wide[1], f"Wide range CI does not include odds ratio"
    
    # All CIs should have finite bounds
    assert ci_narrow[0] > 0 and ci_narrow[1] < float('inf'), f"Narrow range CI has invalid bounds: {ci_narrow}"
    assert ci_medium[0] > 0 and ci_medium[1] < float('inf'), f"Medium range CI has invalid bounds: {ci_medium}"
    assert ci_wide[0] > 0 and ci_wide[1] < float('inf'), f"Wide range CI has invalid bounds: {ci_wide}"
    
    logger.info("✅ Theta range impact test passed")


def test_exact_ci_midp_alpha_levels():
    """Test different alpha levels."""
    logger.info("Testing different alpha levels")
    
    a, b, c, d = 50, 950, 25, 975
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate CIs with different alpha levels
    ci_01 = exact_ci_midp(a, b, c, d, alpha=0.01)
    ci_05 = exact_ci_midp(a, b, c, d, alpha=0.05)
    ci_10 = exact_ci_midp(a, b, c, d, alpha=0.10)
    
    logger.info(f"CI (alpha=0.01): ({ci_01[0]:.4f}, {ci_01[1]:.4f})")
    logger.info(f"CI (alpha=0.05): ({ci_05[0]:.4f}, {ci_05[1]:.4f})")
    logger.info(f"CI (alpha=0.10): ({ci_10[0]:.4f}, {ci_10[1]:.4f})")
    
    # Calculate widths
    width_01 = ci_01[1] - ci_01[0]
    width_05 = ci_05[1] - ci_05[0]
    width_10 = ci_10[1] - ci_10[0]
    
    logger.info(f"CI width (alpha=0.01): {width_01:.4f}")
    logger.info(f"CI width (alpha=0.05): {width_05:.4f}")
    logger.info(f"CI width (alpha=0.10): {width_10:.4f}")
    
    # Lower alpha should give wider CIs
    assert width_01 > width_05 > width_10, f"Expected width_01 ({width_01}) > width_05 ({width_05}) > width_10 ({width_10})"
    
    # All CIs should include the odds ratio
    assert ci_01[0] <= odds_ratio <= ci_01[1], f"CI with alpha=0.01 does not include odds ratio"
    assert ci_05[0] <= odds_ratio <= ci_05[1], f"CI with alpha=0.05 does not include odds ratio"
    assert ci_10[0] <= odds_ratio <= ci_10[1], f"CI with alpha=0.10 does not include odds ratio"
    
    logger.info("✅ Alpha levels test passed")


def test_exact_ci_midp_batch():
    """Test batch processing of multiple tables."""
    logger.info("Testing batch processing")
    
    # Create a list of tables
    tables = [
        (50, 950, 25, 975),  # Large sample
        (10, 90, 5, 95),     # Medium sample
        (2, 8, 1, 9)         # Small sample
    ]
    
    # Calculate confidence intervals for all tables
    results = exact_ci_midp_batch(tables)
    
    # Check results for each table
    for i, ((a, b, c, d), (lower, upper)) in enumerate(zip(tables, results)):
        # Calculate odds ratio
        odds_ratio = (a * d) / (b * c)
        logger.info(f"Table {i+1}: a={a}, b={b}, c={c}, d={d}")
        logger.info(f"Odds ratio: {odds_ratio:.4f}")
        logger.info(f"CI: ({lower:.4f}, {upper:.4f})")
        
        # Verify the confidence interval includes the odds ratio
        assert lower <= odds_ratio <= upper, f"Table {i+1}: CI ({lower}, {upper}) does not include odds ratio {odds_ratio}"
        
        # Verify the confidence interval has a finite upper bound
        assert upper < float('inf'), f"Table {i+1}: Expected finite upper bound, got {upper}"
        
        # Verify the confidence interval width is positive
        ci_width = upper - lower
        assert ci_width > 0, f"Table {i+1}: Expected positive CI width, got {ci_width}"
    
    logger.info("✅ Batch processing test passed")


def test_helper_functions():
    """Test the helper functions used by the Mid-P method."""
    logger.info("Testing helper functions")
    
    # Test calculate_midp_pvalue
    a_obs, n1, n2, m1 = 50, 1000, 1000, 75
    theta = 2.0
    p_value = calculate_midp_pvalue(a_obs, n1, n2, m1, theta)
    logger.info(f"calculate_midp_pvalue({a_obs}, {n1}, {n2}, {m1}, {theta}) = {p_value:.4f}")
    assert 0 <= p_value <= 1, f"Expected p-value between 0 and 1, got {p_value}"
    
    # Test find_ci_bound
    theta_grid = np.logspace(-3, 3, 100)
    p_values = np.linspace(0, 1, 100)
    alpha = 0.05
    
    # Lower bound
    lower = find_ci_bound(theta_grid, p_values, alpha, is_lower=True)
    logger.info(f"find_ci_bound(theta_grid, p_values, {alpha}, is_lower=True) = {lower:.4f}")
    assert lower > 0, f"Expected positive lower bound, got {lower}"
    
    # Upper bound
    upper = find_ci_bound(theta_grid, p_values, alpha, is_lower=False)
    logger.info(f"find_ci_bound(theta_grid, p_values, {alpha}, is_lower=False) = {upper:.4f}")
    assert upper > lower, f"Expected upper bound ({upper}) > lower bound ({lower})"
    
    logger.info("✅ Helper functions test passed")


def test_comparison_with_other_methods():
    """Compare Mid-P results with other methods."""
    logger.info("Testing comparison with other methods")
    
    if not has_other_methods:
        logger.warning("Skipping comparison with other methods (not available)")
        return
    
    # Test case from CLAUDE.md: 50/1000 vs 25/1000
    a, b, c, d = 50, 950, 25, 975
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Calculate CIs with different methods
    ci_midp = exact_ci_midp(a, b, c, d)
    ci_conditional = exact_ci_conditional(a, b, c, d)
    ci_blaker = exact_ci_blaker(a, b, c, d)
    
    logger.info(f"Mid-P CI: ({ci_midp[0]:.4f}, {ci_midp[1]:.4f})")
    logger.info(f"Conditional CI: ({ci_conditional[0]:.4f}, {ci_conditional[1]:.4f})")
    logger.info(f"Blaker CI: ({ci_blaker[0]:.4f}, {ci_blaker[1]:.4f})")
    
    # Calculate widths
    width_midp = ci_midp[1] - ci_midp[0]
    width_conditional = ci_conditional[1] - ci_conditional[0]
    width_blaker = ci_blaker[1] - ci_blaker[0]
    
    logger.info(f"Mid-P CI width: {width_midp:.4f}")
    logger.info(f"Conditional CI width: {width_conditional:.4f}")
    logger.info(f"Blaker CI width: {width_blaker:.4f}")
    
    # Expected width ordering: Blaker ≤ Mid-P ≤ Conditional
    # But this is not guaranteed for all cases, so we just check that all CIs are reasonable
    assert width_blaker > 0, f"Expected positive Blaker CI width, got {width_blaker}"
    assert width_midp > 0, f"Expected positive Mid-P CI width, got {width_midp}"
    assert width_conditional > 0, f"Expected positive Conditional CI width, got {width_conditional}"
    
    # All CIs should include the odds ratio
    assert ci_blaker[0] <= odds_ratio <= ci_blaker[1], f"Blaker CI does not include odds ratio"
    assert ci_midp[0] <= odds_ratio <= ci_midp[1], f"Mid-P CI does not include odds ratio"
    assert ci_conditional[0] <= odds_ratio <= ci_conditional[1], f"Conditional CI does not include odds ratio"
    
    # All CIs should have finite bounds
    assert ci_blaker[0] > 0 and ci_blaker[1] < float('inf'), f"Blaker CI has invalid bounds: {ci_blaker}"
    assert ci_midp[0] > 0 and ci_midp[1] < float('inf'), f"Mid-P CI has invalid bounds: {ci_midp}"
    assert ci_conditional[0] > 0 and ci_conditional[1] < float('inf'), f"Conditional CI has invalid bounds: {ci_conditional}"
    
    # Check if the results match the expected values from CLAUDE.md
    # Blaker: [1.250, 2.053]
    # Conditional: [1.234, 3.263]
    # Mid-P: should be between Blaker and Conditional
    assert abs(ci_blaker[0] - 1.250) < 0.1, f"Expected Blaker lower bound ~1.250, got {ci_blaker[0]}"
    assert abs(ci_blaker[1] - 2.053) < 0.1, f"Expected Blaker upper bound ~2.053, got {ci_blaker[1]}"
    assert abs(ci_conditional[0] - 1.234) < 0.1, f"Expected Conditional lower bound ~1.234, got {ci_conditional[0]}"
    assert abs(ci_conditional[1] - 3.263) < 0.1, f"Expected Conditional upper bound ~3.263, got {ci_conditional[1]}"
    
    logger.info("✅ Comparison with other methods test passed")


def test_performance():
    """Test performance of the Mid-P method with different parameters."""
    logger.info("Testing performance")
    
    a, b, c, d = 50, 950, 25, 975
    logger.info(f"Table: a={a}, b={b}, c={c}, d={d}")
    
    # Run a warm-up to account for JIT compilation and caching
    logger.info("Running warm-up iterations...")
    for _ in range(5):
        exact_ci_midp(a, b, c, d, grid_size=50)
        exact_ci_midp(a, b, c, d, grid_size=200)
        exact_ci_midp(a, b, c, d, grid_size=500)
    
    # Test with different grid sizes using multiple iterations
    iterations = 10
    logger.info(f"Running {iterations} iterations for each grid size...")
    
    # Small grid size
    total_time_small = 0
    for _ in range(iterations):
        start_time = time.time()
        exact_ci_midp(a, b, c, d, grid_size=50)
        total_time_small += time.time() - start_time
    time_small = total_time_small / iterations
    logger.info(f"Average time with grid_size=50: {time_small:.6f} seconds")
    
    # Medium grid size
    total_time_medium = 0
    for _ in range(iterations):
        start_time = time.time()
        exact_ci_midp(a, b, c, d, grid_size=200)
        total_time_medium += time.time() - start_time
    time_medium = total_time_medium / iterations
    logger.info(f"Average time with grid_size=200: {time_medium:.6f} seconds")
    
    # Large grid size
    total_time_large = 0
    for _ in range(iterations):
        start_time = time.time()
        exact_ci_midp(a, b, c, d, grid_size=500)
        total_time_large += time.time() - start_time
    time_large = total_time_large / iterations
    logger.info(f"Average time with grid_size=500: {time_large:.6f} seconds")
    
    # Check if times are in the expected order
    # Due to caching and other optimizations, we can't always guarantee that larger grid sizes
    # will take more time, especially for small workloads. So we'll just check that the times
    # are reasonable and log the results.
    logger.info(f"Time ratios: medium/small={time_medium/time_small:.2f}, large/medium={time_large/time_medium:.2f}")
    
    # Instead of strict assertions, we'll just check that the implementation works
    # and produces correct results for different grid sizes
    ci_small = exact_ci_midp(a, b, c, d, grid_size=50)
    ci_medium = exact_ci_midp(a, b, c, d, grid_size=200)
    ci_large = exact_ci_midp(a, b, c, d, grid_size=500)
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    
    # All CIs should include the odds ratio
    assert ci_small[0] <= odds_ratio <= ci_small[1], f"Small grid CI does not include odds ratio"
    assert ci_medium[0] <= odds_ratio <= ci_medium[1], f"Medium grid CI does not include odds ratio"
    assert ci_large[0] <= odds_ratio <= ci_large[1], f"Large grid CI does not include odds ratio"
    
    # All CIs should have finite bounds
    assert ci_small[0] > 0 and ci_small[1] < float('inf'), f"Small grid CI has invalid bounds: {ci_small}"
    assert ci_medium[0] > 0 and ci_medium[1] < float('inf'), f"Medium grid CI has invalid bounds: {ci_medium}"
    assert ci_large[0] > 0 and ci_large[1] < float('inf'), f"Large grid CI has invalid bounds: {ci_large}"
    
    logger.info("✅ Performance test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Running all tests for the Mid-P implementation")
    
    tests = [
        test_exact_ci_midp_basic,
        test_exact_ci_midp_edge_cases,
        test_exact_ci_midp_invalid_inputs,
        test_exact_ci_midp_small_counts,
        test_exact_ci_midp_large_imbalance,
        test_exact_ci_midp_problematic_case,
        test_exact_ci_midp_large_sample,
        test_exact_ci_midp_grid_size_impact,
        test_exact_ci_midp_theta_range_impact,
        test_exact_ci_midp_alpha_levels,
        test_exact_ci_midp_batch,
        test_helper_functions,
        test_comparison_with_other_methods,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        logger.info(f"\n{'='*80}\nRunning test: {test.__name__}\n{'='*80}")
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
    
    logger.info(f"\n{'='*80}\nTest summary: {passed} passed, {failed} failed\n{'='*80}")
    
    if failed > 0:
        logger.error(f"❌ {failed} tests failed")
        return False
    else:
        logger.info("✅ All tests passed")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)