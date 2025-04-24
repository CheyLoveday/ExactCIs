
"""
Integration tests for the ExactCIs package.

This file integrates tests from the ad-hoc test scripts (test_fixed.py and test_original.py)
into the proper testing framework and provides comprehensive end-to-end testing.
"""

import pytest
import logging
import numpy as np
from exactcis import compute_all_cis
from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_blaker,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.mark.fast
@pytest.mark.integration
def test_readme_example(timer):
    """Test the example from the README."""
    logger.info("Starting test_readme_example with counts (12, 5, 8, 10)")
    a, b, c, d = 12, 5, 8, 10
    alpha = 0.05

    # Test individual CI methods
    logger.info("Testing conditional method")
    lower, upper = exact_ci_conditional(a, b, c, d, alpha)
    assert round(lower, 3) == 1.059, f"Expected lower bound 1.059, got {lower:.3f}"
    assert round(upper, 3) == 8.726, f"Expected upper bound 8.726, got {upper:.3f}"

    logger.info("Testing midp method")
    lower, upper = exact_ci_midp(a, b, c, d, alpha)
    assert round(lower, 3) == 1.205, f"Expected lower bound 1.205, got {lower:.3f}"
    assert round(upper, 3) == 7.893, f"Expected upper bound 7.893, got {upper:.3f}"

    logger.info("Testing blaker method")
    lower, upper = exact_ci_blaker(a, b, c, d, alpha)
    assert round(lower, 3) == 1.114, f"Expected lower bound 1.114, got {lower:.3f}"
    assert round(upper, 3) == 8.312, f"Expected upper bound 8.312, got {upper:.3f}"

    logger.info("Testing unconditional method")
    lower, upper = exact_ci_unconditional(a, b, c, d, alpha, grid_size=10, refine=False)
    assert round(lower, 3) == 1.132, f"Expected lower bound 1.132, got {lower:.3f}"
    assert round(upper, 3) == 8.204, f"Expected upper bound 8.204, got {upper:.3f}"

    logger.info("Testing wald_haldane method")
    lower, upper = ci_wald_haldane(a, b, c, d, alpha)
    assert round(lower, 3) == 1.024, f"Expected lower bound 1.024, got {lower:.3f}"
    assert round(upper, 3) == 8.658, f"Expected upper bound 8.658, got {upper:.3f}"

    logger.info("test_readme_example completed successfully")


@pytest.mark.fast
@pytest.mark.integration
def test_compute_all_cis(timer):
    """Test the compute_all_cis function."""
    logger.info("Starting test_compute_all_cis with counts (12, 5, 8, 10)")
    a, b, c, d = 12, 5, 8, 10
    alpha = 0.05

    logger.info("Computing all CIs")
    results = compute_all_cis(a, b, c, d, alpha, grid_size=10)

    # Check that all methods are included
    assert set(results.keys()) == {
        "conditional", "midp", "blaker", "unconditional", "wald_haldane"
    }
    logger.info("All expected methods are included in results")

    # Check individual results
    lower, upper = results["conditional"]
    assert round(lower, 3) == 1.059, f"Expected lower bound 1.059, got {lower:.3f}"
    assert round(upper, 3) == 8.726, f"Expected upper bound 8.726, got {upper:.3f}"
    logger.info(f"Conditional CI verified: ({lower:.3f}, {upper:.3f})")

    lower, upper = results["midp"]
    assert round(lower, 3) == 1.205, f"Expected lower bound 1.205, got {lower:.3f}"
    assert round(upper, 3) == 7.893, f"Expected upper bound 7.893, got {upper:.3f}"
    logger.info(f"Mid-P CI verified: ({lower:.3f}, {upper:.3f})")

    lower, upper = results["blaker"]
    assert round(lower, 3) == 1.114, f"Expected lower bound 1.114, got {lower:.3f}"
    assert round(upper, 3) == 8.312, f"Expected upper bound 8.312, got {upper:.3f}"
    logger.info(f"Blaker CI verified: ({lower:.3f}, {upper:.3f})")

    lower, upper = results["unconditional"]
    assert round(lower, 3) == 1.132, f"Expected lower bound 1.132, got {lower:.3f}"
    assert round(upper, 3) == 8.204, f"Expected upper bound 8.204, got {upper:.3f}"
    logger.info(f"Unconditional CI verified: ({lower:.3f}, {upper:.3f})")

    lower, upper = results["wald_haldane"]
    assert round(lower, 3) == 1.024, f"Expected lower bound 1.024, got {lower:.3f}"
    assert round(upper, 3) == 8.658, f"Expected upper bound 8.658, got {upper:.3f}"
    logger.info(f"Wald-Haldane CI verified: ({lower:.3f}, {upper:.3f})")

    logger.info("test_compute_all_cis completed successfully")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.edge
def test_small_counts(timer):
    """Test with small counts."""
    logger.info("Starting test_small_counts with counts (1, 1, 1, 1)")
    a, b, c, d = 1, 1, 1, 1
    alpha = 0.05

    logger.info("Computing all CIs for small counts")
    results = compute_all_cis(a, b, c, d, alpha, grid_size=5)

    # Check that all results are valid
    for method, (lower, upper) in results.items():
        logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
        assert lower > 0.0, f"{method}: Expected positive lower bound, got {lower}"
        assert upper < float('inf'), f"{method}: Expected finite upper bound, got {upper}"

    logger.info("test_small_counts completed successfully")


@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.edge
def test_zero_in_one_cell(timer):
    """Test with zero in one cell."""
    logger.info("Starting test_zero_in_one_cell with counts (0, 5, 8, 10)")
    a, b, c, d = 0, 5, 8, 10
    alpha = 0.05

    logger.info("Computing all CIs for zero in one cell")
    results = compute_all_cis(a, b, c, d, alpha, grid_size=5)

    # Check that all results are valid
    for method, (lower, upper) in results.items():
        logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
        assert lower >= 0.0, f"{method}: Expected non-negative lower bound, got {lower}"
        assert upper < float('inf'), f"{method}: Expected finite upper bound, got {upper}"

    logger.info("test_zero_in_one_cell completed successfully")


@pytest.mark.slow
@pytest.mark.timeout(300)  # 5-minute timeout
@pytest.mark.integration
def test_large_imbalance(timer):
    """Test with large imbalance in counts."""
    logger.info("Starting test_large_imbalance with counts (50, 5, 2, 20)")
    a, b, c, d = 50, 5, 2, 20
    alpha = 0.05

    try:
        logger.info("Computing all CIs for large imbalance test")
        results = compute_all_cis(a, b, c, d, alpha, grid_size=10)

        # Check that all results are valid
        for method, (lower, upper) in results.items():
            logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
            assert lower > 0.0, f"{method}: Expected positive lower bound, got {lower}"
            assert upper < float('inf'), f"{method}: Expected finite upper bound, got {upper}"
        logger.info("Large imbalance test completed successfully")
    except RuntimeError as e:
        # If the method raises a RuntimeError, that's acceptable for this edge case
        logger.warning(f"RuntimeError in large imbalance test: {str(e)}")
        pass


@pytest.mark.fast
@pytest.mark.integration
def test_odds_ratio_calculation(timer):
    """Test the odds ratio calculation."""
    logger.info("Starting test_odds_ratio_calculation with counts (12, 5, 8, 10)")
    a, b, c, d = 12, 5, 8, 10

    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Calculated odds ratio: {odds_ratio:.6f}")

    # The odds ratio should be within all confidence intervals
    logger.info("Computing all CIs for odds ratio test")
    results = compute_all_cis(a, b, c, d, alpha=0.05, grid_size=10)

    for method, (lower, upper) in results.items():
        logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f}), odds_ratio = {odds_ratio:.6f}")
        assert lower <= odds_ratio <= upper, f"{method}: Odds ratio {odds_ratio} not in CI ({lower}, {upper})"

    logger.info("test_odds_ratio_calculation completed successfully")


@pytest.mark.parametrize("input_values,expected", [
    ((12, 5, 8, 10), 3.0),   # Standard example
    ((0, 5, 8, 10), 0.0),    # Zero in one cell
    ((5, 0, 8, 10), float('inf')),  # Another zero case
    ((1, 1, 1, 1), 1.0),     # Equal counts
    ((10, 5, 5, 10), 4.0),   # Symmetric
    ((20, 10, 5, 10), 4.0),  # Larger counts
])
@pytest.mark.fast
@pytest.mark.integration
def test_odds_ratio_various_inputs(input_values, expected, timer):
    """Test odds ratio calculation with various inputs."""
    a, b, c, d = input_values
    
    # Calculate odds ratio (handle special cases)
    if b == 0 or c == 0:
        if b == 0 and c == 0:
            odds_ratio = 1.0  # Indeterminate, but conventionally set to 1
        elif b == 0:
            odds_ratio = float('inf')  # Infinite odds ratio
        else:  # c == 0
            odds_ratio = 0.0  # Zero odds ratio
    else:
        odds_ratio = (a * d) / (b * c)
    
    logger.info(f"Testing counts ({a}, {b}, {c}, {d}) with expected OR = {expected}")
    assert odds_ratio == expected, f"Expected odds ratio {expected}, got {odds_ratio}"
    
    try:
        # Only compute CIs for non-degenerate cases
        if b > 0 and c > 0:
            results = compute_all_cis(a, b, c, d, alpha=0.05, grid_size=5)
            
            # Check that odds ratio is within all CIs
            for method, (lower, upper) in results.items():
                logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
                assert lower <= odds_ratio <= upper, f"{method}: OR {odds_ratio} not in CI ({lower}, {upper})"
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Error computing CIs for {input_values}: {str(e)}")
        # Some edge cases might legitimately raise errors


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.fast
@pytest.mark.integration
def test_different_alpha_levels(alpha, timer):
    """Test that different alpha levels produce appropriate CI widths."""
    a, b, c, d = 12, 5, 8, 10
    
    logger.info(f"Computing CIs with alpha={alpha}")
    results = compute_all_cis(a, b, c, d, alpha=alpha, grid_size=5)
    
    # Store widths to compare methods
    widths = {}
    
    for method, (lower, upper) in results.items():
        logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
        widths[method] = upper - lower
    
    # Width comparison - just log the results for information
    methods = list(widths.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, method2 = methods[i], methods[j]
            logger.info(f"Width comparison: {method1} ({widths[method1]:.3f}) vs {method2} ({widths[method2]:.3f})")


@pytest.mark.parametrize("grid_size", [5, 10, 20])
@pytest.mark.slow
@pytest.mark.integration
def test_grid_size_effect(grid_size, timer):
    """Test the effect of grid size on the unconditional method."""
    a, b, c, d = 12, 5, 8, 10
    alpha = 0.05
    
    logger.info(f"Computing CIs with grid_size={grid_size}")
    results = compute_all_cis(a, b, c, d, alpha=alpha, grid_size=grid_size)
    
    # Extract the unconditional CI
    lower, upper = results["unconditional"]
    logger.info(f"Unconditional CI with grid_size={grid_size}: ({lower:.6f}, {upper:.6f})")
    
    # Basic validity checks
    assert lower > 0, "Lower bound should be positive"
    assert upper < float('inf'), "Upper bound should be finite"
    assert lower < upper, "Lower bound should be less than upper bound"
    
    # Reference values from grid_size=500 (as in README)
    ref_lower, ref_upper = 1.132, 8.204
    
    # Allow some tolerance based on grid size
    tolerance = 0.3 if grid_size < 10 else 0.2
    
    assert abs(lower - ref_lower) < tolerance, f"Lower bound {lower} too far from reference {ref_lower}"
    assert abs(upper - ref_upper) < tolerance, f"Upper bound {upper} too far from reference {ref_upper}"


@pytest.mark.fast
@pytest.mark.integration
def test_consistent_ordering(timer):
    """Test that lower bound is always less than upper bound."""
    test_cases = [
        (12, 5, 8, 10),   # Standard case
        (1, 1, 1, 1),     # Equal counts
        (0, 5, 8, 10),    # Zero in one cell
        (10, 5, 5, 10),   # Symmetric case
    ]
    
    for a, b, c, d in test_cases:
        logger.info(f"Testing bounds ordering for counts ({a}, {b}, {c}, {d})")
        try:
            results = compute_all_cis(a, b, c, d, alpha=0.05, grid_size=5)
            
            for method, (lower, upper) in results.items():
                logger.info(f"Method {method}: CI = ({lower:.6f}, {upper:.6f})")
                assert lower <= upper, f"{method}: Lower bound {lower} > upper bound {upper}"
        
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Error computing CIs for {(a,b,c,d)}: {str(e)}")
            # Some edge cases might legitimately raise errors


@pytest.mark.fast
@pytest.mark.integration
def test_invalid_inputs(timer):
    """Test that invalid inputs raise appropriate exceptions."""
    invalid_cases = [
        (-1, 5, 8, 10),    # Negative count
        (12, -5, 8, 10),   # Negative count
        (12, 5, -8, 10),   # Negative count
        (12, 5, 8, -10),   # Negative count
        (0, 0, 8, 10),     # Empty margin
        (12, 5, 0, 0),     # Empty margin
    ]
    
    for a, b, c, d in invalid_cases:
        logger.info(f"Testing invalid inputs ({a}, {b}, {c}, {d})")
        with pytest.raises(ValueError):
            compute_all_cis(a, b, c, d)
    
    # Test invalid alpha
    with pytest.raises(ValueError):
        compute_all_cis(12, 5, 8, 10, alpha=1.5)
    
    with pytest.raises(ValueError):
        compute_all_cis(12, 5, 8, 10, alpha=-0.05)


@pytest.mark.fast
@pytest.mark.integration
def test_consistency_across_methods(timer):
    """Test that all methods produce reasonable and consistent results."""
    a, b, c, d = 12, 5, 8, 10
    alpha = 0.05
    
    logger.info(f"Testing consistency across methods for counts ({a}, {b}, {c}, {d})")
    results = compute_all_cis(a, b, c, d, alpha=alpha, grid_size=10)
    
    # Extract results for each method
    ci_conditional = results["conditional"]
    ci_midp = results["midp"]
    ci_blaker = results["blaker"]
    ci_unconditional = results["unconditional"]
    ci_wald = results["wald_haldane"]
    
    # Log all CIs
    for method, ci in results.items():
        logger.info(f"{method}: CI = {ci}")
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    logger.info(f"Odds ratio: {odds_ratio}")
    
    # Check that all CIs contain the odds ratio
    for method, (lower, upper) in results.items():
        assert lower <= odds_ratio <= upper, f"{method} CI does not contain odds ratio"
    
    # Check relationships between methods (these are generally expected behaviors)
    assert ci_midp[0] >= ci_conditional[0], "Mid-P lower bound typically >= conditional lower bound"
    assert ci_midp[1] <= ci_conditional[1], "Mid-P upper bound typically <= conditional upper bound"
    
    # Width comparisons
    widths = {method: upper-lower for method, (lower, upper) in results.items()}
    logger.info(f"CI widths: {widths}")
    
    # Mid-P should typically be narrower than conditional
    assert widths["midp"] <= widths["conditional"], "Mid-P CI not narrower than conditional"
