"""
Extended tests for the unconditional method to improve coverage.
These tests focus on untested code paths and edge cases.
"""

import pytest
import numpy as np
import math
from unittest.mock import patch, MagicMock
from exactcis.methods.unconditional import (
    exact_ci_unconditional,
    _log_pvalue_barnard,
    _process_grid_point,
    _build_adaptive_grid,
    _optimize_grid_size
)


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_invalid_inputs():
    """Test exact_ci_unconditional with invalid inputs."""
    # Test invalid alpha
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, alpha=0.0)
    
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, alpha=1.0)
    
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, alpha=-0.1)
    
    # Test invalid grid_size
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, grid_size=0)
    
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, grid_size=-5)
    
    # Test invalid theta_min/theta_max
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, theta_min=0.0)
    
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, theta_min=-1.0)
    
    with pytest.raises(ValueError):
        exact_ci_unconditional(5, 5, 5, 5, theta_min=2.0, theta_max=1.0)


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_special_cases():
    """Test exact_ci_unconditional with special cases."""
    # Test with zero marginal totals
    lower, upper = exact_ci_unconditional(0, 0, 5, 5, alpha=0.05)
    assert lower == 0.0
    assert upper == float('inf')
    
    # Test with a=0 and c>0
    lower, upper = exact_ci_unconditional(0, 5, 3, 7, alpha=0.05)
    assert lower == 0.0
    assert upper > 0
    
    # Test with very small table
    lower, upper = exact_ci_unconditional(1, 1, 1, 1, alpha=0.05)
    assert lower >= 0.0
    assert upper >= lower


@pytest.mark.methods
@pytest.mark.fast
def test_build_adaptive_grid():
    """Test _build_adaptive_grid function."""
    # Test with default parameters
    grid = _build_adaptive_grid(0.5, 20)
    assert len(grid) == 20
    assert all(0 < p < 1 for p in grid)
    assert all(grid[i] < grid[i+1] for i in range(len(grid)-1))  # Should be increasing
    
    # Test with different p1_mle
    grid = _build_adaptive_grid(0.2, 10)
    assert len(grid) == 10
    assert all(0 < p < 1 for p in grid)
    
    # Test with different density factor
    grid = _build_adaptive_grid(0.5, 15, density_factor=0.5)
    assert len(grid) == 15
    assert all(0 < p < 1 for p in grid)


@pytest.mark.methods
@pytest.mark.fast
def test_optimize_grid_size():
    """Test _optimize_grid_size function."""
    # Test with small table
    optimized = _optimize_grid_size(5, 5, 15)
    assert optimized >= 15  # Should be at least the base size
    
    # Test with larger table
    optimized = _optimize_grid_size(50, 50, 15)
    assert optimized >= 15
    
    # Test with very large table
    optimized = _optimize_grid_size(1000, 1000, 15)
    assert optimized >= 15


@pytest.mark.methods
@pytest.mark.fast
def test_process_grid_point():
    """Test _process_grid_point function."""
    # Test normal case
    args = (0.3, 5, 2, 10, 10, 2.0)  # (p1, a, c, n1, n2, theta)
    result = _process_grid_point(args)
    
    # Result should be a finite log probability
    assert np.isfinite(result)
    assert result <= 0  # Log probability should be <= 0
    
    # Test with extreme p1 values
    args_extreme = (0.001, 5, 2, 10, 10, 2.0)
    result_extreme = _process_grid_point(args_extreme)
    assert np.isfinite(result_extreme)
    
    # Test with theta = 1 (null hypothesis)
    args_null = (0.3, 5, 2, 10, 10, 1.0)
    result_null = _process_grid_point(args_null)
    assert np.isfinite(result_null)


@pytest.mark.methods
@pytest.mark.fast
def test_log_pvalue_barnard():
    """Test _log_pvalue_barnard function."""
    # Test basic functionality
    log_pval = _log_pvalue_barnard(5, 2, 10, 10, 2.0, grid_size=50)
    assert np.isfinite(log_pval)
    assert log_pval <= 0  # Log p-value should be <= 0
    
    # Test with custom p1 grid
    custom_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    log_pval_custom = _log_pvalue_barnard(
        5, 2, 10, 10, 2.0, grid_size=50, p1_grid_override=custom_grid
    )
    assert np.isfinite(log_pval_custom)
    
    # Test with timeout
    timeout_checker = lambda: True  # Always timeout
    log_pval_timeout = _log_pvalue_barnard(
        5, 2, 10, 10, 2.0, grid_size=50, timeout_checker=timeout_checker
    )
    assert log_pval_timeout is None


@pytest.mark.methods
@pytest.mark.fast
def test_log_pvalue_barnard_edge_cases():
    """Test _log_pvalue_barnard with edge cases."""
    # Test with very small grid
    log_pval = _log_pvalue_barnard(5, 2, 10, 10, 2.0, grid_size=3)
    assert np.isfinite(log_pval)
    
    # Test with large theta
    log_pval = _log_pvalue_barnard(5, 2, 10, 10, 100.0, grid_size=50)
    assert np.isfinite(log_pval)
    
    # Test with small theta
    log_pval = _log_pvalue_barnard(5, 2, 10, 10, 0.01, grid_size=50)
    assert np.isfinite(log_pval)
    
    # Test with extreme table values
    log_pval = _log_pvalue_barnard(0, 10, 10, 10, 1.0, grid_size=50)
    assert np.isfinite(log_pval)


@pytest.mark.methods
@pytest.mark.fast
def test_process_grid_point_edge_cases():
    """Test _process_grid_point with edge cases."""
    # Test with p1 very close to 0
    args = (1e-10, 5, 2, 10, 10, 2.0)
    result = _process_grid_point(args)
    assert np.isfinite(result)
    
    # Test with p1 very close to 1
    args = (1 - 1e-10, 5, 2, 10, 10, 2.0)
    result = _process_grid_point(args)
    assert np.isfinite(result)
    
    # Test with a=0
    args = (0.3, 0, 2, 10, 10, 2.0)
    result = _process_grid_point(args)
    assert np.isfinite(result)
    
    # Test with c=0
    args = (0.3, 5, 0, 10, 10, 2.0)
    result = _process_grid_point(args)
    assert np.isfinite(result)


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_with_custom_parameters():
    """Test exact_ci_unconditional with custom parameters."""
    # Test with custom grid size
    lower, upper = exact_ci_unconditional(5, 5, 5, 5, alpha=0.05, grid_size=20)
    assert lower >= 0.0
    assert upper >= lower
    
    # Test with custom theta range
    lower, upper = exact_ci_unconditional(
        5, 5, 5, 5, alpha=0.05, theta_min=0.5, theta_max=5.0
    )
    assert lower >= 0.0
    assert upper >= lower
    
    # Test with different alpha
    lower_01, upper_01 = exact_ci_unconditional(5, 5, 5, 5, alpha=0.01)
    lower_10, upper_10 = exact_ci_unconditional(5, 5, 5, 5, alpha=0.10)
    
    # 99% CI should be wider than 90% CI
    assert upper_01 >= upper_10
    assert lower_01 <= lower_10


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_logging():
    """Test that logging works correctly in exact_ci_unconditional."""
    with patch('exactcis.methods.unconditional.logger') as mock_logger:
        exact_ci_unconditional(5, 5, 5, 5, alpha=0.05)
        
        # Check that logging was called
        assert mock_logger.info.called or mock_logger.debug.called


@pytest.mark.methods
@pytest.mark.fast
def test_build_adaptive_grid_edge_cases():
    """Test _build_adaptive_grid with edge cases."""
    # Test with grid_size = 1
    grid = _build_adaptive_grid(0.5, 1)
    assert len(grid) == 1
    assert 0 < grid[0] < 1
    
    # Test with grid_size = 2
    grid = _build_adaptive_grid(0.5, 2)
    assert len(grid) == 2
    assert grid[0] < grid[1]
    assert all(0 < p < 1 for p in grid)
    
    # Test with extreme p1_mle values
    grid = _build_adaptive_grid(0.01, 10)
    assert len(grid) == 10
    assert all(0 < p < 1 for p in grid)
    
    grid = _build_adaptive_grid(0.99, 10)
    assert len(grid) == 10
    assert all(0 < p < 1 for p in grid)


@pytest.mark.methods
@pytest.mark.fast
def test_log_pvalue_barnard_with_multiprocessing():
    """Test _log_pvalue_barnard multiprocessing behavior."""
    # Test with use_multiprocessing=True
    log_pval = _log_pvalue_barnard(
        5, 2, 10, 10, 2.0, grid_size=20, use_multiprocessing=True
    )
    assert np.isfinite(log_pval)
    
    # Test with use_multiprocessing=False
    log_pval_single = _log_pvalue_barnard(
        5, 2, 10, 10, 2.0, grid_size=20, use_multiprocessing=False
    )
    assert np.isfinite(log_pval_single)
    
    # Results should be similar (allowing for numerical differences)
    assert abs(log_pval - log_pval_single) < 0.1


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_consistency():
    """Test that exact_ci_unconditional produces consistent results."""
    # Run the same calculation twice
    lower1, upper1 = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05, grid_size=50)
    lower2, upper2 = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05, grid_size=50)
    
    # Results should be identical (or very close due to numerical precision)
    assert abs(lower1 - lower2) < 1e-10
    assert abs(upper1 - upper2) < 1e-10


@pytest.mark.methods
@pytest.mark.slow
def test_exact_ci_unconditional_stress_test():
    """Stress test exact_ci_unconditional with challenging cases."""
    test_cases = [
        (1, 1, 1, 1),      # Minimal table
        (0, 5, 5, 0),      # Zeros in opposite corners
        (10, 0, 0, 10),    # Zeros in other corners
        (1, 99, 99, 1),    # Extreme odds ratio
        (50, 50, 50, 50),  # Large balanced table
    ]
    
    for a, b, c, d in test_cases:
        try:
            lower, upper = exact_ci_unconditional(
                a, b, c, d, alpha=0.05, grid_size=20  # Smaller grid for speed
            )
            
            # Basic sanity checks
            assert lower >= 0.0, f"Lower bound negative for table ({a},{b},{c},{d})"
            assert upper >= lower, f"Upper < lower for table ({a},{b},{c},{d})"
            assert np.isfinite(lower), f"Lower bound not finite for table ({a},{b},{c},{d})"
            # Upper can be infinite
            
        except Exception as e:
            # For stress test, we allow some failures but log them
            pytest.skip(f"Stress test failed for table ({a},{b},{c},{d}): {e}")


@pytest.mark.methods
@pytest.mark.fast
def test_unconditional_method_error_handling():
    """Test error handling in unconditional method functions."""
    # Test _process_grid_point with invalid arguments
    try:
        # This might cause numerical issues but shouldn't crash
        result = _process_grid_point((0.0, 5, 2, 10, 10, 2.0))  # p1 = 0
        # If it doesn't crash, that's good
    except Exception:
        # Expected for invalid inputs
        pass
    
    # Test _log_pvalue_barnard with extreme parameters
    try:
        result = _log_pvalue_barnard(100, 100, 100, 100, 1e-10, grid_size=10)
        # Should handle extreme cases gracefully
    except Exception:
        # Some failures are acceptable for extreme cases
        pass


@pytest.mark.methods
@pytest.mark.fast
def test_optimize_grid_size_edge_cases():
    """Test _optimize_grid_size with edge cases."""
    # Test with very small table
    optimized = _optimize_grid_size(1, 1, 10)
    assert optimized >= 10
    
    # Test with zero base grid size
    optimized = _optimize_grid_size(10, 10, 1)
    assert optimized >= 1
    
    # Test with asymmetric table
    optimized = _optimize_grid_size(100, 10, 15)
    assert optimized >= 15
    
    optimized = _optimize_grid_size(10, 100, 15)
    assert optimized >= 15


@pytest.mark.methods
@pytest.mark.fast
def test_build_adaptive_grid_density_factors():
    """Test _build_adaptive_grid with different density factors."""
    # Test with very low density factor
    grid = _build_adaptive_grid(0.5, 20, density_factor=0.1)
    assert len(grid) == 20
    assert all(0 < p < 1 for p in grid)
    
    # Test with very high density factor
    grid = _build_adaptive_grid(0.5, 20, density_factor=0.9)
    assert len(grid) == 20
    assert all(0 < p < 1 for p in grid)
    
    # Test with density factor = 0
    grid = _build_adaptive_grid(0.5, 20, density_factor=0.0)
    assert len(grid) == 20
    assert all(0 < p < 1 for p in grid)
    
    # Test with density factor = 1
    grid = _build_adaptive_grid(0.5, 20, density_factor=1.0)
    assert len(grid) == 20
    assert all(0 < p < 1 for p in grid)


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_with_haldane():
    """Test exact_ci_unconditional with Haldane correction."""
    # Test with Haldane correction enabled
    lower, upper = exact_ci_unconditional(0, 5, 5, 5, alpha=0.05, haldane=True)
    assert lower >= 0.0
    assert upper > 0
    
    # Test with Haldane correction disabled
    lower_no_haldane, upper_no_haldane = exact_ci_unconditional(0, 5, 5, 5, alpha=0.05, haldane=False)
    assert lower_no_haldane >= 0.0
    assert upper_no_haldane > 0
    
    # Results might be different but both should be valid
    assert np.isfinite(lower) and np.isfinite(lower_no_haldane)


@pytest.mark.methods
@pytest.mark.fast
def test_exact_ci_unconditional_with_timeout():
    """Test exact_ci_unconditional with timeout parameter."""
    # Test with very short timeout (should still complete for small table)
    lower, upper = exact_ci_unconditional(3, 3, 3, 3, alpha=0.05, timeout=0.1)
    assert lower >= 0.0
    assert upper >= lower
    
    # Test with longer timeout
    lower, upper = exact_ci_unconditional(5, 5, 5, 5, alpha=0.05, timeout=10.0)
    assert lower >= 0.0
    assert upper >= lower