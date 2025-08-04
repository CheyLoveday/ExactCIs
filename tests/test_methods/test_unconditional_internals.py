"""
Tests for internal functions of the unconditional exact confidence interval method.

This file contains tests for the internal helper functions used by the
unconditional exact confidence interval method, focusing on improving
test coverage for these functions.
"""

import sys
import numpy as np
import time
import logging
from unittest.mock import patch, MagicMock

# Import pytest - ensure it's installed
try:
    import pytest
except ImportError:
    print("pytest not found. Please install it with 'pip install pytest'")
    sys.exit(1)

# Import the internal functions directly from the module
from exactcis.methods.unconditional import (
    _build_adaptive_grid,
    _optimize_grid_size,
    _process_grid_point,
    _log_binom_pmf,
    _log_pvalue_barnard,
    has_numba
)

# Try to import Numba-specific functions, which may not be available if Numba is not installed
try:
    from exactcis.methods.unconditional import (
        _log_binom_coeff_numba,
        _log_binom_pmf_numba,
        _logsumexp_numba,
        _process_grid_point_numba,
        _process_grid_point_numba_core
    )
    has_numba_imports = True
except ImportError:
    has_numba_imports = False

# Configure logging
logger = logging.getLogger(__name__)


# Tests for grid building and optimization functions
@pytest.mark.fast
def test_build_adaptive_grid_basic():
    """Test basic functionality of _build_adaptive_grid."""
    # Test with MLE in the middle
    grid = _build_adaptive_grid(p1_mle=0.5, grid_size=10)
    
    # Check that the grid has reasonable properties
    assert len(grid) > 0, "Grid should not be empty"
    assert min(grid) >= 0, "Grid points should be non-negative"
    assert max(grid) <= 1, "Grid points should be at most 1"
    assert 0.5 in grid, "Grid should include the MLE"
    
    # Check that the grid is sorted
    assert all(grid[i] <= grid[i+1] for i in range(len(grid)-1)), "Grid should be sorted"
    
    # Check that there are more points near the MLE
    points_near_mle = sum(1 for p in grid if abs(p - 0.5) < 0.2)
    points_far_from_mle = sum(1 for p in grid if abs(p - 0.5) >= 0.2)
    assert points_near_mle > points_far_from_mle, "There should be more points near the MLE"


@pytest.mark.fast
def test_build_adaptive_grid_edge_cases():
    """Test edge cases for _build_adaptive_grid."""
    # Test with MLE at the boundary
    grid_low = _build_adaptive_grid(p1_mle=0.01, grid_size=10)
    grid_high = _build_adaptive_grid(p1_mle=0.99, grid_size=10)
    
    # Check that the grids have reasonable properties
    assert len(grid_low) > 0 and len(grid_high) > 0, "Grids should not be empty"
    assert min(grid_low) > 0 and min(grid_high) > 0, "Grid points should be positive"
    assert max(grid_low) < 1 and max(grid_high) < 1, "Grid points should be less than 1"
    
    # Check that the grids include points near the MLE
    assert any(p < 0.05 for p in grid_low), "Low MLE grid should include points near 0"
    assert any(p > 0.95 for p in grid_high), "High MLE grid should include points near 1"


@pytest.mark.fast
def test_build_adaptive_grid_different_sizes():
    """Test _build_adaptive_grid with different grid sizes."""
    # Test with different grid sizes
    grid_small = _build_adaptive_grid(p1_mle=0.5, grid_size=5)
    grid_medium = _build_adaptive_grid(p1_mle=0.5, grid_size=15)
    grid_large = _build_adaptive_grid(p1_mle=0.5, grid_size=30)
    
    # Check that larger grid sizes produce more points
    assert len(grid_small) < len(grid_medium) < len(grid_large), "Larger grid sizes should produce more points"


@pytest.mark.fast
def test_build_adaptive_grid_density_factor():
    """Test _build_adaptive_grid with different density factors."""
    # Test with different density factors
    grid_low_density = _build_adaptive_grid(p1_mle=0.5, grid_size=15, density_factor=0.1)
    grid_high_density = _build_adaptive_grid(p1_mle=0.5, grid_size=15, density_factor=0.5)
    
    # Count points near MLE
    points_near_mle_low = sum(1 for p in grid_low_density if abs(p - 0.5) < 0.2)
    points_near_mle_high = sum(1 for p in grid_high_density if abs(p - 0.5) < 0.2)
    
    # Higher density factor should result in more points near MLE
    assert points_near_mle_high >= points_near_mle_low, "Higher density factor should result in more points near MLE"


@pytest.mark.fast
def test_optimize_grid_size():
    """Test _optimize_grid_size with different table sizes."""
    # Test with small tables
    small_grid = _optimize_grid_size(n1=5, n2=5, base_grid_size=20)
    assert small_grid <= 20, "Grid size for small tables should be at most the base grid size"
    
    # Test with medium tables
    medium_grid = _optimize_grid_size(n1=30, n2=30, base_grid_size=20)
    assert medium_grid <= 20, "Grid size for medium tables should be at most the base grid size"
    
    # Test with large tables
    large_grid = _optimize_grid_size(n1=100, n2=100, base_grid_size=20)
    assert large_grid <= 20, "Grid size for large tables should be at most the base grid size"
    
    # Check that larger tables get smaller grid sizes for efficiency
    assert small_grid >= medium_grid >= large_grid, "Larger tables should get smaller grid sizes"


# Tests for grid point processing functions
@pytest.mark.fast
def test_process_grid_point_basic():
    """Test basic functionality of _process_grid_point."""
    # Test with very small inputs to avoid timeout
    result = _process_grid_point((0.5, 2, 2, 5, 5, 1.0))
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    assert result <= 0, "Log p-value should be non-positive"


@pytest.mark.fast
def test_process_grid_point_with_timeout():
    """Test _process_grid_point with timeout."""
    # Test with timeout that won't be reached - use very small inputs
    start_time = time.time()
    result = _process_grid_point((0.5, 2, 2, 5, 5, 1.0, start_time, 5.0))
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    assert result <= 0, "Log p-value should be non-positive"
    
    # Test with timeout that will be reached immediately
    start_time = time.time() - 6.0  # 6 seconds ago
    result = _process_grid_point((0.5, 2, 2, 5, 5, 1.0, start_time, 5.0))
    
    # Check that the result is None due to timeout
    assert result is None, "Result should be None due to timeout"


@pytest.mark.fast
def test_process_grid_point_invalid_args():
    """Test _process_grid_point with invalid arguments."""
    # Test with wrong number of arguments
    with pytest.raises(ValueError):
        _process_grid_point((0.5, 5, 5, 10, 10))  # Missing theta
    
    with pytest.raises(ValueError):
        _process_grid_point((0.5, 5, 5, 10, 10, 1.0, time.time()))  # Missing timeout


# Tests for Numba-optimized functions
@pytest.mark.skipif(not has_numba_imports, reason="Numba-specific functions not available")
@pytest.mark.fast
def test_log_binom_coeff_numba():
    """Test _log_binom_coeff_numba with various inputs."""
    # Test with valid inputs
    result = _log_binom_coeff_numba(10, 5)
    expected = np.log(252)  # 10 choose 5 = 252
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    # Test with edge cases
    assert _log_binom_coeff_numba(10, 0) == 0.0, "log(10 choose 0) should be 0"
    assert _log_binom_coeff_numba(10, 10) == 0.0, "log(10 choose 10) should be 0"
    
    # Test with invalid inputs
    assert _log_binom_coeff_numba(10, 11) == float('-inf'), "log(10 choose 11) should be -inf"
    assert _log_binom_coeff_numba(10, -1) == float('-inf'), "log(10 choose -1) should be -inf"


@pytest.mark.skipif(not has_numba_imports, reason="Numba-specific functions not available")
@pytest.mark.fast
def test_log_binom_pmf_numba():
    """Test _log_binom_pmf_numba with various inputs."""
    # Test with valid inputs
    result = _log_binom_pmf_numba(10, 5, 0.5)
    expected = np.log(252) + 5 * np.log(0.5) + 5 * np.log(0.5)  # log(10 choose 5) + 5*log(0.5) + 5*log(0.5)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    # Test with edge cases
    assert _log_binom_pmf_numba(10, 0, 0.5) == 10 * np.log(0.5), "log(P(X=0)) should be 10*log(0.5)"
    assert _log_binom_pmf_numba(10, 10, 0.5) == 10 * np.log(0.5), "log(P(X=10)) should be 10*log(0.5)"


@pytest.mark.skipif(not has_numba_imports, reason="Numba-specific functions not available")
@pytest.mark.fast
def test_logsumexp_numba():
    """Test _logsumexp_numba with various inputs."""
    # Test with valid inputs
    log_values = np.array([-1.0, -2.0, -3.0])
    result = _logsumexp_numba(log_values)
    expected = np.log(np.exp(-1.0) + np.exp(-2.0) + np.exp(-3.0))
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    # Test with edge case: single value
    assert _logsumexp_numba(np.array([-1.0])) == -1.0, "logsumexp of single value should be the value itself"
    
    # Test with edge case: empty array
    assert _logsumexp_numba(np.array([])) == float('-inf'), "logsumexp of empty array should be -inf"
    
    # Test with edge case: array with -inf
    log_values_with_inf = np.array([-1.0, float('-inf'), -3.0])
    result_with_inf = _logsumexp_numba(log_values_with_inf)
    expected_with_inf = np.log(np.exp(-1.0) + np.exp(-3.0))
    assert abs(result_with_inf - expected_with_inf) < 1e-10, f"Expected {expected_with_inf}, got {result_with_inf}"


# Tests for p-value calculation
@pytest.mark.fast
def test_log_pvalue_barnard_basic():
    """Test basic functionality of _log_pvalue_barnard."""
    # Test with very small inputs and minimal grid size
    result = _log_pvalue_barnard(a=2, c=2, n1=5, n2=5, theta=1.0, grid_size=3)
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    # Allow for small floating-point errors (the result should be close to 0)
    assert result <= 1e-10 or np.isclose(result, 0.0, atol=1e-10), f"Log p-value should be close to 0 or negative, got {result}"


@pytest.mark.fast
def test_log_pvalue_barnard_with_timeout():
    """Test _log_pvalue_barnard with timeout."""
    # Test with timeout that won't be reached - use very small inputs
    start_time = time.time()
    result = _log_pvalue_barnard(
        a=2, c=2, n1=5, n2=5, theta=1.0, grid_size=3,
        start_time=start_time, timeout=5.0
    )
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    # Allow for small floating-point errors (the result should be close to 0)
    assert result <= 1e-10 or np.isclose(result, 0.0, atol=1e-10), f"Log p-value should be close to 0 or negative, got {result}"
    
    # Test with timeout that will be reached immediately
    start_time = time.time() - 6.0  # 6 seconds ago
    result = _log_pvalue_barnard(
        a=2, c=2, n1=5, n2=5, theta=1.0, grid_size=3,
        start_time=start_time, timeout=5.0
    )
    
    # Check that the result is None or a conservative value due to timeout
    # The function may return None or 0.0 depending on the implementation
    assert result is None or (isinstance(result, (int, float)) and np.isclose(result, 0.0, atol=1e-10)), \
        f"Result should be None or close to 0.0 (log(1.0)) due to timeout, got {result}"


@pytest.mark.fast
def test_log_pvalue_barnard_with_progress_callback():
    """Test _log_pvalue_barnard with progress callback."""
    # Create a mock progress callback
    mock_callback = MagicMock()
    
    # Test with progress callback - use very small inputs
    result = _log_pvalue_barnard(
        a=2, c=2, n1=5, n2=5, theta=1.0, grid_size=3,
        progress_callback=mock_callback
    )
    
    # Check that the callback was called at least once
    assert mock_callback.call_count > 0, "Progress callback should be called at least once"
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    # Allow for small floating-point errors (the result should be close to 0)
    assert result <= 1e-10 or np.isclose(result, 0.0, atol=1e-10), f"Log p-value should be close to 0 or negative, got {result}"


@pytest.mark.fast
def test_log_pvalue_barnard_with_custom_grid():
    """Test _log_pvalue_barnard with custom grid."""
    # Create a minimal custom grid
    custom_grid = [0.2, 0.5, 0.8]
    
    # Test with custom grid - use very small inputs
    result = _log_pvalue_barnard(
        a=2, c=2, n1=5, n2=5, theta=1.0, grid_size=3,
        p1_grid_override=custom_grid
    )
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    assert result <= 0, "Log p-value should be non-positive"


# Tests for NumPy and Numba paths
@pytest.mark.fast
def test_process_grid_point_numpy_path():
    """Test _process_grid_point with NumPy path."""
    # Ensure NumPy is available
    with patch('exactcis.methods.unconditional.has_numpy', True):
        # Test with very small inputs to avoid timeout
        result = _process_grid_point((0.5, 2, 2, 5, 5, 1.0))
        
        # Check that the result is a valid log p-value
        assert result is not None, "Result should not be None"
        assert result <= 0, "Log p-value should be non-positive"


@pytest.mark.skipif(not has_numba, reason="Numba not available")
@pytest.mark.fast
def test_process_grid_point_numba_path():
    """Test _process_grid_point with Numba path."""
    # Ensure NumPy and Numba are available
    with patch('exactcis.methods.unconditional.has_numpy', True):
        with patch('exactcis.methods.unconditional.has_numba', True):
            # Force Numba path with a small problem by mocking the threshold check
            with patch('exactcis.methods.unconditional.n1', 1000):  # Mock large n1
                with patch('exactcis.methods.unconditional.n2', 1000):  # Mock large n2
                    result = _process_grid_point((0.5, 2, 2, 5, 5, 1.0))
                    
                    # Check that the result is a valid log p-value
                    assert result is not None, "Result should not be None"
                    assert result <= 0, "Log p-value should be non-positive"


@pytest.mark.skipif(not has_numba_imports, reason="Numba-specific functions not available")
@pytest.mark.fast
def test_process_grid_point_numba_core():
    """Test _process_grid_point_numba_core directly."""
    # Test with very small inputs to avoid timeout
    result = _process_grid_point_numba_core(
        p1=0.5, a=2, c=2, n1=5, n2=5, theta=1.0,
        log_p_obs=_log_binom_pmf(5, 2, 0.5) + _log_binom_pmf(5, 2, 0.5)
    )
    
    # Check that the result is a valid log p-value
    assert result is not None, "Result should not be None"
    assert result <= 0, "Log p-value should be non-positive"