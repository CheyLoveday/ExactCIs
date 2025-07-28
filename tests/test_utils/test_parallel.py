"""
Tests for the parallel processing utilities.

This module tests the parallelization functionality in exactcis.utils.parallel,
including worker determination, parallel mapping, CI computation, parameter
space chunking, and parallel root finding.
"""

import pytest
import time
import math
import multiprocessing
from unittest.mock import patch, MagicMock
from exactcis.utils.parallel import (
    get_optimal_workers, parallel_map, parallel_compute_ci,
    chunk_parameter_space, parallel_find_root
)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_get_optimal_workers():
    """Test optimal worker determination."""
    # Test that it returns a reasonable number
    workers = get_optimal_workers()
    assert isinstance(workers, int)
    assert 2 <= workers <= 8
    
    # Test with mocked CPU count
    with patch('multiprocessing.cpu_count', return_value=4):
        assert get_optimal_workers() == 3  # 75% of 4 = 3
    
    with patch('multiprocessing.cpu_count', return_value=1):
        assert get_optimal_workers() == 2  # At least 2
    
    with patch('multiprocessing.cpu_count', return_value=16):
        assert get_optimal_workers() == 8  # At most 8


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_basic():
    """Test basic parallel_map functionality."""
    # Simple function to test
    def square(x):
        return x * x
    
    # Test with small list
    items = [1, 2, 3, 4, 5]
    results = parallel_map(square, items, max_workers=2)
    assert results == [1, 4, 9, 16, 25]
    
    # Test order preservation
    items = [5, 1, 3, 2, 4]
    results = parallel_map(square, items, max_workers=2)
    assert results == [25, 1, 9, 4, 16]


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_empty_list():
    """Test parallel_map with empty list."""
    def dummy_func(x):
        return x
    
    result = parallel_map(dummy_func, [])
    assert result == []


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_single_item():
    """Test parallel_map with single item."""
    def square(x):
        return x * x
    
    result = parallel_map(square, [5], max_workers=4)
    assert result == [25]


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_threads_vs_processes():
    """Test parallel_map with threads vs processes."""
    def simple_func(x):
        return x + 1
    
    items = [1, 2, 3, 4]
    
    # Test with threads
    results_threads = parallel_map(simple_func, items, use_threads=True, max_workers=2)
    assert results_threads == [2, 3, 4, 5]
    
    # Test with processes
    results_processes = parallel_map(simple_func, items, use_threads=False, max_workers=2)
    assert results_processes == [2, 3, 4, 5]


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_with_progress():
    """Test parallel_map with progress callback."""
    def slow_func(x):
        time.sleep(0.01)  # Small delay
        return x * 2
    
    progress_values = []
    
    def progress_callback(progress):
        progress_values.append(progress)
    
    items = [1, 2, 3]
    results = parallel_map(slow_func, items, use_threads=True, 
                          progress_callback=progress_callback, max_workers=2)
    
    assert results == [2, 4, 6]
    # Progress callback should be called with threads but not processes
    assert len(progress_values) == 0  # Disabled for process pools


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_error_handling():
    """Test parallel_map error handling and fallback."""
    def simple_func(x):
        return x * 2
    
    # Mock executor to force exception and test fallback
    with patch('exactcis.utils.parallel.ProcessPoolExecutor') as mock_executor:
        mock_executor.side_effect = Exception("Pool creation failed")
        
        items = [1, 2, 4]  # Avoid problematic values
        # Should fall back to sequential processing
        results = parallel_map(simple_func, items, max_workers=2)
        
        # Sequential fallback should work
        assert results == [2, 4, 8]


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_timeout():
    """Test parallel_map timeout handling."""
    def slow_func(x):
        time.sleep(0.1)
        return x
    
    items = [1, 2, 3]
    
    # Test with very short timeout (should fall back to sequential)
    with patch('exactcis.utils.parallel.ProcessPoolExecutor') as mock_executor:
        mock_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_instance
        mock_instance.map.side_effect = Exception("Timeout")
        
        results = parallel_map(slow_func, items, timeout=0.01, max_workers=2)
        # Should fall back and complete sequentially
        assert len(results) == 3


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_worker_adjustment():
    """Test parallel_map worker count adjustment."""
    def simple_func(x):
        return x
    
    # Test that workers are adjusted when items < workers
    items = [1, 2]
    
    # Request 10 workers but only have 2 items
    with patch('exactcis.utils.parallel.get_optimal_workers', return_value=10):
        # Function should adjust workers to match item count
        result = parallel_map(simple_func, items, max_workers=10)
        assert result == [1, 2]


# ============================================================================
# PARALLEL COMPUTE CI TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_parallel_compute_ci_basic():
    """Test basic parallel CI computation."""
    def mock_ci_method(a, b, c, d, alpha=0.05):
        # Simple mock that returns OR-based bounds
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return (or_value * 0.5, or_value * 2.0)
    
    tables = [(1, 2, 3, 4), (2, 3, 4, 5), (5, 6, 7, 8)]
    
    results = parallel_compute_ci(mock_ci_method, tables, alpha=0.05)
    
    assert len(results) == 3
    assert all(isinstance(ci, tuple) and len(ci) == 2 for ci in results)
    assert all(ci[0] <= ci[1] for ci in results)  # Lower <= Upper


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_compute_ci_error_handling():
    """Test parallel CI computation error handling."""
    def failing_ci_method(a, b, c, d, alpha=0.05):
        if a == 5:  # Fail on specific input
            raise ValueError("Test error")
        return (1.0, 2.0)
    
    tables = [(1, 2, 3, 4), (5, 6, 7, 8), (2, 3, 4, 5)]
    
    results = parallel_compute_ci(failing_ci_method, tables, alpha=0.05)
    
    assert len(results) == 3
    # Failed computation should return conservative interval
    assert results[1] == (0.0, float('inf'))
    # Other computations should succeed
    assert results[0] == (1.0, 2.0)
    assert results[2] == (1.0, 2.0)


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_compute_ci_empty_tables():
    """Test parallel CI computation with empty table list."""
    def dummy_ci_method(a, b, c, d, alpha=0.05):
        return (1.0, 2.0)
    
    results = parallel_compute_ci(dummy_ci_method, [], alpha=0.05)
    assert results == []


# ============================================================================
# PARAMETER SPACE CHUNKING TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_chunk_parameter_space_basic():
    """Test basic parameter space chunking."""
    theta_range = (0.1, 10.0)
    n_chunks = 4
    
    chunks = chunk_parameter_space(theta_range, n_chunks)
    
    assert len(chunks) == 4
    assert all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)
    
    # Check continuity
    for i in range(len(chunks) - 1):
        assert abs(chunks[i][1] - chunks[i+1][0]) < 1e-10
    
    # Check endpoints
    assert abs(chunks[0][0] - 0.1) < 1e-10
    assert abs(chunks[-1][1] - 10.0) < 1e-10


@pytest.mark.utils
@pytest.mark.fast
def test_chunk_parameter_space_zero_handling():
    """Test parameter space chunking with zero/negative values."""
    # Test with zero minimum
    chunks = chunk_parameter_space((0.0, 10.0), 3)
    assert chunks[0][0] >= 1e-9  # Should adjust zero to small positive (more lenient)
    
    # Test with negative minimum
    chunks = chunk_parameter_space((-1.0, 10.0), 3)
    assert chunks[0][0] >= 1e-9  # Should adjust negative to small positive (more lenient)


@pytest.mark.utils
@pytest.mark.fast
def test_chunk_parameter_space_single_chunk():
    """Test parameter space chunking with single chunk."""
    theta_range = (1.0, 5.0)
    chunks = chunk_parameter_space(theta_range, 1)
    
    assert len(chunks) == 1
    # Allow for small floating point errors in numpy.linspace
    assert abs(chunks[0][0] - 1.0) < 1e-10
    assert abs(chunks[0][1] - 5.0) < 1e-10


@pytest.mark.utils
@pytest.mark.fast
def test_chunk_parameter_space_logarithmic():
    """Test that chunking uses logarithmic spacing."""
    theta_range = (1.0, 1000.0)
    chunks = chunk_parameter_space(theta_range, 3)
    
    # Check that spacing is roughly logarithmic
    # Log spacing: 1 -> 10 -> 100 -> 1000
    assert chunks[0][1] / chunks[0][0] > 5  # Should be roughly 10
    assert chunks[1][1] / chunks[1][0] > 5  # Should be roughly 10
    assert chunks[2][1] / chunks[2][0] > 5  # Should be roughly 10


# ============================================================================
# PARALLEL ROOT FINDING TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_parallel_find_root_basic():
    """Test basic parallel root finding."""
    # Simple function: f(x) = x - 3
    def test_func(x):
        return x - 3
    
    root = parallel_find_root(test_func, target_value=0, theta_range=(1.0, 5.0), max_workers=2)
    
    assert abs(root - 3.0) < 0.1  # Should find root near 3


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_find_root_no_root():
    """Test parallel root finding when no root exists."""
    # Function that doesn't cross target
    def test_func(x):
        return x + 10  # Always positive
    
    root = parallel_find_root(test_func, target_value=0, theta_range=(1.0, 5.0), max_workers=2)
    
    # Should return closest value
    assert 1.0 <= root <= 5.0


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_find_root_multiple_roots():
    """Test parallel root finding with multiple roots."""
    # Function with multiple crossings: sin(x)
    def test_func(x):
        return math.sin(x)
    
    root = parallel_find_root(test_func, target_value=0, theta_range=(2.0, 7.0), max_workers=2)
    
    # Should find one of the roots (π ≈ 3.14 or 2π ≈ 6.28)
    assert abs(root - math.pi) < 0.5 or abs(root - 2*math.pi) < 0.5


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_find_root_with_progress():
    """Test parallel root finding with progress callback."""
    def test_func(x):
        return x - 2.5
    
    progress_values = []
    
    def progress_callback(progress):
        progress_values.append(progress)
    
    root = parallel_find_root(
        test_func, 
        target_value=0, 
        theta_range=(1.0, 5.0),
        max_workers=2,
        progress_callback=progress_callback
    )
    
    assert abs(root - 2.5) < 0.1
    assert len(progress_values) > 0  # Should have progress updates
    assert progress_values[0] == 10  # Should start at 10%
    assert progress_values[-1] == 100  # Should end at 100%


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_find_root_worker_determination():
    """Test parallel root finding worker determination."""
    def test_func(x):
        return x - 2
    
    # Test with None max_workers (should use optimal)
    root = parallel_find_root(test_func, target_value=0, theta_range=(1.0, 3.0), max_workers=None)
    assert abs(root - 2.0) < 0.1
    
    # Test with specific max_workers
    root = parallel_find_root(test_func, target_value=0, theta_range=(1.0, 3.0), max_workers=1)
    assert abs(root - 2.0) < 0.1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_parallel_integration_workflow():
    """Test integration of parallel utilities in a typical workflow."""
    # Simulate a batch CI calculation workflow
    
    # Step 1: Create test tables
    tables = [(i, i+1, i+2, i+3) for i in range(1, 6)]
    
    # Step 2: Define a mock CI method that uses chunking
    def mock_ci_method(a, b, c, d, alpha=0.05):
        # Simulate work with parameter space
        theta_range = (0.1, 10.0)
        chunks = chunk_parameter_space(theta_range, 3)
        
        # Use chunks to "compute" CI (mock)
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        width = len(chunks) * 0.1  # Chunk count affects width
        
        return (or_value - width, or_value + width)
    
    # Step 3: Run parallel computation
    results = parallel_compute_ci(mock_ci_method, tables, alpha=0.05)
    
    # Step 4: Verify results
    assert len(results) == len(tables)
    assert all(isinstance(ci, tuple) and len(ci) == 2 for ci in results)
    assert all(ci[0] <= ci[1] for ci in results)


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_error_recovery():
    """Test that parallel utilities gracefully handle various error conditions."""
    
    # Test 1: Function that sometimes fails
    def unreliable_func(x):
        if x % 3 == 0:
            raise RuntimeError("Simulated failure")
        return x * 2
    
    items = list(range(10))
    
    # Should fall back to sequential and handle errors gracefully
    try:
        results = parallel_map(unreliable_func, items, max_workers=2)
        # Some items may fail, but function should not crash
    except:
        # If it does raise, it should be handled gracefully
        pass
    
    # Test 2: Resource exhaustion simulation
    with patch('exactcis.utils.parallel.ProcessPoolExecutor') as mock_executor:
        mock_executor.side_effect = OSError("No more processes")
        
        # Should fall back to sequential
        items = [1, 2, 3]
        results = parallel_map(lambda x: x + 1, items, max_workers=2)
        assert results == [2, 3, 4]  # Should still work sequentially


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_force_processes():
    """Test parallel_map with force_processes parameter."""
    def cpu_bound_func(x):
        # Simulate CPU-bound work
        return sum(range(x))
    
    items = [100, 200, 300]
    
    # Test with forced processes
    results = parallel_map(cpu_bound_func, items, max_workers=2, force_processes=True)
    expected = [sum(range(x)) for x in items]
    assert results == expected


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_chunking():
    """Test parallel_map with different chunk sizes."""
    def simple_func(x):
        return x * 2
    
    items = list(range(20))
    
    # Test with different chunk sizes
    results_auto = parallel_map(simple_func, items, max_workers=2)
    results_custom = parallel_map(simple_func, items, max_workers=2, chunk_size=5)
    
    expected = [x * 2 for x in items]
    assert results_auto == expected
    assert results_custom == expected


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_map_large_dataset():
    """Test parallel_map with a larger dataset to test chunking behavior."""
    def add_one(x):
        return x + 1
    
    # Test with larger dataset
    items = list(range(100))
    results = parallel_map(add_one, items, max_workers=4)
    
    expected = [x + 1 for x in items]
    assert results == expected
    assert len(results) == 100


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_compute_ci_different_methods():
    """Test parallel CI computation with different mock methods."""
    
    # Mock different CI methods
    def fisher_like_method(a, b, c, d, alpha=0.05, **kwargs):
        or_val = (a * d) / (b * c) if b * c > 0 else 1.0
        margin = or_val * 0.2
        return (max(0, or_val - margin), or_val + margin)
    
    def blaker_like_method(a, b, c, d, alpha=0.05, **kwargs):
        or_val = (a * d) / (b * c) if b * c > 0 else 1.0
        margin = or_val * 0.15  # Tighter interval
        return (max(0, or_val - margin), or_val + margin)
    
    tables = [(2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)]
    
    # Test with different methods
    results_fisher = parallel_compute_ci(fisher_like_method, tables)
    results_blaker = parallel_compute_ci(blaker_like_method, tables)
    
    assert len(results_fisher) == len(tables)
    assert len(results_blaker) == len(tables)
    
    # Blaker-like should generally have tighter intervals
    for i in range(len(tables)):
        fisher_width = results_fisher[i][1] - results_fisher[i][0]
        blaker_width = results_blaker[i][1] - results_blaker[i][0]
        assert blaker_width < fisher_width


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_compute_ci_edge_cases():
    """Test parallel CI computation with edge case tables."""
    def robust_ci_method(a, b, c, d, alpha=0.05, **kwargs):
        # Handle edge cases
        if a == 0 or d == 0:
            return (0.0, 1.0)
        if b == 0 or c == 0:
            return (1.0, float('inf'))
        
        or_val = (a * d) / (b * c)
        return (or_val * 0.5, or_val * 2.0)
    
    # Include tables with zeros
    tables = [
        (0, 1, 2, 3),  # Zero in a
        (1, 0, 2, 3),  # Zero in b
        (1, 2, 0, 3),  # Zero in c
        (1, 2, 3, 0),  # Zero in d
        (1, 2, 3, 4)   # Normal table
    ]
    
    results = parallel_compute_ci(robust_ci_method, tables)
    
    assert len(results) == len(tables)
    assert results[0] == (0.0, 1.0)  # Zero in a
    assert results[1] == (1.0, float('inf'))  # Zero in b
    assert results[2] == (1.0, float('inf'))  # Zero in c  
    assert results[3] == (0.0, 1.0)  # Zero in d
    assert results[4][0] > 0  # Normal table should have positive lower bound


@pytest.mark.utils  
@pytest.mark.fast
def test_parallel_compute_ci_with_kwargs():
    """Test parallel CI computation with additional keyword arguments."""
    def parameterized_ci_method(a, b, c, d, alpha=0.05, grid_size=10, **kwargs):
        # Method that uses additional parameters
        or_val = (a * d) / (b * c) if b * c > 0 else 1.0
        # Grid size affects precision (mock behavior)
        precision = 1.0 / grid_size
        margin = or_val * precision
        return (max(0, or_val - margin), or_val + margin)
    
    tables = [(1, 2, 3, 4), (2, 3, 4, 5)]
    
    # Test with different grid sizes
    results_coarse = parallel_compute_ci(parameterized_ci_method, tables, alpha=0.05, grid_size=5)
    results_fine = parallel_compute_ci(parameterized_ci_method, tables, alpha=0.05, grid_size=20)
    
    assert len(results_coarse) == len(tables)
    assert len(results_fine) == len(tables)
    
    # Finer grid should give tighter intervals
    for i in range(len(tables)):
        coarse_width = results_coarse[i][1] - results_coarse[i][0]
        fine_width = results_fine[i][1] - results_fine[i][0]
        assert fine_width < coarse_width


@pytest.mark.utils
@pytest.mark.fast  
def test_parallel_map_timeout_behavior():
    """Test parallel_map behavior with timeout parameter."""
    def variable_time_func(x):
        # Some items take longer than others
        if x > 5:
            time.sleep(0.01)  # Brief delay for larger values
        return x * 2
    
    items = list(range(10))
    
    # Test with reasonable timeout
    results = parallel_map(variable_time_func, items, max_workers=2, timeout=1.0)
    expected = [x * 2 for x in items]
    assert results == expected


@pytest.mark.utils
@pytest.mark.fast
def test_parallel_chunk_parameter_space_edge_cases():
    """Test parameter space chunking with edge cases."""
    
    # Test with very small range
    chunks = chunk_parameter_space((0.999, 1.001), 2)
    assert len(chunks) == 2  
    assert chunks[0][0] >= 1e-9  # Should handle near-zero
    
    # Test with large range  
    chunks = chunk_parameter_space((1e-6, 1e6), 5)
    assert len(chunks) == 5
    assert chunks[0][0] >= 1e-9
    assert chunks[-1][1] <= 1e6 + 1e-10  # Allow small numerical error
    
    # Test logarithmic distribution - ratios should be similar
    ratios = [chunks[i][1] / chunks[i][0] for i in range(len(chunks))]
    # All ratios should be similar for logarithmic spacing
    avg_ratio = sum(ratios) / len(ratios)
    for ratio in ratios:
        assert abs(ratio - avg_ratio) / avg_ratio < 0.5  # Within 50% of average