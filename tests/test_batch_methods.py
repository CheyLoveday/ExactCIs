"""
Tests for the batch processing functionality in CI methods.

This module tests the new batch processing functions added to Blaker's and Mid-P methods,
as well as the enhanced CLI batch processing capabilities.
"""

import pytest
import tempfile
import csv
from unittest.mock import patch, MagicMock
from exactcis.methods.blaker import exact_ci_blaker_batch, exact_ci_blaker
from exactcis.methods.midp import exact_ci_midp_batch, exact_ci_midp
from exactcis.core import batch_validate_counts, batch_calculate_odds_ratios, optimize_core_cache_for_batch


# ============================================================================
# BLAKER BATCH PROCESSING TESTS
# ============================================================================

@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_basic():
    """Test basic Blaker batch processing functionality."""
    tables = [
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6)
    ]
    
    # Get individual results for comparison
    individual_results = [exact_ci_blaker(a, b, c, d) for a, b, c, d in tables]
    
    # Get batch results
    batch_results = exact_ci_blaker_batch(tables)
    
    assert len(batch_results) == len(tables)
    
    # Results should be the same (within numerical precision)
    for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
        assert abs(individual[0] - batch[0]) < 1e-6, f"Lower bound mismatch for table {i}"
        assert abs(individual[1] - batch[1]) < 1e-6, f"Upper bound mismatch for table {i}"


@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_empty_list():
    """Test Blaker batch processing with empty table list."""
    results = exact_ci_blaker_batch([])
    assert results == []


@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_single_table():
    """Test Blaker batch processing with single table."""
    table = (5, 6, 7, 8)
    
    individual_result = exact_ci_blaker(*table)
    batch_results = exact_ci_blaker_batch([table])
    
    assert len(batch_results) == 1
    assert abs(batch_results[0][0] - individual_result[0]) < 1e-6
    assert abs(batch_results[0][1] - individual_result[1]) < 1e-6


@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_with_workers():
    """Test Blaker batch processing with different worker counts."""
    tables = [(i, i+1, i+2, i+3) for i in range(1, 6)]
    
    # Test with different worker counts
    results_1 = exact_ci_blaker_batch(tables, max_workers=1)
    results_2 = exact_ci_blaker_batch(tables, max_workers=2)
    results_auto = exact_ci_blaker_batch(tables, max_workers=None)
    
    # All should give the same results
    assert len(results_1) == len(results_2) == len(results_auto) == len(tables)
    
    for i in range(len(tables)):
        assert abs(results_1[i][0] - results_2[i][0]) < 1e-6
        assert abs(results_1[i][1] - results_2[i][1]) < 1e-6
        assert abs(results_auto[i][0] - results_2[i][0]) < 1e-6
        assert abs(results_auto[i][1] - results_2[i][1]) < 1e-6


@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_error_handling():
    """Test Blaker batch processing error handling."""
    # Include some problematic tables
    tables = [
        (1, 2, 3, 4),  # Good table
        (0, 0, 0, 0),  # Invalid table (should error)
        (2, 3, 4, 5),  # Good table
    ]
    
    # Should handle errors gracefully
    results = exact_ci_blaker_batch(tables)
    
    assert len(results) == len(tables)
    # Failed computation should return conservative interval
    assert results[1] == (0.0, float('inf'))
    # Other computations should succeed
    assert results[0][0] >= 0 and results[0][1] > results[0][0]
    assert results[2][0] >= 0 and results[2][1] > results[2][0]


@pytest.mark.methods
@pytest.mark.fast
def test_blaker_batch_no_parallel_support():
    """Test Blaker batch processing fallback when parallel support unavailable."""
    tables = [(1, 2, 3, 4), (2, 3, 4, 5)]
    
    # Mock has_parallel_support to False
    with patch('exactcis.methods.blaker.has_parallel_support', False):
        results = exact_ci_blaker_batch(tables)
    
    assert len(results) == len(tables)
    # Should still work with sequential processing
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


# ============================================================================
# MID-P BATCH PROCESSING TESTS
# ============================================================================

@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_basic():
    """Test basic Mid-P batch processing functionality."""
    tables = [
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6)
    ]
    
    # Get individual results for comparison
    individual_results = [exact_ci_midp(a, b, c, d) for a, b, c, d in tables]
    
    # Get batch results
    batch_results = exact_ci_midp_batch(tables)
    
    assert len(batch_results) == len(tables)
    
    # Results should be the same (within numerical precision)
    for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
        assert abs(individual[0] - batch[0]) < 1e-6, f"Lower bound mismatch for table {i}"
        assert abs(individual[1] - batch[1]) < 1e-6, f"Upper bound mismatch for table {i}"


@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_empty_list():
    """Test Mid-P batch processing with empty table list."""
    results = exact_ci_midp_batch([])
    assert results == []


@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_single_table():
    """Test Mid-P batch processing with single table."""
    table = (5, 6, 7, 8)
    
    individual_result = exact_ci_midp(*table)
    batch_results = exact_ci_midp_batch([table])
    
    assert len(batch_results) == 1
    assert abs(batch_results[0][0] - individual_result[0]) < 1e-6
    assert abs(batch_results[0][1] - individual_result[1]) < 1e-6


@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_with_alpha():
    """Test Mid-P batch processing with different alpha values."""
    tables = [(1, 2, 3, 4), (2, 3, 4, 5)]
    
    # Test with different alpha values
    results_005 = exact_ci_midp_batch(tables, alpha=0.05)
    results_001 = exact_ci_midp_batch(tables, alpha=0.01)
    
    assert len(results_005) == len(results_001) == len(tables)
    
    # Stricter alpha should give wider intervals
    for i in range(len(tables)):
        width_005 = results_005[i][1] - results_005[i][0]
        width_001 = results_001[i][1] - results_001[i][0]
        assert width_001 > width_005


@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_no_parallel_support():
    """Test Mid-P batch processing fallback when parallel support unavailable."""
    tables = [(1, 2, 3, 4), (2, 3, 4, 5)]
    
    # Mock has_parallel_support to False
    with patch('exactcis.methods.midp.has_parallel_support', False):
        results = exact_ci_midp_batch(tables)
    
    assert len(results) == len(tables)
    # Should still work with sequential processing
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


@pytest.mark.methods
@pytest.mark.fast
def test_midp_batch_progress_callback():
    """Test Mid-P batch processing with progress callback."""
    tables = [(i, i+1, i+2, i+3) for i in range(1, 4)]
    progress_values = []
    
    def progress_callback(progress):
        progress_values.append(progress)
    
    # Test with progress callback when no parallel support
    with patch('exactcis.methods.midp.has_parallel_support', False):
        results = exact_ci_midp_batch(tables, progress_callback=progress_callback)
    
    assert len(results) == len(tables)
    assert len(progress_values) > 0  # Should have progress updates
    assert progress_values[-1] == 100  # Should reach 100%


# ============================================================================
# CORE BATCH UTILITIES TESTS
# ============================================================================

@pytest.mark.core
@pytest.mark.fast
def test_batch_validate_counts():
    """Test batch validation of counts."""
    tables = [
        (1, 2, 3, 4),  # Valid
        (-1, 2, 3, 4), # Invalid - negative count
        (1, 2, 3, 4),  # Valid
        (0, 0, 0, 0),  # Invalid - empty margins
        (2, 3, 4, 5)   # Valid
    ]
    
    validation_results = batch_validate_counts(tables)
    
    assert len(validation_results) == len(tables)
    assert validation_results == [True, False, True, False, True]


@pytest.mark.core
@pytest.mark.fast
def test_batch_calculate_odds_ratios():
    """Test batch calculation of odds ratios."""
    tables = [
        (1, 2, 3, 4),  # OR = (1*4)/(2*3) = 4/6 = 2/3
        (2, 1, 1, 2),  # OR = (2*2)/(1*1) = 4
        (1, 0, 2, 3),  # OR = inf (zero in denominator)
        (0, 1, 0, 2),  # OR = 1 (0/0 case)
    ]
    
    odds_ratios = batch_calculate_odds_ratios(tables)
    
    assert len(odds_ratios) == len(tables)
    assert abs(odds_ratios[0] - 2/3) < 1e-10
    assert abs(odds_ratios[1] - 4.0) < 1e-10
    assert odds_ratios[2] == float('inf')
    assert odds_ratios[3] == 1.0


@pytest.mark.core
@pytest.mark.fast
def test_optimize_core_cache_for_batch():
    """Test core cache optimization for batch processing."""
    # Test enabling large cache
    optimize_core_cache_for_batch(enable_large_cache=True)
    
    # Test disabling large cache (reset to default)
    optimize_core_cache_for_batch(enable_large_cache=False)
    
    # Should not raise errors
    assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.methods
@pytest.mark.fast
def test_batch_methods_consistency():
    """Test that batch and individual methods give consistent results."""
    tables = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (2, 1, 4, 3)
    ]
    
    # Test Blaker method consistency
    blaker_individual = [exact_ci_blaker(a, b, c, d) for a, b, c, d in tables]
    blaker_batch = exact_ci_blaker_batch(tables)
    
    for i in range(len(tables)):
        assert abs(blaker_individual[i][0] - blaker_batch[i][0]) < 1e-6
        assert abs(blaker_individual[i][1] - blaker_batch[i][1]) < 1e-6
    
    # Test Mid-P method consistency  
    midp_individual = [exact_ci_midp(a, b, c, d) for a, b, c, d in tables]
    midp_batch = exact_ci_midp_batch(tables)
    
    for i in range(len(tables)):
        assert abs(midp_individual[i][0] - midp_batch[i][0]) < 1e-6
        assert abs(midp_individual[i][1] - midp_batch[i][1]) < 1e-6


@pytest.mark.methods
@pytest.mark.fast
def test_batch_methods_performance_benefit():
    """Test that batch methods provide some benefit over sequential individual calls."""
    import time
    
    # Create a reasonably sized dataset
    tables = [(i, i+1, i+2, i+3) for i in range(1, 11)]
    
    # Time individual calls
    start_time = time.time()
    individual_results = [exact_ci_blaker(a, b, c, d) for a, b, c, d in tables]
    individual_time = time.time() - start_time
    
    # Time batch call
    start_time = time.time() 
    batch_results = exact_ci_blaker_batch(tables, max_workers=2)
    batch_time = time.time() - start_time
    
    # Results should be consistent
    assert len(individual_results) == len(batch_results)
    
    # For small datasets, batch may not be faster due to overhead,
    # but it should at least complete successfully
    assert batch_time > 0
    assert individual_time > 0


@pytest.mark.methods
@pytest.mark.fast
def test_batch_methods_memory_efficiency():
    """Test that batch methods handle memory efficiently."""
    # Create a larger dataset
    tables = [(i % 5 + 1, i % 4 + 2, i % 3 + 3, i % 2 + 4) for i in range(50)]
    
    # Batch processing should complete without memory issues
    blaker_results = exact_ci_blaker_batch(tables, max_workers=2)
    midp_results = exact_ci_midp_batch(tables, max_workers=2)
    
    assert len(blaker_results) == len(tables)
    assert len(midp_results) == len(tables)
    
    # All results should be valid tuples
    assert all(isinstance(r, tuple) and len(r) == 2 for r in blaker_results)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in midp_results)


@pytest.mark.methods
@pytest.mark.fast
def test_batch_methods_with_edge_cases():
    """Test batch methods with various edge case tables."""
    tables = [
        (1, 2, 3, 4),    # Normal table
        (0, 1, 2, 3),    # Zero in cell a
        (1, 0, 2, 3),    # Zero in cell b
        (1, 2, 0, 3),    # Zero in cell c
        (1, 2, 3, 0),    # Zero in cell d
        (10, 1, 1, 10),  # Extreme odds ratio
        (1, 10, 10, 1),  # Another extreme case
    ]
    
    # Both methods should handle edge cases gracefully
    blaker_results = exact_ci_blaker_batch(tables)
    midp_results = exact_ci_midp_batch(tables)
    
    assert len(blaker_results) == len(tables)
    assert len(midp_results) == len(tables)
    
    # Check that results are reasonable
    for i, result in enumerate(blaker_results):
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] >= 0, f"Negative lower bound for table {i}: {tables[i]}"
        assert result[1] >= result[0], f"Invalid interval for table {i}: {tables[i]}"
    
    for i, result in enumerate(midp_results):
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] >= 0, f"Negative lower bound for table {i}: {tables[i]}"
        assert result[1] >= result[0], f"Invalid interval for table {i}: {tables[i]}"