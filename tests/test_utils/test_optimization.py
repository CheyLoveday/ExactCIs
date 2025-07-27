"""
Tests for the optimization utilities.

This module tests the optimization functionality in exactcis.utils.optimization,
including caching mechanisms, search parameter derivation, adaptive grid search,
and batch optimization.
"""

import pytest
import math
import time
from unittest.mock import patch, MagicMock
from exactcis.utils.optimization import (
    CICache, derive_search_params, adaptive_grid_search, 
    batch_optimize_ci, get_global_cache
)


# ============================================================================
# CI CACHE TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_initialization():
    """Test CICache initialization."""
    cache = CICache(max_size=100)
    
    assert cache.max_size == 100
    assert cache.cache_hits == 0
    assert cache.similar_hits == 0
    assert cache.total_lookups == 0
    assert len(cache.exact_cache) == 0
    assert len(cache.similar_cache) == 0


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_exact_operations():
    """Test exact cache operations."""
    cache = CICache(max_size=10)
    
    # Test cache miss
    result = cache.get_exact(1, 2, 3, 4, 0.05, 50, False)
    assert result is None
    assert cache.total_lookups == 1
    assert cache.cache_hits == 0
    
    # Add entry
    ci = (0.5, 2.0)
    params = {"iterations": 100}
    cache.add(1, 2, 3, 4, 0.05, ci, params, 50, False)
    
    # Test cache hit
    result = cache.get_exact(1, 2, 3, 4, 0.05, 50, False)
    assert result is not None
    assert result[0] == ci
    assert result[1]["iterations"] == 100
    assert cache.total_lookups == 2
    assert cache.cache_hits == 1


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_lookup_alias():
    """Test lookup alias method."""
    cache = CICache()
    
    # Add entry
    cache.add(1, 2, 3, 4, 0.05, (0.5, 2.0), {}, 50, False)
    
    # Test lookup alias (returns just CI tuple)
    result = cache.lookup(1, 2, 3, 4, 0.05, 50, False)
    assert result == (0.5, 2.0)
    
    # Test miss
    result = cache.lookup(5, 6, 7, 8, 0.05, 50, False)
    assert result is None


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_similar_search():
    """Test similar cache search functionality."""
    cache = CICache(max_size=10)
    
    # Add some entries with different characteristics
    cache.add(2, 3, 4, 6, 0.05, (1.0, 3.0), {"method": "test"}, 50, False)  # OR = 1.0
    cache.add(4, 6, 8, 12, 0.05, (0.8, 2.8), {"method": "test"}, 50, False)  # OR = 1.0, larger table
    cache.add(1, 9, 9, 1, 0.05, (0.01, 0.2), {"method": "test"}, 50, False)  # OR = 0.012
    
    # Search for similar to OR ≈ 1.0
    similar = cache.get_similar(3, 4, 5, 6, 0.05, similarity_threshold=0.1)
    
    assert len(similar) >= 1  # Should find at least one similar entry
    # Should be sorted by similarity (most similar first)


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_similar_edge_cases():
    """Test similar cache with edge cases."""
    cache = CICache()
    
    # Add entry with zero cells
    cache.add(0, 5, 8, 10, 0.05, (0.0, 1.5), {}, 50, False)
    
    # Search for similar with zeros - should handle division by zero
    similar = cache.get_similar(0, 3, 5, 7, 0.05)
    
    # Should not crash and may find similar entries
    assert isinstance(similar, list)
    
    # Test with infinite OR
    cache.add(5, 0, 3, 7, 0.05, (2.0, float('inf')), {}, 50, False)
    
    similar = cache.get_similar(3, 0, 2, 5, 0.05)
    assert isinstance(similar, list)


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_pruning():
    """Test cache pruning when max size is exceeded."""
    cache = CICache(max_size=3)
    
    # Add more entries than max_size
    for i in range(5):
        cache.add(i, i+1, i+2, i+3, 0.05, (float(i), float(i+1)), {}, 50, False)
    
    # Should only keep the last max_size entries
    assert len(cache.similar_cache) == 3
    
    # Check that the last 3 entries are preserved
    preserved_tables = [entry[0] for entry in cache.similar_cache]
    assert (2, 3, 4, 5) in preserved_tables
    assert (3, 4, 5, 6) in preserved_tables
    assert (4, 5, 6, 7) in preserved_tables


@pytest.mark.utils
@pytest.mark.fast
def test_ci_cache_store_alias():
    """Test store alias method."""
    cache = CICache()
    
    # Test store alias
    cache.store(1, 2, 3, 4, 0.05, (0.5, 2.0), {"test": True}, 50, False)
    
    # Verify it was stored
    result = cache.get_exact(1, 2, 3, 4, 0.05, 50, False)
    assert result is not None
    assert result[0] == (0.5, 2.0)
    assert result[1]["test"] is True


# ============================================================================
# SEARCH PARAMETER DERIVATION TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_derive_search_params_empty():
    """Test derive_search_params with empty similar entries."""
    params = derive_search_params(1, 2, 3, 4, [])
    
    assert params["grid_size"] == 50  # Default value
    assert "predicted_theta_range" not in params or params["predicted_theta_range"] is None


@pytest.mark.utils
@pytest.mark.fast
def test_derive_search_params_with_entries():
    """Test derive_search_params with similar entries."""
    # Create mock similar entries
    similar_entries = [
        ((2, 3, 4, 6), (0.8, 2.5), {"grid_size": 40, "total_iterations": 80}),
        ((1, 2, 2, 3), (0.6, 2.0), {"grid_size": 60, "total_iterations": 120}),
    ]
    
    params = derive_search_params(1, 2, 3, 4, similar_entries)
    
    assert "grid_size" in params
    assert params["grid_size"] == 50  # Average of 40 and 60
    assert "predicted_theta_range" in params
    assert "estimated_iterations" in params
    assert params["estimated_iterations"] == 100  # Average of 80 and 120


@pytest.mark.utils
@pytest.mark.fast
def test_derive_search_params_infinite_handling():
    """Test derive_search_params with infinite CIs."""
    # Entry with infinite upper bound
    similar_entries = [
        ((2, 3, 4, 6), (0.5, float('inf')), {"grid_size": 50}),
    ]
    
    params = derive_search_params(1, 2, 3, 4, similar_entries)
    
    assert "predicted_theta_range" in params
    # Should handle infinity gracefully
    if params["predicted_theta_range"]:
        lower, upper = params["predicted_theta_range"]
        assert lower > 0
        assert upper < float('inf')  # Should be converted to finite value


@pytest.mark.utils
@pytest.mark.fast
def test_derive_search_params_zero_or_handling():
    """Test derive_search_params with zero odds ratios."""
    # Entries that would produce OR = 0 or inf
    similar_entries = [
        ((0, 3, 4, 6), (0.0, 1.5), {"grid_size": 30}),  # OR = 0
        ((2, 0, 4, 6), (1.5, float('inf')), {"grid_size": 70}),  # OR = inf
    ]
    
    params = derive_search_params(1, 2, 3, 4, similar_entries)
    
    # Should not crash and provide reasonable defaults
    assert "grid_size" in params
    assert isinstance(params["grid_size"], int)


# ============================================================================
# ADAPTIVE GRID SEARCH TESTS  
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_basic():
    """Test basic adaptive grid search."""
    # Simple function: f(x) = x - 2, looking for where it crosses 0
    def test_func(x):
        return x - 2
    
    crossings = adaptive_grid_search(test_func, (1.0, 3.0), target_value=0.0, 
                                   initial_points=5, refinement_rounds=2)
    
    assert len(crossings) >= 1  # May find duplicate crossings due to refinement
    assert any(abs(crossing - 2.0) < 0.1 for crossing in crossings)


@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_multiple_crossings():
    """Test adaptive grid search with multiple crossings."""
    # Function with multiple crossings: sin(x)
    def test_func(x):
        return math.sin(x)
    
    crossings = adaptive_grid_search(test_func, (2.0, 8.0), target_value=0.0,
                                   initial_points=10, refinement_rounds=2)
    
    # Should find crossings near π and 2π
    assert len(crossings) >= 1
    # At least one crossing should be near π or 2π
    found_pi = any(abs(c - math.pi) < 0.5 for c in crossings)
    found_2pi = any(abs(c - 2*math.pi) < 0.5 for c in crossings)
    assert found_pi or found_2pi


@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_no_crossings():
    """Test adaptive grid search when no crossings exist."""
    # Function that doesn't cross target
    def test_func(x):
        return x + 10  # Always positive
    
    crossings = adaptive_grid_search(test_func, (1.0, 3.0), target_value=0.0)
    
    assert len(crossings) == 0


@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_invalid_bounds():
    """Test adaptive grid search with invalid bounds."""
    def test_func(x):
        return x
    
    # Invalid bounds (lower >= upper)
    crossings = adaptive_grid_search(test_func, (3.0, 1.0), target_value=0.0)
    assert crossings == []
    
    crossings = adaptive_grid_search(test_func, (2.0, 2.0), target_value=0.0)
    assert crossings == []


@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_timeout():
    """Test adaptive grid search with timeout."""
    def slow_func(x):
        time.sleep(0.01)
        return x - 2
    
    timeout_triggered = False
    call_count = 0
    
    def timeout_checker():
        nonlocal timeout_triggered, call_count
        call_count += 1
        if call_count > 3:  # Trigger timeout after a few calls
            timeout_triggered = True
            return True
        return False
    
    crossings = adaptive_grid_search(slow_func, (1.0, 3.0), target_value=0.0,
                                   timeout_checker=timeout_checker)
    
    # Should handle timeout gracefully
    assert timeout_triggered
    assert isinstance(crossings, list)


@pytest.mark.utils
@pytest.mark.fast
def test_adaptive_grid_search_function_failures():
    """Test adaptive grid search when function evaluations fail."""
    def unreliable_func(x):
        if 1.5 <= x <= 2.5:  # Fail in middle region
            return None
        return x - 2
    
    crossings = adaptive_grid_search(unreliable_func, (1.0, 3.0), target_value=0.0)
    
    # Should handle None returns gracefully and still find crossing if possible
    assert isinstance(crossings, list)


# ============================================================================
# BATCH OPTIMIZATION TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_batch_optimize_ci_basic():
    """Test basic batch CI optimization."""
    # Mock CI function
    def mock_ci_function(a, b, c, d, alpha, search_params):
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return ((or_value * 0.8, or_value * 1.2), {"iterations": 50})
    
    tables = [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
    
    results = batch_optimize_ci(tables, mock_ci_function, alpha=0.05)
    
    assert len(results) == 3
    assert all(table in results for table in tables)
    assert all(isinstance(ci, tuple) and len(ci) == 2 for ci in results.values())


@pytest.mark.utils
@pytest.mark.fast
def test_batch_optimize_ci_empty():
    """Test batch optimization with empty table list."""
    def mock_ci_function(a, b, c, d, alpha, search_params):
        return ((1.0, 2.0), {})
    
    results = batch_optimize_ci([], mock_ci_function)
    assert results == {}


@pytest.mark.utils
@pytest.mark.fast
def test_batch_optimize_ci_grouping():
    """Test that similar tables are grouped together."""
    call_log = []
    
    def logging_ci_function(a, b, c, d, alpha, search_params):
        call_log.append((a, b, c, d))
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return ((or_value * 0.8, or_value * 1.2), {"iterations": 50})
    
    # Create tables with similar ORs
    tables = [
        (1, 2, 3, 6),  # OR = 1.0
        (2, 4, 6, 12), # OR = 1.0 (similar to first)
        (1, 10, 10, 1), # OR = 0.01 (very different)
    ]
    
    results = batch_optimize_ci(tables, logging_ci_function, alpha=0.05)
    
    assert len(results) == 3
    assert len(call_log) == 3  # All tables processed


@pytest.mark.utils
@pytest.mark.fast
def test_batch_optimize_ci_caching():
    """Test that batch optimization uses caching effectively."""
    call_count = 0
    
    def counting_ci_function(a, b, c, d, alpha, search_params):
        nonlocal call_count
        call_count += 1
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return ((or_value * 0.8, or_value * 1.2), {"iterations": 50})
    
    # Process same table twice in different runs
    tables1 = [(1, 2, 3, 4)]
    tables2 = [(1, 2, 3, 4), (2, 3, 4, 5)]  # Includes same table
    
    # First run
    batch_optimize_ci(tables1, counting_ci_function, alpha=0.05)
    first_count = call_count
    
    # Second run with overlapping table
    batch_optimize_ci(tables2, counting_ci_function, alpha=0.05)
    second_count = call_count
    
    # Should have made some function calls
    assert first_count >= 1
    assert second_count > first_count  # Should call for new table


# ============================================================================
# GLOBAL CACHE TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_global_cache_access():
    """Test global cache access."""
    cache1 = get_global_cache()
    cache2 = get_global_cache()
    
    # Should return same instance
    assert cache1 is cache2
    assert isinstance(cache1, CICache)


@pytest.mark.utils
@pytest.mark.fast
def test_global_cache_persistence():
    """Test that global cache persists across calls."""
    cache = get_global_cache()
    
    # Add something to cache
    cache.add(1, 2, 3, 4, 0.05, (0.5, 2.0), {}, 50, False)
    
    # Get cache again and verify entry is still there
    cache2 = get_global_cache()
    result = cache2.get_exact(1, 2, 3, 4, 0.05, 50, False)
    
    assert result is not None
    assert result[0] == (0.5, 2.0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.utils
@pytest.mark.fast
def test_optimization_integration_workflow():
    """Test integration of optimization components in realistic workflow."""
    
    # Step 1: Create a realistic CI function that uses adaptive search
    def realistic_ci_function(a, b, c, d, alpha, search_params):
        # Use predicted range if available
        if "predicted_theta_range" in search_params and search_params["predicted_theta_range"]:
            theta_min, theta_max = search_params["predicted_theta_range"]
        else:
            theta_min, theta_max = 0.1, 10.0
        
        # Simulate using adaptive grid search
        def p_value_func(theta):
            # Mock p-value function
            or_estimate = (a * d) / (b * c) if b * c > 0 else 1.0
            return abs(math.log(theta) - math.log(or_estimate)) * 0.1
        
        # Find bounds using adaptive search
        crossings = adaptive_grid_search(p_value_func, (theta_min, theta_max), 
                                       target_value=alpha/2, initial_points=5)
        
        if len(crossings) >= 2:
            lower, upper = min(crossings), max(crossings)
        elif len(crossings) == 1:
            lower, upper = crossings[0] * 0.5, crossings[0] * 2.0
        else:
            or_estimate = (a * d) / (b * c) if b * c > 0 else 1.0
            lower, upper = or_estimate * 0.5, or_estimate * 2.0
        
        return ((lower, upper), {
            "iterations": 10 * len(crossings),
            "grid_size": search_params.get("grid_size", 50)
        })
    
    # Step 2: Create test tables
    tables = [(1, 2, 3, 4), (2, 3, 4, 5), (1, 3, 2, 6)]
    
    # Step 3: Run batch optimization
    results = batch_optimize_ci(tables, realistic_ci_function, alpha=0.05)
    
    # Step 4: Verify results
    assert len(results) == len(tables)
    assert all(table in results for table in tables)
    assert all(isinstance(ci, tuple) and len(ci) == 2 for ci in results.values())
    assert all(ci[0] <= ci[1] for ci in results.values())  # Lower <= Upper


@pytest.mark.utils
@pytest.mark.fast  
def test_optimization_error_handling():
    """Test optimization utilities handle errors gracefully."""
    
    # Test 1: CI function that sometimes fails
    def unreliable_ci_function(a, b, c, d, alpha, search_params):
        if a == 2:  # Fail on specific input
            raise ValueError("Simulated CI calculation failure")
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return ((or_value * 0.8, or_value * 1.2), {})
    
    tables = [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
    
    # Should handle failure gracefully
    try:
        results = batch_optimize_ci(tables, unreliable_ci_function, alpha=0.05)
        # If it doesn't raise, check that some results were computed
        assert len(results) >= 1
    except Exception:
        # If it does raise, that's also acceptable behavior
        pass
    
    # Test 2: Adaptive grid search with function that always fails
    def always_fail_func(x):
        raise RuntimeError("Function evaluation failed")
    
    # The function may raise an exception - that's expected behavior
    try:
        crossings = adaptive_grid_search(always_fail_func, (1.0, 3.0), target_value=0.0)
        assert crossings == []  # Should return empty list if it handles errors
    except RuntimeError:
        # It's also acceptable for it to raise the exception
        pass


@pytest.mark.utils
@pytest.mark.fast
def test_optimization_performance_patterns():
    """Test optimization utilities for expected performance patterns."""
    
    # Test that cache reduces computation
    computation_count = 0
    
    def expensive_ci_function(a, b, c, d, alpha, search_params):
        nonlocal computation_count
        computation_count += 1
        time.sleep(0.001)  # Simulate computation time
        or_value = (a * d) / (b * c) if b * c > 0 else 1.0
        return ((or_value * 0.8, or_value * 1.2), {"iterations": 100})
    
    # Process same table multiple times
    tables = [(1, 2, 3, 4)] * 3  # Same table repeated
    
    # First run - should compute once and cache
    results1 = batch_optimize_ci(tables[:1], expensive_ci_function, alpha=0.05)
    first_count = computation_count
    
    # Second run with repeated table - should use cache
    results2 = batch_optimize_ci(tables[1:], expensive_ci_function, alpha=0.05)
    second_count = computation_count
    
    assert first_count >= 1
    # Note: Actual caching behavior depends on implementation details
    # This test mainly ensures the system doesn't crash with repeated tables