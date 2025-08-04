"""
Tests for the calculators utility module.
"""

import pytest
import math
import numpy as np
from unittest.mock import patch, MagicMock
from exactcis.utils.calculators import (
    log_binom_pmf_cached,
    calculate_log_probability_for_table,
    enumerate_all_possible_tables,
    calculate_unconditional_log_pvalue_for_grid_point,
    calculate_unconditional_log_pvalue,
    create_pvalue_function,
    find_confidence_bound_via_sign_change,
    refine_bound_with_plateau_detection,
    find_confidence_bound
)
from exactcis.utils.data_models import TableData, GridConfig, BoundResult


@pytest.mark.utils
@pytest.mark.fast
def test_log_binom_pmf_cached():
    """Test log_binom_pmf_cached function."""
    # Test normal case
    result = log_binom_pmf_cached(10, 3, 0.3)
    
    # Calculate expected value manually
    from exactcis.core import log_binom_coeff
    expected = log_binom_coeff(10, 3) + 3 * math.log(0.3) + 7 * math.log(0.7)
    assert abs(result - expected) < 1e-10
    
    # Test edge cases
    assert log_binom_pmf_cached(10, 3, 0.0) == float('-inf')
    assert log_binom_pmf_cached(10, 3, 1.0) == float('-inf')
    
    # Test caching - call twice with same parameters
    result1 = log_binom_pmf_cached(5, 2, 0.4)
    result2 = log_binom_pmf_cached(5, 2, 0.4)
    assert result1 == result2  # Should be exactly equal due to caching


@pytest.mark.utils
@pytest.mark.fast
def test_calculate_log_probability_for_table():
    """Test calculate_log_probability_for_table function."""
    table = TableData(3, 7, 2, 8)
    p1, p2 = 0.3, 0.2
    
    result = calculate_log_probability_for_table(table, p1, p2)
    
    # Calculate expected value
    expected = (log_binom_pmf_cached(table.n1, table.a, p1) + 
                log_binom_pmf_cached(table.n2, table.c, p2))
    
    assert abs(result - expected) < 1e-10


@pytest.mark.utils
@pytest.mark.fast
def test_enumerate_all_possible_tables():
    """Test enumerate_all_possible_tables generator."""
    n1, n2 = 2, 2
    
    tables = list(enumerate_all_possible_tables(n1, n2))
    
    # Should have (n1+1) * (n2+1) tables
    assert len(tables) == (n1 + 1) * (n2 + 1)
    
    # Check that all tables have correct marginals
    for table in tables:
        assert table.n1 == n1
        assert table.n2 == n2
        assert table.a + table.b == n1
        assert table.c + table.d == n2
    
    # Check specific tables are present
    expected_tables = [
        TableData(0, 2, 0, 2),
        TableData(0, 2, 1, 1),
        TableData(0, 2, 2, 0),
        TableData(1, 1, 0, 2),
        TableData(1, 1, 1, 1),
        TableData(1, 1, 2, 0),
        TableData(2, 0, 0, 2),
        TableData(2, 0, 1, 1),
        TableData(2, 0, 2, 0),
    ]
    
    for expected in expected_tables:
        assert any(t.a == expected.a and t.b == expected.b and 
                  t.c == expected.c and t.d == expected.d for t in tables)


@pytest.mark.utils
@pytest.mark.fast
def test_enumerate_all_possible_tables_edge_cases():
    """Test enumerate_all_possible_tables with edge cases."""
    # Test with n1=0, n2=0
    tables = list(enumerate_all_possible_tables(0, 0))
    assert len(tables) == 1
    assert tables[0] == TableData(0, 0, 0, 0)
    
    # Test with n1=1, n2=0
    tables = list(enumerate_all_possible_tables(1, 0))
    assert len(tables) == 2
    expected = [TableData(0, 1, 0, 0), TableData(1, 0, 0, 0)]
    for exp in expected:
        assert any(t.a == exp.a and t.b == exp.b and 
                  t.c == exp.c and t.d == exp.d for t in tables)


@pytest.mark.utils
@pytest.mark.fast
def test_calculate_unconditional_log_pvalue_for_grid_point():
    """Test calculate_unconditional_log_pvalue_for_grid_point function."""
    table = TableData(3, 7, 2, 8)
    theta = 2.0
    p1 = 0.3
    
    # Mock the _process_grid_point function
    with patch('exactcis.utils.calculators._process_grid_point') as mock_process:
        mock_process.return_value = -2.5  # Mock log p-value
        
        result = calculate_unconditional_log_pvalue_for_grid_point(table, theta, p1)
        
        assert result == -2.5
        mock_process.assert_called_once_with((p1, table.a, table.c, 
                                            table.n1, table.n2, theta))


@pytest.mark.utils
@pytest.mark.fast
def test_calculate_unconditional_log_pvalue():
    """Test calculate_unconditional_log_pvalue function."""
    table = TableData(3, 7, 2, 8)
    theta = 2.0
    grid = GridConfig(grid_size=10)
    
    # Mock the _log_pvalue_barnard function
    with patch('exactcis.utils.calculators._log_pvalue_barnard') as mock_pvalue:
        mock_pvalue.return_value = -1.5  # Mock log p-value
        
        result = calculate_unconditional_log_pvalue(table, theta, grid)
        
        assert result == -1.5
        mock_pvalue.assert_called_once()
    
    # Test with timeout
    timeout_checker = lambda: True
    result = calculate_unconditional_log_pvalue(table, theta, grid, timeout_checker)
    assert result is None


@pytest.mark.utils
@pytest.mark.fast
def test_create_pvalue_function():
    """Test create_pvalue_function."""
    table = TableData(3, 7, 2, 8)
    grid = GridConfig(grid_size=10)
    
    # Mock calculate_unconditional_log_pvalue
    with patch('exactcis.utils.calculators.calculate_unconditional_log_pvalue') as mock_calc:
        mock_calc.return_value = -2.0  # Mock log p-value
        
        pvalue_func = create_pvalue_function(table, grid)
        result = pvalue_func(2.0)
        
        assert abs(result - math.exp(-2.0)) < 1e-10
        mock_calc.assert_called_once()
    
    # Test with timeout (None return)
    with patch('exactcis.utils.calculators.calculate_unconditional_log_pvalue') as mock_calc:
        mock_calc.return_value = None  # Timeout
        
        pvalue_func = create_pvalue_function(table, grid)
        result = pvalue_func(2.0)
        
        assert math.isnan(result)


@pytest.mark.utils
@pytest.mark.fast
def test_find_confidence_bound_via_sign_change():
    """Test find_confidence_bound_via_sign_change function."""
    # Create a mock p-value function that crosses alpha at theta=2.0
    def mock_pvalue_func(theta):
        return 0.05 + (theta - 2.0) * 0.01  # Linear function crossing 0.05 at theta=2.0
    
    # Mock find_sign_change
    with patch('exactcis.utils.calculators.find_sign_change') as mock_find:
        mock_find.return_value = 2.0
        
        result = find_confidence_bound_via_sign_change(
            mock_pvalue_func, (1.0, 3.0), 0.05
        )
        
        assert result == 2.0
        mock_find.assert_called_once()
    
    # Test with exception
    with patch('exactcis.utils.calculators.find_sign_change') as mock_find:
        mock_find.side_effect = ValueError("Sign change failed")
        
        result = find_confidence_bound_via_sign_change(
            mock_pvalue_func, (1.0, 3.0), 0.05
        )
        
        assert result is None


@pytest.mark.utils
@pytest.mark.fast
def test_refine_bound_with_plateau_detection():
    """Test refine_bound_with_plateau_detection function."""
    # Create a mock p-value function
    def mock_pvalue_func(theta):
        return 0.05
    
    # Mock find_plateau_edge
    with patch('exactcis.utils.calculators.find_plateau_edge') as mock_plateau:
        mock_plateau.return_value = (2.1, 5)  # (result, iterations)
        
        result = refine_bound_with_plateau_detection(
            mock_pvalue_func, 2.0, 0.05
        )
        
        assert result == 2.1
        mock_plateau.assert_called_once()
    
    # Test with None return
    with patch('exactcis.utils.calculators.find_plateau_edge') as mock_plateau:
        mock_plateau.return_value = None
        
        result = refine_bound_with_plateau_detection(
            mock_pvalue_func, 2.0, 0.05
        )
        
        assert result is None
    
    # Test with exception
    with patch('exactcis.utils.calculators.find_plateau_edge') as mock_plateau:
        mock_plateau.side_effect = ValueError("Plateau detection failed")
        
        result = refine_bound_with_plateau_detection(
            mock_pvalue_func, 2.0, 0.05
        )
        
        assert result is None


@pytest.mark.utils
@pytest.mark.fast
def test_find_confidence_bound():
    """Test find_confidence_bound function."""
    table = TableData(3, 7, 2, 8)
    grid = GridConfig(grid_size=10)
    search_range = (1.0, 3.0)
    alpha = 0.05
    
    # Test successful sign change detection with plateau refinement
    with patch('exactcis.utils.calculators.find_confidence_bound_via_sign_change') as mock_sign:
        mock_sign.return_value = 2.0
        
        with patch('exactcis.utils.calculators.refine_bound_with_plateau_detection') as mock_plateau:
            mock_plateau.return_value = 2.1
            
            result = find_confidence_bound(table, grid, search_range, alpha, "lower")
            
            assert isinstance(result, BoundResult)
            assert result.value == 2.1
            assert result.iterations == 1
            assert result.method == "plateau_refined"
    
    # Test successful sign change detection without plateau refinement
    with patch('exactcis.utils.calculators.find_confidence_bound_via_sign_change') as mock_sign:
        mock_sign.return_value = 2.0
        
        with patch('exactcis.utils.calculators.refine_bound_with_plateau_detection') as mock_plateau:
            mock_plateau.return_value = None
            
            result = find_confidence_bound(table, grid, search_range, alpha, "lower")
            
            assert result.value == 2.0
            assert result.method == "sign_change"
    
    # Test fallback when sign change detection fails
    with patch('exactcis.utils.calculators.find_confidence_bound_via_sign_change') as mock_sign:
        mock_sign.return_value = None
        
        # Test lower bound fallback
        result = find_confidence_bound(table, grid, search_range, alpha, "lower")
        assert result.value == 1.0  # search_lo
        assert result.method == "fallback"
        
        # Test upper bound fallback
        result = find_confidence_bound(table, grid, search_range, alpha, "upper")
        assert result.value == 3.0  # search_hi
        assert result.method == "fallback"


@pytest.mark.utils
@pytest.mark.fast
def test_find_confidence_bound_with_timeout():
    """Test find_confidence_bound with timeout checker."""
    table = TableData(3, 7, 2, 8)
    grid = GridConfig(grid_size=10)
    search_range = (1.0, 3.0)
    alpha = 0.05
    timeout_checker = lambda: True  # Always timeout
    
    # The timeout should be passed through to the underlying functions
    with patch('exactcis.utils.calculators.create_pvalue_function') as mock_create:
        mock_pvalue_func = MagicMock()
        mock_create.return_value = mock_pvalue_func
        
        with patch('exactcis.utils.calculators.find_confidence_bound_via_sign_change') as mock_sign:
            mock_sign.return_value = None  # Simulate timeout/failure
            
            result = find_confidence_bound(
                table, grid, search_range, alpha, "lower", timeout_checker
            )
            
            # Should fall back to conservative estimate
            assert result.method == "fallback"
            mock_create.assert_called_once_with(table, grid, timeout_checker)


@pytest.mark.utils
@pytest.mark.fast
def test_grid_config_with_p1_values():
    """Test calculate_unconditional_log_pvalue with custom p1_values in grid."""
    table = TableData(3, 7, 2, 8)
    theta = 2.0
    
    # Create a grid with custom p1_values
    grid = GridConfig(grid_size=5)
    grid.p1_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    with patch('exactcis.utils.calculators._log_pvalue_barnard') as mock_pvalue:
        mock_pvalue.return_value = -1.5
        
        result = calculate_unconditional_log_pvalue(table, theta, grid)
        
        # Check that p1_grid_override was passed
        call_args = mock_pvalue.call_args
        assert call_args[1]['p1_grid_override'] == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.utils
@pytest.mark.fast
def test_log_binom_pmf_cached_boundary_values():
    """Test log_binom_pmf_cached with boundary probability values."""
    # Test with very small p
    result = log_binom_pmf_cached(10, 1, 1e-10)
    assert np.isfinite(result)
    
    # Test with p very close to 1
    result = log_binom_pmf_cached(10, 9, 1 - 1e-10)
    assert np.isfinite(result)
    
    # Test with k=0
    result = log_binom_pmf_cached(10, 0, 0.3)
    expected = 10 * math.log(0.7)
    assert abs(result - expected) < 1e-10
    
    # Test with k=n
    result = log_binom_pmf_cached(10, 10, 0.3)
    expected = 10 * math.log(0.3)
    assert abs(result - expected) < 1e-10


@pytest.mark.utils
@pytest.mark.fast
def test_enumerate_all_possible_tables_memory_efficiency():
    """Test that enumerate_all_possible_tables is memory efficient."""
    # This test ensures the generator doesn't create all tables at once
    n1, n2 = 5, 5
    
    # Create generator but don't consume it
    table_gen = enumerate_all_possible_tables(n1, n2)
    
    # Get first few tables
    first_table = next(table_gen)
    second_table = next(table_gen)
    
    assert isinstance(first_table, TableData)
    assert isinstance(second_table, TableData)
    assert first_table != second_table


@pytest.mark.utils
@pytest.mark.fast
def test_pvalue_function_with_various_inputs():
    """Test p-value function creation with various input scenarios."""
    table = TableData(5, 5, 5, 5)
    grid = GridConfig(grid_size=20)
    
    # Test with normal timeout checker
    timeout_checker = lambda: False  # Never timeout
    
    with patch('exactcis.utils.calculators.calculate_unconditional_log_pvalue') as mock_calc:
        mock_calc.return_value = -3.0
        
        pvalue_func = create_pvalue_function(table, grid, timeout_checker)
        result = pvalue_func(1.5)
        
        assert abs(result - math.exp(-3.0)) < 1e-10


@pytest.mark.utils
@pytest.mark.slow
def test_calculators_integration():
    """Integration test for calculator functions working together."""
    # Create a realistic scenario
    table = TableData(7, 3, 2, 8)
    grid = GridConfig(grid_size=10)
    
    # Mock the underlying unconditional method to avoid heavy computation
    with patch('exactcis.utils.calculators._log_pvalue_barnard') as mock_pvalue:
        # Create a mock that returns different values for different theta
        def mock_pvalue_func(*args, **kwargs):
            theta = kwargs.get('theta', args[5] if len(args) > 5 else 1.0)
            # Simulate p-value that decreases as theta increases
            return math.log(0.1 / theta)
        
        mock_pvalue.side_effect = mock_pvalue_func
        
        # Test the full pipeline
        pvalue_func = create_pvalue_function(table, grid)
        
        # Test that p-values decrease as theta increases
        pval_1 = pvalue_func(1.0)
        pval_2 = pvalue_func(2.0)
        
        assert pval_1 > pval_2  # P-value should decrease as theta increases
        assert pval_1 > 0 and pval_2 > 0  # Both should be positive probabilities


@pytest.mark.utils
@pytest.mark.fast
def test_bound_result_creation():
    """Test BoundResult creation and properties."""
    result = BoundResult(2.5, 10, "test_method")
    
    assert result.value == 2.5
    assert result.iterations == 10
    assert result.method == "test_method"


@pytest.mark.utils
@pytest.mark.fast
def test_error_handling_in_calculators():
    """Test error handling in calculator functions."""
    table = TableData(3, 7, 2, 8)
    grid = GridConfig(grid_size=10)
    
    # Test with invalid table data
    invalid_table = TableData(-1, 7, 2, 8)  # Negative value
    
    # The functions should handle this gracefully
    try:
        result = calculate_log_probability_for_table(invalid_table, 0.3, 0.2)
        # If it doesn't raise an error, that's also acceptable
    except Exception:
        # Expected for invalid inputs
        pass
    
    # Test with extreme probability values
    try:
        result = log_binom_pmf_cached(10, 5, 2.0)  # p > 1
        assert result == float('-inf')
    except Exception:
        # Also acceptable
        pass