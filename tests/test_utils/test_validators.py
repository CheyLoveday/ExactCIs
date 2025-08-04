"""
Tests for the validators utility module.
"""

import pytest
import math
from exactcis.utils.validators import (
    validate_table_data,
    validate_alpha,
    has_zero_marginal_totals,
    has_zero_in_cell_a_with_nonzero_c,
    is_valid_theta_range,
    is_finite_positive
)
from exactcis.utils.data_models import TableData


@pytest.mark.utils
@pytest.mark.fast
def test_validate_table_data():
    """Test validate_table_data function."""
    # Test valid table
    table = TableData(5, 3, 2, 8)
    assert validate_table_data(table) is True
    
    # Test table with negative values
    invalid_table = TableData(-1, 3, 2, 8)
    with pytest.raises(ValueError):
        validate_table_data(invalid_table)
    
    # Test table with empty margins
    empty_row_table = TableData(0, 0, 2, 8)
    with pytest.raises(ValueError):
        validate_table_data(empty_row_table)
    
    empty_col_table = TableData(0, 3, 0, 8)
    with pytest.raises(ValueError):
        validate_table_data(empty_col_table)
    
    # Test valid table with zeros (but not empty margins)
    valid_zero_table = TableData(0, 3, 2, 8)
    assert validate_table_data(valid_zero_table) is True


@pytest.mark.utils
@pytest.mark.fast
def test_validate_alpha():
    """Test validate_alpha function."""
    # Test valid alpha values
    assert validate_alpha(0.05) is True
    assert validate_alpha(0.01) is True
    assert validate_alpha(0.1) is True
    assert validate_alpha(0.5) is True
    assert validate_alpha(0.99) is True
    
    # Test invalid alpha values
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        validate_alpha(0.0)
    
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        validate_alpha(1.0)
    
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        validate_alpha(-0.1)
    
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        validate_alpha(1.5)
    
    # Test edge cases very close to boundaries
    with pytest.raises(ValueError):
        validate_alpha(1e-15)  # Very close to 0
    
    with pytest.raises(ValueError):
        validate_alpha(1 - 1e-15)  # Very close to 1


@pytest.mark.utils
@pytest.mark.fast
def test_has_zero_marginal_totals():
    """Test has_zero_marginal_totals function."""
    # Test table with no zero marginals
    table = TableData(5, 3, 2, 8)
    assert has_zero_marginal_totals(table) is False
    
    # Test table with zero first row (n1 = a + b = 0)
    table_zero_n1 = TableData(0, 0, 2, 8)
    assert has_zero_marginal_totals(table_zero_n1) is True
    
    # Test table with zero second row (n2 = c + d = 0)
    table_zero_n2 = TableData(5, 3, 0, 0)
    assert has_zero_marginal_totals(table_zero_n2) is True
    
    # Test table with both rows zero
    table_both_zero = TableData(0, 0, 0, 0)
    assert has_zero_marginal_totals(table_both_zero) is True
    
    # Test table with single zero cell (but non-zero marginals)
    table_single_zero = TableData(0, 3, 2, 8)
    assert has_zero_marginal_totals(table_single_zero) is False


@pytest.mark.utils
@pytest.mark.fast
def test_has_zero_in_cell_a_with_nonzero_c():
    """Test has_zero_in_cell_a_with_nonzero_c function."""
    # Test case where a=0 and c>0
    table_a0_c_pos = TableData(0, 3, 2, 8)
    assert has_zero_in_cell_a_with_nonzero_c(table_a0_c_pos) is True
    
    # Test case where a=0 and c=0
    table_a0_c0 = TableData(0, 3, 0, 8)
    assert has_zero_in_cell_a_with_nonzero_c(table_a0_c0) is False
    
    # Test case where a>0 and c>0
    table_a_pos_c_pos = TableData(5, 3, 2, 8)
    assert has_zero_in_cell_a_with_nonzero_c(table_a_pos_c_pos) is False
    
    # Test case where a>0 and c=0
    table_a_pos_c0 = TableData(5, 3, 0, 8)
    assert has_zero_in_cell_a_with_nonzero_c(table_a_pos_c0) is False
    
    # Test edge case with large c
    table_a0_large_c = TableData(0, 3, 100, 8)
    assert has_zero_in_cell_a_with_nonzero_c(table_a0_large_c) is True


@pytest.mark.utils
@pytest.mark.fast
def test_is_valid_theta_range():
    """Test is_valid_theta_range function."""
    # Test valid ranges
    assert is_valid_theta_range(0.1, 10.0) is True
    assert is_valid_theta_range(1.0, 2.0) is True
    assert is_valid_theta_range(0.001, 1000.0) is True
    
    # Test invalid ranges - theta_min <= 0
    with pytest.raises(ValueError, match="theta_min must be positive"):
        is_valid_theta_range(0.0, 10.0)
    
    with pytest.raises(ValueError, match="theta_min must be positive"):
        is_valid_theta_range(-1.0, 10.0)
    
    # Test invalid ranges - theta_max <= theta_min
    with pytest.raises(ValueError, match="theta_max must be greater than theta_min"):
        is_valid_theta_range(10.0, 10.0)  # Equal
    
    with pytest.raises(ValueError, match="theta_max must be greater than theta_min"):
        is_valid_theta_range(10.0, 5.0)  # theta_max < theta_min
    
    # Test edge cases
    assert is_valid_theta_range(1e-10, 1e-9) is True  # Very small but valid
    assert is_valid_theta_range(1e6, 1e7) is True     # Very large but valid


@pytest.mark.utils
@pytest.mark.fast
def test_is_finite_positive():
    """Test is_finite_positive function."""
    # Test positive finite values
    assert is_finite_positive(1.0) is True
    assert is_finite_positive(0.1) is True
    assert is_finite_positive(1000.0) is True
    assert is_finite_positive(1e-10) is True
    assert is_finite_positive(1e10) is True
    
    # Test zero
    assert is_finite_positive(0.0) is False
    
    # Test negative values
    assert is_finite_positive(-1.0) is False
    assert is_finite_positive(-0.1) is False
    
    # Test infinite values
    assert is_finite_positive(float('inf')) is False
    assert is_finite_positive(float('-inf')) is False
    
    # Test NaN
    assert is_finite_positive(float('nan')) is False
    
    # Test very small positive value
    assert is_finite_positive(1e-100) is True
    
    # Test very large positive value
    assert is_finite_positive(1e100) is True


@pytest.mark.utils
@pytest.mark.fast
def test_validate_table_data_edge_cases():
    """Test validate_table_data with edge cases."""
    # Test table with float values
    float_table = TableData(5.0, 3.0, 2.0, 8.0)
    assert validate_table_data(float_table) is True
    
    # Test table with very large values
    large_table = TableData(1000000, 2000000, 3000000, 4000000)
    assert validate_table_data(large_table) is True
    
    # Test minimal valid table
    minimal_table = TableData(1, 0, 0, 1)
    assert validate_table_data(minimal_table) is True
    
    # Test another minimal valid table
    minimal_table2 = TableData(0, 1, 1, 0)
    assert validate_table_data(minimal_table2) is True


@pytest.mark.utils
@pytest.mark.fast
def test_has_zero_marginal_totals_edge_cases():
    """Test has_zero_marginal_totals with edge cases."""
    # Test with very small non-zero values
    small_table = TableData(1e-10, 1e-10, 1e-10, 1e-10)
    assert has_zero_marginal_totals(small_table) is False
    
    # Test with mixed zero and non-zero
    mixed_table1 = TableData(0, 5, 3, 0)  # n1=5, n2=3, both non-zero
    assert has_zero_marginal_totals(mixed_table1) is False
    
    mixed_table2 = TableData(5, 0, 0, 3)  # n1=5, n2=3, both non-zero
    assert has_zero_marginal_totals(mixed_table2) is False


@pytest.mark.utils
@pytest.mark.fast
def test_has_zero_in_cell_a_with_nonzero_c_edge_cases():
    """Test has_zero_in_cell_a_with_nonzero_c with edge cases."""
    # Test with very small c
    small_c_table = TableData(0, 3, 1e-10, 8)
    assert has_zero_in_cell_a_with_nonzero_c(small_c_table) is True
    
    # Test with float values
    float_table = TableData(0.0, 3.0, 2.0, 8.0)
    assert has_zero_in_cell_a_with_nonzero_c(float_table) is True
    
    # Test with a being very small but non-zero
    small_a_table = TableData(1e-10, 3, 2, 8)
    assert has_zero_in_cell_a_with_nonzero_c(small_a_table) is False


@pytest.mark.utils
@pytest.mark.fast
def test_is_valid_theta_range_edge_cases():
    """Test is_valid_theta_range with edge cases."""
    # Test with very small positive theta_min
    assert is_valid_theta_range(1e-100, 1e-99) is True
    
    # Test with theta_min very close to zero but positive
    assert is_valid_theta_range(1e-15, 1.0) is True
    
    # Test with very large theta values
    assert is_valid_theta_range(1e10, 1e11) is True
    
    # Test with theta_max only slightly larger than theta_min
    assert is_valid_theta_range(1.0, 1.0000001) is True


@pytest.mark.utils
@pytest.mark.fast
def test_is_finite_positive_special_values():
    """Test is_finite_positive with special floating point values."""
    # Test with different representations of zero
    assert is_finite_positive(0.0) is False
    assert is_finite_positive(-0.0) is False
    
    # Test with very small numbers
    assert is_finite_positive(1e-323) is True  # Near smallest positive float
    
    # Test with numbers very close to zero
    assert is_finite_positive(1e-308) is True
    
    # Test with largest finite float
    assert is_finite_positive(1.7976931348623157e+308) is True


@pytest.mark.utils
@pytest.mark.fast
def test_validator_functions_integration():
    """Test validator functions working together."""
    # Create a valid table and test all validators
    table = TableData(5, 3, 2, 8)
    
    # All these should pass
    assert validate_table_data(table) is True
    assert validate_alpha(0.05) is True
    assert has_zero_marginal_totals(table) is False
    assert has_zero_in_cell_a_with_nonzero_c(table) is False
    assert is_valid_theta_range(0.1, 10.0) is True
    assert is_finite_positive(2.5) is True
    
    # Test with a problematic table
    problem_table = TableData(0, 3, 2, 8)
    
    assert validate_table_data(problem_table) is True  # Still valid
    assert has_zero_marginal_totals(problem_table) is False  # Marginals are non-zero
    assert has_zero_in_cell_a_with_nonzero_c(problem_table) is True  # Special case


@pytest.mark.utils
@pytest.mark.fast
def test_error_messages():
    """Test that error messages are informative."""
    # Test alpha validation error messages
    try:
        validate_alpha(0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "alpha must be between 0 and 1" in str(e)
    
    try:
        validate_alpha(1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "alpha must be between 0 and 1" in str(e)
    
    # Test theta range validation error messages
    try:
        is_valid_theta_range(0.0, 1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "theta_min must be positive" in str(e)
    
    try:
        is_valid_theta_range(2.0, 1.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "theta_max must be greater than theta_min" in str(e)


@pytest.mark.utils
@pytest.mark.fast
def test_validators_with_boundary_conditions():
    """Test validators with boundary conditions."""
    # Test alpha at boundaries (should fail)
    boundary_alphas = [0.0, 1.0, -1e-15, 1 + 1e-15]
    for alpha in boundary_alphas:
        with pytest.raises(ValueError):
            validate_alpha(alpha)
    
    # Test theta range at boundaries
    with pytest.raises(ValueError):
        is_valid_theta_range(0.0, 1.0)  # theta_min = 0
    
    with pytest.raises(ValueError):
        is_valid_theta_range(1.0, 1.0)  # theta_min = theta_max
    
    # Test is_finite_positive at boundaries
    assert is_finite_positive(1e-323) is True  # Smallest positive
    assert is_finite_positive(0.0) is False    # Zero
    assert is_finite_positive(-1e-323) is False  # Smallest negative