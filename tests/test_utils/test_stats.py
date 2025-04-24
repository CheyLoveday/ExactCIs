"""
Tests for the statistical utility functions.
"""

import pytest
import math
from exactcis.utils.stats import normal_quantile


def test_normal_quantile_basic():
    """Test basic functionality of normal_quantile."""
    # Test median (should be 0)
    assert normal_quantile(0.5) == 0.0
    
    # Test standard values
    assert abs(normal_quantile(0.975) - 1.96) < 0.01, "Expected ~1.96 for 97.5th percentile"
    assert abs(normal_quantile(0.025) + 1.96) < 0.01, "Expected ~-1.96 for 2.5th percentile"
    assert abs(normal_quantile(0.95) - 1.645) < 0.01, "Expected ~1.645 for 95th percentile"
    assert abs(normal_quantile(0.05) + 1.645) < 0.01, "Expected ~-1.645 for 5th percentile"


def test_normal_quantile_symmetry():
    """Test symmetry property of normal_quantile."""
    for p in [0.01, 0.1, 0.25, 0.4, 0.45]:
        assert abs(normal_quantile(p) + normal_quantile(1-p)) < 1e-10, f"Symmetry failed for p={p}"


def test_normal_quantile_extreme_values():
    """Test normal_quantile with extreme values."""
    # Very small probabilities
    assert normal_quantile(0.001) < -3.0
    assert normal_quantile(0.999) > 3.0
    
    # Approaching limits
    p_small = 1e-6
    assert normal_quantile(p_small) < -4.0
    assert normal_quantile(1 - p_small) > 4.0


def test_normal_quantile_invalid_inputs():
    """Test that invalid inputs raise appropriate exceptions."""
    # Out of range probabilities
    with pytest.raises(ValueError):
        normal_quantile(0)
    
    with pytest.raises(ValueError):
        normal_quantile(1)
    
    with pytest.raises(ValueError):
        normal_quantile(-0.1)
    
    with pytest.raises(ValueError):
        normal_quantile(1.1)


def test_normal_quantile_accuracy():
    """Test accuracy of normal_quantile against known values."""
    # Known values from statistical tables
    known_values = [
        (0.1, -1.282),
        (0.2, -0.842),
        (0.3, -0.524),
        (0.4, -0.253),
        (0.6, 0.253),
        (0.7, 0.524),
        (0.8, 0.842),
        (0.9, 1.282)
    ]
    
    for p, expected in known_values:
        assert abs(normal_quantile(p) - expected) < 0.01, f"Expected {expected} for p={p}, got {normal_quantile(p)}"


def test_normal_quantile_monotonicity():
    """Test that normal_quantile is monotonically increasing."""
    ps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    values = [normal_quantile(p) for p in ps]
    
    for i in range(1, len(values)):
        assert values[i] > values[i-1], f"Monotonicity failed between p={ps[i-1]} and p={ps[i]}"