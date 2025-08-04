"""
Tests for version information.
"""

import pytest
from exactcis._version import __version__


def test_version_exists():
    """Test that version is defined and is a string."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format():
    """Test that version follows semantic versioning format."""
    # Should be in format X.Y.Z or X.Y.Z-suffix
    parts = __version__.split('.')
    assert len(parts) >= 2, f"Version {__version__} should have at least major.minor format"
    
    # First part should be numeric
    assert parts[0].isdigit(), f"Major version should be numeric, got {parts[0]}"
    
    # Second part should be numeric
    assert parts[1].isdigit(), f"Minor version should be numeric, got {parts[1]}"


def test_version_matches_expected():
    """Test that version matches expected value from pyproject.toml."""
    # This should match the version in pyproject.toml
    assert __version__ == "0.1.0"