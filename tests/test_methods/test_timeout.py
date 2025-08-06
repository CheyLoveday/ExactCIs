"""Tests for timeout functionality in confidence interval methods."""

import pytest
import time
from exactcis.methods import exact_ci_unconditional
from exactcis.core import find_root_log, find_plateau_edge


def create_timeout_checker(max_time=0.5):
    """Create a timeout checker function that times out after max_time seconds."""
    start_time = time.time()
    
    def timeout_checker():
        return time.time() - start_time > max_time
    
    return timeout_checker


class TestTimeout:
    """Test timeout functionality in confidence interval methods."""
    
    def test_find_root_log_timeout(self):
        """Test that find_root_log respects timeout."""
        # Define a slow function that always returns the same value
        # This will prevent convergence and force timeout
        def slow_func(x):
            time.sleep(0.1)  # Make the function slow
            return 1.0  # Always returns the same value
        
        # Create timeout checker that times out quickly
        timeout_checker = create_timeout_checker(0.3)
        
        # Call function with timeout checker
        result = find_root_log(
            slow_func, 
            lo=1e-8, 
            hi=1.0, 
            timeout_checker=timeout_checker
        )
        
        # Function should return None on timeout
        assert result is None
    
    def test_find_plateau_edge_timeout(self):
        """Test that find_plateau_edge respects timeout."""
        # Create a timeout checker that immediately returns True (timed out)
        def immediate_timeout_checker():
            return True
            
        # Call function with immediate timeout checker
        result = find_plateau_edge(
            lambda x: 1.0,  # Function doesn't matter as we'll timeout immediately
            lo=0.1, 
            hi=10.0, 
            target=0.5,
            timeout_checker=immediate_timeout_checker
        )
        
        # Function should return None on timeout
        assert result is None
    
    @pytest.mark.slow
    def test_unconditional_timeout(self):
        """Test that exact_ci_unconditional respects timeout."""
        # Create a table that will be slow to compute
        a, b, c, d = 20, 30, 25, 35
        
        # Set a very short timeout
        timeout = 0.1
        
        # Measure actual time
        start_time = time.time()
        
        # Call function with short timeout
        try:
            result = exact_ci_unconditional(
                a, b, c, d, 
                alpha=0.05, 
                timeout=timeout
            )
            
            # If we reach here, the function returned normally
            # The result might be None or a partial CI
            elapsed = time.time() - start_time
            
            # Should not take much longer than the timeout
            assert elapsed < timeout + 1.0
            
        except Exception as e:
            # If an exception occurred, make sure it's timeout related
            # and the execution time was close to the timeout
            elapsed = time.time() - start_time
            assert elapsed < timeout + 1.0
            assert "timeout" in str(e).lower() or "time limit" in str(e).lower()
    
    def test_timeout_functionality_basic(self):
        """
        Test basic timeout functionality with core functions.
        """
        # Test that timeout checkers work with core functions
        timeout_checker = create_timeout_checker(0.1)
        
        # Test find_root_log with timeout
        def test_func(x):
            return x - 0.5  # Simple function with root at 0.5
        
        # This should work normally without timeout
        result = find_root_log(test_func, lo=0.1, hi=1.0)
        assert result is not None
        
        # Test find_plateau_edge with timeout
        result = find_plateau_edge(
            lambda x: 1.0,  # Constant function
            lo=0.1, 
            hi=10.0, 
            target=0.5,
            timeout_checker=lambda: False  # No timeout
        )
        assert result is not None
        
        # Test with immediate timeout
        result = find_plateau_edge(
            lambda x: 1.0,
            lo=0.1, 
            hi=10.0, 
            target=0.5,
            timeout_checker=lambda: True  # Immediate timeout
        )
        assert result is None
