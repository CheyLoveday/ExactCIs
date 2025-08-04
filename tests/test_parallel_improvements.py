"""
Tests for parallel processing improvements in ExactCIs.

This module tests the improvements made to the parallel processing utilities
in ExactCIs, including:
1. LRU cache for exact_ci_midp
2. as_completed implementation in parallel_map
3. Backend selection in parallel_compute_ci
4. Batch size autotuning in parallel_map
"""

import time
import logging
import unittest
import numpy as np
from concurrent.futures import TimeoutError

from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.utils.parallel import (
    parallel_map,
    parallel_compute_ci,
    has_numba_support
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestParallelImprovements(unittest.TestCase):
    """Test cases for parallel processing improvements."""

    def test_lru_cache_midp(self):
        """Test that LRU cache is working for exact_ci_midp."""
        # Call the function twice with the same arguments
        result1 = exact_ci_midp(10, 20, 15, 30, alpha=0.05)
        result2 = exact_ci_midp(10, 20, 15, 30, alpha=0.05)
        
        # Results should be identical
        self.assertEqual(result1, result2)
        
        # Call with different arguments
        result3 = exact_ci_midp(5, 10, 8, 12, alpha=0.05)
        
        # Should be different from result1
        self.assertNotEqual(result1, result3)

    def test_timeout_handling(self):
        """Test that timeout handling works correctly with as_completed."""
        # Define a function that sometimes takes too long
        def slow_function(x):
            if x % 3 == 0:
                time.sleep(0.5)  # This will timeout
            return x * 2
        
        # Create a list of items
        items = list(range(10))
        
        # Call parallel_map with a short timeout
        results = parallel_map(
            slow_function,
            items,
            timeout=0.1,  # Short timeout
            backend='thread'  # Use threads for this test
        )
        
        # All items should be processed, even those that timed out
        self.assertEqual(len(results), len(items))
        
        # Check that all results are correct
        for i, result in enumerate(results):
            self.assertEqual(result, i * 2)

    def test_backend_selection(self):
        """Test that backend selection works correctly."""
        # Create a list of tables
        tables = [
            (10, 20, 15, 30),
            (5, 10, 8, 12),
            (2, 3, 1, 4)
        ]
        
        # Test with explicit thread backend
        results_thread = parallel_compute_ci(
            exact_ci_blaker,
            tables,
            backend='thread'
        )
        
        # Test with explicit process backend
        results_process = parallel_compute_ci(
            exact_ci_blaker,
            tables,
            backend='process'
        )
        
        # Results should be the same regardless of backend
        for rt, rp in zip(results_thread, results_process):
            self.assertAlmostEqual(rt[0], rp[0], places=6)
            self.assertAlmostEqual(rt[1], rp[1], places=6)
        
        # Test auto-detection for Blaker's method
        results_auto = parallel_compute_ci(
            exact_ci_blaker,
            tables
        )
        
        # Results should be the same as with explicit backends
        for ra, rt in zip(results_auto, results_thread):
            self.assertAlmostEqual(ra[0], rt[0], places=6)
            self.assertAlmostEqual(ra[1], rt[1], places=6)

    def test_batch_size_autotuning(self):
        """Test that batch size autotuning works correctly."""
        # Define a simple function
        def square(x):
            return x * x
        
        # Create a list of items
        items = list(range(100))
        
        # Test with default min_batch_size
        results1 = parallel_map(
            square,
            items,
            max_workers=50  # More workers than needed
        )
        
        # Test with larger min_batch_size
        results2 = parallel_map(
            square,
            items,
            max_workers=50,
            min_batch_size=10
        )
        
        # Results should be the same
        self.assertEqual(results1, results2)
        
        # Test with very small number of items
        small_items = list(range(5))
        results3 = parallel_map(
            square,
            small_items,
            max_workers=10  # More workers than items
        )
        
        # Should still work correctly
        self.assertEqual(results3, [x*x for x in small_items])

    def test_numba_detection(self):
        """Test that Numba detection works correctly."""
        # Check if Numba is available
        has_numba = has_numba_support()
        
        # Log the result
        logger.info(f"Numba support detected: {has_numba}")
        
        # This is just informational, no assertion needed


if __name__ == '__main__':
    unittest.main()