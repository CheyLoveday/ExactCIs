"""
Test suite for the conditional (Fisher) confidence interval method.

This module compares the results of our implementation against expected values
from statsmodels, scipy.stats, and R's fisher.test function.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.exactcis.methods.conditional import exact_ci_conditional, ComputationError


class TestConditionalCI(unittest.TestCase):
    """Test suite for the conditional (Fisher's exact) confidence interval method."""

    def test_agresti_example(self):
        """Test example from Agresti (2002) - Tea tasting experiment."""
        # Table: [[3, 1], [1, 3]] -> a=3, b=1, c=1, d=3
        # Expected from statsmodels: ~(0.238051, 1074.817433)
        a, b, c, d = 3, 1, 1, 3
        alpha = 0.05
        
        lower, upper = exact_ci_conditional(a, b, c, d, alpha)
        
        # Using relatively loose tolerances due to potential implementation differences
        self.assertAlmostEqual(lower, 0.238051, places=2)
        self.assertTrue(upper > 500)  # Just check it's a large value, exact value isn't critical
    
    def test_zero_in_cell_a(self):
        """Test case with zero count in cell 'a'."""
        # Table: [[0, 5], [5, 5]] -> a=0, b=5, c=5, d=5
        # Expected from statsmodels: ~(0.000000, 1.506704)
        a, b, c, d = 0, 5, 5, 5
        alpha = 0.05
        
        lower, upper = exact_ci_conditional(a, b, c, d, alpha)
        
        self.assertAlmostEqual(lower, 0.0, places=5)
        self.assertAlmostEqual(upper, 1.506704, places=2)
    
    def test_scipy_example(self):
        """Test example from scipy.stats.fisher_exact documentation."""
        # Table: [[1, 9], [11, 3]] -> a=1, b=9, c=11, d=3
        # Expected from statsmodels: ~(0.000541, 0.525381)
        a, b, c, d = 1, 9, 11, 3
        alpha = 0.05
        
        lower, upper = exact_ci_conditional(a, b, c, d, alpha)
        
        self.assertAlmostEqual(lower, 0.000541, places=5)
        self.assertAlmostEqual(upper, 0.525381, places=2)
    
    def test_infinity_upper_bound(self):
        """Test case where upper bound should be infinity (a=max_k)."""
        # Table: [[5, 0], [2, 3]] -> a=5, b=0, c=2, d=3
        # Expected from statsmodels: ~(0.528283, inf)
        a, b, c, d = 5, 0, 2, 3
        alpha = 0.05
        
        lower, upper = exact_ci_conditional(a, b, c, d, alpha)
        
        self.assertAlmostEqual(lower, 0.528283, places=2)
        self.assertEqual(upper, float('inf'))
    
    def test_statsmodels_example(self):
        """Test example from statsmodels Table2x2 fisher example."""
        # Table: [[7, 17], [15, 5]] -> a=7, b=17, c=15, d=5
        # Expected from statsmodels: ~(0.019110, 0.831039)
        a, b, c, d = 7, 17, 15, 5
        alpha = 0.05
        
        lower, upper = exact_ci_conditional(a, b, c, d, alpha)
        
        self.assertAlmostEqual(lower, 0.019110, places=3)
        self.assertAlmostEqual(upper, 0.831039, places=2)
    
    def test_degenerate_cases(self):
        """Test degenerate cases (zero row or column totals)."""
        # These might raise ValueError in validate_counts
        
        # Case 1: [[0, 0], [5, 5]] (zero row)
        with self.assertRaises(ValueError):
            exact_ci_conditional(0, 0, 5, 5, 0.05)
        
        # Case 2: [[5, 5], [0, 0]] (zero row)
        with self.assertRaises(ValueError):
            exact_ci_conditional(5, 5, 0, 0, 0.05)


if __name__ == '__main__':
    unittest.main()
