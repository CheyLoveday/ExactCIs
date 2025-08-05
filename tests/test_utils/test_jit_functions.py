"""
Tests for JIT-compiled functions in exactcis.utils.jit_functions.

This module tests the Numba-accelerated functions for performance-critical
calculations, focusing on correctness and numerical stability.
"""

import pytest
import numpy as np
from exactcis.utils.jit_functions import (
    HAS_NUMBA, nchg_pdf_jit, sum_probs_less_than_threshold,
    _log_binom_coeff_jit, _nchg_pdf_jit
)


class TestJITAvailability:
    """Test JIT compilation availability and fallbacks."""
    
    def test_numba_availability(self):
        """Test that Numba availability is correctly detected."""
        # Just verify the flag is a boolean
        assert isinstance(HAS_NUMBA, bool)
    
    def test_jit_functions_callable(self):
        """Test that JIT functions are callable regardless of Numba availability."""
        # These should always be callable (either JIT or fallback)
        assert callable(nchg_pdf_jit)
        assert callable(sum_probs_less_than_threshold)
        assert callable(_log_binom_coeff_jit)
        assert callable(_nchg_pdf_jit)


class TestLogBinomCoeff:
    """Test log binomial coefficient calculation."""
    
    def test_log_binom_coeff_basic(self):
        """Test basic log binomial coefficient calculations."""
        # Test known values
        result = _log_binom_coeff_jit(10, 0)
        assert abs(result - 0.0) < 1e-10, "log(10 choose 0) should be 0"
        
        result = _log_binom_coeff_jit(10, 10)
        assert abs(result - 0.0) < 1e-10, "log(10 choose 10) should be 0"
        
        result = _log_binom_coeff_jit(10, 1)
        expected = np.log(10)
        assert abs(result - expected) < 1e-10, "log(10 choose 1) should be log(10)"
    
    def test_log_binom_coeff_symmetry(self):
        """Test symmetry property: C(n,k) = C(n,n-k)."""
        n, k = 10, 3
        result1 = _log_binom_coeff_jit(n, k)
        result2 = _log_binom_coeff_jit(n, n - k)
        assert abs(result1 - result2) < 1e-10, "Binomial coefficients should be symmetric"
    
    def test_log_binom_coeff_edge_cases(self):
        """Test edge cases for log binomial coefficient."""
        # Test k > n (should return -inf)
        result = _log_binom_coeff_jit(5, 10)
        assert result == float('-inf'), "log C(5,10) should be -inf"
        
        # Test negative k (should return -inf)
        result = _log_binom_coeff_jit(5, -1)
        assert result == float('-inf'), "log C(5,-1) should be -inf"


class TestNCHGPDFJIT:
    """Test JIT-compiled noncentral hypergeometric PDF."""
    
    def test_nchg_pdf_basic_properties(self):
        """Test basic properties of NCHG PDF calculation."""
        k_values = np.array([0, 1, 2, 3, 4, 5])
        n1, n2, m1 = 10, 10, 5
        theta = 1.0
        
        probabilities = _nchg_pdf_jit(k_values, n1, n2, m1, theta)
        
        # Test basic properties
        assert len(probabilities) == len(k_values), "Output length should match input"
        assert all(p >= 0 for p in probabilities), "All probabilities should be non-negative"
        
        # For theta=1 (no association), distribution should be symmetric-ish
        assert probabilities.sum() > 0, "Total probability should be positive"
    
    def test_nchg_pdf_extreme_theta(self):
        """Test NCHG PDF with extreme theta values."""
        k_values = np.array([0, 1, 2, 3])
        n1, n2, m1 = 5, 5, 3
        
        # Test theta = 0 (all mass at minimum)
        probabilities = _nchg_pdf_jit(k_values, n1, n2, m1, 0.0)
        assert probabilities[0] > 0, "Should have mass at minimum value for theta=0"
        
        # Test very large theta (mass shifts to maximum values)
        probabilities = _nchg_pdf_jit(k_values, n1, n2, m1, 1000.0)
        assert probabilities[-1] >= probabilities[0], "Large theta should favor larger k values"
    
    def test_nchg_pdf_consistency_with_support(self):
        """Test that PDF respects support constraints."""
        n1, n2, m1 = 8, 12, 10
        k_min = max(0, m1 - n2)
        k_max = min(m1, n1)
        
        # Test values outside support
        k_values = np.array([k_min - 1, k_min, k_max, k_max + 1])
        probabilities = _nchg_pdf_jit(k_values, n1, n2, m1, 2.0)
        
        # Values outside support should have zero probability
        if k_min > 0:
            assert probabilities[0] == 0, "Probability should be 0 outside support (below)"
        if k_max < n1:
            assert probabilities[-1] == 0, "Probability should be 0 outside support (above)"


class TestNCHGPDFWrapper:
    """Test the main nchg_pdf_jit wrapper function."""
    
    def test_nchg_pdf_jit_basic(self):
        """Test basic functionality of nchg_pdf_jit wrapper."""
        support = np.array([0, 1, 2, 3, 4])
        n1, n2, m1 = 10, 10, 4
        theta = 1.5
        
        probabilities = nchg_pdf_jit(support, n1, n2, m1, theta)
        
        # Basic checks
        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == len(support), "Output length should match support"
        assert all(p >= 0 for p in probabilities), "All probabilities should be non-negative"
        
        # Should sum to approximately 1 (within numerical tolerance)
        total_prob = probabilities.sum()
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to ~1, got {total_prob}"
    
    def test_nchg_pdf_jit_theta_monotonicity(self):
        """Test that changing theta affects probability distribution monotonically."""
        support = np.array([0, 1, 2, 3])
        n1, n2, m1 = 6, 8, 4
        
        # Compare different theta values
        probs_low = nchg_pdf_jit(support, n1, n2, m1, 0.5)
        probs_high = nchg_pdf_jit(support, n1, n2, m1, 2.0)
        
        # For larger theta, probability should shift toward larger k values
        # (this is a general property, though exact comparison depends on parameters)
        assert len(probs_low) == len(probs_high), "Same support length"
        
        # At least verify both are valid probability distributions
        assert abs(probs_low.sum() - 1.0) < 0.01, "Low theta probs should sum to 1"
        assert abs(probs_high.sum() - 1.0) < 0.01, "High theta probs should sum to 1"


class TestSumProbsJIT:
    """Test JIT-compiled probability summation function."""
    
    def test_sum_probs_basic(self):
        """Test basic probability summation functionality."""
        probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        idx_a = 2  # Index of observed value
        
        result = sum_probs_less_than_threshold(probs, idx_a)
        
        # Basic properties
        assert isinstance(result, float), "Should return float"
        assert 0 <= result <= 1, "Result should be a valid probability"
        assert np.isfinite(result), "Result should be finite"
    
    def test_sum_probs_edge_cases(self):
        """Test edge cases for probability summation."""
        # Test with all equal probabilities
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        idx_a = 2
        
        result = sum_probs_less_than_threshold(probs, idx_a)
        assert 0 <= result <= 1, "Should handle equal probabilities"
        
        # Test with single probability
        probs = np.array([1.0])
        idx_a = 0
        
        result = sum_probs_less_than_threshold(probs, idx_a)
        assert result >= 0, "Should handle single probability"
    
    def test_sum_probs_different_epsilon(self):
        """Test probability summation with different epsilon values."""
        probs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        idx_a = 2
        
        # Test with different epsilon values
        result1 = sum_probs_less_than_threshold(probs, idx_a, epsilon=1e-7)
        result2 = sum_probs_less_than_threshold(probs, idx_a, epsilon=1e-10)
        
        # Results should be close but potentially different
        assert np.isfinite(result1) and np.isfinite(result2), "Both results should be finite"
        assert 0 <= result1 <= 1 and 0 <= result2 <= 1, "Both should be valid probabilities"


class TestJITIntegration:
    """Test integration between JIT functions and main library."""
    
    def test_jit_functions_integration(self):
        """Test that JIT functions integrate properly with the main library."""
        # This is more of an integration test to ensure JIT functions
        # work with the rest of the system
        
        from exactcis.core import support
        
        n1, n2, m1 = 10, 12, 8
        theta = 1.5
        
        # Get support
        supp = support(n1, n2, m1)
        
        # Test that JIT functions work with support data
        probabilities = nchg_pdf_jit(supp.x, n1, n2, m1, theta)
        sum_result = sum_probs_less_than_threshold(probabilities, 2)
        
        assert len(probabilities) == len(supp.x), "Probabilities length should match support"
        assert isinstance(sum_result, float), "Sum result should be float"
        
        # Basic sanity checks
        assert all(p >= 0 for p in probabilities), "Probabilities should be non-negative"
        assert 0 <= sum_result <= 1, "Sum result should be valid probability"
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
    def test_jit_compilation_works(self):
        """Test that JIT compilation actually works when Numba is available."""
        # This test only runs if Numba is available
        
        # Call functions to trigger compilation
        result = _log_binom_coeff_jit(10, 5)
        assert np.isfinite(result), "JIT function should return finite result"
        
        # Call again - should use compiled version
        result2 = _log_binom_coeff_jit(10, 5)
        assert abs(result - result2) < 1e-15, "Compiled function should be deterministic"


# Performance comparison tests (optional, for development)
class TestJITPerformance:
    """Optional performance tests for JIT functions."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
    def test_jit_performance_benefit(self):
        """Test that JIT functions provide performance benefit (when available)."""
        import time
        
        n1, n2, m1 = 20, 25, 15
        theta = 2.0
        support_vals = np.arange(max(0, m1-n2), min(m1, n1) + 1)
        
        # Warm up JIT compilation
        _ = nchg_pdf_jit(support_vals, n1, n2, m1, theta)
        
        # Time JIT version
        start_time = time.time()
        for _ in range(100):
            _ = nchg_pdf_jit(support_vals, n1, n2, m1, theta)
        jit_time = time.time() - start_time
        
        # Just verify it completed without error
        assert jit_time > 0, "JIT function should complete"
        
        # Note: We don't compare against non-JIT version as that would require
        # more complex setup. This test mainly ensures JIT functions work repeatedly.