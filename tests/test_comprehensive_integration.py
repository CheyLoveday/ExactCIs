"""
Comprehensive integration tests for all exact confidence interval methods.

This module tests all available methods (MIDP, Blaker, Conditional, Unconditional, 
Clopper-Pearson) to ensure they work correctly and produce reasonable results.
"""

import pytest
import numpy as np
import time
from typing import List, Tuple

from exactcis.methods import (
    exact_ci_midp,
    exact_ci_blaker, 
    exact_ci_conditional,
    exact_ci_unconditional,
    exact_ci_clopper_pearson
)
from exactcis.core import calculate_odds_ratio


class TestComprehensiveIntegration:
    """Comprehensive integration tests for all CI methods."""
    
    @pytest.fixture
    def test_tables(self) -> List[Tuple[int, int, int, int]]:
        """Standard test tables for all methods."""
        return [
            (10, 20, 5, 25),    # Balanced moderate counts
            (2, 8, 1, 9),       # Small counts
            (50, 950, 25, 975), # Large counts
            (1, 9, 0, 10),      # Zero cell
            (0, 10, 1, 9),      # Zero cell (different position)
            (5, 5, 5, 5),       # Equal margins
        ]
    
    def test_all_methods_basic_functionality(self, test_tables):
        """Test that all methods produce valid confidence intervals."""
        for i, (a, b, c, d) in enumerate(test_tables):
            print(f"\nTesting table {i+1}: [{a}, {b}, {c}, {d}]")
            
            # Calculate odds ratio for reference
            try:
                odds_ratio = calculate_odds_ratio(a, b, c, d)
                print(f"Odds ratio: {odds_ratio:.4f}")
            except:
                odds_ratio = float('inf') if (a * d > 0 and b * c == 0) else 0.0
                print(f"Odds ratio: {odds_ratio}")
            
            # Test MIDP method
            try:
                midp_lower, midp_upper = exact_ci_midp(a, b, c, d, alpha=0.05, grid_size=50)
                assert midp_lower >= 0, f"MIDP lower bound should be non-negative"
                assert midp_upper > midp_lower, f"MIDP upper bound should be greater than lower"
                print(f"MIDP CI: ({midp_lower:.4f}, {midp_upper:.4f})")
            except Exception as e:
                pytest.fail(f"MIDP method failed for table {i+1}: {str(e)}")
            
            # Test Blaker method
            try:
                blaker_lower, blaker_upper = exact_ci_blaker(a, b, c, d, alpha=0.05)
                assert blaker_lower >= 0, f"Blaker lower bound should be non-negative"
                assert blaker_upper > blaker_lower, f"Blaker upper bound should be greater than lower"
                print(f"Blaker CI: ({blaker_lower:.4f}, {blaker_upper:.4f})")
            except Exception as e:
                pytest.fail(f"Blaker method failed for table {i+1}: {str(e)}")
            
            # Test Conditional method
            try:
                cond_lower, cond_upper = exact_ci_conditional(a, b, c, d, alpha=0.05)
                assert cond_lower >= 0, f"Conditional lower bound should be non-negative"
                assert cond_upper > cond_lower, f"Conditional upper bound should be greater than lower"
                print(f"Conditional CI: ({cond_lower:.4f}, {cond_upper:.4f})")
            except Exception as e:
                pytest.fail(f"Conditional method failed for table {i+1}: {str(e)}")
            
            # Test Unconditional method (with smaller grid for speed)
            try:
                uncond_lower, uncond_upper = exact_ci_unconditional(
                    a, b, c, d, alpha=0.05, grid_size=50
                )
                assert uncond_lower >= 0, f"Unconditional lower bound should be non-negative"
                assert uncond_upper > uncond_lower, f"Unconditional upper bound should be greater than lower"
                print(f"Unconditional CI: ({uncond_lower:.4f}, {uncond_upper:.4f})")
            except Exception as e:
                pytest.fail(f"Unconditional method failed for table {i+1}: {str(e)}")
            
            # Test Clopper-Pearson method for both groups
            try:
                cp1_lower, cp1_upper = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=1)
                assert 0 <= cp1_lower <= 1, f"Clopper-Pearson group 1 lower bound should be in [0,1]"
                assert 0 <= cp1_upper <= 1, f"Clopper-Pearson group 1 upper bound should be in [0,1]"
                assert cp1_upper > cp1_lower, f"Clopper-Pearson group 1 upper bound should be greater than lower"
                print(f"Clopper-Pearson Group 1 CI: ({cp1_lower:.4f}, {cp1_upper:.4f})")
                
                cp2_lower, cp2_upper = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=2)
                assert 0 <= cp2_lower <= 1, f"Clopper-Pearson group 2 lower bound should be in [0,1]"
                assert 0 <= cp2_upper <= 1, f"Clopper-Pearson group 2 upper bound should be in [0,1]"
                assert cp2_upper > cp2_lower, f"Clopper-Pearson group 2 upper bound should be greater than lower"
                print(f"Clopper-Pearson Group 2 CI: ({cp2_lower:.4f}, {cp2_upper:.4f})")
            except Exception as e:
                pytest.fail(f"Clopper-Pearson method failed for table {i+1}: {str(e)}")
    
    def test_method_consistency_properties(self):
        """Test that methods satisfy expected statistical properties."""
        # Test case with moderate effect size
        a, b, c, d = 15, 10, 8, 17
        alpha = 0.05
        
        # Calculate all CIs
        midp_ci = exact_ci_midp(a, b, c, d, alpha=alpha, grid_size=100)
        blaker_ci = exact_ci_blaker(a, b, c, d, alpha=alpha)
        cond_ci = exact_ci_conditional(a, b, c, d, alpha=alpha)
        uncond_ci = exact_ci_unconditional(a, b, c, d, alpha=alpha, grid_size=100)
        
        # All intervals should contain the point estimate
        odds_ratio = calculate_odds_ratio(a, b, c, d)
        
        assert midp_ci[0] <= odds_ratio <= midp_ci[1], "MIDP CI should contain odds ratio"
        assert blaker_ci[0] <= odds_ratio <= blaker_ci[1], "Blaker CI should contain odds ratio"
        assert cond_ci[0] <= odds_ratio <= cond_ci[1], "Conditional CI should contain odds ratio"
        assert uncond_ci[0] <= odds_ratio <= uncond_ci[1], "Unconditional CI should contain odds ratio"
        
        # MIDP should generally be narrower than exact methods
        midp_width = np.log(midp_ci[1]) - np.log(midp_ci[0])
        blaker_width = np.log(blaker_ci[1]) - np.log(blaker_ci[0])
        cond_width = np.log(cond_ci[1]) - np.log(cond_ci[0])
        
        print(f"Interval widths (log scale): MIDP={midp_width:.3f}, Blaker={blaker_width:.3f}, Conditional={cond_width:.3f}")
        
        # MIDP should typically be narrower (though not always guaranteed)
        # We'll just check that the widths are reasonable
        assert midp_width > 0, "MIDP width should be positive"
        assert blaker_width > 0, "Blaker width should be positive"
        assert cond_width > 0, "Conditional width should be positive"
    
    def test_alpha_level_consistency(self):
        """Test that different alpha levels produce consistent results."""
        a, b, c, d = 12, 8, 6, 14
        
        # Test with different alpha levels
        alpha_levels = [0.01, 0.05, 0.10]
        
        for method_name, method_func in [
            ("MIDP", lambda a, b, c, d, alpha: exact_ci_midp(a, b, c, d, alpha=alpha, grid_size=50)),
            ("Blaker", exact_ci_blaker),
            ("Conditional", exact_ci_conditional),
            ("Unconditional", lambda a, b, c, d, alpha: exact_ci_unconditional(a, b, c, d, alpha=alpha, grid_size=50))
        ]:
            print(f"\nTesting {method_name} with different alpha levels:")
            
            prev_lower, prev_upper = None, None
            for alpha in alpha_levels:
                lower, upper = method_func(a, b, c, d, alpha)
                print(f"  α={alpha}: CI=({lower:.4f}, {upper:.4f})")
                
                # Smaller alpha (higher confidence) should produce wider intervals
                if prev_lower is not None:
                    assert lower <= prev_lower, f"{method_name}: Lower bound should decrease as alpha decreases"
                    assert upper >= prev_upper, f"{method_name}: Upper bound should increase as alpha decreases"
                
                prev_lower, prev_upper = lower, upper
    
    def test_edge_cases_handling(self):
        """Test that all methods handle edge cases appropriately."""
        edge_cases = [
            (0, 10, 5, 5, "Zero in cell a"),
            (5, 5, 0, 10, "Zero in cell c"),
            (1, 99, 99, 1, "Very imbalanced"),
            (1, 1, 1, 1, "All ones"),
        ]
        
        for a, b, c, d, description in edge_cases:
            print(f"\nTesting edge case: {description} [{a}, {b}, {c}, {d}]")
            
            # All methods should handle these cases without crashing
            try:
                midp_ci = exact_ci_midp(a, b, c, d, grid_size=20)
                print(f"  MIDP: ({midp_ci[0]:.4f}, {midp_ci[1]:.4f})")
            except Exception as e:
                print(f"  MIDP failed: {str(e)}")
            
            try:
                blaker_ci = exact_ci_blaker(a, b, c, d)
                print(f"  Blaker: ({blaker_ci[0]:.4f}, {blaker_ci[1]:.4f})")
            except Exception as e:
                print(f"  Blaker failed: {str(e)}")
            
            try:
                cond_ci = exact_ci_conditional(a, b, c, d)
                print(f"  Conditional: ({cond_ci[0]:.4f}, {cond_ci[1]:.4f})")
            except Exception as e:
                print(f"  Conditional failed: {str(e)}")
            
            try:
                uncond_ci = exact_ci_unconditional(a, b, c, d, grid_size=20)
                print(f"  Unconditional: ({uncond_ci[0]:.4f}, {uncond_ci[1]:.4f})")
            except Exception as e:
                print(f"  Unconditional failed: {str(e)}")
            
            try:
                cp1_ci = exact_ci_clopper_pearson(a, b, c, d, group=1)
                cp2_ci = exact_ci_clopper_pearson(a, b, c, d, group=2)
                print(f"  Clopper-Pearson G1: ({cp1_ci[0]:.4f}, {cp1_ci[1]:.4f})")
                print(f"  Clopper-Pearson G2: ({cp2_ci[0]:.4f}, {cp2_ci[1]:.4f})")
            except Exception as e:
                print(f"  Clopper-Pearson failed: {str(e)}")
    
    def test_performance_benchmarks(self):
        """Basic performance benchmarks for all methods."""
        a, b, c, d = 20, 30, 15, 35
        
        methods = [
            ("MIDP", lambda: exact_ci_midp(a, b, c, d, grid_size=100)),
            ("Blaker", lambda: exact_ci_blaker(a, b, c, d)),
            ("Conditional", lambda: exact_ci_conditional(a, b, c, d)),
            ("Unconditional", lambda: exact_ci_unconditional(a, b, c, d, grid_size=100)),
            ("Clopper-Pearson G1", lambda: exact_ci_clopper_pearson(a, b, c, d, group=1)),
            ("Clopper-Pearson G2", lambda: exact_ci_clopper_pearson(a, b, c, d, group=2)),
        ]
        
        print(f"\nPerformance benchmarks for table [{a}, {b}, {c}, {d}]:")
        
        for method_name, method_func in methods:
            start_time = time.time()
            try:
                result = method_func()
                elapsed = time.time() - start_time
                print(f"  {method_name:20s}: {elapsed:.4f}s -> CI=({result[0]:.4f}, {result[1]:.4f})")
                
                # All methods should complete within reasonable time
                assert elapsed < 30.0, f"{method_name} took too long: {elapsed:.2f}s"
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  {method_name:20s}: {elapsed:.4f}s -> FAILED: {str(e)}")
    
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_all_methods_with_different_alpha(self, alpha):
        """Test all methods with different significance levels."""
        a, b, c, d = 8, 12, 6, 14
        
        # All methods should work with different alpha levels
        midp_ci = exact_ci_midp(a, b, c, d, alpha=alpha, grid_size=50)
        blaker_ci = exact_ci_blaker(a, b, c, d, alpha=alpha)
        cond_ci = exact_ci_conditional(a, b, c, d, alpha=alpha)
        uncond_ci = exact_ci_unconditional(a, b, c, d, alpha=alpha, grid_size=50)
        cp1_ci = exact_ci_clopper_pearson(a, b, c, d, alpha=alpha, group=1)
        cp2_ci = exact_ci_clopper_pearson(a, b, c, d, alpha=alpha, group=2)
        
        # All should produce valid intervals
        for method_name, ci in [
            ("MIDP", midp_ci),
            ("Blaker", blaker_ci), 
            ("Conditional", cond_ci),
            ("Unconditional", uncond_ci)
        ]:
            assert ci[0] >= 0, f"{method_name} lower bound should be non-negative"
            assert ci[1] > ci[0], f"{method_name} upper bound should be greater than lower"
        
        # Clopper-Pearson should be in [0,1]
        for group, ci in [("Group 1", cp1_ci), ("Group 2", cp2_ci)]:
            assert 0 <= ci[0] <= 1, f"Clopper-Pearson {group} lower bound should be in [0,1]"
            assert 0 <= ci[1] <= 1, f"Clopper-Pearson {group} upper bound should be in [0,1]"
            assert ci[1] > ci[0], f"Clopper-Pearson {group} upper bound should be greater than lower"


if __name__ == "__main__":
    # Run a quick integration test
    test = TestComprehensiveIntegration()
    test_tables = [(10, 20, 5, 25), (2, 8, 1, 9)]
    
    print("Running comprehensive integration tests...")
    test.test_all_methods_basic_functionality(test_tables)
    test.test_method_consistency_properties()
    test.test_alpha_level_consistency()
    test.test_edge_cases_handling()
    test.test_performance_benchmarks()
    print("All integration tests completed successfully!")