"""
Golden parity tests for ExactCIs refactoring safety.

These tests ensure that refactored code produces identical outputs to the original
implementation by comparing against golden fixtures. Critical for maintaining
correctness during incremental refactoring.

TIERED WARNING SYSTEM:
    🟢 LOW (< 0.1%):      Minor precision difference (likely acceptable refactoring artifact)
    🟡 MEDIUM (< 1%):     Moderate precision difference (may indicate algorithm change)
    🟠 HIGH (< 10%):      Significant precision difference (requires investigation)
    🔴 CRITICAL (≥ 10%):  Major numerical difference (likely bug or substantial algorithm change)

Environment variables:
    EXACTCIS_STRICT_PARITY=1        - Enable strict parity mode (no tolerances)
    EXACTCIS_REFACTORING_MODE=1     - Allow warnings for differences < 10% (fail for CRITICAL only)
    EXACTCIS_REGEN_GOLDEN=1         - Regenerate golden fixtures before testing
    EXACTCIS_REL_TOL=1e-9           - Custom relative tolerance (default: 1e-9)
    EXACTCIS_ABS_TOL=1e-12          - Custom absolute tolerance (default: 1e-12)

Usage during refactoring:
    # Standard mode - fail on any difference beyond tight tolerances
    uv run pytest tests/test_golden_parity.py
    
    # Refactoring mode - warn on differences, fail only on critical (>= 10%)
    EXACTCIS_REFACTORING_MODE=1 uv run pytest tests/test_golden_parity.py
    
    # Investigate specific large differences
    EXACTCIS_REFACTORING_MODE=1 uv run pytest tests/test_golden_parity.py -s -v
"""

import json
import os
import pytest
import warnings
from typing import Dict, Any, Tuple
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from exactcis import compute_all_cis, compute_all_rr_cis

# TODO: Normalize boolean env var parsing across test suite. Accept common truthy strings.
def _parse_env_bool(value: str, default: bool = False) -> bool:
    """Parse environment boolean values robustly.
    Accepts: 1, true, yes, on (case-insensitive) as True; 0, false, no, off as False.
    If value is None or unrecognized, returns default.
    """
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if v in {"0", "false", "no", "off", "n", "f"}:
        return False
    return default


class GoldenParityTester:
    """Manages golden fixture loading and parity validation."""
    
    def __init__(self):
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
        self.golden_file = os.path.join(self.fixtures_dir, 'golden_outputs.json')
        self.metadata_file = os.path.join(self.fixtures_dir, 'golden_metadata.json')
        # NOTE: Booleans accept 1/true/yes/on (case-insensitive)
        self.strict_parity = _parse_env_bool(os.getenv('EXACTCIS_STRICT_PARITY', '0'))
        self.regen_golden = _parse_env_bool(os.getenv('EXACTCIS_REGEN_GOLDEN', '0'))
        
        # Flexible tolerance configuration
        self.rel_tolerance = float(os.getenv('EXACTCIS_REL_TOL', '1e-9'))
        self.abs_tolerance = float(os.getenv('EXACTCIS_ABS_TOL', '1e-12'))
        
    def load_golden_fixtures(self) -> Dict[str, Any]:
        """Load golden fixtures, regenerating if requested or missing."""
        if self.regen_golden or not os.path.exists(self.golden_file):
            print("🔄 Regenerating golden fixtures...")
            from tests.fixtures.generate_golden_fixtures import generate_golden_fixtures
            generate_golden_fixtures()
            
        if not os.path.exists(self.golden_file):
            pytest.skip("Golden fixtures not available. Run generate_golden_fixtures.py first.")
            
        with open(self.golden_file, 'r') as f:
            return json.load(f)
    
    def compare_ci_results(self, expected: Dict[str, Tuple[float, float]], 
                          actual: Dict[str, Tuple[float, float]], 
                          table: Tuple[int, int, int, int], 
                          alpha: float, 
                          method_family: str) -> None:
        """
        Compare CI results with appropriate tolerance handling.
        
        Args:
            expected: Golden fixture results
            actual: Current implementation results
            table: The 2x2 table (a,b,c,d)
            alpha: Significance level
            method_family: 'OR' or 'RR' for error reporting
        """
        # Check method coverage
        expected_methods = set(expected.keys())
        actual_methods = set(actual.keys())
        
        # TODO: In refactoring mode, consider warning (not failing) on method-set differences
        # to allow transitional phases where methods are added/renamed intentionally.
        assert expected_methods == actual_methods, (
            f"{method_family} method mismatch for table {table}, alpha={alpha}. "
            f"Expected: {sorted(expected_methods)}, Got: {sorted(actual_methods)}"
        )
        
        # Compare each method's results
        for method in expected_methods:
            exp_lower, exp_upper = expected[method]
            act_lower, act_upper = actual[method]
            
            if self.strict_parity:
                # Strict mode: exact match required
                assert exp_lower == act_lower and exp_upper == act_upper, (
                    f"{method_family} {method} parity failure for table {table}, alpha={alpha}. "
                    f"Expected: ({exp_lower}, {exp_upper}), Got: ({act_lower}, {act_upper})"
                )
            else:
                # Standard mode: handle floating point precision and infinities
                self._assert_ci_close(exp_lower, act_lower, method, method_family, table, alpha, "lower")
                self._assert_ci_close(exp_upper, act_upper, method, method_family, table, alpha, "upper")
    
    def _assert_ci_close(self, expected: float, actual: float, method: str, 
                        method_family: str, table: Tuple[int, int, int, int], 
                        alpha: float, bound: str) -> None:
        """Assert that CI bounds are numerically close with proper inf handling."""
        # Handle infinity cases
        if expected == float('inf') and actual == float('inf'):
            return
        if expected == float('-inf') and actual == float('-inf'):
            return
        if expected == float('inf') or actual == float('inf'):
            assert False, (
                f"{method_family} {method} {bound} bound infinity mismatch for table {table}, "
                f"alpha={alpha}. Expected: {expected}, Got: {actual}"
            )
        
        # Handle finite values with configurable tolerance
        # Note (Aug 2025): Default 1e-9 relative tolerance accommodates tiny, acceptable
        # numerical differences from Phase 1 solver improvements (centralized bracketing, 
        # enhanced root finding, plateau detection). Configurable via environment variables:
        # EXACTCIS_REL_TOL=1e-10 for tighter tolerance, or EXACTCIS_STRICT_PARITY=1 for exact match.
        rel_tol = self.rel_tolerance
        abs_tol = self.abs_tolerance
        
        if abs(expected) < abs_tol and abs(actual) < abs_tol:
            return  # Both effectively zero
            
        if abs(expected - actual) <= abs_tol + rel_tol * max(abs(expected), abs(actual)):
            return  # Within tolerance
        
        # Tiered warning system for refactoring-induced differences
        rel_diff = abs(expected - actual) / max(abs(expected), abs(actual))
        abs_diff = abs(expected - actual)
        
        # Determine concern level based on magnitude of difference
        if rel_diff < 0.001:  # < 0.1% difference
            concern_level = "🟢 LOW"
            message = "Minor precision difference (likely acceptable refactoring artifact)"
        elif rel_diff < 0.01:  # < 1% difference  
            concern_level = "🟡 MEDIUM"
            message = "Moderate precision difference (may indicate algorithm change)"
        elif rel_diff < 0.1:   # < 10% difference
            concern_level = "🟠 HIGH"
            message = "Significant precision difference (requires investigation)"
        else:  # >= 10% difference
            concern_level = "🔴 CRITICAL"
            message = "Major numerical difference (likely indicates bug or substantial algorithm change)"
        
        # In refactoring mode, warn but don't fail for acceptable differences
        refactoring_mode = _parse_env_bool(os.getenv('EXACTCIS_REFACTORING_MODE', '0'))
        
        warning_msg = (
            f"\n{'='*80}\n"
            f"GOLDEN PARITY DIFFERENCE DETECTED [{concern_level}]\n"
            f"{'='*80}\n"
            f"Method: {method_family}.{method}.{bound}\n"
            f"Table: {table}, Alpha: {alpha}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            f"Abs Diff: {abs_diff:.2e}\n" 
            f"Rel Diff: {rel_diff:.4%}\n"
            f"Assessment: {message}\n"
            f"{'='*80}"
        )
        
        if refactoring_mode and rel_diff < 0.1:  # Allow up to 10% in refactoring mode
            print(warning_msg)
            warnings.warn(f"Golden parity difference: {rel_diff:.4%} - {message}")
            return
        else:
            # Fail the test with detailed information
            assert False, warning_msg


@pytest.fixture(scope="session")
def golden_tester():
    """Session-wide golden parity tester instance."""
    return GoldenParityTester()


@pytest.fixture(scope="session") 
def golden_fixtures(golden_tester):
    """Load golden fixtures once per test session."""
    return golden_tester.load_golden_fixtures()


def extract_test_cases(golden_fixtures):
    """Extract individual test cases from golden fixtures."""
    test_cases = []
    for table_key, table_data in golden_fixtures.items():
        table = table_data['table']
        for alpha_key, alpha_data in table_data['alpha_results'].items():
            alpha = float(alpha_key.replace('alpha_', ''))
            # Skip cases with errors in the golden fixtures
            if alpha_data.get('errors'):
                continue
            test_cases.append((table, alpha, alpha_data))
    return test_cases


class TestGoldenParity:
    """Golden parity tests for all methods and configurations."""
    
    @pytest.mark.parametrize("table,alpha,expected_data", 
                           extract_test_cases(GoldenParityTester().load_golden_fixtures()))
    def test_or_methods_parity(self, table, alpha, expected_data, golden_tester):
        """Test that all OR methods produce identical results to golden fixtures."""
        a, b, c, d = table
        
        # Compute current results
        try:
            actual_results = compute_all_cis(a, b, c, d, alpha)
        except Exception as e:
            pytest.fail(f"OR methods failed for table {table}, alpha={alpha}: {e}")
            
        # Compare with golden fixtures
        expected_results = expected_data['or_results']
        golden_tester.compare_ci_results(expected_results, actual_results, table, alpha, "OR")
    
    @pytest.mark.parametrize("table,alpha,expected_data",
                           extract_test_cases(GoldenParityTester().load_golden_fixtures()))
    def test_rr_methods_parity(self, table, alpha, expected_data, golden_tester):
        """Test that all RR methods produce identical results to golden fixtures."""
        a, b, c, d = table
        
        # Compute current results
        try:
            actual_results = compute_all_rr_cis(a, b, c, d, alpha)
        except Exception as e:
            pytest.fail(f"RR methods failed for table {table}, alpha={alpha}: {e}")
            
        # Compare with golden fixtures
        expected_results = expected_data['rr_results']  
        golden_tester.compare_ci_results(expected_results, actual_results, table, alpha, "RR")
    
    def test_golden_fixtures_metadata(self, golden_fixtures):
        """Validate that golden fixtures have expected structure and metadata."""
        # Check that we have reasonable test coverage
        assert len(golden_fixtures) >= 30, "Should have at least 30 test tables"
        
        # Check structure of a sample fixture
        sample_key = next(iter(golden_fixtures.keys()))
        sample_data = golden_fixtures[sample_key]
        
        assert 'table' in sample_data
        assert 'alpha_results' in sample_data
        assert len(sample_data['table']) == 4  # (a,b,c,d)
        
        # Check that we test multiple alpha levels
        alpha_count = len(sample_data['alpha_results'])
        assert alpha_count >= 2, "Should test multiple alpha levels"
    
    @pytest.mark.slow
    def test_comprehensive_parity_check(self, golden_tester):
        """
        Comprehensive parity check that can be run independently.
        Tests all fixtures in a single test for CI efficiency.
        """
        golden_fixtures = golden_tester.load_golden_fixtures()
        
        or_failures = []
        rr_failures = []
        
        for table_key, table_data in golden_fixtures.items():
            table = table_data['table']
            a, b, c, d = table
            
            for alpha_key, expected_data in table_data['alpha_results'].items():
                alpha = float(alpha_key.replace('alpha_', ''))
                
                # Skip cases with errors in golden fixtures
                if expected_data.get('errors'):
                    continue
                
                # Test OR methods
                if expected_data['or_results']:
                    try:
                        actual_or = compute_all_cis(a, b, c, d, alpha)
                        golden_tester.compare_ci_results(
                            expected_data['or_results'], actual_or, table, alpha, "OR"
                        )
                    except Exception as e:
                        or_failures.append(f"Table {table}, alpha={alpha}: {e}")
                
                # Test RR methods  
                if expected_data['rr_results']:
                    try:
                        actual_rr = compute_all_rr_cis(a, b, c, d, alpha)
                        golden_tester.compare_ci_results(
                            expected_data['rr_results'], actual_rr, table, alpha, "RR"
                        )
                    except Exception as e:
                        rr_failures.append(f"Table {table}, alpha={alpha}: {e}")
        
        # Report all failures at once
        if or_failures or rr_failures:
            failure_msg = []
            if or_failures:
                failure_msg.append(f"OR method failures ({len(or_failures)}):")
                failure_msg.extend(f"  - {f}" for f in or_failures[:10])  # Show first 10
                if len(or_failures) > 10:
                    failure_msg.append(f"  ... and {len(or_failures) - 10} more")
            
            if rr_failures:
                failure_msg.append(f"RR method failures ({len(rr_failures)}):")
                failure_msg.extend(f"  - {f}" for f in rr_failures[:10])  # Show first 10
                if len(rr_failures) > 10:
                    failure_msg.append(f"  ... and {len(rr_failures) - 10} more")
                    
            pytest.fail("\n".join(failure_msg))


if __name__ == "__main__":
    # Allow running this module directly for quick parity checks
    import subprocess
    import sys
    
    print("🔍 Running golden parity tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", 
        "--tb=short", "-x"  # Stop on first failure for quick feedback
    ])
    sys.exit(result.returncode)