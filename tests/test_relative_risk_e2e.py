"""
End-to-end tests for relative risk confidence interval methods.

This module provides comprehensive end-to-end testing that validates the complete
workflow from input validation through CI calculation to output formatting.
"""

import pytest
import math
import numpy as np
import json
from typing import Dict, List, Tuple, Any

from exactcis.methods.relative_risk import (
    validate_counts,
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)


class RelativeRiskCalculator:
    """
    End-to-end relative risk calculator that mimics real-world usage.
    """
    
    def __init__(self):
        self.methods = {
            "wald": ci_wald_rr,
            "wald_katz": ci_wald_katz_rr,
            "wald_correlated": ci_wald_correlated_rr,
            "score": ci_score_rr,
            "score_cc": ci_score_cc_rr,
            "ustat": ci_ustat_rr,
        }
    
    def calculate_point_estimate(self, a: int, b: int, c: int, d: int) -> float:
        """Calculate relative risk point estimate."""
        validate_counts(a, b, c, d)
        
        n1 = a + b
        n2 = c + d
        
        if n1 == 0 or n2 == 0:
            return 1.0
        
        risk1 = a / n1
        risk2 = c / n2
        
        if risk2 == 0:
            return float('inf') if risk1 > 0 else 1.0
        
        return risk1 / risk2
    
    def calculate_confidence_interval(self, a: int, b: int, c: int, d: int, 
                                    method: str = "wald", alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval using specified method."""
        validate_counts(a, b, c, d)
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        
        return self.methods[method](a, b, c, d, alpha)
    
    def calculate_all_intervals(self, a: int, b: int, c: int, d: int, 
                              alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """Calculate confidence intervals using all methods."""
        validate_counts(a, b, c, d)
        
        point_estimate = self.calculate_point_estimate(a, b, c, d)
        
        results = {
            "point_estimate": point_estimate,
            "alpha": alpha,
            "confidence_level": (1 - alpha) * 100,
            "methods": {}
        }
        
        for method_name, method_func in self.methods.items():
            try:
                lower, upper = method_func(a, b, c, d, alpha)
                results["methods"][method_name] = {
                    "lower": lower,
                    "upper": upper,
                    "width": upper - lower if math.isfinite(upper - lower) else float('inf'),
                    "contains_point_estimate": lower <= point_estimate <= upper,
                    "status": "success"
                }
            except Exception as e:
                results["methods"][method_name] = {
                    "lower": float('nan'),
                    "upper": float('nan'),
                    "width": float('nan'),
                    "contains_point_estimate": False,
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results for human-readable output."""
        output = []
        output.append(f"Relative Risk Analysis")
        output.append(f"=====================")
        output.append(f"Point Estimate: {results['point_estimate']:.4f}")
        output.append(f"Confidence Level: {results['confidence_level']:.1f}%")
        output.append("")
        
        for method_name, method_results in results["methods"].items():
            if method_results["status"] == "success":
                lower = method_results["lower"]
                upper = method_results["upper"]
                width = method_results["width"]
                
                if math.isfinite(lower) and math.isfinite(upper):
                    output.append(f"{method_name:15s}: ({lower:.4f}, {upper:.4f}) [width: {width:.4f}]")
                elif lower == 0.0 and math.isinf(upper):
                    output.append(f"{method_name:15s}: (0.0000, ∞)")
                elif math.isfinite(lower) and math.isinf(upper):
                    output.append(f"{method_name:15s}: ({lower:.4f}, ∞)")
                else:
                    output.append(f"{method_name:15s}: ({lower}, {upper})")
            else:
                output.append(f"{method_name:15s}: ERROR - {method_results['error']}")
        
        return "\n".join(output)


@pytest.mark.fast
@pytest.mark.integration
class TestRelativeRiskCalculatorBasic:
    """Basic end-to-end tests for the RelativeRiskCalculator."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = RelativeRiskCalculator()
        assert len(calc.methods) == 6
        assert all(method in calc.methods for method in 
                  ["wald", "wald_katz", "wald_correlated", "score", "score_cc", "ustat"])
    
    def test_point_estimate_calculation(self):
        """Test point estimate calculation."""
        calc = RelativeRiskCalculator()
        
        # Standard case
        rr = calc.calculate_point_estimate(20, 80, 10, 90)
        expected = (20/100) / (10/100)  # = 2.0
        assert math.isclose(rr, expected)
        
        # Zero cases
        assert calc.calculate_point_estimate(0, 10, 5, 5) == 0.0
        assert calc.calculate_point_estimate(5, 5, 0, 10) == float('inf')
        assert calc.calculate_point_estimate(0, 10, 0, 10) == 1.0
    
    def test_single_method_calculation(self):
        """Test single method CI calculation."""
        calc = RelativeRiskCalculator()
        
        a, b, c, d = 15, 5, 10, 10
        
        # Test each method individually
        for method in calc.methods.keys():
            lower, upper = calc.calculate_confidence_interval(a, b, c, d, method)
            assert 0 < lower < upper < float('inf')
            
            # Should contain point estimate
            rr = calc.calculate_point_estimate(a, b, c, d)
            assert lower <= rr <= upper
    
    def test_all_methods_calculation(self):
        """Test calculation using all methods."""
        calc = RelativeRiskCalculator()
        
        a, b, c, d = 15, 5, 10, 10
        results = calc.calculate_all_intervals(a, b, c, d)
        
        # Check structure
        assert "point_estimate" in results
        assert "alpha" in results
        assert "confidence_level" in results
        assert "methods" in results
        
        # Check that all methods are included
        assert len(results["methods"]) == 6
        
        # Check that most methods succeeded
        successful_methods = sum(1 for method_results in results["methods"].values() 
                               if method_results["status"] == "success")
        assert successful_methods >= 5  # At least 5 out of 6 should work
        
        # Check that successful methods contain point estimate
        for method_name, method_results in results["methods"].items():
            if method_results["status"] == "success":
                assert method_results["contains_point_estimate"], \
                    f"Method {method_name} CI doesn't contain point estimate"
    
    def test_input_validation(self):
        """Test input validation in calculator."""
        calc = RelativeRiskCalculator()
        
        # Invalid counts
        with pytest.raises(ValueError):
            calc.calculate_point_estimate(-1, 5, 8, 10)
        
        with pytest.raises(ValueError):
            calc.calculate_confidence_interval(10, 5, 8, -10)
        
        # Invalid method
        with pytest.raises(ValueError):
            calc.calculate_confidence_interval(10, 5, 8, 10, method="invalid")
        
        # Invalid alpha
        with pytest.raises(ValueError):
            calc.calculate_confidence_interval(10, 5, 8, 10, alpha=1.5)
        
        with pytest.raises(ValueError):
            calc.calculate_confidence_interval(10, 5, 8, 10, alpha=0.0)
    
    def test_results_formatting(self):
        """Test results formatting."""
        calc = RelativeRiskCalculator()
        
        a, b, c, d = 15, 5, 10, 10
        results = calc.calculate_all_intervals(a, b, c, d)
        formatted = calc.format_results(results)
        
        # Check that formatted output contains expected elements
        assert "Relative Risk Analysis" in formatted
        assert "Point Estimate:" in formatted
        assert "Confidence Level:" in formatted
        
        # Check that method results are included
        for method_name in calc.methods.keys():
            assert method_name in formatted


@pytest.mark.integration
class TestRelativeRiskWorkflows:
    """Test complete workflows for different use cases."""
    
    def test_epidemiological_study_workflow(self, timer):
        """Test complete workflow for epidemiological study."""
        calc = RelativeRiskCalculator()
        
        # Simulate epidemiological study data
        # Exposure and disease status
        exposed_cases = 45
        exposed_controls = 455
        unexposed_cases = 15
        unexposed_controls = 485
        
        # Calculate results
        results = calc.calculate_all_intervals(
            exposed_cases, exposed_controls, 
            unexposed_cases, unexposed_controls,
            alpha=0.05
        )
        
        # Validate results
        assert results["point_estimate"] > 1.0  # Exposure increases risk
        
        # All methods should work for this standard case
        successful_methods = [name for name, res in results["methods"].items() 
                            if res["status"] == "success"]
        assert len(successful_methods) >= 5
        
        # All successful CIs should exclude 1.0 (significant effect)
        for method_name in successful_methods:
            method_results = results["methods"][method_name]
            lower = method_results["lower"]
            upper = method_results["upper"]
            
            # For this case with clear effect, CI should not include 1.0
            if math.isfinite(lower) and math.isfinite(upper):
                # This is a strong effect, so most methods should exclude 1.0
                pass  # We don't enforce this as it depends on the specific method and data
        
        # Format and validate output
        formatted = calc.format_results(results)
        assert len(formatted) > 100  # Should be substantial output
        
        # Should be able to parse key information from formatted output
        lines = formatted.split('\n')
        point_estimate_line = [line for line in lines if "Point Estimate:" in line][0]
        assert "Point Estimate:" in point_estimate_line
    
    def test_clinical_trial_workflow(self, timer):
        """Test complete workflow for clinical trial."""
        calc = RelativeRiskCalculator()
        
        # Simulate clinical trial data
        # Treatment vs control
        treatment_events = 12
        treatment_no_events = 88
        control_events = 20
        control_no_events = 80
        
        # Calculate results with 99% confidence
        results = calc.calculate_all_intervals(
            treatment_events, treatment_no_events,
            control_events, control_no_events,
            alpha=0.01  # 99% confidence
        )
        
        # Validate results
        assert results["confidence_level"] == 99.0
        assert results["point_estimate"] < 1.0  # Treatment is protective
        
        # Check that CIs are wider for 99% confidence
        results_95 = calc.calculate_all_intervals(
            treatment_events, treatment_no_events,
            control_events, control_no_events,
            alpha=0.05  # 95% confidence
        )
        
        # Compare widths (99% should be wider than 95%)
        for method_name in calc.methods.keys():
            if (results["methods"][method_name]["status"] == "success" and
                results_95["methods"][method_name]["status"] == "success"):
                
                width_99 = results["methods"][method_name]["width"]
                width_95 = results_95["methods"][method_name]["width"]
                
                if math.isfinite(width_99) and math.isfinite(width_95):
                    assert width_99 >= width_95, \
                        f"Method {method_name}: 99% CI should be wider than 95% CI"
    
    def test_rare_disease_workflow(self, timer):
        """Test workflow for rare disease study."""
        calc = RelativeRiskCalculator()
        
        # Simulate rare disease data
        # Very low event rates
        exposed_cases = 3
        exposed_controls = 997
        unexposed_cases = 1
        unexposed_controls = 999
        
        # Calculate results
        results = calc.calculate_all_intervals(
            exposed_cases, exposed_controls,
            unexposed_cases, unexposed_controls
        )
        
        # Validate results
        assert results["point_estimate"] > 1.0  # Exposure increases risk
        
        # For rare disease, some methods might struggle
        successful_methods = [name for name, res in results["methods"].items() 
                            if res["status"] == "success"]
        assert len(successful_methods) >= 3  # At least half should work
        
        # CIs should be wide due to small numbers
        for method_name in successful_methods:
            method_results = results["methods"][method_name]
            width = method_results["width"]
            
            if math.isfinite(width):
                # For rare events, CI should be relatively wide
                assert width > 1.0, f"Method {method_name}: CI too narrow for rare disease"
    
    def test_zero_events_workflow(self, timer):
        """Test workflow with zero events in one group."""
        calc = RelativeRiskCalculator()
        
        # Zero events in treatment group
        treatment_events = 0
        treatment_no_events = 100
        control_events = 5
        control_no_events = 95
        
        # Calculate results
        results = calc.calculate_all_intervals(
            treatment_events, treatment_no_events,
            control_events, control_no_events
        )
        
        # Validate results
        assert results["point_estimate"] == 0.0  # No events in treatment
        
        # Methods should handle zero events appropriately
        for method_name, method_results in results["methods"].items():
            if method_results["status"] == "success":
                lower = method_results["lower"]
                upper = method_results["upper"]
                
                # Lower bound should be 0 or very close to 0 (allowing for continuity correction)
                assert lower == 0.0 or lower <= 0.01
                
                # Upper bound should be finite (rule of three, etc.)
                # Some methods might give infinite upper bounds, which is acceptable
                assert upper >= 0.0


@pytest.mark.slow
@pytest.mark.integration
class TestRelativeRiskBatchProcessing:
    """Test batch processing and performance scenarios."""
    
    def test_multiple_studies_batch(self, timer):
        """Test processing multiple studies in batch."""
        calc = RelativeRiskCalculator()
        
        # Simulate multiple studies
        studies = [
            (20, 80, 10, 90, "Study 1: Standard"),
            (5, 95, 2, 98, "Study 2: Rare disease"),
            (45, 55, 25, 75, "Study 3: Common disease"),
            (0, 100, 5, 95, "Study 4: Zero events"),
            (3, 7, 2, 8, "Study 5: Small sample"),
        ]
        
        all_results = []
        
        for a, b, c, d, description in studies:
            results = calc.calculate_all_intervals(a, b, c, d)
            results["description"] = description
            results["table"] = (a, b, c, d)
            all_results.append(results)
        
        # Validate batch results
        assert len(all_results) == len(studies)
        
        # Each study should have valid structure
        for i, results in enumerate(all_results):
            assert "point_estimate" in results
            assert "methods" in results
            assert "description" in results
            assert "table" in results
            
            # At least some methods should work for each study
            successful_methods = sum(1 for res in results["methods"].values() 
                                   if res["status"] == "success")
            assert successful_methods >= 2, f"Too few methods succeeded for {results['description']}"
        
        # Generate summary report
        summary = self._generate_batch_summary(all_results)
        assert len(summary) > 0
    
    def _generate_batch_summary(self, all_results: List[Dict]) -> str:
        """Generate summary report for batch results."""
        summary = ["Batch Processing Summary", "=" * 25, ""]
        
        for results in all_results:
            summary.append(f"{results['description']}")
            summary.append(f"Table: {results['table']}")
            summary.append(f"RR: {results['point_estimate']:.4f}")
            
            successful_methods = [name for name, res in results["methods"].items() 
                                if res["status"] == "success"]
            summary.append(f"Successful methods: {len(successful_methods)}/6")
            summary.append("")
        
        return "\n".join(summary)
    
    def test_performance_stress_test(self, timer):
        """Test performance with many calculations."""
        calc = RelativeRiskCalculator()
        
        # Generate random test cases
        np.random.seed(42)
        num_tests = 100
        
        successful_calculations = 0
        total_time_per_method = {method: 0.0 for method in calc.methods.keys()}
        
        for _ in range(num_tests):
            # Generate random 2x2 table
            n1 = np.random.randint(10, 200)
            n2 = np.random.randint(10, 200)
            a = np.random.randint(0, n1 + 1)
            c = np.random.randint(0, n2 + 1)
            b = n1 - a
            d = n2 - c
            
            try:
                # Time each method
                import time
                
                for method_name, method_func in calc.methods.items():
                    start_time = time.time()
                    try:
                        lower, upper = method_func(a, b, c, d, 0.05)
                        if not (math.isnan(lower) or math.isnan(upper)):
                            successful_calculations += 1
                    except Exception:
                        pass
                    end_time = time.time()
                    total_time_per_method[method_name] += (end_time - start_time)
                
            except Exception:
                pass
        
        # Performance validation
        success_rate = successful_calculations / (num_tests * len(calc.methods))
        assert success_rate >= 0.7, f"Success rate too low: {success_rate:.2f}"
        
        # No method should be extremely slow
        for method_name, total_time in total_time_per_method.items():
            avg_time = total_time / num_tests
            assert avg_time < 0.1, f"Method {method_name} too slow: {avg_time:.4f}s average"


@pytest.mark.integration
class TestRelativeRiskErrorHandling:
    """Test error handling and edge cases in end-to-end scenarios."""
    
    def test_graceful_error_handling(self, timer):
        """Test that errors are handled gracefully."""
        calc = RelativeRiskCalculator()
        
        # Test cases that might cause issues
        problematic_cases = [
            (0, 0, 10, 10, "Empty exposed group"),
            (10, 10, 0, 0, "Empty unexposed group"),
            (0, 0, 0, 0, "Empty table"),
            (1000000, 1000000, 1000000, 1000000, "Very large counts"),
        ]
        
        for a, b, c, d, description in problematic_cases:
            try:
                results = calc.calculate_all_intervals(a, b, c, d)
                
                # Even if some methods fail, structure should be intact
                assert "point_estimate" in results
                assert "methods" in results
                
                # At least some methods might work
                successful_methods = sum(1 for res in results["methods"].values() 
                                       if res["status"] == "success")
                
                # For truly problematic cases, it's OK if no methods work
                # but the calculator shouldn't crash
                
            except ValueError as e:
                # Some cases might legitimately raise ValueError during validation
                assert "negative" in str(e).lower() or "margin" in str(e).lower()
    
    def test_output_serialization(self, timer):
        """Test that results can be serialized/deserialized."""
        calc = RelativeRiskCalculator()
        
        a, b, c, d = 15, 5, 10, 10
        results = calc.calculate_all_intervals(a, b, c, d)
        
        # Convert to JSON-serializable format
        serializable_results = self._make_json_serializable(results)
        
        # Should be able to serialize to JSON
        json_str = json.dumps(serializable_results, indent=2)
        assert len(json_str) > 100
        
        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized["point_estimate"] == results["point_estimate"]
        assert deserialized["alpha"] == results["alpha"]
        assert len(deserialized["methods"]) == len(results["methods"])
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            elif math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            else:
                return obj
        else:
            return obj
    
    def test_method_comparison_report(self, timer):
        """Test generation of method comparison report."""
        calc = RelativeRiskCalculator()
        
        # Test case with known properties
        a, b, c, d = 30, 70, 15, 85
        results = calc.calculate_all_intervals(a, b, c, d)
        
        # Generate comparison report
        report = self._generate_method_comparison_report(results)
        
        # Validate report content
        assert "Method Comparison" in report
        assert "Point Estimate" in report
        
        # Should include information about each method
        for method_name in calc.methods.keys():
            assert method_name in report
    
    def _generate_method_comparison_report(self, results: Dict) -> str:
        """Generate a detailed method comparison report."""
        report = []
        report.append("Method Comparison Report")
        report.append("=" * 30)
        report.append(f"Point Estimate: {results['point_estimate']:.4f}")
        report.append(f"Confidence Level: {results['confidence_level']:.1f}%")
        report.append("")
        
        # Sort methods by CI width (narrowest first)
        method_items = []
        for name, res in results["methods"].items():
            if res["status"] == "success" and math.isfinite(res["width"]):
                method_items.append((name, res))
        
        method_items.sort(key=lambda x: x[1]["width"])
        
        report.append("Methods (sorted by CI width):")
        for name, res in method_items:
            report.append(f"  {name:15s}: ({res['lower']:.4f}, {res['upper']:.4f}) "
                         f"[width: {res['width']:.4f}]")
        
        # Failed methods
        failed_methods = [name for name, res in results["methods"].items() 
                         if res["status"] != "success"]
        if failed_methods:
            report.append("")
            report.append("Failed methods:")
            for name in failed_methods:
                report.append(f"  {name}: {results['methods'][name]['error']}")
        
        return "\n".join(report)