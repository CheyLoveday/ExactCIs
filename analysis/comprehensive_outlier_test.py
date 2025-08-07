#!/usr/bin/env python3
"""
Comprehensive Outlier Detection Test
Run all CI methods across multiple table configurations to identify remaining outliers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import sys
import os

# Import our ExactCIs implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from exactcis import compute_all_cis

# Import gold standard calculator
from test_ci_methods import CICalculator


def define_test_scenarios() -> List[Dict]:
    """Define comprehensive test scenarios covering various edge cases."""
    
    scenarios = [
        # Original three scenarios
        {
            "name": "Small (N=200)",
            "description": "10/100 vs 2/100",
            "a": 10, "b": 90, "c": 2, "d": 98,
            "or_expected": 5.44
        },
        {
            "name": "Medium (N=2000)", 
            "description": "50/1000 vs 10/1000",
            "a": 50, "b": 950, "c": 10, "d": 990,
            "or_expected": 5.21
        },
        {
            "name": "Large (N=4000)",
            "description": "100/2000 vs 20/2000", 
            "a": 100, "b": 1900, "c": 20, "d": 1980,
            "or_expected": 5.21
        },
        
        # Additional edge cases
        {
            "name": "Very Small (N=40)",
            "description": "5/15 vs 2/18",
            "a": 5, "b": 10, "c": 2, "d": 16,
            "or_expected": 4.0
        },
        {
            "name": "Balanced Small",
            "description": "8/42 vs 4/46", 
            "a": 8, "b": 42, "c": 4, "d": 46,
            "or_expected": 2.19
        },
        {
            "name": "High OR Large",
            "description": "80/920 vs 10/990",
            "a": 80, "b": 920, "c": 10, "d": 990,
            "or_expected": 8.7
        },
        {
            "name": "Low OR Medium", 
            "description": "15/485 vs 30/470",
            "a": 15, "b": 485, "c": 30, "d": 470,
            "or_expected": 0.48
        },
        {
            "name": "Near Equal",
            "description": "25/475 vs 24/476",
            "a": 25, "b": 475, "c": 24, "d": 476,
            "or_expected": 1.04
        },
        {
            "name": "Zero Cell Small",
            "description": "0/20 vs 5/15", 
            "a": 0, "b": 20, "c": 5, "d": 15,
            "or_expected": 0.0
        },
        {
            "name": "Single Count",
            "description": "1/19 vs 5/15",
            "a": 1, "b": 19, "c": 5, "d": 15, 
            "or_expected": 0.16
        }
    ]
    
    return scenarios


def run_gold_standard_for_scenario(scenario: Dict) -> Dict:
    """Run gold standard calculations for a scenario."""
    table = np.array([[scenario["a"], scenario["b"]], 
                     [scenario["c"], scenario["d"]]])
    calculator = CICalculator(table, alpha=0.05)
    
    results = {}
    try:
        or_wald, or_wald_ci = calculator.odds_ratio_wald()
        results['OR Wald'] = {"estimate": or_wald, "ci": or_wald_ci}
    except:
        results['OR Wald'] = {"estimate": float('nan'), "ci": (float('nan'), float('nan'))}
    
    try:
        or_fisher, or_fisher_ci = calculator.odds_ratio_exact_fisher()
        results['OR Fisher Exact'] = {"estimate": or_fisher, "ci": or_fisher_ci}
    except:
        results['OR Fisher Exact'] = {"estimate": float('nan'), "ci": (float('nan'), float('nan'))}
        
    return results


def run_exactcis_for_scenario(scenario: Dict) -> Dict:
    """Run ExactCIs calculations for a scenario."""
    a, b, c, d = scenario["a"], scenario["b"], scenario["c"], scenario["d"]
    
    # Calculate point estimate
    if b * c == 0:
        or_estimate = float('inf') if a > 0 else 0.0
    else:
        or_estimate = (a * d) / (b * c)
    
    try:
        ci_results = compute_all_cis(a, b, c, d, alpha=0.05)
        
        results = {}
        for method, (lower, upper) in ci_results.items():
            results[method] = {"estimate": or_estimate, "ci": (lower, upper)}
        
        return results
    except Exception as e:
        print(f"Error in scenario {scenario['name']}: {e}")
        return {}


def identify_outliers(gold_results: Dict, exactcis_results: Dict, scenario: Dict) -> List[str]:
    """Identify outliers by comparing with gold standards and other methods."""
    outliers = []
    
    # Get gold standard Fisher as reference
    if 'OR Fisher Exact' in gold_results:
        gold_fisher_ci = gold_results['OR Fisher Exact']['ci']
        gold_lower, gold_upper = gold_fisher_ci
        
        # Check each of our methods
        for method_name, method_data in exactcis_results.items():
            if 'conditional' in method_name:  # Skip conditional vs gold Fisher comparison
                continue
                
            our_lower, our_upper = method_data['ci']
            
            # Check for severe outliers (>3x deviation from gold standard)
            if not (np.isnan(gold_lower) or np.isnan(gold_upper) or 
                   np.isinf(our_lower) or np.isinf(our_upper)):
                
                if gold_upper > 0:  # Avoid division by zero
                    upper_ratio = our_upper / gold_upper
                    if upper_ratio > 3.0:
                        outliers.append(f"{method_name}: Upper bound {our_upper:.2f} is {upper_ratio:.1f}x gold standard {gold_upper:.2f}")
                
                if gold_lower > 0:  # Avoid division by zero  
                    lower_ratio = our_lower / gold_lower if our_lower > 0 else 0
                    if our_lower > 0 and lower_ratio > 3.0:
                        outliers.append(f"{method_name}: Lower bound {our_lower:.2f} is {lower_ratio:.1f}x gold standard {gold_lower:.2f}")
    
    # Check for internal inconsistencies (method bounds way off from each other)
    method_uppers = []
    method_names = []
    for method_name, method_data in exactcis_results.items():
        our_lower, our_upper = method_data['ci']
        if np.isfinite(our_upper) and our_upper < 1000:  # Reasonable finite bounds
            method_uppers.append(our_upper)
            method_names.append(method_name)
    
    if len(method_uppers) >= 3:  # Need at least 3 methods to compare
        median_upper = np.median(method_uppers)
        for i, (method_name, upper) in enumerate(zip(method_names, method_uppers)):
            if median_upper > 0 and upper / median_upper > 2.5:
                outliers.append(f"{method_name}: Upper bound {upper:.2f} is {upper/median_upper:.1f}x median of methods ({median_upper:.2f})")
    
    return outliers


def format_results_table(scenario: Dict, gold_results: Dict, exactcis_results: Dict) -> str:
    """Format results into a readable table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"SCENARIO: {scenario['name']} - {scenario['description']}")
    lines.append(f"Expected OR: {scenario['or_expected']:.2f}")
    lines.append(f"{'='*80}")
    
    lines.append(f"{'Method':<25} {'Estimate':>10} {'Lower CI':>12} {'Upper CI':>12} {'Source':>10}")
    lines.append("-" * 80)
    
    # Gold standard results
    for method, data in gold_results.items():
        est = data['estimate']
        lower, upper = data['ci']
        est_str = f"{est:.4f}" if np.isfinite(est) else "inf/nan"
        lower_str = f"{lower:.4f}" if np.isfinite(lower) else "inf/nan"
        upper_str = f"{upper:.4f}" if np.isfinite(upper) else "inf/nan"
        lines.append(f"{method + ' (Gold)':<25} {est_str:>10} {lower_str:>12} {upper_str:>12} {'Gold':>10}")
    
    # Our results
    for method, data in exactcis_results.items():
        est = data['estimate'] 
        lower, upper = data['ci']
        est_str = f"{est:.4f}" if np.isfinite(est) else "inf/nan"
        lower_str = f"{lower:.4f}" if np.isfinite(lower) else "inf/nan"
        upper_str = f"{upper:.4f}" if np.isfinite(upper) else "inf/nan"
        lines.append(f"{method:<25} {est_str:>10} {lower_str:>12} {upper_str:>12} {'ExactCIs':>10}")
    
    return "\n".join(lines)


def main():
    """Run comprehensive outlier detection across all scenarios."""
    print("="*80)
    print("COMPREHENSIVE OUTLIER DETECTION TEST")
    print("Testing all CI methods across multiple table configurations")
    print("="*80)
    
    scenarios = define_test_scenarios()
    all_outliers = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nProcessing scenario {i+1}/{len(scenarios)}: {scenario['name']}")
        
        # Run gold standard calculations
        try:
            gold_results = run_gold_standard_for_scenario(scenario)
        except Exception as e:
            print(f"Error in gold standard for {scenario['name']}: {e}")
            continue
            
        # Run our implementations
        try:
            exactcis_results = run_exactcis_for_scenario(scenario)
        except Exception as e:
            print(f"Error in ExactCIs for {scenario['name']}: {e}")
            continue
        
        # Print detailed results
        results_table = format_results_table(scenario, gold_results, exactcis_results)
        print(results_table)
        
        # Identify outliers
        outliers = identify_outliers(gold_results, exactcis_results, scenario)
        if outliers:
            print(f"\n🚨 OUTLIERS DETECTED in {scenario['name']}:")
            for outlier in outliers:
                print(f"  - {outlier}")
                all_outliers.append(f"{scenario['name']}: {outlier}")
        else:
            print(f"\n✅ No significant outliers detected in {scenario['name']}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL OUTLIER SUMMARY")
    print(f"{'='*80}")
    
    if all_outliers:
        print(f"Found {len(all_outliers)} outlier(s) across all scenarios:")
        for i, outlier in enumerate(all_outliers, 1):
            print(f"{i:2d}. {outlier}")
    else:
        print("✅ NO SIGNIFICANT OUTLIERS DETECTED across all scenarios!")
        print("All methods are performing within expected ranges.")
    
    return all_outliers


if __name__ == "__main__":
    outliers = main()