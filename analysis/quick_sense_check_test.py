"""
Quick Sense Check Test: Compare ExactCIs implementation with gold standard calculations
across three 2x2 tables with same odds ratio pattern but different sample sizes.

Tables designed with OR ≈ 5.2:
1. Small: 10/100 vs 2/100 
2. Medium: 50/1000 vs 10/1000
3. Large: 100/2000 vs 20/2000
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


def define_test_tables() -> List[Dict]:
    """Define three 2x2 tables with same OR pattern, different sample sizes."""
    tables = [
        {
            "name": "Small (N=200)",
            "description": "10/100 vs 2/100",
            "table": np.array([[10, 90], [2, 98]]),  # OR = (10*98)/(90*2) = 5.44
            "a": 10, "b": 90, "c": 2, "d": 98
        },
        {
            "name": "Medium (N=2000)", 
            "description": "50/1000 vs 10/1000",
            "table": np.array([[50, 950], [10, 990]]),  # OR = (50*990)/(950*10) = 5.21
            "a": 50, "b": 950, "c": 10, "d": 990
        },
        {
            "name": "Large (N=4000)",
            "description": "100/2000 vs 20/2000", 
            "table": np.array([[100, 1900], [20, 1980]]),  # OR = (100*1980)/(1900*20) = 5.21
            "a": 100, "b": 1900, "c": 20, "d": 1980
        }
    ]
    return tables


def run_gold_standard_calculations(table_info: Dict) -> Dict:
    """Run gold standard CI calculations for a given table."""
    calculator = CICalculator(table_info["table"], alpha=0.05)
    
    results = {}
    
    # Odds Ratio methods
    or_wald, or_wald_ci = calculator.odds_ratio_wald()
    results['OR Wald'] = {"estimate": or_wald, "ci": or_wald_ci}
    
    or_fisher, or_fisher_ci = calculator.odds_ratio_exact_fisher()
    results['OR Fisher Exact'] = {"estimate": or_fisher, "ci": or_fisher_ci}
    
    # Relative Risk methods
    rr_wald, rr_wald_ci = calculator.relative_risk_wald()
    results['RR Wald'] = {"estimate": rr_wald, "ci": rr_wald_ci}
    
    rr_logbin, rr_logbin_ci = calculator.relative_risk_log_binomial()
    results['RR Log-Binomial'] = {"estimate": rr_logbin, "ci": rr_logbin_ci}
    
    # Risk Difference
    rd_wilson, rd_wilson_ci = calculator.wilson_score_difference()
    results['Risk Diff Wilson'] = {"estimate": rd_wilson, "ci": rd_wilson_ci}
    
    return results


def run_exactcis_calculations(table_info: Dict) -> Dict:
    """Run ALL ExactCIs implementation methods for a given table."""
    a, b, c, d = table_info["a"], table_info["b"], table_info["c"], table_info["d"]
    
    # Calculate point estimates for comparison
    or_estimate = (a * d) / (b * c) if b * c != 0 else float('inf')
    rr_estimate = (a / (a + b)) / (c / (c + d)) if c + d != 0 and c != 0 else float('inf')
    rd_estimate = (a / (a + b)) - (c / (c + d)) if (a + b) != 0 and (c + d) != 0 else 0
    
    # Get ALL CI results from our implementation
    ci_results = compute_all_cis(a, b, c, d, alpha=0.05)
    
    # Format to match gold standard structure
    results = {}
    for method, (lower, upper) in ci_results.items():
        # Determine appropriate point estimate based on method type
        if any(keyword in method.lower() for keyword in ['conditional', 'midp', 'blaker', 'unconditional', 'wald']):
            estimate = or_estimate
        elif 'rr' in method.lower() or 'relative_risk' in method.lower():
            estimate = rr_estimate  
        elif 'rd' in method.lower() or 'risk_diff' in method.lower():
            estimate = rd_estimate
        elif 'clopper' in method.lower():
            # Clopper-Pearson typically for proportions - use appropriate estimate
            estimate = a / (a + b) if (a + b) != 0 else 0  # Exposed group proportion
        else:
            estimate = or_estimate  # Default to OR
            
        results[method] = {"estimate": estimate, "ci": (lower, upper)}
    
    return results


def create_comparison_table(table_info: Dict, gold_results: Dict, exactcis_results: Dict) -> pd.DataFrame:
    """Create a detailed comparison table for one 2x2 table."""
    
    comparison_data = []
    
    # Add gold standard results
    for method, data in gold_results.items():
        comparison_data.append({
            "Method": f"{method} (Gold)",
            "Estimate": f"{data['estimate']:.4f}" if data['estimate'] != float('inf') else "∞",
            "Lower CI": f"{data['ci'][0]:.4f}" if data['ci'][0] != float('inf') else "∞",
            "Upper CI": f"{data['ci'][1]:.4f}" if data['ci'][1] != float('inf') else "∞",
            "Source": "Gold Standard"
        })
    
    # Add our implementation results
    for method, data in exactcis_results.items():
        comparison_data.append({
            "Method": f"{method} (ExactCIs)",
            "Estimate": f"{data['estimate']:.4f}" if data['estimate'] != float('inf') else "∞",
            "Lower CI": f"{data['ci'][0]:.4f}" if data['ci'][0] != float('inf') else "∞", 
            "Upper CI": f"{data['ci'][1]:.4f}" if data['ci'][1] != float('inf') else "∞",
            "Source": "ExactCIs"
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def calculate_differences(gold_results: Dict, exactcis_results: Dict) -> Dict:
    """Calculate differences between gold standard and our implementation."""
    
    # Method mapping: gold standard name -> our method name
    method_mapping = {
        "OR Fisher Exact": "conditional",  # Our conditional = Fisher's exact
        "OR Wald": "wald_haldane"         # Our Wald-Haldane ≈ Wald
    }
    
    differences = {}
    
    for gold_method, exactcis_method in method_mapping.items():
        if gold_method in gold_results and exactcis_method in exactcis_results:
            gold_ci = gold_results[gold_method]["ci"]
            exactcis_ci = exactcis_results[exactcis_method]["ci"]
            
            lower_diff = abs(gold_ci[0] - exactcis_ci[0]) if gold_ci[0] != float('inf') and exactcis_ci[0] != float('inf') else float('inf')
            upper_diff = abs(gold_ci[1] - exactcis_ci[1]) if gold_ci[1] != float('inf') and exactcis_ci[1] != float('inf') else float('inf')
            
            differences[f"{gold_method} vs {exactcis_method}"] = {
                "lower_diff": lower_diff,
                "upper_diff": upper_diff,
                "gold_ci": gold_ci,
                "exactcis_ci": exactcis_ci
            }
    
    return differences


def run_comprehensive_comparison():
    """Run comprehensive comparison across all three tables."""
    
    print("="*80)
    print("QUICK SENSE CHECK TEST: ExactCIs vs Gold Standard")
    print("Comparing confidence interval methods across different sample sizes")
    print("="*80)
    print()
    
    tables = define_test_tables()
    all_results = {}
    
    for table_info in tables:
        print(f"\n{'='*60}")
        print(f"TABLE: {table_info['name']} - {table_info['description']}")
        print(f"{'='*60}")
        
        # Display the 2x2 table
        a, b, c, d = table_info["a"], table_info["b"], table_info["c"], table_info["d"]
        or_calc = (a * d) / (b * c)
        
        print(f"\n2x2 Contingency Table:")
        print(f"                Success  Failure   Total")
        print(f"Exposed           {a:>7d}  {b:>7d}  {a+b:>7d}")
        print(f"Unexposed         {c:>7d}  {d:>7d}  {c+d:>7d}")
        print(f"Total             {a+c:>7d}  {b+d:>7d}  {a+b+c+d:>7d}")
        print(f"Odds Ratio: {or_calc:.4f}")
        print()
        
        # Run calculations
        print("Running gold standard calculations...")
        gold_results = run_gold_standard_calculations(table_info)
        
        print("Running ExactCIs calculations...")
        exactcis_results = run_exactcis_calculations(table_info)
        
        # Create comparison table
        comparison_df = create_comparison_table(table_info, gold_results, exactcis_results)
        
        print("\nCOMPARISON RESULTS:")
        print(comparison_df.to_string(index=False))
        
        # Calculate and display differences
        differences = calculate_differences(gold_results, exactcis_results)
        
        if differences:
            print(f"\nKEY DIFFERENCES:")
            print("-" * 50)
            for comparison, diff_data in differences.items():
                print(f"\n{comparison}:")
                print(f"  Gold CI:     ({diff_data['gold_ci'][0]:.4f}, {diff_data['gold_ci'][1]:.4f})")
                print(f"  ExactCIs CI: ({diff_data['exactcis_ci'][0]:.4f}, {diff_data['exactcis_ci'][1]:.4f})")
                print(f"  Lower diff:  {diff_data['lower_diff']:.4f}")
                print(f"  Upper diff:  {diff_data['upper_diff']:.4f}")
        
        all_results[table_info["name"]] = {
            "gold": gold_results,
            "exactcis": exactcis_results,
            "differences": differences
        }
    
    # Summary across all tables
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL TABLES")
    print(f"{'='*80}")
    
    print("\nFisher's Exact (Gold vs ExactCIs conditional method):")
    print(f"{'Table':<20} {'Gold Lower':<12} {'Gold Upper':<12} {'ExactCIs Lower':<15} {'ExactCIs Upper':<15} {'Diff L':<8} {'Diff U':<8}")
    print("-" * 100)
    
    for table_name, results in all_results.items():
        if "OR Fisher Exact vs conditional" in results["differences"]:
            diff_data = results["differences"]["OR Fisher Exact vs conditional"]
            gold_ci = diff_data["gold_ci"]
            exactcis_ci = diff_data["exactcis_ci"]
            
            print(f"{table_name:<20} {gold_ci[0]:<12.4f} {gold_ci[1]:<12.4f} "
                  f"{exactcis_ci[0]:<15.4f} {exactcis_ci[1]:<15.4f} "
                  f"{diff_data['lower_diff']:<8.4f} {diff_data['upper_diff']:<8.4f}")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_comparison()