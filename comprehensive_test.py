#!/usr/bin/env python3
"""
Comprehensive test of ALL CI methods on 50/1000 vs 10/1000 table
"""

from src.exactcis.methods.conditional import exact_ci_conditional
from src.exactcis.methods.midp import exact_ci_midp
from src.exactcis.methods.blaker import exact_ci_blaker
from src.exactcis.methods.unconditional import exact_ci_unconditional
from src.exactcis.methods.wald import ci_wald_haldane
from src.exactcis.methods.clopper_pearson import exact_ci_clopper_pearson
from src.exactcis.core import calculate_odds_ratio
from src.exactcis import compute_all_cis
import time

def print_table_info():
    """Print information about the 2x2 table"""
    a, b, c, d = 50, 950, 10, 990
    
    print("2×2 Contingency Table: 50/1000 vs 10/1000")
    print("=" * 60)
    print(f"           Event   No Event   Total")
    print(f"Group 1      {a:2d}      {b:3d}     {a+b:4d}")
    print(f"Group 2      {c:2d}      {d:3d}     {c+d:4d}")
    print(f"Total        {a+c:2d}     {b+d:4d}     {a+b+c+d:4d}")
    print()
    
    # Calculate key statistics
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    p1 = a / (a + b)
    p2 = c / (c + d)
    risk_ratio = p1 / p2 if p2 > 0 else float('inf')
    
    print("Key Statistics:")
    print(f"  Sample Odds Ratio:  {odds_ratio:.4f}")
    print(f"  Group 1 proportion: {p1:.4f} ({a}/{a+b})")
    print(f"  Group 2 proportion: {p2:.4f} ({c}/{c+d})")
    print(f"  Risk Ratio:         {risk_ratio:.4f}")
    print()
    
    return a, b, c, d, odds_ratio

def test_all_methods():
    """Test all CI methods individually"""
    a, b, c, d, odds_ratio = print_table_info()
    alpha = 0.05
    
    print("95% Confidence Intervals - Individual Method Tests:")
    print("=" * 80)
    
    methods = [
        ('Conditional (Fisher)', exact_ci_conditional),
        ('Mid-P', exact_ci_midp),
        ('Blaker', exact_ci_blaker),
        ('Unconditional (Barnard)', exact_ci_unconditional),
        ('Wald-Haldane', ci_wald_haldane),
        ('Clopper-Pearson Group 1', lambda a,b,c,d,alpha: exact_ci_clopper_pearson(a,b,c,d,alpha,group=1)),
        ('Clopper-Pearson Group 2', lambda a,b,c,d,alpha: exact_ci_clopper_pearson(a,b,c,d,alpha,group=2)),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"{method_name:25s}:", end=" ")
        start_time = time.time()
        
        try:
            if 'Mid-P' in method_name:
                # Use moderate grid size for balance of speed and accuracy
                lower, upper = method_func(a, b, c, d, alpha, grid_size=100)
            elif 'Unconditional' in method_name:
                # Use moderate grid size and timeout
                lower, upper = method_func(a, b, c, d, alpha, grid_size=100, timeout=60)
            else:
                lower, upper = method_func(a, b, c, d, alpha)
            
            end_time = time.time()
            
            # Check if OR/proportion is contained in CI
            if 'Clopper-Pearson' in method_name:
                # For Clopper-Pearson, check if the proportion is contained
                if 'Group 1' in method_name:
                    p_obs = a / (a + b)
                    param_name = f"p1={p_obs:.4f}"
                else:
                    p_obs = c / (c + d) 
                    param_name = f"p2={p_obs:.4f}"
                contains_param = lower <= p_obs <= upper
            else:
                # For OR methods, check if OR is contained
                contains_param = lower <= odds_ratio <= upper
                param_name = f"OR={odds_ratio:.4f}"
            
            width = upper - lower if upper != float('inf') else float('inf')
            status = "✓" if contains_param else "✗"
            
            print(f"({lower:8.4f}, {upper:8.4f})  Width: {width:8.4f}  {status}  Time: {end_time-start_time:5.2f}s")
            
            results[method_name] = {
                'interval': (lower, upper),
                'width': width,
                'contains_param': contains_param,
                'param_name': param_name,
                'time': end_time - start_time
            }
            
        except Exception as e:
            end_time = time.time()
            print(f"ERROR after {end_time-start_time:.2f}s: {str(e)[:50]}...")
            results[method_name] = {
                'interval': (None, None),
                'error': str(e),
                'time': end_time - start_time
            }
    
    return results

def test_compute_all_cis():
    """Test the main compute_all_cis function"""
    a, b, c, d, odds_ratio = 50, 950, 10, 990, 5.2105
    
    print("\n" + "=" * 80)
    print("Testing compute_all_cis() function:")
    print("=" * 80)
    
    start_time = time.time()
    try:
        results = compute_all_cis(a, b, c, d, alpha=0.05)
        end_time = time.time()
        
        print(f"Total computation time: {end_time - start_time:.2f} seconds")
        print()
        print(f"{'Method':<15} {'Lower':<10} {'Upper':<10} {'Width':<10} {'Contains OR':<12} {'Status'}")
        print("-" * 80)
        
        for method, (lower, upper) in results.items():
            width = upper - lower if upper != float('inf') else float('inf')
            contains_or = lower <= odds_ratio <= upper
            status = "✓" if contains_or else "✗"
            
            print(f"{method:<15} {lower:<10.4f} {upper:<10.4f} {width:<10.4f} {odds_ratio:<12.4f} {status}")
        
        return results
        
    except Exception as e:
        end_time = time.time()
        print(f'ERROR in compute_all_cis after {end_time - start_time:.2f}s: {e}')
        import traceback
        traceback.print_exc()
        return None

def create_summary_table(individual_results, all_results):
    """Create a comprehensive summary table"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Method':<25} {'Individual Test':<20} {'compute_all_cis':<20} {'Match':<8} {'Comments'}")
    print("-" * 100)
    
    # OR-based methods
    or_methods = ['Conditional (Fisher)', 'Mid-P', 'Blaker', 'Unconditional (Barnard)', 'Wald-Haldane']
    
    for method in or_methods:
        if method in individual_results and 'interval' in individual_results[method]:
            ind_result = individual_results[method]['interval']
            
            # Map method names
            method_key_map = {
                'Conditional (Fisher)': 'conditional',
                'Mid-P': 'midp', 
                'Blaker': 'blaker',
                'Unconditional (Barnard)': 'unconditional',
                'Wald-Haldane': 'wald_haldane'
            }
            
            method_key = method_key_map.get(method)
            
            if all_results and method_key in all_results:
                all_result = all_results[method_key]
                
                if ind_result[0] is not None and all_result is not None:
                    match = (abs(ind_result[0] - all_result[0]) < 0.001 and 
                            abs(ind_result[1] - all_result[1]) < 0.001)
                    match_str = "✓" if match else "✗"
                    
                    ind_str = f"({ind_result[0]:.4f}, {ind_result[1]:.4f})"
                    all_str = f"({all_result[0]:.4f}, {all_result[1]:.4f})"
                    
                    comment = "Grid size difference" if not match else "Consistent"
                    
                else:
                    match_str = "✗"
                    ind_str = "Failed"
                    all_str = "Failed" 
                    comment = "Computation failed"
            else:
                match_str = "N/A"
                ind_str = f"({ind_result[0]:.4f}, {ind_result[1]:.4f})" if ind_result[0] is not None else "Failed"
                all_str = "Not in compute_all_cis"
                comment = "Not included in main function"
            
            print(f"{method:<25} {ind_str:<20} {all_str:<20} {match_str:<8} {comment}")
    
    # Clopper-Pearson methods
    cp_methods = ['Clopper-Pearson Group 1', 'Clopper-Pearson Group 2']
    for method in cp_methods:
        if method in individual_results:
            if 'interval' in individual_results[method]:
                ind_result = individual_results[method]['interval']
                ind_str = f"({ind_result[0]:.4f}, {ind_result[1]:.4f})" if ind_result[0] is not None else "Failed"
                comment = "Binomial proportions (separate from OR methods)"
            else:
                ind_str = "Failed"
                comment = individual_results[method].get('error', 'Unknown error')[:40]
            
            print(f"{method:<25} {ind_str:<20} {'Not applicable':<20} {'N/A':<8} {comment}")

def main():
    print("COMPREHENSIVE EXACTCIS TESTING: 50/1000 vs 10/1000")
    print("=" * 100)
    
    # Test individual methods
    individual_results = test_all_methods()
    
    # Test compute_all_cis
    all_results = test_compute_all_cis()
    
    # Create summary
    create_summary_table(individual_results, all_results)
    
    print(f"\n{'='*100}")
    print("TESTING COMPLETE")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()