#!/usr/bin/env python3
"""
Complete test script to run ALL CI methods on the test case: 50/1000 vs 10/1000
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

def test_individual_methods():
    """Test each method individually with detailed error handling"""
    a, b, c, d = 50, 950, 10, 990
    print(f'Testing 2x2 table:')
    print(f'         Event   No Event   Total')
    print(f'Group1     {a:2d}      {b:3d}     {a+b:4d}')
    print(f'Group2     {c:2d}      {d:3d}     {c+d:4d}')
    print(f'Total      {a+c:2d}     {b+d:4d}     {a+b+c+d:4d}')
    print()

    # Calculate odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    print(f'Sample odds ratio: {odds_ratio:.4f}')
    print()

    # Test individual methods with timing
    methods = [
        ('conditional', exact_ci_conditional),
        ('midp', exact_ci_midp),
        ('blaker', exact_ci_blaker),
        ('unconditional', exact_ci_unconditional),
        ('wald_haldane', ci_wald_haldane),
        ('clopper_pearson', exact_ci_clopper_pearson),
    ]
    
    print('95% Confidence Intervals - Individual Method Tests:')
    print('=' * 70)
    results = {}
    
    for method_name, method_func in methods:
        print(f'Running {method_name}...', end=' ')
        start_time = time.time()
        try:
            if method_name == 'midp':
                # Use smaller grid for faster computation
                lower, upper = method_func(a, b, c, d, alpha=0.05, grid_size=50)
            elif method_name == 'unconditional':
                # Use small grid and timeout for unconditional to avoid long wait
                lower, upper = method_func(a, b, c, d, alpha=0.05, grid_size=20, timeout=30)
            elif method_name == 'clopper_pearson':
                # Clopper-Pearson is for binomial proportions, test both groups
                print("\n  Group 1 (50/1000):", end=" ")
                lower1, upper1 = method_func(50, 1000, alpha=0.05)
                print(f"({lower1:.4f}, {upper1:.4f})")
                print(f"  Group 2 (10/1000):", end=" ")
                lower2, upper2 = method_func(10, 1000, alpha=0.05)
                print(f"({lower2:.4f}, {upper2:.4f})")
                # Skip normal processing for Clopper-Pearson
                end_time = time.time()
                print(f"  Time: {end_time - start_time:.2f}s")
                results[method_name] = ((lower1, upper1), (lower2, upper2))
                continue
            else:
                lower, upper = method_func(a, b, c, d, alpha=0.05)
            
            end_time = time.time()
            width = upper - lower if upper != float('inf') else float('inf')
            results[method_name] = (lower, upper)
            
            print(f'({lower:8.4f}, {upper:8.4f})  Width: {width:8.4f}  Time: {end_time - start_time:.2f}s')
            
        except Exception as e:
            end_time = time.time()
            print(f'ERROR after {end_time - start_time:.2f}s: {e}')
            results[method_name] = (None, None)
    
    print()
    print('Summary:')
    print('-' * 70)
    for method_name, result in results.items():
        if method_name == 'clopper_pearson':
            print(f'{method_name:15s}: Binomial proportions (not odds ratios)')
        elif result[0] is not None and result[1] is not None:
            lower, upper = result
            contains_or = lower <= odds_ratio <= upper
            status = "✓" if contains_or else "✗"
            print(f'{method_name:15s}: {status} Contains OR={odds_ratio:.4f}')
        else:
            print(f'{method_name:15s}: ✗ Failed to compute')
    
    return results

def test_compute_all_cis():
    """Test the main compute_all_cis function"""
    a, b, c, d = 50, 950, 10, 990
    print('\n' + '='*70)
    print('Testing compute_all_cis() function:')
    print('='*70)
    
    start_time = time.time()
    try:
        results = compute_all_cis(a, b, c, d, alpha=0.05, grid_size=50)
        end_time = time.time()
        
        print(f'Total computation time: {end_time - start_time:.2f} seconds')
        print()
        
        odds_ratio = calculate_odds_ratio(a, b, c, d)
        
        for method, (lower, upper) in results.items():
            width = upper - lower if upper != float('inf') else float('inf')
            contains_or = lower <= odds_ratio <= upper
            status = "✓" if contains_or else "✗"
            print(f'{method:15s}: ({lower:8.4f}, {upper:8.4f})  Width: {width:8.4f}  {status}')
        
        return results
        
    except Exception as e:
        end_time = time.time()
        print(f'ERROR in compute_all_cis after {end_time - start_time:.2f}s: {e}')
        import traceback
        traceback.print_exc()
        return None

def main():
    print("COMPLETE EXACTCIS METHOD TESTING")
    print("="*70)
    
    # Test 1: Individual methods
    individual_results = test_individual_methods()
    
    # Test 2: compute_all_cis function  
    all_results = test_compute_all_cis()
    
    # Compare results
    if all_results:
        print('\n' + '='*70)
        print('COMPARISON: Individual vs compute_all_cis')
        print('='*70)
        
        for method in ['conditional', 'midp', 'blaker', 'unconditional', 'wald_haldane']:
            if method in individual_results and method in all_results:
                ind_result = individual_results[method]
                all_result = all_results[method]
                
                if ind_result[0] is not None and all_result is not None:
                    match = abs(ind_result[0] - all_result[0]) < 0.001 and abs(ind_result[1] - all_result[1]) < 0.001
                    status = "✓ MATCH" if match else "✗ DIFFER"
                    print(f'{method:15s}: {status}')
                else:
                    print(f'{method:15s}: ✗ One or both failed')

if __name__ == '__main__':
    main()