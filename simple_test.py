#!/usr/bin/env python3
"""
Simple test script to run key CI methods on the test case: 50/1000 vs 10/1000
"""

from src.exactcis.methods.conditional import exact_ci_conditional
from src.exactcis.methods.midp import exact_ci_midp
from src.exactcis.methods.blaker import exact_ci_blaker
from src.exactcis.methods.wald import ci_wald_haldane
from src.exactcis.core import calculate_odds_ratio
import time

def main():
    # Test case: 50/1000 vs 10/1000 
    # This represents a 2x2 table:
    #         Event   No Event   Total
    # Group1    50      950      1000
    # Group2    10      990      1000

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
        ('wald_haldane', ci_wald_haldane),
    ]
    
    print('95% Confidence Intervals:')
    print('=' * 60)
    results = {}
    
    for method_name, method_func in methods:
        print(f'Running {method_name}...', end=' ')
        start_time = time.time()
        try:
            if method_name == 'midp':
                # Use smaller grid for faster computation
                lower, upper = method_func(a, b, c, d, alpha=0.05, grid_size=50)
            else:
                lower, upper = method_func(a, b, c, d, alpha=0.05)
            
            end_time = time.time()
            width = upper - lower if upper != float('inf') else float('inf')
            results[method_name] = (lower, upper)
            
            print(f'({lower:8.4f}, {upper:8.4f})  Width: {width:8.4f}  Time: {end_time - start_time:.2f}s')
            
        except Exception as e:
            print(f'ERROR: {e}')
            results[method_name] = (None, None)
    
    print()
    print('Summary:')
    print('-' * 60)
    for method_name, (lower, upper) in results.items():
        if lower is not None and upper is not None:
            contains_or = lower <= odds_ratio <= upper
            status = "✓" if contains_or else "✗"
            print(f'{method_name:12s}: {status} Contains OR={odds_ratio:.4f}')
        else:
            print(f'{method_name:12s}: ✗ Failed to compute')

if __name__ == '__main__':
    main()