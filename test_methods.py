#!/usr/bin/env python3
"""
Test script to run all CI methods on the test case: 50/1000 vs 10/1000
"""

from src.exactcis import compute_all_cis
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
    odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
    print(f'Sample odds ratio: {odds_ratio:.4f}')
    print()

    # Run all methods
    start_time = time.time()
    try:
        results = compute_all_cis(a, b, c, d, alpha=0.05)
        
        print('95% Confidence Intervals:')
        print('=' * 50)
        for method, (lower, upper) in results.items():
            width = upper - lower if upper != float('inf') else float('inf')
            print(f'{method:12s}: ({lower:8.4f}, {upper:8.4f})  Width: {width:8.4f}')
        
        end_time = time.time()
        print(f'\nComputation time: {end_time - start_time:.2f} seconds')
        
        return results
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()