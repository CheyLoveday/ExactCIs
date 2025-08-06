#!/usr/bin/env python3
"""
Test to demonstrate grid size dependency in unconditional method
"""

from src.exactcis.methods.unconditional import exact_ci_unconditional
import time

def test_grid_sizes():
    a, b, c, d = 50, 950, 10, 990
    alpha = 0.05
    
    print("Testing Unconditional Method with Different Grid Sizes")
    print("="*60)
    print(f"Table: a={a}, b={b}, c={c}, d={d}")
    print(f"Sample OR: {(a*d)/(b*c):.4f}")
    print()
    
    grid_sizes = [10, 20, 50, 100, 200, 500]
    
    for grid_size in grid_sizes:
        print(f"Grid size {grid_size:3d}:", end=" ")
        start_time = time.time()
        
        try:
            lower, upper = exact_ci_unconditional(
                a, b, c, d, 
                alpha=alpha, 
                grid_size=grid_size,
                timeout=60  # 1 minute timeout
            )
            end_time = time.time()
            
            width = upper - lower
            print(f"({lower:8.4f}, {upper:8.4f})  Width: {width:7.4f}  Time: {end_time-start_time:5.2f}s")
            
        except Exception as e:
            end_time = time.time()
            print(f"ERROR after {end_time-start_time:.2f}s: {e}")

if __name__ == '__main__':
    test_grid_sizes()