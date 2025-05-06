#!/usr/bin/env python
"""
Optimization script for ExactCIs slow functions.

This script applies performance optimizations to the identified slow functions:
1. Optimizes Blaker's method using memoization and vectorization
2. Improves unconditional method with better grid strategies and caching
"""

import os
import sys
import fileinput
import re
from pathlib import Path
import time
import subprocess

# Paths to the source files
BLAKER_PATH = "src/exactcis/methods/blaker.py"
UNCONDITIONAL_PATH = "src/exactcis/methods/unconditional.py"
CORE_PATH = "src/exactcis/core.py"

# Create backup copies of the original files
def backup_files():
    """Create backup copies of the original files."""
    for path in [BLAKER_PATH, UNCONDITIONAL_PATH, CORE_PATH]:
        backup_path = f"{path}.bak"
        if not os.path.exists(backup_path):
            print(f"Creating backup of {path} -> {backup_path}")
            with open(path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())

# Optimizations for Blaker's method
def optimize_blaker():
    """Apply optimizations to Blaker's method."""
    print("\nOptimizing Blaker's method...")
    
    # Read the current file content
    with open(BLAKER_PATH, 'r') as f:
        content = f.read()

    # Add necessary imports and caching
    if "from functools import lru_cache" not in content:
        content = content.replace(
            "from typing import Tuple",
            "from typing import Tuple, List\nfrom functools import lru_cache\nimport numpy as np"
        )

    # Optimize the blaker_p function
    blaker_p_pattern = re.compile(r'def blaker_p\(theta: float\).*?return.*?(?=\n\s*low)', re.DOTALL)
    optimized_blaker_p = '''
    @lru_cache(maxsize=128)
    def blaker_p(theta: float) -> float:
        """Calculate Blaker's p-value with caching and vectorization."""
        supp, probs = pmf_weights(n1, n2, m, theta)
        
        # Convert to numpy arrays for vectorization
        probs_array = np.array(probs)
        
        # Calculate CDFs efficiently
        cdf_lo = np.cumsum(probs_array)
        cdf_hi = np.cumsum(probs_array[::-1])[::-1]
        
        # Calculate acceptability function values
        fvals = np.minimum(cdf_lo, cdf_hi)
        
        # Find observed table's acceptability
        f_obs = fvals[supp.index(a)]
        
        # Sum probabilities for tables with acceptability <= observed
        return float(np.sum(probs_array[fvals <= f_obs]))
    '''
    
    content = blaker_p_pattern.sub(optimized_blaker_p, content)
    
    # Write the optimized file
    with open(BLAKER_PATH, 'w') as f:
        f.write(content)
    
    print(" Blaker's method optimized")

# Optimizations for unconditional method
def optimize_unconditional():
    """Apply optimizations to unconditional method."""
    print("\nOptimizing unconditional method...")
    
    # Read the current file content
    with open(UNCONDITIONAL_PATH, 'r') as f:
        content = f.read()
    
    # Increase LRU cache size for the binomial PMF
    content = content.replace(
        "@lru_cache(maxsize=256)",
        "@lru_cache(maxsize=2048)"
    )
    
    # Optimize the grid size function
    optimize_grid_pattern = re.compile(r'def _optimize_grid_size.*?return.*?(?=\n\n)', re.DOTALL)
    optimized_grid_fn = '''
def _optimize_grid_size(n1: int, n2: int, base_grid_size: int) -> int:
    """
    Determine optimal grid size based on table dimensions - optimized version.
    """
    # For very small tables, we need fewer points
    if n1 <= 5 and n2 <= 5:
        return min(base_grid_size, 10)
    
    # For small tables
    if n1 <= 10 and n2 <= 10:
        return min(base_grid_size, 15)
    
    # For moderate tables
    if n1 <= 20 and n2 <= 20:
        return min(base_grid_size, 20)
    
    # For larger tables, use fewer points as performance becomes an issue
    return min(base_grid_size, 25)
    '''
    
    content = optimize_grid_pattern.sub(optimized_grid_fn, content)
    
    # Improve the adaptive grid function to use fewer points for small tables
    adaptive_grid_pattern = re.compile(r'def _build_adaptive_grid.*?return sorted\(set\(grid_points\)\)', re.DOTALL)
    
    optimized_adaptive_grid = '''
def _build_adaptive_grid(p1_mle: float, grid_size: int, density_factor: float = 0.3) -> List[float]:
    """
    Build an adaptive grid with more points near the MLE - optimized version.
    """
    eps = 1e-6
    grid_points = []
    
    # Add exact MLE point
    grid_points.append(max(eps, min(1-eps, p1_mle)))
    
    # Use logarithmic spacing for points near boundaries
    # and linear spacing near the MLE
    left_bound = eps
    right_bound = 1 - eps
    
    # Add more points near MLE
    for i in range(grid_size):
        # Parameter t controls spacing (0 to 1)
        t = i / (grid_size - 1) if grid_size > 1 else 0.5
        
        # Use different spacing based on distance from MLE
        if t < 0.33:
            # More points on left side
            p = left_bound + (p1_mle - left_bound) * (3 * t) ** 2
        elif t > 0.67:
            # More points on right side
            p = p1_mle + (right_bound - p1_mle) * ((3 * (t - 0.67)) ** 2)
        else:
            # Linear spacing near MLE
            t_adjusted = (t - 0.33) / 0.34
            p = p1_mle - (density_factor/2) + t_adjusted * density_factor
        
        p = max(left_bound, min(right_bound, p))
        
        # Skip if very close to already added points
        if any(abs(p - p_existing) < 1e-5 for p_existing in grid_points):
            continue
            
        grid_points.append(p)
    
    # For small grid sizes, ensure we have essentials (boundaries and midpoint)
    if grid_size <= 15:
        grid_points.append(left_bound)
        grid_points.append(0.5)
        grid_points.append(right_bound)
        
    # Remove duplicates and sort
    return sorted(set(grid_points))
    '''
    
    content = adaptive_grid_pattern.sub(optimized_adaptive_grid, content)
    
    # Modify the main function to use different defaults for small tables
    content = content.replace(
        "exact_ci_unconditional(a: int, b: int, c: int, d: int,\n                          alpha: float = 0.05, grid_size: int = 50,",
        "exact_ci_unconditional(a: int, b: int, c: int, d: int,\n                          alpha: float = 0.05, grid_size: int = 30,"
    )
    
    content = content.replace(
        "grid_size = _optimize_grid_size(n1, n2, grid_size)",
        "# Adjust grid size based on table dimensions\nif n1 + n2 <= 20:\n        grid_size = min(grid_size, 15)\n    grid_size = _optimize_grid_size(n1, n2, grid_size)"
    )
    
    # Write the optimized file
    with open(UNCONDITIONAL_PATH, 'w') as f:
        f.write(content)

    print(" Unconditional method optimized")

# Core function optimizations
def optimize_core():
    """Optimize core functions that are called frequently."""
    print("\nOptimizing core functions...")
    
    # Read the current file content
    with open(CORE_PATH, 'r') as f:
        content = f.read()
    
    # Add caching to the support function
    if "@lru_cache(maxsize=128)" not in content:
        content = content.replace(
            "from typing import",
            "from typing import"
        )
        content = content.replace(
            "import math",
            "import math\nfrom functools import lru_cache"
        )
        
        # Add caching to support function
        content = content.replace(
            "def support(n1: int, n2: int, m: int):",
            "@lru_cache(maxsize=128)\ndef support(n1: int, n2: int, m: int):"
        )
        
        # Add caching to pmf_weights function
        content = content.replace(
            "def pmf_weights(n1: int, n2: int, m: int, theta: float):",
            "@lru_cache(maxsize=256)\ndef pmf_weights(n1: int, n2: int, m: int, theta: float):"
        )
    
    # Write the optimized file
    with open(CORE_PATH, 'w') as f:
        f.write(content)
    
    print(" Core functions optimized")

# Test the optimized functions
def test_optimizations():
    """Run tests to verify optimizations improved performance."""
    print("\nTesting optimizations...")
    
    # First, run the slow tests to establish baseline
    print("Running Blaker's test (slow version)...")
    cmd = "uv run python -m pytest tests/test_methods/test_blaker.py::test_exact_ci_blaker_basic -v"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    blaker_baseline = time.time() - start_time
    
    print(f"Baseline Blaker test time: {blaker_baseline:.2f} seconds")
    
    # Test unconditional method
    print("Running unconditional test (slow version)...")
    cmd = "uv run python -m pytest tests/test_methods/test_unconditional.py::test_exact_ci_unconditional_small_counts -v"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    unconditional_baseline = time.time() - start_time
    
    print(f"Baseline unconditional test time: {unconditional_baseline:.2f} seconds")
    
    # Apply optimizations
    backup_files()
    optimize_blaker()
    optimize_unconditional()
    optimize_core()
    
    # Run tests again with optimized code
    print("\nRunning Blaker's test (optimized version)...")
    cmd = "uv run python -m pytest tests/test_methods/test_blaker.py::test_exact_ci_blaker_basic -v"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    blaker_optimized = time.time() - start_time
    
    print(f"Optimized Blaker test time: {blaker_optimized:.2f} seconds")
    print(f"Speed improvement: {blaker_baseline/blaker_optimized:.1f}x faster")
    
    # Test unconditional method
    print("\nRunning unconditional test (optimized version)...")
    cmd = "uv run python -m pytest tests/test_methods/test_unconditional.py::test_exact_ci_unconditional_small_counts -v"
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    unconditional_optimized = time.time() - start_time
    
    print(f"Optimized unconditional test time: {unconditional_optimized:.2f} seconds")
    print(f"Speed improvement: {unconditional_baseline/unconditional_optimized:.1f}x faster")

    # Print summary
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Blaker method: {blaker_baseline:.2f}s → {blaker_optimized:.2f}s ({blaker_baseline/blaker_optimized:.1f}x faster)")
    print(f"Unconditional method: {unconditional_baseline:.2f}s → {unconditional_optimized:.2f}s ({unconditional_baseline/unconditional_optimized:.1f}x faster)")

def main():
    """Main function to run optimizations."""
    print("=== ExactCIs Performance Optimization ===")
    
    # Check if source files exist
    for path in [BLAKER_PATH, UNCONDITIONAL_PATH, CORE_PATH]:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            return 1
    
    # Run optimizations and tests
    try:
        test_optimizations()
        return 0
    except Exception as e:
        print(f"Error during optimization: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
