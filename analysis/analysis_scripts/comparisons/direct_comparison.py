#!/usr/bin/env python
# Direct comparison of methods for the specific table we're investigating
import time
import numpy as np
from exactcis.methods.unconditional import exact_ci_unconditional

# Table we're focusing on
a, b, c, d = 7, 3, 2, 8  # The specific table we're investigating
print(f"Direct comparison for table ({a},{b},{c},{d}):")

# Alpha values to test
alpha_values = [0.05, 0.01, 0.1]

# R results for Fisher and Unconditional exact tests (from the R output)
r_results = {
    0.05: {
        "fisher": (0.882117, 127.055842),
        "unconditional": (0.882117, 127.055842)
    },
    0.01: {
        "fisher": (0.521258, 307.936480),
        "unconditional": (0.521258, 307.936480)
    },
    0.1: {
        "fisher": (1.155327, 84.045604),
        "unconditional": (1.155327, 84.045604)
    }
}

print("\n===== COMPARISON OF ALL METHODS =====")
print(f"{'Alpha':<10} {'Method':<25} {'Lower Bound':<15} {'Upper Bound':<15} {'Time (s)':<10}")
print("-" * 80)

for alpha in alpha_values:
    # Get times and results for Python original method
    start_time = time.time()
    py_orig_lower, py_orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
    py_orig_time = time.time() - start_time
    
    # Get times and results for Python improved method
    start_time = time.time()
    py_imp_lower, py_imp_upper = exact_ci_unconditional(a, b, c, d, alpha, adaptive_grid=True, use_cache=True)
    py_imp_time = time.time() - start_time
    
    # Get R results
    r_fisher_lower, r_fisher_upper = r_results[alpha]["fisher"]
    r_uncond_lower, r_uncond_upper = r_results[alpha]["unconditional"]
    
    # Print results
    print(f"{alpha:<10} {'Python Original':<25} {py_orig_lower:<15.6f} {py_orig_upper:<15.6f} {py_orig_time:<10.6f}")
    print(f"{'':<10} {'Python Improved':<25} {py_imp_lower:<15.6f} {py_imp_upper:<15.6f} {py_imp_time:<10.6f}")
    print(f"{'':<10} {'R Fisher\'s Exact':<25} {r_fisher_lower:<15.6f} {r_fisher_upper:<15.6f} {'N/A':<10}")
    print(f"{'':<10} {'R Unconditional Exact':<25} {r_uncond_lower:<15.6f} {r_uncond_upper:<15.6f} {'N/A':<10}")
    print("-" * 80)

print("\n===== DIFFERENCE ANALYSIS =====")
print("Comparing with R's Unconditional Exact method (considered the reference implementation)")
print(f"{'Alpha':<10} {'Method':<25} {'Lower Diff':<15} {'Upper Diff':<15} {'Lower % Diff':<15} {'Upper % Diff':<15}")
print("-" * 95)

for alpha in alpha_values:
    # Get results
    py_orig_lower, py_orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
    py_imp_lower, py_imp_upper = exact_ci_unconditional(a, b, c, d, alpha, adaptive_grid=True, use_cache=True)
    r_uncond_lower, r_uncond_upper = r_results[alpha]["unconditional"]
    
    # Calculate differences
    orig_lower_diff = abs(py_orig_lower - r_uncond_lower)
    orig_upper_diff = abs(py_orig_upper - r_uncond_upper)
    imp_lower_diff = abs(py_imp_lower - r_uncond_lower)
    imp_upper_diff = abs(py_imp_upper - r_uncond_upper)
    
    # Calculate percentage differences
    orig_lower_pct = 100 * orig_lower_diff / r_uncond_lower if r_uncond_lower != 0 else float('inf')
    orig_upper_pct = 100 * orig_upper_diff / r_uncond_upper if r_uncond_upper != 0 else float('inf')
    imp_lower_pct = 100 * imp_lower_diff / r_uncond_lower if r_uncond_lower != 0 else float('inf')
    imp_upper_pct = 100 * imp_upper_diff / r_uncond_upper if r_uncond_upper != 0 else float('inf')
    
    # Print differences
    print(f"{alpha:<10} {'Python Original':<25} {orig_lower_diff:<15.6f} {orig_upper_diff:<15.6f} {orig_lower_pct:<15.2f}% {orig_upper_pct:<15.2f}%")
    print(f"{'':<10} {'Python Improved':<25} {imp_lower_diff:<15.6f} {imp_upper_diff:<15.6f} {imp_lower_pct:<15.2f}% {imp_upper_pct:<15.2f}%")
    print("-" * 95)

print("\n===== CONCLUSION =====")
# Determine which method is closest to R overall
total_orig_diff = 0
total_imp_diff = 0

for alpha in alpha_values:
    py_orig_lower, py_orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
    py_imp_lower, py_imp_upper = exact_ci_unconditional(a, b, c, d, alpha, adaptive_grid=True, use_cache=True)
    r_uncond_lower, r_uncond_upper = r_results[alpha]["unconditional"]
    
    # Sum of absolute differences
    orig_diff = abs(py_orig_lower - r_uncond_lower) + abs(py_orig_upper - r_uncond_upper)
    imp_diff = abs(py_imp_lower - r_uncond_lower) + abs(py_imp_upper - r_uncond_upper)
    
    total_orig_diff += orig_diff
    total_imp_diff += imp_diff

print(f"Total absolute difference from R unconditional method:")
print(f"  Python Original: {total_orig_diff:.6f}")
print(f"  Python Improved: {total_imp_diff:.6f}")

if total_orig_diff < total_imp_diff:
    print("\nPython Original method is closer to R's unconditional exact method overall.")
elif total_imp_diff < total_orig_diff:
    print("\nPython Improved method is closer to R's unconditional exact method overall.")
else:
    print("\nBoth Python methods are equally close to R's unconditional exact method.")
