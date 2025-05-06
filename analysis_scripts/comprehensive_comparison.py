#!/usr/bin/env python
import time
import numpy as np
import pandas as pd
from scipy import stats
from exactcis.methods.unconditional import exact_ci_unconditional, improved_ci_unconditional

def fisher_exact_ci_scipy(a, b, c, d, alpha=0.05):
    """Calculate Fisher's exact confidence interval using SciPy."""
    table = np.array([[a, b], [c, d]])
    
    # Get odds ratio and p-value
    odds_ratio, p_value = stats.fisher_exact(table)
    
    # SciPy doesn't directly calculate CI, so we'll estimate it
    # This is only an approximation - proper Fisher exact CI would require more work
    # We'll compute log-odds and use normal approximation (not ideal but gives an estimate)
    log_odds = np.log(odds_ratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    
    lower = np.exp(log_odds - z * se)
    upper = np.exp(log_odds + z * se)
    
    return lower, upper

def compare_methods_for_table(a, b, c, d, alpha_values=[0.05, 0.01, 0.1]):
    """Compare different methods for a specific 2x2 table."""
    
    # R results from our previous run (exact2x2 package)
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
    
    # Create a DataFrame to store all results
    results = []
    
    for alpha in alpha_values:
        # Compute ExactCIs original method
        start_time = time.time()
        py_orig_lower, py_orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
        py_orig_time = time.time() - start_time
        
        # Compute ExactCIs improved method
        start_time = time.time()
        py_imp_lower, py_imp_upper = improved_ci_unconditional(a, b, c, d, alpha)
        py_imp_time = time.time() - start_time
        
        # Compute SciPy approximation
        start_time = time.time()
        scipy_lower, scipy_upper = fisher_exact_ci_scipy(a, b, c, d, alpha)
        scipy_time = time.time() - start_time
        
        # Get R results
        r_fisher_ci = r_results[alpha]["fisher"]
        r_uncond_ci = r_results[alpha]["unconditional"]
        
        # Add results to DataFrame
        results.append({
            "Alpha": alpha,
            "Method": "ExactCIs Original",
            "Lower": py_orig_lower,
            "Upper": py_orig_upper,
            "Time": py_orig_time
        })
        
        results.append({
            "Alpha": alpha,
            "Method": "ExactCIs Improved",
            "Lower": py_imp_lower,
            "Upper": py_imp_upper,
            "Time": py_imp_time
        })
        
        results.append({
            "Alpha": alpha,
            "Method": "SciPy Approximation",
            "Lower": scipy_lower,
            "Upper": scipy_upper,
            "Time": scipy_time
        })
        
        results.append({
            "Alpha": alpha,
            "Method": "R Fisher's Exact",
            "Lower": r_fisher_ci[0],
            "Upper": r_fisher_ci[1],
            "Time": np.nan
        })
        
        results.append({
            "Alpha": alpha,
            "Method": "R Unconditional",
            "Lower": r_uncond_ci[0],
            "Upper": r_uncond_ci[1],
            "Time": np.nan
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def print_comparison_results(df):
    """Print formatted comparison results."""
    # Group by Alpha
    for alpha, group in df.groupby("Alpha"):
        print(f"\n===== RESULTS FOR ALPHA = {alpha} =====")
        print(f"{'Method':<25} {'Lower Bound':<15} {'Upper Bound':<15} {'CI Width':<15} {'Time (s)':<10}")
        print("-" * 85)
        
        for _, row in group.iterrows():
            ci_width = row["Upper"] - row["Lower"]
            print(f"{row['Method']:<25} {row['Lower']:<15.6f} {row['Upper']:<15.6f} {ci_width:<15.6f} {row['Time']:<10.6f}")

def analyze_differences(df):
    """Analyze differences between methods."""
    print("\n===== DIFFERENCE ANALYSIS =====")
    
    # Calculate differences from R Unconditional (reference)
    reference_method = "R Unconditional"
    comparison_methods = ["ExactCIs Original", "ExactCIs Improved", "SciPy Approximation", "R Fisher's Exact"]
    
    for alpha, group in df.groupby("Alpha"):
        print(f"\nAlpha = {alpha}")
        print("-" * 85)
        
        # Get reference values
        ref_row = group[group["Method"] == reference_method].iloc[0]
        ref_lower = ref_row["Lower"]
        ref_upper = ref_row["Upper"]
        ref_width = ref_upper - ref_lower
        
        print(f"Reference ({reference_method}): ({ref_lower:.6f}, {ref_upper:.6f}), Width: {ref_width:.6f}")
        print()
        
        print(f"{'Method':<25} {'Lower Diff':<15} {'Upper Diff':<15} {'Width Diff':<15} {'Lower % Diff':<15} {'Upper % Diff':<15}")
        print("-" * 100)
        
        for method in comparison_methods:
            try:
                method_row = group[group["Method"] == method].iloc[0]
                
                lower_diff = abs(method_row["Lower"] - ref_lower)
                upper_diff = abs(method_row["Upper"] - ref_upper)
                width_diff = abs((method_row["Upper"] - method_row["Lower"]) - ref_width)
                
                lower_pct = 100 * lower_diff / ref_lower if ref_lower != 0 else float('inf')
                upper_pct = 100 * upper_diff / ref_upper if ref_upper != 0 else float('inf')
                
                print(f"{method:<25} {lower_diff:<15.6f} {upper_diff:<15.6f} {width_diff:<15.6f} {lower_pct:<15.2f}% {upper_pct:<15.2f}%")
            except:
                print(f"{method}: Error calculating differences")

def main():
    # The 2x2 table we're examining: a=7, b=3, c=2, d=8
    a, b, c, d = 7, 3, 2, 8
    print(f"Comprehensive comparison for table ({a},{b},{c},{d}):")
    
    # Calculate CI for all methods and alphas
    results_df = compare_methods_for_table(a, b, c, d)
    
    # Print formatted results
    print_comparison_results(results_df)
    
    # Analyze differences
    analyze_differences(results_df)
    
    # Save results to CSV
    results_df.to_csv("method_comparison_results.csv", index=False)
    print("\nResults saved to method_comparison_results.csv")

if __name__ == "__main__":
    main()
