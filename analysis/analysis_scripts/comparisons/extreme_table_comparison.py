#!/usr/bin/env python
import time
import numpy as np
import pandas as pd
from scipy import stats
from exactcis.methods.unconditional import exact_ci_unconditional, improved_ci_unconditional
import warnings
warnings.filterwarnings('ignore')

def fisher_exact_ci_scipy(a, b, c, d, alpha=0.05):
    """Calculate Fisher's exact confidence interval using SciPy."""
    table = np.array([[a, b], [c, d]])
    
    # Get odds ratio and p-value
    odds_ratio, p_value = stats.fisher_exact(table)
    
    # SciPy doesn't directly calculate CI, so we'll estimate it
    # This is only an approximation - proper Fisher exact CI would require more work
    # We'll compute log-odds and use normal approximation
    log_odds = np.log(odds_ratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    
    lower = np.exp(log_odds - z * se)
    upper = np.exp(log_odds + z * se)
    
    return lower, upper

def normal_approximation_ci(a, b, c, d, alpha=0.05):
    """Calculate CI using normal approximation."""
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    
    # Calculate standard error of log odds ratio
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    # Calculate confidence interval
    z = stats.norm.ppf(1 - alpha/2)
    log_odds = np.log(odds_ratio)
    
    lower = np.exp(log_odds - z * se)
    upper = np.exp(log_odds + z * se)
    
    return lower, upper

def compare_table(a, b, c, d, alpha=0.05):
    """Compare different methods for a specific 2x2 table."""
    print(f"\n===== COMPARISON FOR TABLE ({a},{b},{c},{d}) =====")
    print(f"Row 1: {a} success, {b} failure")
    print(f"Row 2: {c} success, {d} failure")
    print(f"Odds Ratio: {(a*d)/(b*c):.6f}")
    
    results = []
    
    # ExactCIs original
    try:
        start_time = time.time()
        orig_lower, orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
        orig_time = time.time() - start_time
        results.append({
            "Method": "ExactCIs Original",
            "Lower": orig_lower,
            "Upper": orig_upper,
            "Width": orig_upper - orig_lower,
            "Time": orig_time
        })
    except Exception as e:
        print(f"Error in ExactCIs Original: {str(e)}")
        results.append({
            "Method": "ExactCIs Original",
            "Lower": np.nan,
            "Upper": np.nan,
            "Width": np.nan,
            "Time": np.nan
        })
    
    # ExactCIs improved
    try:
        start_time = time.time()
        imp_lower, imp_upper = improved_ci_unconditional(a, b, c, d, alpha)
        imp_time = time.time() - start_time
        results.append({
            "Method": "ExactCIs Improved",
            "Lower": imp_lower,
            "Upper": imp_upper,
            "Width": imp_upper - imp_lower,
            "Time": imp_time
        })
    except Exception as e:
        print(f"Error in ExactCIs Improved: {str(e)}")
        results.append({
            "Method": "ExactCIs Improved",
            "Lower": np.nan,
            "Upper": np.nan,
            "Width": np.nan,
            "Time": np.nan
        })
    
    # SciPy Fisher approximation
    try:
        start_time = time.time()
        scipy_lower, scipy_upper = fisher_exact_ci_scipy(a, b, c, d, alpha)
        scipy_time = time.time() - start_time
        results.append({
            "Method": "SciPy Approximation",
            "Lower": scipy_lower,
            "Upper": scipy_upper,
            "Width": scipy_upper - scipy_lower,
            "Time": scipy_time
        })
    except Exception as e:
        print(f"Error in SciPy Approximation: {str(e)}")
        results.append({
            "Method": "SciPy Approximation",
            "Lower": np.nan,
            "Upper": np.nan,
            "Width": np.nan,
            "Time": np.nan
        })
    
    # Normal approximation
    try:
        start_time = time.time()
        normal_lower, normal_upper = normal_approximation_ci(a, b, c, d, alpha)
        normal_time = time.time() - start_time
        results.append({
            "Method": "Normal Approximation",
            "Lower": normal_lower,
            "Upper": normal_upper,
            "Width": normal_upper - normal_lower,
            "Time": normal_time
        })
    except Exception as e:
        print(f"Error in Normal Approximation: {str(e)}")
        results.append({
            "Method": "Normal Approximation",
            "Lower": np.nan,
            "Upper": np.nan,
            "Width": np.nan,
            "Time": np.nan
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    print(f"\n{'Method':<25} {'Lower Bound':<15} {'Upper Bound':<15} {'CI Width':<15} {'Time (s)':<10}")
    print("-" * 85)
    
    for _, row in df.iterrows():
        lower = row["Lower"] if not np.isnan(row["Lower"]) else "Error"
        upper = row["Upper"] if not np.isnan(row["Upper"]) else "Error"
        width = row["Width"] if not np.isnan(row["Width"]) else "Error"
        time_val = row["Time"] if not np.isnan(row["Time"]) else "Error"
        
        if isinstance(lower, str):
            print(f"{row['Method']:<25} {lower:<15} {upper:<15} {width:<15} {time_val:<10}")
        else:
            print(f"{row['Method']:<25} {lower:<15.6f} {upper:<15.6f} {width:<15.6f} {time_val:<10.6f}")
    
    return df

def main():
    # Extreme tables to test
    tables = [
        (1, 1000, 10, 1000),
        (10, 1000, 1, 1000)
    ]
    
    # Alpha level
    alpha = 0.05
    
    # Compare each table
    all_results = []
    for a, b, c, d in tables:
        df = compare_table(a, b, c, d, alpha)
        df["Table"] = f"({a},{b},{c},{d})"
        all_results.append(df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv("extreme_table_comparison.csv", index=False)
    print("\nResults saved to extreme_table_comparison.csv")
    
    # Summarize trustworthiness
    print("\n===== TRUSTWORTHINESS ANALYSIS =====")
    print("Confidence intervals from ExactCIs methods are compared with other approaches.")
    
    for a, b, c, d in tables:
        table_data = combined_df[combined_df["Table"] == f"({a},{b},{c},{d})"]
        
        # Get ExactCIs improved results if available
        exact_row = table_data[table_data["Method"] == "ExactCIs Improved"]
        if len(exact_row) > 0 and not np.isnan(exact_row.iloc[0]["Lower"]):
            exact_lower = exact_row.iloc[0]["Lower"]
            exact_upper = exact_row.iloc[0]["Upper"]
            exact_width = exact_row.iloc[0]["Width"]
            
            # Compare with normal approximation
            normal_row = table_data[table_data["Method"] == "Normal Approximation"]
            if len(normal_row) > 0 and not np.isnan(normal_row.iloc[0]["Lower"]):
                normal_lower = normal_row.iloc[0]["Lower"]
                normal_upper = normal_row.iloc[0]["Upper"]
                normal_width = normal_row.iloc[0]["Width"]
                
                lower_diff_pct = 100 * abs(exact_lower - normal_lower) / normal_lower
                upper_diff_pct = 100 * abs(exact_upper - normal_upper) / normal_upper
                width_diff_pct = 100 * abs(exact_width - normal_width) / normal_width
                
                print(f"\nFor table ({a},{b},{c},{d}):")
                print(f"  Difference from Normal Approximation:")
                print(f"    Lower bound: {lower_diff_pct:.2f}%")
                print(f"    Upper bound: {upper_diff_pct:.2f}%")
                print(f"    CI Width: {width_diff_pct:.2f}%")
                
                if lower_diff_pct > 50 or upper_diff_pct > 50:
                    print("  LARGE DIFFERENCE DETECTED: ExactCIs differs significantly from normal approximation")
                else:
                    print("  REASONABLE AGREEMENT: ExactCIs is relatively close to normal approximation")

if __name__ == "__main__":
    main()
