#!/usr/bin/env python
"""
Comprehensive comparison of confidence interval methods between ExactCIs and SciPy.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from exactcis.methods.unconditional import exact_ci_unconditional, improved_ci_unconditional
from exactcis.methods.fixed_ci import fisher_approximation
from exactcis.utils.optimization import CICache

# Initialize cache
ci_cache = CICache()

def format_ci(ci):
    """Format confidence interval for display."""
    return f"({ci[0]:.6f}, {ci[1]:.6f})"

def time_execution(func, *args, **kwargs):
    """Time the execution of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

def compare_all_methods(a, b, c, d, alpha=0.05):
    """Compare all confidence interval methods for a given 2x2 table."""
    results = {}
    
    # 1. Original ExactCIs unconditional method
    ci, exec_time = time_execution(
        exact_ci_unconditional, a, b, c, d, alpha=alpha
    )
    results["ExactCIs Original"] = {
        "CI": format_ci(ci),
        "Time (s)": f"{exec_time:.6f}"
    }
    
    # 2. Improved ExactCIs unconditional method
    ci, exec_time = time_execution(
        improved_ci_unconditional, a, b, c, d, alpha=alpha
    )
    results["ExactCIs Improved"] = {
        "CI": format_ci(ci),
        "Time (s)": f"{exec_time:.6f}"
    }
    
    # 3. Fisher approximation
    ci, exec_time = time_execution(
        fisher_approximation, a, b, c, d, alpha=alpha
    )
    results["Fisher Approximation"] = {
        "CI": format_ci(ci),
        "Time (s)": f"{exec_time:.6f}"
    }
    
    # 4. SciPy Fisher Exact CI
    # SciPy doesn't directly provide CIs for Fisher exact test, but we can compute them
    try:
        # The SciPy method returns an odds ratio and p-value
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
        
        # For approximate CI calculation, we use normal approximation on log scale
        # This is not an exact method but commonly used
        alpha_half = alpha / 2
        z = stats.norm.ppf(1 - alpha_half)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        log_odds = np.log(odds_ratio)
        lower = np.exp(log_odds - z * se)
        upper = np.exp(log_odds + z * se)
        
        results["SciPy Fisher Exact (Approx CI)"] = {
            "CI": format_ci((lower, upper)),
            "Time (s)": "N/A"  # We didn't time this
        }
    except (ValueError, ZeroDivisionError):
        results["SciPy Fisher Exact (Approx CI)"] = {
            "CI": "Error - likely due to zero counts",
            "Time (s)": "N/A"
        }
    
    # 5. SciPy Barnard's test
    # SciPy doesn't have a direct implementation of Barnard's test
    
    # 6. SciPy Boschloo's test
    # SciPy doesn't have a direct implementation of Boschloo's test
    
    # 7. Normal approximation for proportion difference
    try:
        p1 = a / (a + b)
        p2 = c / (c + d)
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / (a + b) + p2 * (1 - p2) / (c + d))
        z = stats.norm.ppf(1 - alpha / 2)
        lower = diff - z * se
        upper = diff + z * se
        
        results["Normal Approximation (Difference)"] = {
            "CI": format_ci((lower, upper)),
            "Time (s)": "N/A"
        }
    except (ValueError, ZeroDivisionError):
        results["Normal Approximation (Difference)"] = {
            "CI": "Error - likely due to zero counts",
            "Time (s)": "N/A"
        }
    
    # 8. Normal approximation for risk ratio
    try:
        p1 = a / (a + b)
        p2 = c / (c + d)
        rr = p1 / p2
        log_rr = np.log(rr)
        se = np.sqrt((b / (a * (a + b))) + (d / (c * (c + d))))
        z = stats.norm.ppf(1 - alpha / 2)
        lower = np.exp(log_rr - z * se)
        upper = np.exp(log_rr + z * se)
        
        results["Normal Approximation (Risk Ratio)"] = {
            "CI": format_ci((lower, upper)),
            "Time (s)": "N/A"
        }
    except (ValueError, ZeroDivisionError):
        results["Normal Approximation (Risk Ratio)"] = {
            "CI": "Error - likely due to zero counts",
            "Time (s)": "N/A"
        }
    
    # 9. Normal approximation for odds ratio
    try:
        odds_ratio = (a * d) / (b * c)
        log_odds = np.log(odds_ratio)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = np.exp(log_odds - z * se)
        upper = np.exp(log_odds + z * se)
        
        results["Normal Approximation (Odds Ratio)"] = {
            "CI": format_ci((lower, upper)),
            "Time (s)": "N/A"
        }
    except (ValueError, ZeroDivisionError):
        results["Normal Approximation (Odds Ratio)"] = {
            "CI": "Error - likely due to zero counts",
            "Time (s)": "N/A"
        }
    
    return results

def analyze_differences(results):
    """Analyze and explain differences between methods."""
    analysis = []
    
    # Extract CIs for comparison
    cis = {}
    for method, result in results.items():
        if "Error" not in result["CI"]:
            # Extract the numeric values from formatted string
            ci_str = result["CI"].strip("()")
            lower, upper = map(float, ci_str.split(", "))
            cis[method] = (lower, upper)
    
    # Compare ExactCIs methods with each other
    if "ExactCIs Original" in cis and "ExactCIs Improved" in cis:
        orig = cis["ExactCIs Original"]
        impr = cis["ExactCIs Improved"]
        diff_lower = abs(orig[0] - impr[0])
        diff_upper = abs(orig[1] - impr[1])
        rel_diff_lower = diff_lower / orig[0] if orig[0] != 0 else float('inf')
        rel_diff_upper = diff_upper / orig[1] if orig[1] != 0 else float('inf')
        
        analysis.append("Comparison between ExactCIs Original and Improved:")
        analysis.append(f"  Lower bound absolute difference: {diff_lower:.6f}")
        analysis.append(f"  Upper bound absolute difference: {diff_upper:.6f}")
        analysis.append(f"  Lower bound relative difference: {rel_diff_lower:.2%}")
        analysis.append(f"  Upper bound relative difference: {rel_diff_upper:.2%}")
        
        if rel_diff_lower > 0.01 or rel_diff_upper > 0.01:
            analysis.append("  NOTABLE DIFFERENCE: The improved method differs by more than 1% from the original.")
            analysis.append("  This could be due to improvements in numerical precision or different grid sizes.")
        else:
            analysis.append("  The methods provide very similar results, suggesting good consistency.")
    
    # Compare ExactCIs with Fisher Approximation
    if "ExactCIs Improved" in cis and "Fisher Approximation" in cis:
        exact = cis["ExactCIs Improved"]
        fisher = cis["Fisher Approximation"]
        diff_lower = abs(exact[0] - fisher[0])
        diff_upper = abs(exact[1] - fisher[1])
        rel_diff_lower = diff_lower / exact[0] if exact[0] != 0 else float('inf')
        rel_diff_upper = diff_upper / exact[1] if exact[1] != 0 else float('inf')
        
        analysis.append("\nComparison between ExactCIs Improved and Fisher Approximation:")
        analysis.append(f"  Lower bound absolute difference: {diff_lower:.6f}")
        analysis.append(f"  Upper bound absolute difference: {diff_upper:.6f}")
        analysis.append(f"  Lower bound relative difference: {rel_diff_lower:.2%}")
        analysis.append(f"  Upper bound relative difference: {rel_diff_upper:.2%}")
        
        if rel_diff_lower > 0.05 or rel_diff_upper > 0.05:
            analysis.append("  NOTABLE DIFFERENCE: Fisher approximation differs significantly from exact method.")
            analysis.append("  This is expected as Fisher is an approximation method.")
        else:
            analysis.append("  Fisher approximation is relatively close to the exact method for this example.")
    
    # Compare ExactCIs with SciPy's Fisher Exact
    if "ExactCIs Improved" in cis and "SciPy Fisher Exact (Approx CI)" in cis:
        exact = cis["ExactCIs Improved"]
        scipy = cis["SciPy Fisher Exact (Approx CI)"]
        diff_lower = abs(exact[0] - scipy[0])
        diff_upper = abs(exact[1] - scipy[1])
        rel_diff_lower = diff_lower / exact[0] if exact[0] != 0 else float('inf')
        rel_diff_upper = diff_upper / exact[1] if exact[1] != 0 else float('inf')
        
        analysis.append("\nComparison between ExactCIs Improved and SciPy Fisher Exact:")
        analysis.append(f"  Lower bound absolute difference: {diff_lower:.6f}")
        analysis.append(f"  Upper bound absolute difference: {diff_upper:.6f}")
        analysis.append(f"  Lower bound relative difference: {rel_diff_lower:.2%}")
        analysis.append(f"  Upper bound relative difference: {rel_diff_upper:.2%}")
        
        if rel_diff_lower > 0.10 or rel_diff_upper > 0.10:
            analysis.append("  LARGE DIFFERENCE: SciPy's approximation differs substantially from our exact method.")
            analysis.append("  This is expected as our method uses a more precise calculation approach.")
        else:
            analysis.append("  SciPy's method gives reasonably close results to our exact method for this example.")
    
    # Compare with normal approximations
    exact_methods = ["ExactCIs Original", "ExactCIs Improved"]
    normal_methods = [
        "Normal Approximation (Difference)", 
        "Normal Approximation (Risk Ratio)",
        "Normal Approximation (Odds Ratio)"
    ]
    
    for exact_method in exact_methods:
        if exact_method in cis:
            exact = cis[exact_method]
            for normal_method in normal_methods:
                if normal_method in cis:
                    normal = cis[normal_method]
                    
                    analysis.append(f"\nComparison between {exact_method} and {normal_method}:")
                    
                    # For difference CIs, we need to convert our odds ratio CIs to difference CIs
                    if "Difference" in normal_method:
                        # Here we would need to convert odds ratio to difference
                        # This is complex and requires the original counts
                        analysis.append("  Direct comparison not calculated - requires conversion between measures.")
                    else:
                        diff_lower = abs(exact[0] - normal[0])
                        diff_upper = abs(exact[1] - normal[1])
                        rel_diff_lower = diff_lower / exact[0] if exact[0] != 0 else float('inf')
                        rel_diff_upper = diff_upper / exact[1] if exact[1] != 0 else float('inf')
                        
                        analysis.append(f"  Lower bound absolute difference: {diff_lower:.6f}")
                        analysis.append(f"  Upper bound absolute difference: {diff_upper:.6f}")
                        analysis.append(f"  Lower bound relative difference: {rel_diff_lower:.2%}")
                        analysis.append(f"  Upper bound relative difference: {rel_diff_upper:.2%}")
                        
                        if rel_diff_lower > 0.15 or rel_diff_upper > 0.15:
                            analysis.append("  LARGE DIFFERENCE: Normal approximation is substantially different.")
                            analysis.append("  This is expected for small sample sizes where normal approximation is less accurate.")
                        else:
                            analysis.append("  Normal approximation gives relatively close results for this example.")
    
    return "\n".join(analysis)

def main():
    """Main function to run comparisons on multiple test cases."""
    # Test cases - various 2x2 tables
    test_cases = [
        {"name": "Example 1", "table": (10, 5, 6, 12)},
        {"name": "Example 2 (Balanced)", "table": (20, 20, 20, 20)},
        {"name": "Example 3 (Small counts)", "table": (3, 1, 2, 4)},
        {"name": "Example 4 (Zero cell)", "table": (0, 5, 6, 12)},
        {"name": "Example 5 (Large counts)", "table": (100, 50, 60, 120)},
    ]
    
    # Run comparisons for each test case
    for test_case in test_cases:
        name = test_case["name"]
        a, b, c, d = test_case["table"]
        
        print(f"\n{'=' * 80}")
        print(f"Test Case: {name}")
        print(f"2x2 Table: a={a}, b={b}, c={c}, d={d}")
        print(f"{'=' * 80}")
        
        # Compare all methods
        results = compare_all_methods(a, b, c, d)
        
        # Display results as a table
        df = pd.DataFrame(results).T
        print("\nConfidence Interval Results:")
        print(df)
        
        # Analyze differences
        print("\nAnalysis of Differences:")
        analysis = analyze_differences(results)
        print(analysis)
        
        print("\nSummary Statistics:")
        print(f"Row 1 proportion: {a/(a+b):.4f}")
        print(f"Row 2 proportion: {c/(c+d):.4f}")
        print(f"Odds ratio: {(a*d)/(b*c):.4f}")
        print(f"Risk ratio: {(a/(a+b))/(c/(c+d)):.4f}")
        print(f"Total sample size: {a+b+c+d}")

if __name__ == "__main__":
    main()
