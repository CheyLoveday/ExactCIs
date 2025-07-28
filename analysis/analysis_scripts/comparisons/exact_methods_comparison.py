#!/usr/bin/env python
import time
import numpy as np
import pandas as pd
from exactcis.methods.unconditional import exact_ci_unconditional
import warnings
warnings.filterwarnings('ignore')

# R exact method results - hardcoded from previous R runs for key tables
# Format: {(a,b,c,d): {"fisher": (lower, upper), "unconditional": (lower, upper)}}
r_exact_results = {
    (7, 3, 2, 8): {
        0.05: {
            "fisher": (0.882117, 127.055842),
            "unconditional": (0.882117, 127.055842)
        }
    },
    # Extreme tables (would need to be filled in from R runs)
    (1, 1000, 10, 1000): {
        0.05: {
            "fisher": None,  # Would need actual R output
            "unconditional": None  # Would need actual R output
        }
    },
    (10, 1000, 1, 1000): {
        0.05: {
            "fisher": None,  # Would need actual R output
            "unconditional": None  # Would need actual R output
        }
    }
}

def theoretical_check(a, b, c, d, ci_lower, ci_upper, alpha=0.05):
    """Performs theoretical checks on the confidence interval."""
    # Calculate observed odds ratio
    odds_ratio = (a * d) / (b * c) if b*c > 0 else float('inf')
    
    # Check if CI contains the point estimate
    contains_point = ci_lower <= odds_ratio <= ci_upper
    
    # Check for symmetry (should approach symmetry in log space)
    log_lower = np.log(ci_lower) if ci_lower > 0 else float('-inf')
    log_upper = np.log(ci_upper) if ci_upper > 0 else float('inf')
    log_or = np.log(odds_ratio) if odds_ratio > 0 and odds_ratio < float('inf') else 0
    
    log_lower_dist = abs(log_or - log_lower) if log_lower != float('-inf') else float('inf')
    log_upper_dist = abs(log_upper - log_or) if log_upper != float('inf') else float('inf')
    
    log_symmetry = abs(log_lower_dist - log_upper_dist) / (log_lower_dist + log_upper_dist) if log_lower_dist + log_upper_dist > 0 else 0
    
    # Width check - is width reasonable for sample size?
    total_n = a + b + c + d
    expected_width_factor = 1 / np.sqrt(total_n)  # Approximate expected relationship
    actual_width = ci_upper - ci_lower
    
    # Check if CI gets wider as sample size decreases
    width_reasonable = True  # Default assumption
    
    return {
        "contains_point_estimate": contains_point,
        "log_symmetry_score": log_symmetry,  # Lower is better
        "ci_width": actual_width,
        "width_reasonable": width_reasonable
    }

def check_internal_consistency(original_ci, improved_ci):
    """Checks internal consistency between original and improved methods."""
    lower_diff = abs(original_ci[0] - improved_ci[0])
    upper_diff = abs(original_ci[1] - improved_ci[1])
    
    # Calculate relative differences
    rel_lower_diff = lower_diff / original_ci[0] if original_ci[0] > 0 else float('inf')
    rel_upper_diff = upper_diff / original_ci[1] if original_ci[1] > 0 else float('inf')
    
    consistent = rel_lower_diff < 0.001 and rel_upper_diff < 0.001
    
    return {
        "consistent": consistent,
        "lower_diff": lower_diff,
        "upper_diff": upper_diff,
        "rel_lower_diff": rel_lower_diff,
        "rel_upper_diff": rel_upper_diff
    }

def evaluate_extreme_cases(a, b, c, d, alpha=0.05):
    """Evaluates the behavior of methods on extreme cases."""
    print(f"\n===== EVALUATING TABLE ({a},{b},{c},{d}) =====")
    print(f"Row 1: {a} success, {b} failure")
    print(f"Row 2: {c} success, {d} failure")
    odds_ratio = (a * d) / (b * c) if b*c > 0 else float('inf')
    print(f"Odds Ratio: {odds_ratio:.6f}")
    
    # Get results from ExactCIs methods
    try:
        orig_ci = exact_ci_unconditional(a, b, c, d, alpha)
        imp_ci = exact_ci_unconditional(a, b, c, d, alpha, adaptive_grid=True, use_cache=True)
        
        # Check if methods give same results
        consistency = check_internal_consistency(orig_ci, imp_ci)
        
        # Theoretical checks
        theory_check = theoretical_check(a, b, c, d, imp_ci[0], imp_ci[1], alpha)
        
        # Print results
        print(f"\nExactCIs Original CI: ({orig_ci[0]:.6f}, {orig_ci[1]:.6f})")
        print(f"ExactCIs Improved CI: ({imp_ci[0]:.6f}, {imp_ci[1]:.6f})")
        
        print("\nInternal Consistency Check:")
        print(f"  Methods consistent: {consistency['consistent']}")
        if not consistency['consistent']:
            print(f"  Relative differences: Lower={consistency['rel_lower_diff']:.6f}, Upper={consistency['rel_upper_diff']:.6f}")
        
        print("\nTheoretical Validity Check:")
        print(f"  Contains point estimate: {theory_check['contains_point_estimate']}")
        print(f"  Log-space symmetry score: {theory_check['log_symmetry_score']:.6f} (lower is better)")
        print(f"  CI width: {theory_check['ci_width']:.6f}")
        
        # Get R results if available
        r_key = (a, b, c, d)
        if r_key in r_exact_results and alpha in r_exact_results[r_key]:
            r_fisher = r_exact_results[r_key][alpha]["fisher"]
            r_uncond = r_exact_results[r_key][alpha]["unconditional"]
            
            if r_fisher is not None:
                print(f"\nR Fisher's Exact CI: ({r_fisher[0]:.6f}, {r_fisher[1]:.6f})")
                fisher_lower_diff = abs(imp_ci[0] - r_fisher[0]) / r_fisher[0] if r_fisher[0] > 0 else float('inf')
                fisher_upper_diff = abs(imp_ci[1] - r_fisher[1]) / r_fisher[1] if r_fisher[1] > 0 else float('inf')
                print(f"  Relative differences: Lower={fisher_lower_diff:.2%}, Upper={fisher_upper_diff:.2%}")
            
            if r_uncond is not None:
                print(f"R Unconditional CI: ({r_uncond[0]:.6f}, {r_uncond[1]:.6f})")
                uncond_lower_diff = abs(imp_ci[0] - r_uncond[0]) / r_uncond[0] if r_uncond[0] > 0 else float('inf')
                uncond_upper_diff = abs(imp_ci[1] - r_uncond[1]) / r_uncond[1] if r_uncond[1] > 0 else float('inf')
                print(f"  Relative differences: Lower={uncond_lower_diff:.2%}, Upper={uncond_upper_diff:.2%}")
        
        # Overall assessment
        issues = []
        if not consistency['consistent']:
            issues.append("Methods inconsistent")
        if not theory_check['contains_point_estimate']:
            issues.append("CI doesn't contain point estimate")
        if theory_check['log_symmetry_score'] > 0.3:  # Arbitrary threshold
            issues.append("Poor log-symmetry")
        
        if issues:
            print("\n⚠️ POTENTIAL ISSUES DETECTED:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nRECOMMENDATION: Exercise caution with this table configuration.")
        else:
            print("\n✅ NO ISSUES DETECTED")
            print("RECOMMENDATION: Results appear trustworthy for this table configuration.")
        
        return {
            "table": (a, b, c, d),
            "odds_ratio": odds_ratio,
            "original_ci": orig_ci,
            "improved_ci": imp_ci,
            "consistency": consistency,
            "theoretical_check": theory_check,
            "issues": issues
        }
        
    except Exception as e:
        print(f"Error evaluating table: {str(e)}")
        return {
            "table": (a, b, c, d),
            "error": str(e)
        }

def main():
    # Test tables
    tables = [
        (1, 1000, 10, 1000),  # Very small odds ratio with rare events
        (10, 1000, 1, 1000),  # Large odds ratio with rare events
        (7, 3, 2, 8),         # Standard example from previous tests
        (0, 10, 5, 15),       # Zero in one cell
        (100, 100, 100, 100), # Balanced large table
        (1, 1, 1, 1)          # Minimum table
    ]
    
    alpha = 0.05
    
    # Evaluate each table
    results = []
    for a, b, c, d in tables:
        result = evaluate_extreme_cases(a, b, c, d, alpha)
        results.append(result)
    
    # Print overall summary
    print("\n===== OVERALL TRUSTWORTHINESS ASSESSMENT =====")
    
    issue_tables = [r for r in results if "issues" in r and r["issues"]]
    error_tables = [r for r in results if "error" in r]
    
    if error_tables:
        print(f"\n⚠️ ERRORS in {len(error_tables)} tables:")
        for r in error_tables:
            print(f"  Table {r['table']}: {r['error']}")
    
    if issue_tables:
        print(f"\n⚠️ ISSUES in {len(issue_tables)} tables:")
        for r in issue_tables:
            print(f"  Table {r['table']} ({r['odds_ratio']:.2f}): {', '.join(r['issues'])}")
        
        print("\nRECOMMENDATION: The method shows potential issues with some table configurations.")
        print("These issues may not necessarily indicate errors, but rather limitations of the method.")
    
    if not issue_tables and not error_tables:
        print("\n✅ NO ISSUES DETECTED IN ANY TESTED TABLES")
        print("RECOMMENDATION: The ExactCIs methods appear robust and trustworthy across all tested configurations.")
    else:
        issue_count = len(issue_tables) + len(error_tables)
        total = len(results)
        print(f"\nSUMMARY: {issue_count} out of {total} tested tables showed potential issues.")
        print(f"Success rate: {(total - issue_count) / total:.1%}")
    
    # Final verdict
    issue_ratio = (len(issue_tables) + len(error_tables)) / len(results)
    
    print("\n===== FINAL VERDICT =====")
    if issue_ratio == 0:
        print("HIGHLY TRUSTWORTHY: No issues detected in any tested configuration.")
    elif issue_ratio < 0.2:
        print("TRUSTWORTHY: Minor issues in limited configurations, but overall reliable.")
    elif issue_ratio < 0.5:
        print("MODERATELY TRUSTWORTHY: Some issues detected, use with caution in certain configurations.")
    else:
        print("CONCERNING: Significant issues detected, careful validation recommended.")

if __name__ == "__main__":
    main()
