#!/usr/bin/env python
"""
ExactCIs Logical Comparison Tables

This script generates and analyzes 2x2 contingency tables for testing confidence interval methods.
It includes optimizations for performance, flexible method selection, and detailed reporting.

Features:
- Configurable test cases with different sample sizes
- Support for all confidence interval methods (Wald, conditional, midp, unconditional)
- Performance monitoring with timestamps and execution times
- Customizable parameters for computationally intensive methods
- Comprehensive output in various formats (console, JSON, Markdown)

Usage:
    uv run python analysis/logical_comparison/logical_comparison.py [options]

Options:
    --methods METHOD1,METHOD2,...  Methods to use (default: all)
    --test-cases CASE1,CASE2,...   Test cases to run (default: all)
    --grid-size SIZE               Grid size for unconditional method (default: 20)
    --timeout SECONDS              Timeout in seconds for unconditional method (default: 10)
    --no-haldane                   Disable Haldane correction for unconditional method
    --output-format FORMAT         Output format: console, json, markdown, all (default: console)
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path

# Add the project root to the path so we can import exactcis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from exactcis.core import calculate_odds_ratio
from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Define available test cases
TEST_CASES = {
    "small": {
        "table_id": 1,
        "a": 1, "b": 99, "c": 2, "d": 98,
        "description": "1 vs 2 (sample size 100 each)"
    },
    "medium": {
        "table_id": 2,
        "a": 10, "b": 90, "c": 20, "d": 80,
        "description": "10 vs 20 (sample size 100 each)"
    },
    "medium_large": {
        "table_id": 6,
        "a": 20, "b": 80, "c": 40, "d": 60,
        "description": "20 vs 40 (sample size 100 each)"
    },
    "large": {
        "table_id": 3,
        "a": 100, "b": 900, "c": 200, "d": 800,
        "description": "100 vs 200 (sample size 1000 each)"
    },
    "balanced": {
        "table_id": 4,
        "a": 50, "b": 50, "c": 100, "d": 100,
        "description": "50 vs 100 (balanced exposure)"
    },
    "rare": {
        "table_id": 5,
        "a": 1, "b": 999, "c": 2, "d": 998,
        "description": "1 vs 2 (rare exposure, sample size 1000 each)"
    }
}

# Define available methods with their functions and descriptions
METHODS = {
    "wald": {
        "function": ci_wald_haldane,
        "description": "Wald method with Haldane correction",
        "params": {}
    },
    "conditional": {
        "function": exact_ci_conditional,
        "description": "Exact conditional method",
        "params": {}
    },
    "midp": {
        "function": exact_ci_midp,
        "description": "Mid-P method",
        "params": {}
    },
    "unconditional": {
        "function": exact_ci_unconditional,
        "description": "Exact unconditional method",
        "params": {
            "grid_size": 20,
            "timeout": 10,
            "haldane": True
        }
    }
}

def prepare_tables(tables: List[Dict]) -> List[Dict]:
    """
    Prepare tables by calculating odds ratios and adding metadata.
    
    Args:
        tables: List of table dictionaries with a, b, c, d values
    
    Returns:
        Updated list of table dictionaries with additional metadata
    """
    for table in tables:
        a, b, c, d = table["a"], table["b"], table["c"], table["d"]
        
        # Calculate odds ratio
        or_value = calculate_odds_ratio(a, b, c, d)
        
        # Add metadata
        table["total_cases"] = a + b
        table["total_controls"] = c + d
        table["actual_or"] = or_value
    
    return tables

def calculate_cis_for_tables(
    tables: List[Dict], 
    methods: Dict[str, Dict[str, Any]]
) -> List[Dict]:
    """
    Calculate confidence intervals for each table using specified methods.
    
    Args:
        tables: List of table dictionaries
        methods: Dictionary of method configurations
    
    Returns:
        Updated list of table dictionaries with CI information
    """
    for i, table in enumerate(tables):
        print(f"\nProcessing table {i+1}/{len(tables)}: {table['description']}")
        print(f"a={table['a']}, b={table['b']}, c={table['c']}, d={table['d']}")
        a, b, c, d = table["a"], table["b"], table["c"], table["d"]
        
        # Calculate CIs using different methods
        for method_name, method_config in methods.items():
            print(f"[{time.strftime('%H:%M:%S')}] Starting {method_name} CI calculation...")
            try:
                start_time = time.time()
                
                # Get the function and parameters for this method
                func = method_config["function"]
                params = method_config.get("params", {})
                
                # Call the function with the appropriate parameters
                lower, upper = func(a, b, c, d, **params)
                
                elapsed = time.time() - start_time
                table[f"{method_name}_ci"] = (lower, upper)
                table[f"{method_name}_ci_width"] = upper - lower
                print(f"[{time.strftime('%H:%M:%S')}] Completed {method_name} CI calculation in {elapsed:.2f} seconds: ({lower:.4f}, {upper:.4f})")
            except Exception as e:
                table[f"{method_name}_ci"] = ("error", str(e))
                table[f"{method_name}_ci_width"] = None
                print(f"[{time.strftime('%H:%M:%S')}] Error in {method_name} CI calculation: {str(e)}")
    
    return tables

def print_tables_console(tables: List[Dict], method_names: List[str]):
    """Print the tables in a readable format to the console."""
    print("\n=== Confidence Intervals for 2x2 Tables ===\n")
    
    # Create header
    header = "Table ID | Description | a | b | c | d | OR"
    for method in method_names:
        header += f" | {method.capitalize()} CI"
    
    print(header)
    print("-" * len(header) * 2)  # Double the length for better visibility
    
    # Print each table
    for table in tables:
        row = f"{table['table_id']:8} | {table['description']:20} | {table['a']:3} | {table['b']:3} | {table['c']:3} | {table['d']:3} | {table['actual_or']:.4f}"
        
        for method in method_names:
            ci = table.get(f"{method}_ci", ("N/A", "N/A"))
            if not isinstance(ci[0], str) and not isinstance(ci[1], str):
                ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
            else:
                ci_str = "N/A"
            row += f" | {ci_str:15}"
        
        print(row)
    
    print("\n")

def save_tables_json(tables: List[Dict], output_dir: str):
    """Save the tables to a JSON file."""
    output_path = os.path.join(output_dir, "comparison_tables.json")
    with open(output_path, 'w') as f:
        json.dump(tables, f, indent=2)
    print(f"Tables saved to {output_path}")

def save_tables_markdown(tables: List[Dict], method_names: List[str], output_dir: str):
    """Save the tables to a Markdown file."""
    output_path = os.path.join(output_dir, "comparison_tables.md")
    
    with open(output_path, 'w') as f:
        f.write("# Confidence Intervals for 2x2 Tables\n\n")
        
        # Write table header
        f.write("| Table ID | Description | a | b | c | d | OR |")
        for method in method_names:
            f.write(f" {method.capitalize()} CI |")
        f.write("\n")
        
        # Write separator line
        f.write("|----------|-------------|---|---|---|---|-----|")
        for _ in method_names:
            f.write("------------|")
        f.write("\n")
        
        # Write each table row
        for table in tables:
            f.write(f"| {table['table_id']} | {table['description']} | {table['a']} | {table['b']} | {table['c']} | {table['d']} | {table['actual_or']:.4f} |")
            
            for method in method_names:
                ci = table.get(f"{method}_ci", ("N/A", "N/A"))
                if not isinstance(ci[0], str) and not isinstance(ci[1], str):
                    ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
                else:
                    ci_str = "N/A"
                f.write(f" {ci_str} |")
            
            f.write("\n")
    
    print(f"Tables saved to {output_path}")

def verify_ci_narrowing(tables: List[Dict], method_names: List[str]):
    """
    Verify that CI widths decrease as counts increase for tables with similar OR values.
    
    Args:
        tables: List of table dictionaries with CI information
        method_names: List of method names to check
    
    Returns:
        Dictionary with verification results for each method
    """
    print("\n=== Verifying CI Narrowing with Increasing Counts ===\n")
    
    # Group tables by similar OR values (within 10%)
    or_groups = {}
    for table in tables:
        or_value = table["actual_or"]
        # Round to 1 decimal place for grouping
        or_key = round(or_value, 1)
        if or_key not in or_groups:
            or_groups[or_key] = []
        or_groups[or_key].append(table)
    
    results = {}
    
    # For each OR group, check if CI widths decrease with increasing counts
    for or_key, group_tables in or_groups.items():
        if len(group_tables) < 2:
            print(f"Skipping OR group {or_key} - not enough tables for comparison")
            continue
        
        print(f"\nAnalyzing tables with OR ≈ {or_key}:")
        
        # Sort tables by total count (a+b+c+d)
        sorted_tables = sorted(group_tables, key=lambda t: t["a"] + t["b"] + t["c"] + t["d"])
        
        # Check each method
        for method in method_names:
            print(f"\n  Method: {method}")
            results[method] = {"consistent": True, "anomalies": []}
            
            # Get CI widths for each table
            widths = []
            for table in sorted_tables:
                width_key = f"{method}_ci_width"
                ci_key = f"{method}_ci"
                
                if width_key in table and table[width_key] is not None:
                    # Check if the CI is valid (lower bound <= upper bound)
                    ci = table.get(ci_key, ("N/A", "N/A"))
                    if not isinstance(ci[0], str) and not isinstance(ci[1], str):
                        if ci[0] > ci[1]:
                            print(f"    ⚠️ Table {table['table_id']} ({table['description']}): Invalid CI - lower bound ({ci[0]:.4f}) > upper bound ({ci[1]:.4f})")
                            # Use absolute width for comparison
                            abs_width = abs(table[width_key])
                            widths.append((table["table_id"], table["description"], abs_width, True))
                            print(f"    Table {table['table_id']} ({table['description']}): Using absolute CI width = {abs_width:.4f} (original: {table[width_key]:.4f})")
                        else:
                            widths.append((table["table_id"], table["description"], table[width_key], False))
                            print(f"    Table {table['table_id']} ({table['description']}): CI width = {table[width_key]:.4f}")
            
            # Check if widths decrease as counts increase
            for i in range(1, len(widths)):
                prev_id, prev_desc, prev_width, prev_invalid = widths[i-1]
                curr_id, curr_desc, curr_width, curr_invalid = widths[i]
                
                if curr_width >= prev_width:
                    anomaly_type = "wider" if curr_width > prev_width else "equal"
                    anomaly = f"Anomaly: Table {curr_id} ({curr_desc}) has {anomaly_type} CI than Table {prev_id} ({prev_desc})"
                    if prev_invalid or curr_invalid:
                        anomaly += " (Note: One or both CIs had invalid bounds)"
                    print(f"    ⚠️ {anomaly}")
                    results[method]["consistent"] = False
                    results[method]["anomalies"].append(anomaly)
            
            if results[method]["consistent"]:
                print(f"    ✓ CI widths consistently decrease with increasing counts")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run optimized comparison tables for ExactCIs")
    
    parser.add_argument("--methods", type=str, default="wald,conditional,midp,unconditional",
                        help="Comma-separated list of methods to use (default: all)")
    
    parser.add_argument("--test-cases", type=str, default="small,medium",
                        help="Comma-separated list of test cases to run (default: small,medium)")
    
    parser.add_argument("--grid-size", type=int, default=20,
                        help="Grid size for unconditional method (default: 20)")
    
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds for unconditional method (default: 10)")
    
    parser.add_argument("--no-haldane", action="store_true",
                        help="Disable Haldane correction for unconditional method")
    
    parser.add_argument("--output-format", type=str, default="console",
                        choices=["console", "json", "markdown", "all"],
                        help="Output format (default: console)")
    
    return parser.parse_args()

def main():
    """Generate and test the tables."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine which methods to use
    method_names = args.methods.split(",")
    selected_methods = {}
    for name in method_names:
        if name in METHODS:
            selected_methods[name] = METHODS[name].copy()
            # Update parameters for unconditional method
            if name == "unconditional":
                selected_methods[name]["params"] = {
                    "grid_size": args.grid_size,
                    "timeout": args.timeout,
                    "haldane": not args.no_haldane
                }
    
    # Determine which test cases to use
    test_case_names = args.test_cases.split(",")
    selected_test_cases = []
    for name in test_case_names:
        if name in TEST_CASES:
            selected_test_cases.append(TEST_CASES[name].copy())
    
    # Use the current directory for output
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting comparison tables generation...")
    print(f"[{time.strftime('%H:%M:%S')}] Using methods: {', '.join(method_names)}")
    print(f"[{time.strftime('%H:%M:%S')}] Using test cases: {', '.join(test_case_names)}")
    
    if "unconditional" in method_names:
        print(f"[{time.strftime('%H:%M:%S')}] Unconditional parameters: grid_size={args.grid_size}, timeout={args.timeout}s, haldane={not args.no_haldane}")
    
    start_total = time.time()
    
    print(f"[{time.strftime('%H:%M:%S')}] Preparing test tables...")
    # Prepare the tables
    tables = prepare_tables(selected_test_cases)
    print(f"[{time.strftime('%H:%M:%S')}] Prepared {len(tables)} test tables successfully")
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting CI calculations for all tables...")
    # Calculate CIs for each table
    tables_with_cis = calculate_cis_for_tables(tables, selected_methods)
    print(f"[{time.strftime('%H:%M:%S')}] Completed all CI calculations")
    
    # Output results in the requested format
    if args.output_format in ["console", "all"]:
        print(f"[{time.strftime('%H:%M:%S')}] Printing summary table...")
        print_tables_console(tables_with_cis, method_names)
    
    if args.output_format in ["json", "all"]:
        print(f"[{time.strftime('%H:%M:%S')}] Saving results to JSON file...")
        save_tables_json(tables_with_cis, output_dir)
    
    if args.output_format in ["markdown", "all"]:
        print(f"[{time.strftime('%H:%M:%S')}] Saving results to Markdown file...")
        save_tables_markdown(tables_with_cis, method_names, output_dir)
    
    # Verify that CI widths decrease as counts increase
    print(f"[{time.strftime('%H:%M:%S')}] Verifying CI narrowing with increasing counts...")
    verification_results = verify_ci_narrowing(tables_with_cis, method_names)
    
    # Summarize verification results
    print(f"\n[{time.strftime('%H:%M:%S')}] Verification summary:")
    all_consistent = True
    for method, result in verification_results.items():
        if result["consistent"]:
            print(f"  ✓ {method}: CI widths consistently decrease with increasing counts")
        else:
            all_consistent = False
            print(f"  ⚠️ {method}: Found {len(result['anomalies'])} anomalies where CI widths don't decrease")
            for anomaly in result["anomalies"]:
                print(f"    - {anomaly}")
    
    if all_consistent:
        print(f"\n[{time.strftime('%H:%M:%S')}] All methods show consistent narrowing of CIs with increasing counts")
    else:
        print(f"\n[{time.strftime('%H:%M:%S')}] Some methods show anomalies in CI narrowing pattern")
    
    # Display total execution time
    total_time = time.time() - start_total
    print(f"[{time.strftime('%H:%M:%S')}] All operations completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()