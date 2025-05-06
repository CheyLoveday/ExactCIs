#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from exactcis.methods.unconditional import exact_ci_unconditional, improved_ci_unconditional

# Test cases to match our R comparison
test_cases = [
    # Standard tables
    {"a": 7, "b": 3, "c": 2, "d": 8, "name": "Standard table 1"},
    {"a": 10, "b": 10, "c": 10, "d": 10, "name": "Balanced small table"},
    
    # Tables with zero cells
    {"a": 0, "b": 10, "c": 5, "d": 15, "name": "Table with a zero"},
    {"a": 10, "b": 0, "c": 5, "d": 15, "name": "Table with b zero"},
    
    # Various table sizes and distributions
    {"a": 40, "b": 10, "c": 20, "d": 30, "name": "Unbalanced medium table"},
    {"a": 100, "b": 50, "c": 60, "d": 120, "name": "Large table 1"},
    {"a": 500, "b": 500, "c": 300, "d": 700, "name": "Large table 2"},
    
    # Extreme proportions
    {"a": 99, "b": 1, "c": 50, "d": 50, "name": "Extreme proportion 1"},
    {"a": 1, "b": 99, "c": 50, "d": 50, "name": "Extreme proportion 2"},
    
    # Different group sizes
    {"a": 5, "b": 5, "c": 90, "d": 10, "name": "Very different group sizes 1"},
    {"a": 90, "b": 10, "c": 5, "d": 5, "name": "Very different group sizes 2"}
]

# Alpha values to test
alpha_values = [0.05, 0.01, 0.1]

def run_comparison():
    # Load R results from the JSON file
    try:
        with open('r_comparison_results.json', 'r') as f:
            r_results = json.load(f)
    except FileNotFoundError:
        print("R comparison results file not found.")
        return
    
    # Create DataFrame to store all results
    columns = ['Table', 'Alpha', 
               'Python Original Lower', 'Python Original Upper',
               'Python Improved Lower', 'Python Improved Upper',
               'R Fisher Lower', 'R Fisher Upper',
               'R Unconditional Lower', 'R Unconditional Upper']
    
    results_data = []
    
    # Run through all test cases
    for alpha in alpha_values:
        for test_case in test_cases:
            a, b, c, d = test_case['a'], test_case['b'], test_case['c'], test_case['d']
            name = test_case['name']
            
            if alpha == 0.05:
                display_name = name
            else:
                display_name = f"{name} (alpha={alpha})"
            
            print(f"Processing {display_name}...")
            
            # Get Python results
            try:
                py_orig_lower, py_orig_upper = exact_ci_unconditional(a, b, c, d, alpha)
            except Exception as e:
                print(f"Error in Python original method: {str(e)}")
                py_orig_lower, py_orig_upper = np.nan, np.nan
            
            try:
                py_imp_lower, py_imp_upper = improved_ci_unconditional(a, b, c, d, alpha)
            except Exception as e:
                print(f"Error in Python improved method: {str(e)}")
                py_imp_lower, py_imp_upper = np.nan, np.nan
            
            # Find matching R result
            r_alpha_results = r_results[str(alpha)]
            matching_r_result = None
            
            for result in r_alpha_results:
                if (result['table']['a'] == a and 
                    result['table']['b'] == b and 
                    result['table']['c'] == c and 
                    result['table']['d'] == d):
                    matching_r_result = result
                    break
            
            if matching_r_result:
                r_fisher_lower = matching_r_result['fisher']['lower']
                r_fisher_upper = matching_r_result['fisher']['upper']
                r_exact_lower = matching_r_result['unconditional_exact']['lower']
                r_exact_upper = matching_r_result['unconditional_exact']['upper']
            else:
                r_fisher_lower, r_fisher_upper = np.nan, np.nan
                r_exact_lower, r_exact_upper = np.nan, np.nan
            
            # Add to results
            row = [
                f"({a},{b},{c},{d}) - {display_name}",
                alpha,
                py_orig_lower,
                py_orig_upper,
                py_imp_lower, 
                py_imp_upper,
                r_fisher_lower,
                r_fisher_upper,
                r_exact_lower,
                r_exact_upper
            ]
            
            results_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data, columns=columns)
    df.to_csv('all_method_comparison.csv', index=False)
    
    # Calculate similarity scores
    print("\nCalculating similarity to R's unconditional exact method...")
    
    df['Original_Lower_Diff'] = abs(df['Python Original Lower'] - df['R Unconditional Lower'])
    df['Original_Upper_Diff'] = abs(df['Python Original Upper'] - df['R Unconditional Upper'])
    df['Improved_Lower_Diff'] = abs(df['Python Improved Lower'] - df['R Unconditional Lower'])
    df['Improved_Upper_Diff'] = abs(df['Python Improved Upper'] - df['R Unconditional Upper'])
    
    # For cases where both methods give values
    valid_comparisons = df.dropna(subset=['Python Original Lower', 'Python Improved Lower', 
                                          'R Unconditional Lower', 'R Unconditional Upper'])
    
    print(f"\nBased on {len(valid_comparisons)} valid comparisons:")
    
    # Calculate which method is closer to R's unconditional exact method
    orig_closer_lower = (valid_comparisons['Original_Lower_Diff'] < valid_comparisons['Improved_Lower_Diff']).sum()
    imp_closer_lower = (valid_comparisons['Original_Lower_Diff'] > valid_comparisons['Improved_Lower_Diff']).sum()
    
    orig_closer_upper = (valid_comparisons['Original_Upper_Diff'] < valid_comparisons['Improved_Upper_Diff']).sum()
    imp_closer_upper = (valid_comparisons['Original_Upper_Diff'] > valid_comparisons['Improved_Upper_Diff']).sum()
    
    print(f"\nLower bound comparison to R's unconditional exact:")
    print(f"* Original method closer: {orig_closer_lower} times")
    print(f"* Improved method closer: {imp_closer_lower} times")
    
    print(f"\nUpper bound comparison to R's unconditional exact:")
    print(f"* Original method closer: {orig_closer_upper} times")
    print(f"* Improved method closer: {imp_closer_upper} times")
    
    # Print detailed comparison
    print("\n--------------------------------------------------------------------------------")
    print("Detailed Comparison with R's Unconditional Exact Method")
    print("--------------------------------------------------------------------------------")
    
    for index, row in valid_comparisons.iterrows():
        table = row['Table']
        alpha = row['Alpha']
        
        py_orig = (row['Python Original Lower'], row['Python Original Upper'])
        py_imp = (row['Python Improved Lower'], row['Python Improved Upper'])
        r_exact = (row['R Unconditional Lower'], row['R Unconditional Upper'])
        
        orig_diff = (row['Original_Lower_Diff'], row['Original_Upper_Diff'])
        imp_diff = (row['Improved_Lower_Diff'], row['Improved_Upper_Diff'])
        
        print(f"\nTable: {table}")
        print(f"Alpha: {alpha}")
        print(f"Python Original:    ({py_orig[0]:.6f}, {py_orig[1]:.6f})")
        print(f"Python Improved:    ({py_imp[0]:.6f}, {py_imp[1]:.6f})")
        print(f"R Unconditional:    ({r_exact[0]:.6f}, {r_exact[1]:.6f})")
        print(f"Original Diff:      ({orig_diff[0]:.6f}, {orig_diff[1]:.6f})")
        print(f"Improved Diff:      ({imp_diff[0]:.6f}, {imp_diff[1]:.6f})")
        
        # Determine which is closest overall
        orig_total_diff = orig_diff[0] + orig_diff[1]
        imp_total_diff = imp_diff[0] + imp_diff[1]
        
        if orig_total_diff < imp_total_diff:
            closer = "ORIGINAL"
        else:
            closer = "IMPROVED"
            
        print(f"Closest method:     {closer}")
    
    return df

if __name__ == "__main__":
    print("Comparing Python methods with R results...")
    df = run_comparison()
    print("\nResults saved to all_method_comparison.csv")
