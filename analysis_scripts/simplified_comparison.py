#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from exactcis.methods.unconditional import exact_ci_unconditional, improved_ci_unconditional

def run_comparison():
    # Load R results from the JSON file
    try:
        with open('r_comparison_results.json', 'r') as f:
            r_results = json.load(f)
    except FileNotFoundError:
        print("R comparison results file not found.")
        return
    
    # List to hold all comparison results
    all_results = []
    
    # Alpha values we care about
    alphas = ['0.05', '0.01', '0.1']
    
    # Process each alpha value
    for alpha_str in alphas:
        alpha = float(alpha_str)
        
        # Get R results for this alpha
        if alpha_str not in r_results:
            print(f"No R results for alpha={alpha}")
            continue
            
        r_alpha_results = r_results[alpha_str]
        
        # Process each table in the R results
        for r_result in r_alpha_results:
            # Get table details
            a = r_result['table']['a']
            b = r_result['table']['b']
            c = r_result['table']['c']
            d = r_result['table']['d']
            name = r_result['name']
            
            print(f"Processing table ({a},{b},{c},{d}) - {name}...")
            
            # Get R confidence intervals
            r_fisher_ci = (r_result['fisher']['lower'], r_result['fisher']['upper'])
            r_exact_ci = (r_result['unconditional_exact']['lower'], r_result['unconditional_exact']['upper'])
            
            # Calculate Python confidence intervals
            try:
                py_orig_ci = exact_ci_unconditional(a, b, c, d, alpha)
            except Exception as e:
                print(f"  Error in Python original method: {str(e)}")
                py_orig_ci = (np.nan, np.nan)
            
            try:
                py_imp_ci = improved_ci_unconditional(a, b, c, d, alpha)
            except Exception as e:
                print(f"  Error in Python improved method: {str(e)}")
                py_imp_ci = (np.nan, np.nan)
            
            # Add result to list
            result = {
                'Table': f"({a},{b},{c},{d}) - {name}",
                'Alpha': alpha,
                'Python Original Lower': py_orig_ci[0],
                'Python Original Upper': py_orig_ci[1],
                'Python Improved Lower': py_imp_ci[0],
                'Python Improved Upper': py_imp_ci[1],
                'R Fisher Lower': r_fisher_ci[0],
                'R Fisher Upper': r_fisher_ci[1],
                'R Unconditional Lower': r_exact_ci[0],
                'R Unconditional Upper': r_exact_ci[1]
            }
            
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    df.to_csv('method_comparison.csv', index=False)
    
    # Print summary of results
    print("\n=== Summary of Comparisons with R's Unconditional Method ===")
    
    # Calculate differences
    df['Orig_Lower_Diff'] = np.abs(df['Python Original Lower'] - df['R Unconditional Lower'])
    df['Orig_Upper_Diff'] = np.abs(df['Python Original Upper'] - df['R Unconditional Upper'])
    df['Imp_Lower_Diff'] = np.abs(df['Python Improved Lower'] - df['R Unconditional Lower'])
    df['Imp_Upper_Diff'] = np.abs(df['Python Improved Upper'] - df['R Unconditional Upper'])
    
    # For valid comparisons only
    valid_df = df.dropna(subset=['Python Original Lower', 'Python Improved Lower', 
                                'R Unconditional Lower', 'R Unconditional Upper'])
    
    print(f"\nBased on {len(valid_df)} valid comparisons:")
    
    # Calculate average absolute differences
    orig_avg_diff = (valid_df['Orig_Lower_Diff'].mean() + valid_df['Orig_Upper_Diff'].mean()) / 2
    imp_avg_diff = (valid_df['Imp_Lower_Diff'].mean() + valid_df['Imp_Upper_Diff'].mean()) / 2
    
    print(f"\nAverage absolute difference from R's unconditional method:")
    print(f"  Python Original: {orig_avg_diff:.6f}")
    print(f"  Python Improved: {imp_avg_diff:.6f}")
    
    # Count which method is closer
    orig_closer = ((valid_df['Orig_Lower_Diff'] < valid_df['Imp_Lower_Diff']) & 
                   (valid_df['Orig_Upper_Diff'] < valid_df['Imp_Upper_Diff'])).sum()
    imp_closer = ((valid_df['Orig_Lower_Diff'] > valid_df['Imp_Lower_Diff']) & 
                  (valid_df['Orig_Upper_Diff'] > valid_df['Imp_Upper_Diff'])).sum()
    mixed = len(valid_df) - orig_closer - imp_closer
    
    print(f"\nOverall comparison:")
    print(f"  Python Original closer to R: {orig_closer} tables")
    print(f"  Python Improved closer to R: {imp_closer} tables")
    print(f"  Mixed results: {mixed} tables")
    
    # Print the top 5 most divergent cases
    print("\nTop 5 most divergent cases between Python methods and R:")
    df['Total_Orig_Diff'] = df['Orig_Lower_Diff'] + df['Orig_Upper_Diff']
    df['Total_Imp_Diff'] = df['Imp_Lower_Diff'] + df['Imp_Upper_Diff']
    
    top_orig = df.nlargest(5, 'Total_Orig_Diff')
    top_imp = df.nlargest(5, 'Total_Imp_Diff')
    
    print("\nTop divergent cases for Original method:")
    for _, row in top_orig.iterrows():
        print(f"  Table: {row['Table']}, Alpha: {row['Alpha']}")
        print(f"    Python: ({row['Python Original Lower']:.6f}, {row['Python Original Upper']:.6f})")
        print(f"    R Uncond: ({row['R Unconditional Lower']:.6f}, {row['R Unconditional Upper']:.6f})")
        print(f"    Difference: {row['Total_Orig_Diff']:.6f}")
    
    print("\nTop divergent cases for Improved method:")
    for _, row in top_imp.iterrows():
        print(f"  Table: {row['Table']}, Alpha: {row['Alpha']}")
        print(f"    Python: ({row['Python Improved Lower']:.6f}, {row['Python Improved Upper']:.6f})")
        print(f"    R Uncond: ({row['R Unconditional Lower']:.6f}, {row['R Unconditional Upper']:.6f})")
        print(f"    Difference: {row['Total_Imp_Diff']:.6f}")
    
    return df

if __name__ == "__main__":
    print("Running simplified comparison between Python and R methods...")
    df = run_comparison()
    print("\nResults saved to method_comparison.csv")
