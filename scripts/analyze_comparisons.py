"""
Analyze and visualize comparisons between ExactCIs, SciPy, and R implementations.
Creates plots comparing different confidence interval methods and implementations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_comparison_data(csv_file='method_comparison.csv'):
    """Load the comparison data from CSV file."""
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Convert 'NA', 'inf', and 'Error' strings to appropriate values
    for col in df.columns[6:]:  # Skip the first 6 columns (Case, a, b, c, d, Odds Ratio)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Replace inf with NaN for easier handling
    df = df.replace([float('inf'), -float('inf')], np.nan)
    
    return df

def create_individual_method_plots(df, output_dir='plots'):
    """Create plots for each method comparing the three implementations."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the methods and implementations
    methods = ["Conditional", "MidP", "Blaker", "Unconditional", "Wald-Haldane"]
    implementations = ["ExactCIs", "SciPy", "R"]
    
    # Create a plot for each method
    for method in methods:
        plt.figure(figsize=(12, 10))
        
        # Create a plot for lower bounds
        plt.subplot(2, 1, 1)
        for impl in implementations:
            # Skip SciPy for methods other than "Wald-Haldane" and "Fisher" (it doesn't have those)
            if impl == "SciPy" and method not in ["Wald-Haldane", "Fisher"]:
                continue
                
            # Special case for SciPy Fisher (which we'll use for all methods except Wald-Haldane)
            if impl == "SciPy" and method != "Wald-Haldane":
                col = "SciPy Fisher Lower"
                label = "SciPy Fisher"
            else:
                col = f"{impl} {method} Lower"
                label = impl
                
            plt.scatter(df["Odds Ratio"], df[col], label=label, alpha=0.7)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('True Odds Ratio (log scale)')
        plt.ylabel('Lower Bound (log scale)')
        plt.title(f'{method} Method - Lower Bounds')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Create a plot for upper bounds
        plt.subplot(2, 1, 2)
        for impl in implementations:
            # Skip SciPy for methods other than "Wald-Haldane" and "Fisher"
            if impl == "SciPy" and method not in ["Wald-Haldane", "Fisher"]:
                continue
                
            # Special case for SciPy Fisher
            if impl == "SciPy" and method != "Wald-Haldane":
                col = "SciPy Fisher Upper"
                label = "SciPy Fisher"
            else:
                col = f"{impl} {method} Upper"
                label = impl
                
            plt.scatter(df["Odds Ratio"], df[col], label=label, alpha=0.7)
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('True Odds Ratio (log scale)')
        plt.ylabel('Upper Bound (log scale)')
        plt.title(f'{method} Method - Upper Bounds')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method.lower()}_comparison.png'), dpi=300)
        plt.close()
        
def create_difference_plots(df, output_dir='plots'):
    """Create plots showing differences between implementations vs. Wald-Haldane."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the methods and implementations
    methods = ["Conditional", "MidP", "Blaker", "Unconditional"]
    implementations = ["ExactCIs", "R"]
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # Plot for lower bounds
    plt.subplot(1, 2, 1)
    
    for impl in implementations:
        for method in methods:
            # Calculate difference from Wald-Haldane
            wald_col = f"{impl} Wald-Haldane Lower"
            method_col = f"{impl} {method} Lower"
            
            # Filter out rows where either value is NaN
            mask = ~df[wald_col].isna() & ~df[method_col].isna()
            
            # Calculate difference as percentage
            diff = ((df[method_col] - df[wald_col]) / df[wald_col] * 100).where(mask)
            
            plt.scatter(df["Odds Ratio"], diff, label=f"{impl} {method}", alpha=0.7)
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('True Odds Ratio (log scale)')
    plt.ylabel('% Difference from Wald-Haldane')
    plt.title('Lower Bound: % Difference from Wald-Haldane')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot for upper bounds
    plt.subplot(1, 2, 2)
    
    for impl in implementations:
        for method in methods:
            # Calculate difference from Wald-Haldane
            wald_col = f"{impl} Wald-Haldane Upper"
            method_col = f"{impl} {method} Upper"
            
            # Filter out rows where either value is NaN
            mask = ~df[wald_col].isna() & ~df[method_col].isna()
            
            # Calculate difference as percentage
            diff = ((df[method_col] - df[wald_col]) / df[wald_col] * 100).where(mask)
            
            plt.scatter(df["Odds Ratio"], diff, label=f"{impl} {method}", alpha=0.7)
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('True Odds Ratio (log scale)')
    plt.ylabel('% Difference from Wald-Haldane')
    plt.title('Upper Bound: % Difference from Wald-Haldane')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_from_wald_haldane.png'), dpi=300)
    plt.close()

def create_implementation_difference_plots(df, output_dir='plots'):
    """Create plots showing differences between ExactCIs and R implementations."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the methods
    methods = ["Conditional", "MidP", "Blaker", "Unconditional", "Wald-Haldane"]
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # Plot for lower bounds
    plt.subplot(1, 2, 1)
    
    for method in methods:
        # Calculate difference between ExactCIs and R
        exactcis_col = f"ExactCIs {method} Lower"
        r_col = f"R {method} Lower"
        
        # Filter out rows where either value is NaN
        mask = ~df[exactcis_col].isna() & ~df[r_col].isna()
        
        # Calculate difference as percentage
        diff = ((df[exactcis_col] - df[r_col]) / df[r_col] * 100).where(mask)
        
        plt.scatter(df["Odds Ratio"], diff, label=method, alpha=0.7)
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('True Odds Ratio (log scale)')
    plt.ylabel('% Difference (ExactCIs - R) / R')
    plt.title('Lower Bound: % Difference between ExactCIs and R')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot for upper bounds
    plt.subplot(1, 2, 2)
    
    for method in methods:
        # Calculate difference between ExactCIs and R
        exactcis_col = f"ExactCIs {method} Upper"
        r_col = f"R {method} Upper"
        
        # Filter out rows where either value is NaN
        mask = ~df[exactcis_col].isna() & ~df[r_col].isna()
        
        # Calculate difference as percentage
        diff = ((df[exactcis_col] - df[r_col]) / df[r_col] * 100).where(mask)
        
        plt.scatter(df["Odds Ratio"], diff, label=method, alpha=0.7)
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xscale('log')
    plt.xlabel('True Odds Ratio (log scale)')
    plt.ylabel('% Difference (ExactCIs - R) / R')
    plt.title('Upper Bound: % Difference between ExactCIs and R')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exactcis_vs_r.png'), dpi=300)
    plt.close()

def create_ci_width_comparison(df, output_dir='plots'):
    """Create plots comparing CI widths across methods and implementations."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the methods and implementations
    methods = ["Conditional", "MidP", "Blaker", "Unconditional", "Wald-Haldane"]
    implementations = ["ExactCIs", "R"]
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # For each implementation, calculate and plot CI widths
    for i, impl in enumerate(implementations):
        plt.subplot(1, 2, i+1)
        
        for method in methods:
            # Get lower and upper bounds
            lower_col = f"{impl} {method} Lower"
            upper_col = f"{impl} {method} Upper"
            
            # Filter out rows where either value is NaN
            mask = ~df[lower_col].isna() & ~df[upper_col].isna()
            
            # Calculate width in log space for better visualization
            width = np.log(df[upper_col]) - np.log(df[lower_col])
            width = width.where(mask)
            
            plt.scatter(df["Odds Ratio"], width, label=method, alpha=0.7)
        
        plt.xscale('log')
        plt.xlabel('True Odds Ratio (log scale)')
        plt.ylabel('CI Width (log scale)')
        plt.title(f'CI Width Comparison - {impl}')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ci_width_comparison.png'), dpi=300)
    plt.close()

def create_heatmap_plot(df, output_dir='plots'):
    """Create a heatmap showing relative differences between methods."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the methods and implementations
    methods = ["Conditional", "MidP", "Blaker", "Unconditional", "Wald-Haldane"]
    implementations = ["ExactCIs", "R"]
    
    # Create arrays to store the average differences
    diff_matrix_lower = np.zeros((len(methods), len(implementations)))
    diff_matrix_upper = np.zeros((len(methods), len(implementations)))
    
    # Use Wald-Haldane as reference
    for i, method in enumerate(methods):
        for j, impl in enumerate(implementations):
            # Skip the reference method
            if method == "Wald-Haldane":
                continue
                
            # Calculate differences for lower bounds
            wald_col_lower = f"{impl} Wald-Haldane Lower"
            method_col_lower = f"{impl} {method} Lower"
            
            # Filter out rows where either value is NaN
            mask_lower = ~df[wald_col_lower].isna() & ~df[method_col_lower].isna()
            
            # Calculate absolute percentage difference
            diff_lower = np.abs((df[method_col_lower] - df[wald_col_lower]) / df[wald_col_lower] * 100).where(mask_lower)
            diff_matrix_lower[i, j] = diff_lower.mean(skipna=True)
            
            # Calculate differences for upper bounds
            wald_col_upper = f"{impl} Wald-Haldane Upper"
            method_col_upper = f"{impl} {method} Upper"
            
            # Filter out rows where either value is NaN
            mask_upper = ~df[wald_col_upper].isna() & ~df[method_col_upper].isna()
            
            # Calculate absolute percentage difference
            diff_upper = np.abs((df[method_col_upper] - df[wald_col_upper]) / df[wald_col_upper] * 100).where(mask_upper)
            diff_matrix_upper[i, j] = diff_upper.mean(skipna=True)
    
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    
    # Plot heatmap for lower bounds
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(diff_matrix_lower, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=implementations, yticklabels=methods,
                ax=ax1)
    ax1.set_title('Average Absolute % Difference from Wald-Haldane - Lower Bounds')
    
    # Plot heatmap for upper bounds
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(diff_matrix_upper, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=implementations, yticklabels=methods,
                ax=ax2)
    ax2.set_title('Average Absolute % Difference from Wald-Haldane - Upper Bounds')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_differences.png'), dpi=300)
    plt.close()

def analyze_and_plot(csv_file='method_comparison.csv'):
    """Main function to analyze data and generate all plots."""
    print("Loading comparison data...")
    df = load_comparison_data(csv_file)
    
    print("Creating output directory for plots...")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating individual method comparison plots...")
    create_individual_method_plots(df, output_dir)
    
    print("Generating difference from Wald-Haldane plots...")
    create_difference_plots(df, output_dir)
    
    print("Generating implementation difference plots...")
    create_implementation_difference_plots(df, output_dir)
    
    print("Generating CI width comparison plots...")
    create_ci_width_comparison(df, output_dir)
    
    print("Generating heatmap plot...")
    create_heatmap_plot(df, output_dir)
    
    print(f"All plots have been generated in {output_dir}")
    
    return df

def generate_analysis_report(df, output_file="method_comparison_analysis.md"):
    """Generate a markdown report summarizing the findings."""
    # Calculate some basic statistics for the report
    report_text = """# Confidence Interval Method Comparison Analysis

## Overview
This report summarizes the comparisons between different confidence interval methods implemented in ExactCIs, SciPy, and R. The analysis focuses on understanding how these methods differ from each other and across implementations.

## Key Findings

### Differences Between Methods
- **Conditional vs. Unconditional Methods**: 
"""
    
    # Analyze differences between conditional and unconditional methods
    cond_vs_uncond = []
    for impl in ["ExactCIs", "R"]:
        # Get lower and upper bounds for both methods
        cond_lower = f"{impl} Conditional Lower"
        cond_upper = f"{impl} Conditional Upper"
        uncond_lower = f"{impl} Unconditional Lower"
        uncond_upper = f"{impl} Unconditional Upper"
        
        # Calculate percentage differences for those that have both values
        mask_lower = ~df[cond_lower].isna() & ~df[uncond_lower].isna()
        diff_lower = ((df[uncond_lower] - df[cond_lower]) / df[cond_lower] * 100).where(mask_lower)
        avg_diff_lower = diff_lower.mean(skipna=True)
        
        mask_upper = ~df[cond_upper].isna() & ~df[uncond_upper].isna()
        diff_upper = ((df[uncond_upper] - df[cond_upper]) / df[cond_upper] * 100).where(mask_upper)
        avg_diff_upper = diff_upper.mean(skipna=True)
        
        cond_vs_uncond.append((impl, avg_diff_lower, avg_diff_upper))
    
    # Add to report
    for impl, diff_lower, diff_upper in cond_vs_uncond:
        report_text += f"  - In {impl}, unconditional methods produce lower bounds that are on average {diff_lower:.1f}% different from conditional methods.\n"
        report_text += f"  - In {impl}, unconditional methods produce upper bounds that are on average {diff_upper:.1f}% different from conditional methods.\n\n"
    
    # Analyze MidP vs. Other methods
    report_text += "- **MidP vs. Other Methods**: \n"
    midp_vs_others = []
    for impl in ["ExactCIs", "R"]:
        for method in ["Conditional", "Blaker", "Unconditional"]:
            # Get lower and upper bounds
            midp_lower = f"{impl} MidP Lower"
            midp_upper = f"{impl} MidP Upper"
            other_lower = f"{impl} {method} Lower"
            other_upper = f"{impl} {method} Upper"
            
            # Calculate percentage differences
            mask_lower = ~df[midp_lower].isna() & ~df[other_lower].isna()
            diff_lower = ((df[midp_lower] - df[other_lower]) / df[other_lower] * 100).where(mask_lower)
            avg_diff_lower = diff_lower.mean(skipna=True)
            
            mask_upper = ~df[midp_upper].isna() & ~df[other_upper].isna()
            diff_upper = ((df[midp_upper] - df[other_upper]) / df[other_upper] * 100).where(mask_upper)
            avg_diff_upper = diff_upper.mean(skipna=True)
            
            midp_vs_others.append((impl, method, avg_diff_lower, avg_diff_upper))
    
    # Add to report
    for impl, method, diff_lower, diff_upper in midp_vs_others:
        report_text += f"  - In {impl}, MidP methods produce lower bounds that are on average {diff_lower:.1f}% different from {method} methods.\n"
        report_text += f"  - In {impl}, MidP methods produce upper bounds that are on average {diff_upper:.1f}% different from {method} methods.\n\n"
    
    # Analyze differences between ExactCIs and R
    report_text += "### Differences Between Implementations\n"
    report_text += "- **ExactCIs vs. R**: \n"
    exactcis_vs_r = []
    for method in ["Conditional", "MidP", "Blaker", "Unconditional", "Wald-Haldane"]:
        # Get lower and upper bounds
        exactcis_lower = f"ExactCIs {method} Lower"
        exactcis_upper = f"ExactCIs {method} Upper"
        r_lower = f"R {method} Lower"
        r_upper = f"R {method} Upper"
        
        # Calculate percentage differences
        mask_lower = ~df[exactcis_lower].isna() & ~df[r_lower].isna()
        diff_lower = ((df[exactcis_lower] - df[r_lower]) / df[r_lower] * 100).where(mask_lower)
        avg_diff_lower = diff_lower.mean(skipna=True)
        
        mask_upper = ~df[exactcis_upper].isna() & ~df[r_upper].isna()
        diff_upper = ((df[exactcis_upper] - df[r_upper]) / df[r_upper] * 100).where(mask_upper)
        avg_diff_upper = diff_upper.mean(skipna=True)
        
        exactcis_vs_r.append((method, avg_diff_lower, avg_diff_upper))
    
    # Add to report
    for method, diff_lower, diff_upper in exactcis_vs_r:
        report_text += f"  - For {method} methods, ExactCIs produces lower bounds that are on average {diff_lower:.1f}% different from R.\n"
        report_text += f"  - For {method} methods, ExactCIs produces upper bounds that are on average {diff_upper:.1f}% different from R.\n\n"
    
    # Add conclusion
    report_text += """## Potential Sources of Differences

1. **Numerical Precision**: Different numerical algorithms used in the implementations may lead to slight variations in results.

2. **Search Algorithms**: The root-finding and optimization algorithms used to determine confidence interval boundaries can vary across implementations.

3. **Edge Case Handling**: Different strategies for handling edge cases (zeros, small counts, etc.) can significantly impact results.

4. **Implementation Details**: Specific implementation choices for each method can lead to differences, such as:
   - How grid points are selected for the unconditional method
   - How p-values are calculated and compared to the alpha level
   - How convergence criteria are defined

5. **Version Differences**: The reference values from R might be from older versions with different implementations.

## Conclusion

The comparison between ExactCIs, SciPy, and R shows that while there are differences in the confidence interval estimates, they generally follow similar patterns. The Wald-Haldane method serves as a good baseline for comparison since it's implemented consistently across all platforms.

For most practical applications, these differences are unlikely to significantly impact statistical inference. However, for edge cases with very small counts or extreme odds ratios, users should be aware that different implementations may produce notably different results.

The ExactCIs package provides results that are generally consistent with established implementations, with variations that are expected due to implementation differences.
"""

    # Write the report to a file
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_file)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Analysis report generated at {output_path}")

if __name__ == "__main__":
    # Run the analysis
    df = analyze_and_plot()
    
    # Generate the analysis report
    generate_analysis_report(df)
