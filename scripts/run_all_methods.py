"""
Run a single 2x2 table across all available confidence interval methods.

This script runs a 2x2 table (50/1000 vs 10/1000) through all available
confidence interval methods and reports the results in a tabular format.
Results are saved to a markdown file for easy viewing.
"""

import sys
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the compute_all_cis function
try:
    from exactcis import compute_all_cis
    from exactcis.core import calculate_odds_ratio
except ImportError as e:
    logger.error(f"Error importing exactcis: {e}")
    logger.error("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def run_all_methods(a, b, c, d, alpha=0.05, grid_size=200):
    """
    Run all confidence interval methods on a single 2x2 table.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        grid_size: Grid size for methods that use grid search (default: 200)
        
    Returns:
        Dictionary with method names as keys and confidence intervals as values
    """
    logger.info(f"Running all methods on table: a={a}, b={b}, c={c}, d={d}")
    
    # Calculate the odds ratio
    odds_ratio = calculate_odds_ratio(a, b, c, d)
    logger.info(f"Odds ratio: {odds_ratio:.4f}")
    
    # Run all methods
    results = compute_all_cis(a, b, c, d, alpha, grid_size=grid_size)
    
    return results, odds_ratio

def format_results(results, odds_ratio):
    """
    Format the results in a tabular format for console output.
    
    Args:
        results: Dictionary with method names as keys and confidence intervals as values
        odds_ratio: The odds ratio for the 2x2 table
        
    Returns:
        Formatted table as a string
    """
    # Prepare the table data
    table_data = []
    
    for method, ci in results.items():
        lower, upper = ci
        
        # Calculate the width of the confidence interval
        width = upper - lower if upper < float('inf') else float('inf')
        
        # Check if the confidence interval excludes 1 (indicating statistical significance)
        excludes_one = "Yes" if lower > 1 or upper < 1 else "No"
        
        # Format the confidence interval
        ci_formatted = f"({lower:.4f}, {upper:.4f})" if upper < float('inf') else f"({lower:.4f}, inf)"
        
        # Add a row to the table
        table_data.append([
            method.capitalize(),
            ci_formatted,
            f"{width:.4f}" if width < float('inf') else "inf",
            excludes_one,
            "Yes" if lower <= odds_ratio <= upper else "No"
        ])
    
    # Sort the table by method name
    table_data.sort(key=lambda x: x[0])
    
    # Create the table using string formatting
    headers = ["Method", "95% CI", "Width", "Excludes OR=1", "Includes True OR"]
    
    # Determine column widths
    col_widths = [max(len(str(row[i])) for row in table_data + [headers]) for i in range(len(headers))]
    
    # Create the table header
    header_row = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    separator = "-+-".join("-" * width for width in col_widths)
    
    # Create the table rows
    table_rows = []
    for row in table_data:
        table_rows.append(" | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))))
    
    # Combine all parts of the table
    table = f"{header_row}\n{separator}\n" + "\n".join(table_rows)
    
    return table

def format_results_markdown(results, odds_ratio):
    """
    Format the results in a markdown table format.
    
    Args:
        results: Dictionary with method names as keys and confidence intervals as values
        odds_ratio: The odds ratio for the 2x2 table
        
    Returns:
        Formatted table as a markdown string
    """
    # Prepare the table data
    table_data = []
    
    for method, ci in results.items():
        lower, upper = ci
        
        # Calculate the width of the confidence interval
        width = upper - lower if upper < float('inf') else float('inf')
        
        # Check if the confidence interval excludes 1 (indicating statistical significance)
        excludes_one = "Yes" if lower > 1 or upper < 1 else "No"
        
        # Format the confidence interval
        ci_formatted = f"({lower:.4f}, {upper:.4f})" if upper < float('inf') else f"({lower:.4f}, inf)"
        
        # Add a row to the table
        table_data.append([
            method.capitalize(),
            ci_formatted,
            f"{width:.4f}" if width < float('inf') else "inf",
            excludes_one,
            "Yes" if lower <= odds_ratio <= upper else "No"
        ])
    
    # Sort the table by method name
    table_data.sort(key=lambda x: x[0])
    
    # Create the markdown table
    headers = ["Method", "95% CI", "Width", "Excludes OR=1", "Includes True OR"]
    
    # Create the table header
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---" for _ in headers]) + " |"
    
    # Create the table rows
    table_rows = []
    for row in table_data:
        table_rows.append("| " + " | ".join(str(item) for item in row) + " |")
    
    # Combine all parts of the table
    table = f"{header_row}\n{separator}\n" + "\n".join(table_rows)
    
    return table

def write_results_to_markdown(a, b, c, d, results, odds_ratio, filename="ci_results.md"):
    """
    Write the results to a markdown file.
    
    Args:
        a, b, c, d: Counts in the 2x2 table
        results: Dictionary with method names as keys and confidence intervals as values
        odds_ratio: The odds ratio for the 2x2 table
        filename: Name of the markdown file to write to
        
    Returns:
        Path to the created markdown file
    """
    # Format the results in markdown
    table = format_results_markdown(results, odds_ratio)
    
    # Create the markdown content
    content = f"""# Confidence Interval Results for 2x2 Table

## Table Information

### 2x2 Table
|             | Cases | Non-cases |
|-------------|-------|-----------|
| Exposed     | {a}   | {b}       |
| Unexposed   | {c}   | {d}       |

### Odds Ratio
The odds ratio is **{odds_ratio:.4f}**.

## Results

{table}

## Interpretation

1. The odds ratio is {odds_ratio:.4f}, indicating that the odds of being a case is about {odds_ratio:.1f} times higher in the exposed group.
2. Methods that exclude OR=1 in their confidence interval indicate a statistically significant association.
3. Narrower confidence intervals indicate more precise estimates.

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # Write the content to the file
    with open(filename, "w") as f:
        f.write(content)
    
    logger.info(f"Results written to {filename}")
    
    return filename

def main():
    """Run the main script."""
    logger.info("Starting run_all_methods.py")
    
    # Define the 2x2 table (50/1000 vs 10/1000)
    a = 50    # Cases in exposed group
    b = 950   # Non-cases in exposed group
    c = 10    # Cases in unexposed group
    d = 990   # Non-cases in unexposed group
    
    # Run all methods
    results, odds_ratio = run_all_methods(a, b, c, d)
    
    # Format and print the results to console
    table = format_results(results, odds_ratio)
    print("\n2x2 Table:")
    print(f"             Cases    Non-cases")
    print(f"Exposed      {a}        {b}")
    print(f"Unexposed    {c}        {d}")
    print(f"\nOdds Ratio: {odds_ratio:.4f}")
    print("\nResults:")
    print(table)
    
    # Provide a brief interpretation
    print("\nInterpretation:")
    print("1. The odds ratio is 5.2105, indicating that the odds of being a case is about 5.2 times higher in the exposed group.")
    print("2. Methods that exclude OR=1 in their confidence interval indicate a statistically significant association.")
    print("3. Narrower confidence intervals indicate more precise estimates.")
    
    # Write results to markdown file
    output_file = "confidence_interval_results.md"
    md_file = write_results_to_markdown(a, b, c, d, results, odds_ratio, filename=output_file)
    print(f"\nResults have been saved to {md_file}")
    
    logger.info("run_all_methods.py completed successfully")

if __name__ == "__main__":
    main()