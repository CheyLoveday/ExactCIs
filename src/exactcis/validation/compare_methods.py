#!/usr/bin/env python3
"""
Validation script to compare different confidence interval methods
for odds ratio estimation in 2x2 contingency tables.

This script compares the unconditional, mid-p, and conditional (Fisher's) exact
confidence interval methods to validate their relative behavior.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our CI methods
from exactcis.methods.conditional import exact_ci_conditional
from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.wald import ci_wald_haldane

def format_ci(ci: Tuple[float, float]) -> str:
    """Format a confidence interval for nice printing"""
    if ci is None:
        return "(None, None)"
    if np.isinf(ci[1]):
        return f"({ci[0]:.3f}, ∞)"
    return f"({ci[0]:.3f}, {ci[1]:.3f})"

def run_ci_method(method_func, method_name, a, b, c, d, alpha):
    """Run a CI method with detailed error handling and debugging"""
    try:
        logger.info(f"Starting {method_name} CI calculation for [{a}, {b}; {c}, {d}]")
        result = method_func(a, b, c, d, alpha)
        
        if result is None:
            logger.error(f"{method_name} CI returned None")
            return (0.0, float('inf'))
        
        if not isinstance(result, tuple) or len(result) != 2:
            logger.error(f"{method_name} CI returned invalid format: {result}")
            return (0.0, float('inf'))
        
        logger.info(f"{method_name} CI: {format_ci(result)}")
        return result
    
    except Exception as e:
        logger.error(f"Error in {method_name} CI: {str(e)}")
        logger.debug(traceback.format_exc())
        return (0.0, float('inf'))

def compare_methods_for_table(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict:
    """
    Compare different CI methods for a single 2x2 table.
    
    Args:
        a, b, c, d: Cell counts for the 2x2 table
        alpha: Significance level
        
    Returns:
        Dictionary of method names and their confidence intervals
    """
    logger.info(f"Analyzing table: [{a}, {b}; {c}, {d}]")
    
    ci_conditional = run_ci_method(exact_ci_conditional, "Conditional", a, b, c, d, alpha)
    ci_midp = run_ci_method(exact_ci_midp, "Mid-P", a, b, c, d, alpha)
    ci_unconditional = run_ci_method(exact_ci_unconditional, "Unconditional", a, b, c, d, alpha)
    ci_blaker = run_ci_method(exact_ci_blaker, "Blaker", a, b, c, d, alpha)
    ci_wald = run_ci_method(ci_wald_haldane, "Wald", a, b, c, d, alpha)
        
    return {
        'Conditional': ci_conditional,
        'Mid-P': ci_midp,
        'Unconditional': ci_unconditional,
        'Blaker': ci_blaker,
        'Wald': ci_wald
    }

def validate_ci_relationships(results: Dict) -> bool:
    """
    Validate the known theoretical relationships between different CI methods.
    
    We expect:
    1. Conditional CIs to be the widest (most conservative)
    2. Mid-P CIs to be narrower than conditional
    3. Unconditional CIs to be wider than Mid-P but narrower than conditional
    4. Blaker CIs to typically be narrower than conditional
    
    Returns:
        True if all expected relationships hold
    """
    # First, check that all CIs are valid tuples
    for method, ci in results.items():
        if ci is None or not isinstance(ci, tuple) or len(ci) != 2:
            logger.warning(f"Invalid CI for {method}: {ci}")
            return False
    
    valid = True
    
    # Lower bounds: conditional ≤ unconditional ≤ mid-p
    if results['Conditional'][0] > results['Unconditional'][0]:
        logger.warning(f"Lower bound relationship violated: Conditional {results['Conditional'][0]:.4f} > Unconditional {results['Unconditional'][0]:.4f}")
        valid = False
        
    if results['Unconditional'][0] > results['Mid-P'][0]:
        logger.warning(f"Lower bound relationship violated: Unconditional {results['Unconditional'][0]:.4f} > Mid-P {results['Mid-P'][0]:.4f}")
        valid = False
    
    # Upper bounds: conditional ≥ unconditional ≥ mid-p
    if not np.isinf(results['Conditional'][1]) and not np.isinf(results['Unconditional'][1]):
        if results['Conditional'][1] < results['Unconditional'][1]:
            logger.warning(f"Upper bound relationship violated: Conditional {results['Conditional'][1]:.4f} < Unconditional {results['Unconditional'][1]:.4f}")
            valid = False
    
    if not np.isinf(results['Unconditional'][1]) and not np.isinf(results['Mid-P'][1]):
        if results['Unconditional'][1] < results['Mid-P'][1]:
            logger.warning(f"Upper bound relationship violated: Unconditional {results['Unconditional'][1]:.4f} < Mid-P {results['Mid-P'][1]:.4f}")
            valid = False
    
    return valid

def run_validation(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05) -> pd.DataFrame:
    """
    Run validation on a list of 2x2 tables.
    
    Args:
        tables: List of (a,b,c,d) tuples representing 2x2 tables
        alpha: Significance level
        
    Returns:
        DataFrame with validation results
    """
    results = []
    
    for i, (a, b, c, d) in enumerate(tables):
        logger.info(f"Table {i+1}/{len(tables)}")
        
        # Get CIs from all methods
        cis = compare_methods_for_table(a, b, c, d, alpha)
        
        # Validate relationships
        valid = validate_ci_relationships(cis)
        
        # Calculate CI widths
        widths = {}
        for method, ci in cis.items():
            if np.isinf(ci[1]):
                widths[f"{method}_width"] = float('inf')
            else:
                widths[f"{method}_width"] = ci[1] - ci[0]
        
        # Store results
        result = {
            'table_id': i+1,
            'a': a, 'b': b, 'c': c, 'd': d,
            'valid_relationships': valid
        }
        
        # Add CIs and widths
        for method, ci in cis.items():
            result[f"{method}_lower"] = ci[0]
            result[f"{method}_upper"] = ci[1]
        
        result.update(widths)
        results.append(result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define a set of test tables
    test_tables = [
        (5, 10, 15, 20),  # Balanced table
        (0, 10, 5, 20),   # Zero in cell a
        (10, 0, 5, 20),   # Zero in cell b
        (10, 5, 0, 20),   # Zero in cell c
        (10, 5, 15, 0),   # Zero in cell d
        (1, 1, 1, 1),     # Small counts
        (50, 50, 50, 50), # Large balanced counts
        (5, 50, 5, 50),   # Unbalanced rows
        (5, 5, 50, 50)    # Unbalanced columns
    ]
    
    # Run validation
    results_df = run_validation(test_tables)
    
    # Display summary
    print("\nValidation Summary:")
    print(f"Total tables tested: {len(results_df)}")
    print(f"Tables with valid relationships: {results_df['valid_relationships'].sum()}")
    print(f"Tables with invalid relationships: {len(results_df) - results_df['valid_relationships'].sum()}")
    
    # Display any tables with invalid relationships
    if len(results_df) - results_df['valid_relationships'].sum() > 0:
        print("\nTables with invalid relationships:")
        invalid_tables = results_df[~results_df['valid_relationships']]
        for _, row in invalid_tables.iterrows():
            print(f"Table {row['table_id']}: [{row['a']}, {row['b']}; {row['c']}, {row['d']}]")
