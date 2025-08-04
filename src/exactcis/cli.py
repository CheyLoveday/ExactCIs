"""
Command-line interface for ExactCIs package.

This module provides a command-line interface to the confidence interval methods
implemented in the ExactCIs package, allowing users to compute confidence intervals
without writing Python code.
"""

import argparse
import sys
import csv
import json
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)
from exactcis.core import validate_counts, apply_haldane_correction
from exactcis.utils.parallel import parallel_compute_ci, get_optimal_workers

# Configure logging
logger = logging.getLogger(__name__)

# Map method names to functions
METHOD_MAP = {
    'blaker': exact_ci_blaker,
    'conditional': exact_ci_conditional,
    'midp': exact_ci_midp,
    'unconditional': exact_ci_unconditional,
    'wald': ci_wald_haldane
}


def validate_counts_cli(a: int, b: int, c: int, d: int) -> None:
    """
    CLI-specific validation for counts that provides user-friendly error messages.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)

    Raises:
        ValueError: With user-friendly message if validation fails
    """
    try:
        validate_counts(a, b, c, d)
    except ValueError as e:
        # Convert the error to a more user-friendly message
        if "negative" in str(e):
            raise ValueError("Error: All cell counts must be non-negative.")
        elif "margin" in str(e):
            raise ValueError("Error: Cannot compute CI with empty margins (row or column totals = 0).")
        else:
            raise ValueError(f"Error: {str(e)}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate exact confidence intervals for odds ratios from 2x2 contingency tables.",
        epilog="Example: exactcis-cli 10 20 15 30 --method blaker --alpha 0.05\n" +
               "Batch mode: exactcis-cli --batch input.csv --output results.csv --method blaker",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Single table arguments (mutually exclusive with batch mode)
    single_group = parser.add_argument_group('single table mode')
    single_group.add_argument("a", type=int, nargs='?', help="Cell a count (exposed cases)")
    single_group.add_argument("b", type=int, nargs='?', help="Cell b count (exposed controls)")
    single_group.add_argument("c", type=int, nargs='?', help="Cell c count (unexposed cases)")
    single_group.add_argument("d", type=int, nargs='?', help="Cell d count (unexposed controls)")
    
    # Batch processing arguments
    batch_group = parser.add_argument_group('batch processing mode')
    batch_group.add_argument(
        "--batch", 
        type=str, 
        help="Input CSV file with columns 'a', 'b', 'c', 'd', and optional 'id' column"
    )
    batch_group.add_argument(
        "--output", 
        type=str, 
        help="Output CSV file for batch results (default: stdout)"
    )
    batch_group.add_argument(
        "--workers", 
        type=int, 
        help="Number of parallel workers for batch processing (default: auto-detected)"
    )
    batch_group.add_argument(
        "--timeout", 
        type=float, 
        help="Timeout in seconds for batch processing operations (default: no timeout)"
    )
    
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.05, 
        help="Significance level (default: 0.05)"
    )
    
    parser.add_argument(
        "--method", 
        type=str, 
        default="blaker", 
        choices=METHOD_MAP.keys(),
        help="CI method to use (default: blaker)"
    )
    
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=20,
        help="Grid size for unconditional method (default: 20, ignored for other methods)"
    )
    
    parser.add_argument(
        "--apply-haldane", 
        action="store_true",
        help="Apply Haldane's correction (add 0.5 to each cell)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print additional information"
    )
    
    parser.add_argument(
        "--format", 
        type=str, 
        choices=['table', 'csv', 'json'],
        default='table',
        help="Output format for batch processing (default: table)"
    )

    parsed = parser.parse_args(args)
    
    # Validate argument combinations
    if parsed.batch:
        if any(x is not None for x in [parsed.a, parsed.b, parsed.c, parsed.d]):
            parser.error("Cannot specify both batch file and individual cell counts")
    else:
        if any(x is None for x in [parsed.a, parsed.b, parsed.c, parsed.d]):
            parser.error("Must specify all four cell counts (a, b, c, d) or use --batch mode")
    
    return parsed


def load_batch_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load batch data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries with table data
    """
    tables = []
    required_columns = {'a', 'b', 'c', 'd'}
    
    try:
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if not required_columns.issubset(set(reader.fieldnames)):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"CSV file missing required columns: {missing}")
            
            for row_num, row in enumerate(reader, 1):
                try:
                    table = {
                        'id': row.get('id', f'table_{row_num}'),
                        'a': int(row['a']),
                        'b': int(row['b']),
                        'c': int(row['c']),
                        'd': int(row['d'])
                    }
                    tables.append(table)
                except ValueError as e:
                    logger.warning(f"Skipping row {row_num}: invalid data - {e}")
                    continue
                    
    except FileNotFoundError:
        raise ValueError(f"Input file '{filepath}' not found")
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")
        
    if not tables:
        raise ValueError("No valid tables found in input file")
        
    logger.info(f"Loaded {len(tables)} tables from {filepath}")
    return tables


def save_batch_results(results: List[Dict[str, Any]], filepath: Optional[str] = None, 
                      format_type: str = 'csv') -> None:
    """
    Save batch results to file or stdout.
    
    Args:
        results: List of result dictionaries
        filepath: Output file path (None for stdout)
        format_type: Output format ('csv', 'json', 'table')
    """
    if format_type == 'json':
        output = json.dumps(results, indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(output)
        else:
            print(output)
            
    elif format_type == 'csv':
        if not results:
            return
            
        fieldnames = list(results[0].keys())
        
        if filepath:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
    else:  # table format
        for result in results:
            print(f"\nTable {result['id']}:")
            print(f"  Input: a={result['a']}, b={result['b']}, c={result['c']}, d={result['d']}")
            print(f"  Method: {result['method']}")
            print(f"  Odds Ratio: {result['odds_ratio']:.4f}")
            print(f"  CI: ({result['lower']:.4f}, {result['upper']:.4f})")
            if 'error' in result:
                print(f"  Error: {result['error']}")


def process_single_table(a: int, b: int, c: int, d: int, method: str, alpha: float, 
                        grid_size: int, apply_haldane: bool, verbose: bool) -> None:
    """
    Process a single table (original CLI functionality).
    """
    try:
        # Apply Haldane's correction if requested
        if apply_haldane:
            original_a, original_b, original_c, original_d = a, b, c, d
            a, b, c, d = apply_haldane_correction(a, b, c, d)
            if verbose:
                print(f"Haldane's correction applied (or attempted):")
                print(f"  Original: a={original_a}, b={original_b}, c={original_c}, d={original_d}")
                print(f"  Resulting values for calculation: a={a}, b={b}, c={c}, d={d}")
    
        # Validate counts
        validate_counts_cli(a, b, c, d)
        
        # Get the appropriate CI function
        ci_function = METHOD_MAP[method]
        
        # Calculate the odds ratio
        odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
        
        # Call the CI function with appropriate arguments
        if method == "unconditional":
            lower, upper = ci_function(a, b, c, d, alpha, grid_size=grid_size)
        else:
            lower, upper = ci_function(a, b, c, d, alpha)
        
        # Print results
        print("\nExactCIs Result:")
        print(f"Method: {method.capitalize()}")
        print(f"Input: a={a}, b={b}, c={c}, d={d}")
        if apply_haldane:
            print(f"Haldane's correction was requested.")
            print(f"  Values used for calculation: a={a}, b={b}, c={c}, d={d}")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"{(1-alpha)*100:.1f}% Confidence Interval: ({lower:.4f}, {upper:.4f})")
        
        if verbose:
            print(f"\nInterval width: {upper - lower:.4f}")
            print(f"Row totals: {a+b}, {c+d}")
            print(f"Column totals: {a+c}, {b+d}")
            print(f"Total observations: {a+b+c+d}")
            
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ZeroDivisionError:
        print("Error: Division by zero occurred. Check that your data is appropriate.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def process_batch_tables(tables: List[Dict[str, Any]], method: str, alpha: float, 
                        grid_size: int, apply_haldane: bool, workers: Optional[int] = None,
                        timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Process multiple tables in parallel.
    
    Args:
        tables: List of table dictionaries
        method: CI method to use
        alpha: Significance level
        grid_size: Grid size for unconditional method
        apply_haldane: Whether to apply Haldane's correction
        workers: Number of parallel workers
        timeout: Timeout in seconds for processing operations
        
    Returns:
        List of result dictionaries
    """
    def process_table(table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single table and return results."""
        table_id = table_data['id']
        a, b, c, d = table_data['a'], table_data['b'], table_data['c'], table_data['d']
        
        result = {
            'id': table_id,
            'a': a, 'b': b, 'c': c, 'd': d,
            'method': method,
            'alpha': alpha
        }
        
        try:
            # Apply Haldane's correction if requested
            if apply_haldane:
                a, b, c, d = apply_haldane_correction(a, b, c, d)
                result.update({'a_corrected': a, 'b_corrected': b, 'c_corrected': c, 'd_corrected': d})
        
            # Validate counts
            validate_counts(a, b, c, d)
            
            # Get the appropriate CI function
            ci_function = METHOD_MAP[method]
            
            # Calculate the odds ratio
            odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
            
            # Call the CI function with appropriate arguments
            if method == "unconditional":
                lower, upper = ci_function(a, b, c, d, alpha, grid_size=grid_size)
            else:
                lower, upper = ci_function(a, b, c, d, alpha)
            
            result.update({
                'odds_ratio': odds_ratio,
                'lower': lower,
                'upper': upper,
                'width': upper - lower
            })
            
        except Exception as e:
            result['error'] = str(e)
            result.update({
                'odds_ratio': None,
                'lower': None,
                'upper': None,
                'width': None
            })
            logger.warning(f"Error processing table {table_id}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Full traceback for table {table_id}:\n{traceback.format_exc()}")
            
        return result
    
    # Determine number of workers
    if workers is None:
        workers = get_optimal_workers()
    
    logger.info(f"Processing {len(tables)} tables with {workers} workers")
    
    # Use parallel processing
    from exactcis.utils.parallel import parallel_map
    results = parallel_map(
        process_table,
        tables,
        max_workers=workers,
        force_processes=True,
        timeout=timeout
    )
    
    return results


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    """
    parsed_args = parse_args(args)
    
    # Configure logging if verbose
    if parsed_args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    alpha = parsed_args.alpha
    method = parsed_args.method
    
    if parsed_args.batch:
        # Batch processing mode
        try:
            # Load tables from CSV
            tables = load_batch_data(parsed_args.batch)
            
            if parsed_args.verbose:
                print(f"Processing {len(tables)} tables using {method} method...")
            
            # Process tables in parallel
            results = process_batch_tables(
                tables=tables,
                method=method,
                alpha=alpha,
                grid_size=parsed_args.grid_size,
                apply_haldane=parsed_args.apply_haldane,
                workers=parsed_args.workers,
                timeout=parsed_args.timeout
            )
            
            # Save results
            save_batch_results(results, parsed_args.output, parsed_args.format)
            
            if parsed_args.verbose:
                successful = sum(1 for r in results if 'error' not in r)
                print(f"\nProcessed {len(results)} tables successfully: {successful}, errors: {len(results) - successful}")
                
        except Exception as e:
            print(f"Batch processing error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Single table processing mode (original functionality)
        process_single_table(
            a=parsed_args.a, b=parsed_args.b, c=parsed_args.c, d=parsed_args.d,
            method=method, alpha=alpha, grid_size=parsed_args.grid_size,
            apply_haldane=parsed_args.apply_haldane, verbose=parsed_args.verbose
        )


if __name__ == "__main__":
    main()
