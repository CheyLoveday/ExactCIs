"""
Command-line interface for ExactCIs package.

This module provides a command-line interface to the confidence interval methods
implemented in the ExactCIs package, allowing users to compute confidence intervals
without writing Python code.
"""

import argparse
import sys
from typing import List, Optional

from exactcis.methods import (
    exact_ci_blaker,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional,
    ci_wald_haldane
)
from exactcis.core import validate_counts, apply_haldane_correction

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
        epilog="Example: exactcis-cli 10 20 15 30 --method blaker --alpha 0.05"
    )

    parser.add_argument("a", type=int, help="Cell a count (exposed cases)")
    parser.add_argument("b", type=int, help="Cell b count (exposed controls)")
    parser.add_argument("c", type=int, help="Cell c count (unexposed cases)")
    parser.add_argument("d", type=int, help="Cell d count (unexposed controls)")
    
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

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    """
    parsed_args = parse_args(args)
    
    # Extract values
    a, b, c, d = parsed_args.a, parsed_args.b, parsed_args.c, parsed_args.d
    alpha = parsed_args.alpha
    method = parsed_args.method
    
    try:
        # Apply Haldane's correction if requested
        if parsed_args.apply_haldane:
            original_a, original_b, original_c, original_d = a, b, c, d
            a, b, c, d = apply_haldane_correction(a, b, c, d)
            if parsed_args.verbose:
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
            lower, upper = ci_function(a, b, c, d, alpha, grid_size=parsed_args.grid_size)
        else:
            lower, upper = ci_function(a, b, c, d, alpha)
        
        # Print results
        print("\nExactCIs Result:")
        print(f"Method: {method.capitalize()}")
        print(f"Input: a={parsed_args.a}, b={parsed_args.b}, c={parsed_args.c}, d={parsed_args.d}")
        if parsed_args.apply_haldane:
            # Use the state of a,b,c,d *after* attempting correction for this message
            print(f"Haldane's correction was requested.")
            print(f"  Values used for calculation: a={a}, b={b}, c={c}, d={d}")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"{(1-alpha)*100:.1f}% Confidence Interval: ({lower:.4f}, {upper:.4f})")
        
        if parsed_args.verbose:
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


if __name__ == "__main__":
    main()
