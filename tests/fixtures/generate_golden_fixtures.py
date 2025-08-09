#!/usr/bin/env python3
"""
Golden fixture generation script for ExactCIs refactoring.

This script generates comprehensive test fixtures by running all methods across
a curated grid of 2x2 tables, including edge cases with zeros and imbalanced tables.
These fixtures serve as the safety net during refactoring to ensure output parity.

Usage:
    python tests/fixtures/generate_golden_fixtures.py
    
Output:
    tests/fixtures/golden_outputs.json - Complete fixture data
    tests/fixtures/golden_metadata.json - Generation metadata and environment info
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any
import traceback

# Add src to path to import exactcis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from exactcis import compute_all_cis, compute_all_rr_cis, __version__


def generate_test_grid() -> List[Tuple[int, int, int, int]]:
    """
    Generate a comprehensive grid of 2x2 table configurations for testing.
    
    Includes:
    - Small balanced tables
    - Zero cells in all positions  
    - Imbalanced tables (rare/common outcomes and exposures)
    - Medium sized tables
    - Extreme ratios
    - Edge cases from literature and previous bug reports
    
    Returns:
        List of (a, b, c, d) tuples representing 2x2 tables
    """
    tables = []
    
    # Small balanced tables
    small_balanced = [
        (1, 1, 1, 1),    # Minimal non-zero
        (2, 2, 2, 2),    # Balanced small
        (3, 2, 2, 3),    # Slightly unbalanced
        (5, 5, 5, 5),    # Balanced medium
        (12, 5, 8, 10),  # Common test case from literature
    ]
    tables.extend(small_balanced)
    
    # Zero cells - systematic coverage
    zero_patterns = [
        (0, 5, 3, 7),    # a=0 (no exposed cases)
        (3, 0, 7, 5),    # b=0 (no exposed controls)
        (7, 3, 0, 5),    # c=0 (no unexposed cases)
        (5, 7, 3, 0),    # d=0 (no unexposed controls)
        (0, 0, 5, 10),   # Row 1 all zeros
        (5, 10, 0, 0),   # Row 2 all zeros
        (0, 5, 0, 10),   # Column 1 all zeros
        (5, 0, 10, 0),   # Column 2 all zeros
    ]
    tables.extend(zero_patterns)
    
    # Imbalanced - rare outcomes
    rare_outcome = [
        (1, 49, 2, 48),  # Rare outcome, similar exposure
        (2, 98, 1, 99),  # Very rare outcome
        (3, 47, 8, 42),  # Moderate imbalance
    ]
    tables.extend(rare_outcome)
    
    # Imbalanced - rare exposures
    rare_exposure = [
        (2, 1, 48, 49),  # Rare exposure, similar outcome rate
        (1, 2, 49, 48),  # Different direction
        (3, 2, 47, 48),  # Moderate rare exposure
    ]
    tables.extend(rare_exposure)
    
    # Medium sized tables
    medium_tables = [
        (20, 30, 15, 35), # Moderate effect
        (40, 10, 20, 30), # Stronger effect
        (25, 25, 25, 25), # Perfect balance, medium size
    ]
    tables.extend(medium_tables)
    
    # Extreme ratios
    extreme_tables = [
        (20, 1, 1, 20),   # Very high OR/RR
        (1, 20, 20, 1),   # Very low OR/RR
        (50, 1, 5, 44),   # High OR, different marginals
        (5, 45, 45, 5),   # Low OR, different marginals
    ]
    tables.extend(extreme_tables)
    
    # Literature examples and edge cases
    literature_cases = [
        (4, 16, 1, 19),   # Agresti example
        (6, 14, 8, 12),   # Balanced moderate
        (15, 5, 10, 10),  # Unequal marginals
        (9, 1, 3, 7),     # Small with strong effect
        (100, 200, 150, 250), # Large table
    ]
    tables.extend(literature_cases)
    
    return tables


def safe_compute_cis(table: Tuple[int, int, int, int], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Safely compute all CIs for a table, handling any exceptions.
    
    Args:
        table: (a, b, c, d) tuple
        alpha: Significance level
        
    Returns:
        Dict with 'or_results', 'rr_results', and 'errors' keys
    """
    a, b, c, d = table
    result = {
        'or_results': {},
        'rr_results': {},
        'errors': {}
    }
    
    # Compute OR methods
    try:
        result['or_results'] = compute_all_cis(a, b, c, d, alpha)
    except Exception as e:
        result['errors']['or_methods'] = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Compute RR methods
    try:
        result['rr_results'] = compute_all_rr_cis(a, b, c, d, alpha)
    except Exception as e:
        result['errors']['rr_methods'] = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
    
    return result


def generate_golden_fixtures():
    """Generate comprehensive golden fixtures for all methods and tables."""
    print("🔄 Generating golden fixtures for ExactCIs refactoring...")
    
    # Create fixtures directory if it doesn't exist
    fixtures_dir = os.path.dirname(__file__)
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Generate test grid
    test_tables = generate_test_grid()
    print(f"📊 Testing {len(test_tables)} table configurations")
    
    # Alpha levels to test
    alpha_levels = [0.05, 0.01, 0.10]
    
    fixtures = {}
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'exactcis_version': __version__,  # capture the package version dynamically
        'python_version': sys.version,
        'total_tables': len(test_tables),
        'alpha_levels': alpha_levels,
        'test_grid_description': 'Comprehensive grid including balanced, zero-cell, imbalanced, and extreme ratio tables',
    }
    
    # Generate fixtures for each table and alpha
    for i, table in enumerate(test_tables, 1):
        table_key = f"table_{table[0]}_{table[1]}_{table[2]}_{table[3]}"
        fixtures[table_key] = {
            'table': table,
            'alpha_results': {}
        }
        
        print(f"⏳ Processing table {i}/{len(test_tables)}: {table}")
        
        for alpha in alpha_levels:
            alpha_key = f"alpha_{alpha}"
            fixtures[table_key]['alpha_results'][alpha_key] = safe_compute_cis(table, alpha)
    
    # Write fixtures to JSON
    fixtures_file = os.path.join(fixtures_dir, 'golden_outputs.json')
    metadata_file = os.path.join(fixtures_dir, 'golden_metadata.json')
    
    with open(fixtures_file, 'w') as f:
        json.dump(fixtures, f, indent=2, default=str)  # default=str handles any numpy types
        
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Golden fixtures generated:")
    print(f"   📄 Fixtures: {fixtures_file}")
    print(f"   📄 Metadata: {metadata_file}")
    print(f"   📈 {len(test_tables)} tables × {len(alpha_levels)} alphas = {len(test_tables) * len(alpha_levels)} test cases")
    
    # Summary statistics
    total_or_methods = 0
    total_rr_methods = 0
    total_errors = 0
    
    for table_data in fixtures.values():
        for alpha_data in table_data['alpha_results'].values():
            total_or_methods += len(alpha_data['or_results'])
            total_rr_methods += len(alpha_data['rr_results'])
            total_errors += len(alpha_data['errors'])
    
    print(f"   🎯 {total_or_methods} OR method results")
    print(f"   🎯 {total_rr_methods} RR method results")
    if total_errors > 0:
        print(f"   ⚠️  {total_errors} errors captured (expected for edge cases)")
    
    return fixtures_file, metadata_file


if __name__ == "__main__":
    generate_golden_fixtures()