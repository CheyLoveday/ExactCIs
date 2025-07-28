"""
Tests for the command-line interface.
"""

import pytest
import sys
import tempfile
import csv
import os
from unittest.mock import patch
from io import StringIO

from exactcis.cli import main, parse_args, load_batch_data, save_batch_results


def test_parse_args_default():
    """Test argument parsing with default values."""
    args = parse_args(["10", "20", "5", "25"])
    assert args.a == 10
    assert args.b == 20
    assert args.c == 5
    assert args.d == 25
    assert args.alpha == 0.05
    assert args.method == "blaker"
    assert args.grid_size == 20
    assert not args.apply_haldane
    assert not args.verbose


def test_parse_args_custom():
    """Test argument parsing with custom values."""
    args = parse_args([
        "10", "20", "5", "25",
        "--alpha", "0.01",
        "--method", "unconditional",
        "--grid-size", "30",
        "--apply-haldane",
        "--verbose"
    ])
    assert args.a == 10
    assert args.b == 20
    assert args.c == 5
    assert args.d == 25
    assert args.alpha == 0.01
    assert args.method == "unconditional"
    assert args.grid_size == 30
    assert args.apply_haldane
    assert args.verbose


@patch('sys.stdout', new_callable=StringIO)
def test_main_blaker(mock_stdout):
    """Test the main function with Blaker's method."""
    main(["10", "20", "5", "25", "--method", "blaker"])
    output = mock_stdout.getvalue()
    assert "Method: Blaker" in output
    assert "Odds Ratio:" in output
    assert "Confidence Interval:" in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_conditional(mock_stdout):
    """Test the main function with conditional method."""
    main(["10", "20", "5", "25", "--method", "conditional"])
    output = mock_stdout.getvalue()
    assert "Method: Conditional" in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_unconditional(mock_stdout):
    """Test the main function with unconditional method."""
    main(["10", "20", "5", "25", "--method", "unconditional", "--grid-size", "10"])
    output = mock_stdout.getvalue()
    assert "Method: Unconditional" in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_midp(mock_stdout):
    """Test the main function with mid-p method."""
    main(["10", "20", "5", "25", "--method", "midp"])
    output = mock_stdout.getvalue()
    assert "Method: Midp" in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_wald(mock_stdout):
    """Test the main function with Wald method."""
    main(["10", "20", "5", "25", "--method", "wald"])
    output = mock_stdout.getvalue()
    assert "Method: Wald" in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_with_haldane(mock_stdout):
    """Test the main function with Haldane's correction."""
    main(["1", "20", "5", "25", "--method", "wald", "--apply-haldane", "--verbose"])
    output = mock_stdout.getvalue()
    assert "Haldane's correction was requested" in output
    assert "Original: a=1" in output
    assert "Values used for calculation: a=1.0" in output


@patch('sys.stderr', new_callable=StringIO)
@patch('sys.stdout', new_callable=StringIO)
def test_main_invalid_input(mock_stdout, mock_stderr):
    """Test the main function with invalid input."""
    with pytest.raises(SystemExit):
        main(["0", "0", "0", "0"])
    assert "Error" in mock_stderr.getvalue()


@patch('sys.stdout', new_callable=StringIO)
def test_main_verbose(mock_stdout):
    """Test the main function with verbose output."""
    main(["10", "20", "5", "25", "--verbose"])
    output = mock_stdout.getvalue()
    assert "Interval width:" in output
    assert "Row totals:" in output
    assert "Column totals:" in output
    assert "Total observations:" in output


# ============================================================================
# BATCH PROCESSING CLI TESTS
# ============================================================================

def test_parse_args_batch_mode():
    """Test argument parsing for batch mode."""
    args = parse_args([
        "--batch", "input.csv",
        "--output", "output.csv", 
        "--method", "blaker",
        "--workers", "4",
        "--format", "csv"
    ])
    
    assert args.batch == "input.csv"
    assert args.output == "output.csv"
    assert args.method == "blaker"
    assert args.workers == 4
    assert args.format == "csv"
    assert args.a is None  # Should be None in batch mode


def test_parse_args_batch_validation():
    """Test argument validation for batch mode."""
    # Should fail when both batch and individual args provided
    with pytest.raises(SystemExit):
        parse_args([
            "1", "2", "3", "4",  # Individual args
            "--batch", "input.csv"  # Batch arg
        ])
    
    # Should fail when individual args are incomplete
    with pytest.raises(SystemExit):
        parse_args(["1", "2", "3"])  # Missing d


def test_load_batch_data():
    """Test loading batch data from CSV file."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
        writer = csv.writer(f)
        writer.writerow(['id', 'a', 'b', 'c', 'd'])
        writer.writerow(['table1', '1', '2', '3', '4'])
        writer.writerow(['table2', '2', '3', '4', '5'])
        writer.writerow(['table3', '3', '4', '5', '6'])
    
    try:
        tables = load_batch_data(csv_file)
        
        assert len(tables) == 3
        assert tables[0] == {'id': 'table1', 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        assert tables[1] == {'id': 'table2', 'a': 2, 'b': 3, 'c': 4, 'd': 5}
        assert tables[2] == {'id': 'table3', 'a': 3, 'b': 4, 'c': 5, 'd': 6}
        
    finally:
        os.unlink(csv_file)


def test_load_batch_data_no_id():
    """Test loading batch data without ID column."""
    # Create temporary CSV file without ID column
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'c', 'd'])
        writer.writerow(['1', '2', '3', '4'])
        writer.writerow(['2', '3', '4', '5'])
    
    try:
        tables = load_batch_data(csv_file)
        
        assert len(tables) == 2
        assert tables[0]['id'] == 'table_1'  # Auto-generated ID
        assert tables[1]['id'] == 'table_2'
        
    finally:
        os.unlink(csv_file)


def test_load_batch_data_missing_columns():
    """Test loading batch data with missing required columns."""
    # Create temporary CSV file missing required columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'c'])  # Missing 'd' column
        writer.writerow(['1', '2', '3'])
    
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_batch_data(csv_file)
            
    finally:
        os.unlink(csv_file)


def test_load_batch_data_invalid_data():
    """Test loading batch data with invalid data."""
    # Create temporary CSV file with invalid data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'c', 'd'])
        writer.writerow(['1', '2', '3', '4'])     # Valid row
        writer.writerow(['x', '2', '3', '4'])     # Invalid row (non-integer)
        writer.writerow(['2', '3', '4', '5'])     # Valid row
    
    try:
        tables = load_batch_data(csv_file)
        
        # Should skip invalid row but load valid ones
        assert len(tables) == 2
        assert tables[0]['a'] == 1
        assert tables[1]['a'] == 2
        
    finally:
        os.unlink(csv_file)


def test_save_batch_results_csv():
    """Test saving batch results to CSV format."""
    results = [
        {'id': 'table1', 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'method': 'blaker', 
         'odds_ratio': 0.667, 'lower': 0.1, 'upper': 2.5},
        {'id': 'table2', 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'method': 'blaker',
         'odds_ratio': 0.833, 'lower': 0.2, 'upper': 3.0}
    ]
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
    
    try:
        save_batch_results(results, csv_file, 'csv')
        
        # Read back and verify
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert rows[0]['id'] == 'table1'
        assert rows[1]['id'] == 'table2'
        
    finally:
        os.unlink(csv_file)


def test_save_batch_results_json():
    """Test saving batch results to JSON format."""
    results = [
        {'id': 'table1', 'lower': 0.1, 'upper': 2.5}
    ]
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        save_batch_results(results, json_file, 'json')
        
        # Read back and verify
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        assert len(data) == 1
        assert data[0]['id'] == 'table1'
        
    finally:
        os.unlink(json_file)


@patch('sys.stdout', new_callable=StringIO)
def test_save_batch_results_table_format(mock_stdout):
    """Test saving batch results to table format (stdout)."""
    results = [
        {'id': 'table1', 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'method': 'blaker',
         'odds_ratio': 0.667, 'lower': 0.1, 'upper': 2.5}
    ]
    
    save_batch_results(results, None, 'table')
    output = mock_stdout.getvalue()
    
    assert 'Table table1:' in output
    assert 'Input: a=1, b=2, c=3, d=4' in output
    assert 'Method: blaker' in output
    assert 'CI: (0.1000, 2.5000)' in output


@patch('sys.stdout', new_callable=StringIO)
def test_main_batch_processing(mock_stdout):
    """Test main function with batch processing."""
    # Create temporary input CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        input_csv = f.name
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'c', 'd'])
        writer.writerow(['1', '2', '3', '4'])
        writer.writerow(['2', '3', '4', '5'])
    
    try:
        # Test batch processing to stdout
        main([
            "--batch", input_csv,
            "--method", "blaker",
            "--format", "table",
            "--workers", "1"
        ])
        
        output = mock_stdout.getvalue()
        assert 'Table table_1:' in output
        assert 'Table table_2:' in output
        
    finally:
        os.unlink(input_csv)


def test_main_batch_processing_to_file():
    """Test main function with batch processing to output file."""
    # Create temporary input CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        input_csv = f.name
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'c', 'd'])
        writer.writerow(['1', '2', '3', '4'])
    
    # Create temporary output CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_csv = f.name
    
    try:
        # Test batch processing to file
        main([
            "--batch", input_csv,
            "--output", output_csv,
            "--method", "blaker",
            "--format", "csv",
            "--workers", "1"
        ])
        
        # Verify output file was created and has content
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        assert 'lower' in rows[0]
        assert 'upper' in rows[0]
        
    finally:
        os.unlink(input_csv)
        os.unlink(output_csv)
