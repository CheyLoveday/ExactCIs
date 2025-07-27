"""
Tests for the command-line interface.
"""

import pytest
import sys
from unittest.mock import patch
from io import StringIO

from exactcis.cli import main, parse_args


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
