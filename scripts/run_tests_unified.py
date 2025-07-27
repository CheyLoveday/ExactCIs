#!/usr/bin/env python3
"""
Unified Test Runner Script for ExactCIs using UV.

This script combines the functionality of run_tests.py, run_tests.sh, and 
run_tests_with_progress.py into a single, configurable test runner.

Usage:
    python run_tests_unified.py [options]

Examples:
    # Run fast tests only (default)
    python run_tests_unified.py
    
    # Run all tests including slow tests
    python run_tests_unified.py --all
    
    # Run with coverage
    python run_tests_unified.py --coverage
    
    # Run with progress bars
    python run_tests_unified.py --progress
    
    # Run specific module or test
    python run_tests_unified.py --module=methods/blaker
    python run_tests_unified.py --test=test_blaker_ci
"""

import argparse
import subprocess
import sys
import time
import os
import json
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import threading
import queue

# Try to import rich for progress visualization
HAVE_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
    from rich.panel import Panel
    from rich.text import Text
    HAVE_RICH = True
except ImportError:
    pass

# Try to import tqdm for alternative progress bars
HAVE_TQDM = False
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    pass


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===\n")
    start_time = time.time()
    process = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time
    print(f"\n=== Completed in {duration:.2f} seconds with exit code {process.returncode} ===\n")
    return process.returncode


def find_test_files(test_dir="tests"):
    """Find all test files in the project."""
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files


def analyze_tests(files):
    """Analyze test files to categorize them by speed and dependencies."""
    fast_tests = []
    slow_tests = []
    other_tests = []
    
    # Dictionary to track dependencies between tests
    dependencies = {}

    for file in files:
        with open(file, "r") as f:
            content = f.read()
            
        file_has_slow = "@pytest.mark.slow" in content
        file_has_fast = "@pytest.mark.fast" in content
        
        # Look for dependencies (tests that must run before others)
        if "# DEPENDS:" in content:
            for line in content.split("\n"):
                if "# DEPENDS:" in line:
                    dependency = line.split("# DEPENDS:")[1].strip()
                    dependencies[file] = dependency
        
        if file_has_slow:
            slow_tests.append(file)
        elif file_has_fast:
            fast_tests.append(file)
        else:
            other_tests.append(file)
            
    return fast_tests, slow_tests, other_tests, dependencies


def get_optimal_worker_count():
    """Determine optimal number of worker processes."""
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of available cores by default, but at least 2 and at most 8
    return max(2, min(8, int(cpu_count * 0.75)))


def run_test(test_file, base_cmd, progress=None, task_id=None):
    """Run a single test file and return the results."""
    start_time = time.time()
    cmd = f"{base_cmd} {test_file}"
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration = time.time() - start_time
    
    # Update progress if provided
    if progress and task_id is not None:
        if HAVE_RICH:
            progress.update(task_id, advance=1)
        elif HAVE_TQDM:
            progress.update(1)
    
    category = os.path.dirname(test_file) or "root"
    return {
        "file": test_file,
        "category": category,
        "duration": duration,
        "exit_code": process.returncode,
        "stdout": process.stdout.decode('utf-8', errors='replace'),
        "stderr": process.stderr.decode('utf-8', errors='replace'),
        "success": process.returncode == 0
    }


def run_tests_simple(args):
    """Run tests using the simple approach from run_tests.py."""
    # Base command using uv
    base_cmd = "uv run pytest"
    
    # Build pytest options
    pytest_options = []
    
    if args.verbose:
        pytest_options.append("-v")
    
    if args.all:
        pytest_options.append("--run-slow")
    
    if args.parallel:
        pytest_options.append("-n auto")
    
    if args.coverage:
        pytest_options.append("--cov=src/exactcis --cov-report=term --cov-report=html")
    
    # If no specific mode is set, default to fast tests only
    if not (args.all or args.coverage or args.module or args.test):
        pytest_options.append("-m fast")
    
    # Add specific module or test
    target = ""
    if args.module:
        if "/" in args.module:
            # Handle nested modules like methods/unconditional
            parts = args.module.split("/")
            target = f"tests/test_{parts[0]}/{parts[1]}.py"
        else:
            target = f"tests/test_{args.module}.py"
        
        # Verify file exists
        if not os.path.exists(target):
            print(f"Error: Test module '{target}' not found")
            return 1
            
    elif args.test:
        # Find the test function in test files
        test_files = []
        for root, _, files in os.walk("tests"):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            if f"def {args.test}" in content:
                                test_files.append(file_path)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        if not test_files:
            print(f"Error: Test function '{args.test}' not found in any test file")
            return 1
        
        if len(test_files) > 1:
            print(f"Warning: Test '{args.test}' found in multiple files: {test_files}")
            print(f"Using the first one: {test_files[0]}")
        
        target = f"{test_files[0]}::{args.test}"
    
    # Build and run the final command
    cmd = f"{base_cmd} {' '.join(pytest_options)} {target}".strip()
    
    print(f"Running tests with command: {cmd}")
    return run_command(cmd, "Running tests")


def run_tests_with_progress(test_files, args):
    """Run tests with rich progress bars."""
    if not HAVE_RICH and not HAVE_TQDM:
        print("Warning: Neither 'rich' nor 'tqdm' is installed. Running without progress bars.")
        return run_tests_simple(args)
    
    # Find and categorize test files
    fast_tests, slow_tests, other_tests, dependencies = analyze_tests(test_files)
    
    if args.verbose:
        print(f"Found {len(fast_tests)} fast tests, {len(slow_tests)} slow tests, and {len(other_tests)} uncategorized tests")
    
    # Determine which tests to run
    tests_to_run = []
    tests_to_run.extend(fast_tests)
    
    if args.all:
        tests_to_run.extend(slow_tests)
        tests_to_run.extend(other_tests)
    
    if not tests_to_run:
        print("No tests found to run!")
        return 1
    
    # Set up base command
    base_cmd = "uv run pytest -xvs"
    
    if args.coverage:
        base_cmd += " --cov=src/exactcis --cov-report=term"
    
    # Set up worker count
    max_workers = args.workers or get_optimal_worker_count()
    
    # Run tests with progress
    results = []
    
    if HAVE_RICH:
        console = Console()
        console.print(f"[bold green]Running {len(tests_to_run)} tests with {max_workers} workers[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task("Running tests...", total=len(tests_to_run))
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {executor.submit(run_test, test, base_cmd, progress, task_id): test for test in tests_to_run}
                
                for future in as_completed(future_to_test):
                    test = future_to_test[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        results.append({
                            "file": test,
                            "error": str(exc),
                            "success": False
                        })
    elif HAVE_TQDM:
        with tqdm(total=len(tests_to_run), desc="Running tests") as progress_bar:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {executor.submit(run_test, test, base_cmd): test for test in tests_to_run}
                
                for future in as_completed(future_to_test):
                    test = future_to_test[future]
                    try:
                        result = future.result()
                        results.append(result)
                        progress_bar.update(1)
                    except Exception as exc:
                        results.append({
                            "file": test,
                            "error": str(exc),
                            "success": False
                        })
                        progress_bar.update(1)
    
    # Display results
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\nTest Results: {successful} passed, {failed} failed")
    
    if failed > 0:
        print("\nFailed tests:")
        for result in results:
            if not result.get('success', False):
                print(f"  {result.get('file', 'Unknown')}")
                if 'stderr' in result and result['stderr']:
                    error_lines = result['stderr'].split('\n')
                    # Print at most 5 error lines
                    for line in error_lines[:5]:
                        if line.strip():
                            print(f"    {line}")
                    if len(error_lines) > 5:
                        print(f"    ... ({len(error_lines) - 5} more lines)")
    
    return 1 if failed > 0 else 0


def display_help():
    """Display help information about installing dependencies."""
    print("\nTo enable progress visualization, install optional dependencies:")
    print("  uv pip install rich")
    print("  uv pip install tqdm\n")


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run tests for ExactCIs with UV")
    parser.add_argument("--all", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", type=str, help="Run tests for a specific module")
    parser.add_argument("--test", type=str, help="Run a specific test")
    parser.add_argument("--progress", action="store_true", help="Use progress bars")
    parser.add_argument("--workers", type=int, help="Number of worker processes to use")
    args = parser.parse_args()
    
    # Find test files
    test_files = find_test_files()
    
    # Run tests
    if args.progress:
        if not (HAVE_RICH or HAVE_TQDM):
            print("Warning: Progress visualization requires 'rich' or 'tqdm' to be installed.")
            display_help()
            return run_tests_simple(args)
        return run_tests_with_progress(test_files, args)
    else:
        return run_tests_simple(args)


if __name__ == "__main__":
    sys.exit(main())
