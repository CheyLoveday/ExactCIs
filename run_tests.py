#!/usr/bin/env python
"""
Test runner script for ExactCIs using UV.

This script provides convenient ways to run the test suite with different options
through the uv run command.
"""

import argparse
import subprocess
import sys
import time
import os


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===\n")
    start_time = time.time()
    process = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time
    print(f"\n=== Completed in {duration:.2f} seconds with exit code {process.returncode} ===\n")
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for ExactCIs with UV")
    parser.add_argument("--all", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", type=str, help="Run tests for a specific module")
    parser.add_argument("--test", type=str, help="Run a specific test")
    args = parser.parse_args()

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
    if not (args.all or args.fast or args.coverage or args.module or args.test):
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
    return run_command(cmd, "Running tests")


if __name__ == "__main__":
    sys.exit(main())