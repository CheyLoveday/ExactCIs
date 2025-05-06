#!/usr/bin/env python
"""
Enhanced test runner script for ExactCIs with progress visualization.

This script provides an improved test running experience with:
1. Progress bars for long-running tests
2. Parallel test execution with dynamic progress updates
3. Prioritized test execution (fast tests first)
4. Adaptive parallelization for optimal resource utilization
5. Summary statistics and timing information
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

from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
from rich.panel import Panel
from rich.text import Text
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


console = Console()


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
        progress.update(task_id, advance=1)
    
    category = os.path.dirname(test_file) or "root"
    return {
        "file": test_file,
        "category": category,
        "status": "pass" if process.returncode == 0 else "fail",
        "returncode": process.returncode,
        "duration": duration,
        "output": process.stdout.decode('utf-8', errors='replace'),
        "error": process.stderr.decode('utf-8', errors='replace')
    }


def run_tests_with_progress(test_files, parallel=False, include_slow=False, verbose=False, max_workers=None):
    """Run tests with rich progress bars."""
    results = defaultdict(lambda: {"pass": 0, "fail": 0, "duration": 0})
    all_results = []
    
    # Analyze test files
    fast_tests, slow_tests, other_tests, dependencies = analyze_tests(test_files)
    
    # Determine optimal number of workers if not specified
    if parallel and max_workers is None:
        max_workers = get_optimal_worker_count()
    else:
        max_workers = 1  # Sequential mode
    
    console.print(f"[blue]Using {'parallel' if parallel else 'sequential'} mode with {max_workers} workers[/blue]")
    
    # Build the list of tests to run based on options
    tests_to_run = fast_tests + other_tests
    if include_slow:
        tests_to_run += slow_tests
    
    # Prepare pytest command base
    base_cmd = "uv run pytest"
    if parallel:
        base_cmd += " -n auto"  # Let pytest handle further parallelization if needed
    if verbose:
        base_cmd += " -v"
    
    # Add JSON output format for parsing results
    base_cmd += " --json-report --json-report-file=.test_results.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        expand=True
    ) as progress:
        # Create tasks for different test categories
        fast_task = progress.add_task(
            "[green]Running fast tests...", 
            total=len(fast_tests), 
            visible=bool(fast_tests)
        )
        
        other_task = progress.add_task(
            "[yellow]Running other tests...", 
            total=len(other_tests), 
            visible=bool(other_tests)
        )
        
        slow_task = progress.add_task(
            "[red]Running slow tests...", 
            total=len(slow_tests),
            visible=include_slow and bool(slow_tests)
        )
        
        # Process tests in optimal order: fast -> other -> slow
        # Each category can be processed in parallel
        
        # Function to process test results
        def process_result(result):
            category = result["category"]
            if result["status"] == "pass":
                results[category]["pass"] += 1
            else:
                results[category]["fail"] += 1
            results[category]["duration"] += result["duration"]
            all_results.append(result)
        
        # Run fast tests with parallel execution if enabled
        if fast_tests:
            if parallel and len(fast_tests) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_test = {
                        executor.submit(run_test, test, base_cmd, progress, fast_task): test 
                        for test in fast_tests
                    }
                    for future in as_completed(future_to_test):
                        process_result(future.result())
            else:
                # Sequential execution
                for test in fast_tests:
                    process_result(run_test(test, base_cmd, progress, fast_task))
        
        # Run other tests with parallel execution if enabled
        if other_tests:
            if parallel and len(other_tests) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_test = {
                        executor.submit(run_test, test, base_cmd, progress, other_task): test 
                        for test in other_tests
                    }
                    for future in as_completed(future_to_test):
                        process_result(future.result())
            else:
                # Sequential execution
                for test in other_tests:
                    process_result(run_test(test, base_cmd, progress, other_task))
        
        # Finally, run slow tests if included
        if include_slow and slow_tests:
            # For slow tests, we use batch execution in parallel mode
            # and individual execution in sequential mode for better progress tracking
            if parallel and len(slow_tests) > 1:
                # Create batches of slow tests to improve parallelization and progress tracking
                num_batches = min(len(slow_tests), max_workers)
                batch_size = (len(slow_tests) + num_batches - 1) // num_batches
                batches = [slow_tests[i:i+batch_size] for i in range(0, len(slow_tests), batch_size)]
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_futures = []
                    
                    for batch in batches:
                        # For each batch, run tests in that batch as a group
                        batch_files = " ".join(batch)
                        
                        def run_batch(batch_files, batch_base_cmd):
                            start_time = time.time()
                            cmd = f"{batch_base_cmd} {batch_files}"
                            process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            duration = time.time() - start_time
                            return {
                                "batch": True,
                                "files": batch_files,
                                "returncode": process.returncode,
                                "duration": duration,
                                "count": len(batch_files.split())
                            }
                        
                        batch_futures.append(executor.submit(run_batch, batch_files, base_cmd))
                    
                    # Process batch results
                    for future in as_completed(batch_futures):
                        batch_result = future.result()
                        # Update progress with the number of tests in the batch
                        progress.update(slow_task, advance=batch_result["count"])
                
                # Try to parse detailed results from JSON output
                try:
                    if os.path.exists(".test_results.json"):
                        with open(".test_results.json", "r") as f:
                            json_results = json.load(f)
                            
                        for test_name, test_result in json_results.get("tests", {}).items():
                            file_path = test_name.split("::")[0]
                            category = os.path.dirname(file_path) or "root"
                            
                            if test_result.get("outcome") == "passed":
                                results[category]["pass"] += 1
                            else:
                                results[category]["fail"] += 1
                            
                            test_duration = test_result.get("duration", 0)
                            results[category]["duration"] += test_duration
                            
                            # Add to all_results for detailed reporting
                            all_results.append({
                                "file": file_path,
                                "category": category,
                                "status": "pass" if test_result.get("outcome") == "passed" else "fail",
                                "duration": test_duration
                            })
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not parse JSON results: {e}[/yellow]")
                    # Fallback - just count the whole run for each file in the batch
                    for test_file in slow_tests:
                        category = os.path.dirname(test_file) or "root"
                        all_results.append({
                            "file": test_file,
                            "category": category,
                            "status": "unknown",
                            "duration": 0  # We don't know individual durations
                        })
            else:
                # Sequential execution of slow tests
                for test in slow_tests:
                    process_result(run_test(test, base_cmd, progress, slow_task))
    
    # Sort results by duration for display
    sorted_results = sorted(all_results, key=lambda x: x.get("duration", 0), reverse=True)
    
    return results, sorted_results


def display_summary(results, sorted_results):
    """Display a summary of test results."""
    console.print("\n[bold]Test Summary:[/bold]")
    
    # Create a table for category summary
    table = Table(show_header=True, header_style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Passed", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Total", style="blue")
    table.add_column("Duration", style="yellow")
    
    total_pass = 0
    total_fail = 0
    total_duration = 0
    
    for category, stats in results.items():
        passed = stats["pass"]
        failed = stats["fail"]
        duration = stats["duration"]
        
        table.add_row(
            category,
            str(passed),
            str(failed),
            str(passed + failed),
            f"{duration:.2f}s"
        )
        
        total_pass += passed
        total_fail += failed
        total_duration += duration
    
    table.add_row(
        "Total",
        str(total_pass),
        str(total_fail),
        str(total_pass + total_fail),
        f"{total_duration:.2f}s",
        style="bold"
    )
    
    console.print(table)
    
    # Show the top 5 slowest tests
    if sorted_results:
        console.print("\n[bold]Top 5 Slowest Tests:[/bold]")
        slowest_table = Table(show_header=True, header_style="bold")
        slowest_table.add_column("Test File", style="dim")
        slowest_table.add_column("Status", style="bold")
        slowest_table.add_column("Duration", style="yellow")
        
        for result in sorted_results[:5]:
            if "duration" not in result:  # Skip if duration not available
                continue
                
            status_style = "green" if result["status"] == "pass" else "red"
            slowest_table.add_row(
                result["file"],
                f"[{status_style}]{result['status']}[/{status_style}]",
                f"{result['duration']:.2f}s"
            )
        
        console.print(slowest_table)
        
        # Recommendation for optimization
        if sorted_results and sorted_results[0].get("duration", 0) > 5.0:
            console.print("\n[yellow]Optimization Opportunity:[/yellow]")
            console.print(f"The test [bold]{sorted_results[0]['file']}[/bold] is taking {sorted_results[0]['duration']:.2f}s to run.")
            console.print("Consider adding parallelization or optimizing the calculations in this test.")


def install_dependencies():
    """Install required dependencies for the test runner."""
    try:
        import rich
        import tqdm
    except ImportError:
        console.print("[yellow]Installing required dependencies...[/yellow]")
        subprocess.run("uv pip install rich tqdm pytest-json-report pytest-xdist", shell=True)
        console.print("[green]Dependencies installed successfully![/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced test runner for ExactCIs with progress visualization"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, help="Number of worker processes for parallel execution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", type=str, help="Run tests for a specific module")
    parser.add_argument("--test", type=str, help="Run a specific test")
    parser.add_argument("--mark", type=str, help="Run tests with a specific pytest mark")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        Text("ExactCIs Enhanced Test Runner", style="bold blue"),
        border_style="blue"
    ))
    
    install_dependencies()
    
    # Handle specific module, test, or mark
    if args.module or args.test or args.mark:
        cmd = "uv run pytest"
        
        if args.verbose:
            cmd += " -v"
        
        if args.parallel:
            cmd += " -n auto"
            
        if args.all:
            cmd += " --run-slow"
            
        target = ""
        if args.mark:
            target = f"-m {args.mark}"
        elif args.module:
            if "/" in args.module:
                parts = args.module.split("/")
                target = f"tests/test_{parts[0]}/{parts[1]}.py"
            else:
                target = f"tests/test_{args.module}.py"
        elif args.test:
            files = find_test_files()
            for file in files:
                with open(file, "r") as f:
                    if f"def {args.test}" in f.read():
                        target = f"{file}::{args.test}"
                        break
        
        if not target:
            console.print("[red]Error: Could not find the specified module or test[/red]")
            return 1
            
        console.print(f"[blue]Running: {cmd} {target}[/blue]")
        return subprocess.run(f"{cmd} {target}", shell=True).returncode
    
    # Find all test files
    test_files = find_test_files()
    console.print(f"[blue]Found {len(test_files)} test files[/blue]")
    
    # Default to running only fast tests unless --all is specified
    include_slow = args.all and not args.fast
    
    start_time = time.time()
    results, sorted_results = run_tests_with_progress(
        test_files, 
        parallel=args.parallel,
        include_slow=include_slow,
        verbose=args.verbose,
        max_workers=args.workers
    )
    total_time = time.time() - start_time
    
    # Display summary
    display_summary(results, sorted_results)
    
    # Clean up temporary files
    if os.path.exists(".test_results.json"):
        os.remove(".test_results.json")
    
    console.print(f"\n[blue]Total execution time: {total_time:.2f}s[/blue]")
    
    # Determine if any tests failed
    any_failed = any(stats["fail"] > 0 for stats in results.values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
