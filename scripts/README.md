# Scripts Directory

This directory contains utility scripts for the ExactCIs project.

## Test Execution

### `run_tests_unified.py` (RECOMMENDED)

**The official test runner for ExactCIs.** This is the single, canonical script for running tests locally.

```bash
# Run fast tests only (default, recommended for development)
python scripts/run_tests_unified.py

# Run all tests including slow tests
python scripts/run_tests_unified.py --all

# Run with coverage reporting
python scripts/run_tests_unified.py --coverage

# Run with progress visualization
python scripts/run_tests_unified.py --progress

# Run in parallel for faster execution
python scripts/run_tests_unified.py --parallel

# Run specific module or test
python scripts/run_tests_unified.py --module=methods/blaker
python scripts/run_tests_unified.py --test=test_blaker_ci
```

**Features:**
- Parallel test execution with progress bars
- Coverage reporting integration
- Fast/slow test categorization
- Rich progress visualization (when `rich` or `tqdm` is installed)
- Comprehensive command-line options

## Development and Analysis

### `build_package.sh`
Script for building the Python package for distribution.

### `setup_dev_env.sh`
Script for setting up the development environment with all necessary dependencies.

### `test_profiler.py`
Performance profiling utility for analyzing test execution times and identifying bottlenecks.

### `compare_methods.py`
Utility for comparing different confidence interval methods and their results.

### `analyze_comparisons.py`
Script for analyzing and visualizing comparison results between different statistical methods.

## Usage Guidelines

1. **For running tests**: Always use `run_tests_unified.py`
2. **For development**: Use the other scripts as needed for specific tasks
3. **For CI/CD**: The CI pipeline uses standard `pytest` commands as defined in `.github/workflows/ci.yml`

## Removed Scripts

The following scripts have been removed to eliminate redundancy and confusion:

- `run_tests.sh` - Functionality merged into `run_tests_unified.py`
- `run_tests.py` - Functionality merged into `run_tests_unified.py`
- `run_tests_with_progress.py` - Functionality merged into `run_tests_unified.py`
- `run_blaker_tests.py` - Moved to `analysis/analysis_scripts/blaker_validation.py`

## Installation of Optional Dependencies

For enhanced progress visualization, install optional dependencies:

```bash
uv add rich tqdm
```

These dependencies enable beautiful progress bars and enhanced terminal output in the unified test runner.