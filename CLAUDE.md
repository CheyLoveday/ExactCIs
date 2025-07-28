# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactCIs is a Python package for computing exact confidence intervals for odds ratios in 2×2 contingency tables. It implements five methods:

- **conditional** (Fisher's exact): Conservative method using noncentral hypergeometric CDF inversion
- **midp**: Mid-P adjusted method that reduces Fisher's conservatism by giving half-weight to observed values
- **blaker**: Blaker's exact method using acceptability function for minimal over-coverage
- **unconditional** (Barnard's): Most powerful exact method optimizing over nuisance parameters
- **wald_haldane**: Asymptotic Wald method with Haldane correction (adds 0.5 to zero cells)

## Development Commands

This project uses **uv** for package management and virtual environment management. Always use `uv` commands instead of pip or python directly.

### Testing
- `uv run pytest` - Run tests quickly (skips slow tests by default)
- `uv run pytest --run-slow` - Run all tests including slow ones
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest tests/test_methods/test_blaker.py` - Run specific test file
- `uv run pytest -k test_exact_ci_blaker_basic` - Run specific test by name pattern
- `uv run pytest --cov=src/exactcis tests/` - Run tests with coverage report
- `uv run pytest tests/test_batch_methods.py` - Run batch processing tests
- `uv run pytest tests/test_utils/test_parallel.py` - Run parallel processing tests

### Code Quality
- `uv run ruff check` - Lint code
- `uv run black .` - Format code
- `uv run mypy src/exactcis` - Type checking
- `make lint` - Run pre-commit hooks (includes formatting, linting, type checking)

### Development Setup
- `uv sync` - Install locked dependencies for development (recommended)
- `uv pip install -e ".[dev]"` - Install package in development mode with dev dependencies  
- `uv pip install -e ".[full]"` - Install with all optional dependencies (pandas, sympy, tqdm, matplotlib)

### Package Building and Documentation  
- `make clean` - Clean build artifacts
- `python -m build` - Build source and wheel distributions
- `make docs` - Generate Sphinx documentation

## Architecture

### Core Module Structure
- `src/exactcis/core.py` - Core mathematical functions, validation, probability calculations
- `src/exactcis/methods/` - Individual CI method implementations:
  - `conditional.py` - Fisher's exact method
  - `midp.py` - Mid-P adjusted method  
  - `blaker.py` - Blaker's exact method
  - `unconditional.py` - Barnard's unconditional method
  - `wald.py` - Wald method with Haldane correction
- `src/exactcis/utils/` - Utility functions for optimization, parallelization, statistics
- `src/exactcis/cli.py` - Command-line interface
- `src/exactcis/__init__.py` - Main API with `compute_all_cis()` orchestrator function

### Key Mathematical Components
- **Support calculation**: `support()` function determines valid range for cell 'a' in hypergeometric distribution
- **PMF calculations**: `pmf_weights()` and `pmf()` compute noncentral hypergeometric probabilities with numerical stability
- **Root finding**: `find_root_log()`, `find_plateau_edge()`, `find_smallest_theta()` for CI inversion
- **Validation**: `validate_counts()` ensures valid 2×2 table inputs

### Testing Strategy
- Tests are organized by method in `tests/test_methods/`
- Slow tests (marked with `@pytest.mark.slow`) are skipped by default
- Integration tests compare against known reference values
- Timeout tests verify computational limits for unconditional method

## Important Development Notes

### Numerical Stability
- All probability calculations use log-space arithmetic to prevent overflow/underflow
- `logsumexp()` function provides stable summation of exponentials
- Root finding uses log-space bisection for wide-range theta values

### Performance Considerations  
- Unconditional method includes timeout protection (default 300s) to prevent infinite computations
- Grid size parameter balances accuracy vs. computation time
- LRU caching used for support calculations

### Concurrency and Parallel Processing
- **Batch Processing**: CLI supports batch processing of multiple tables from CSV files
  - `exactcis-cli --batch input.csv --output results.csv --method blaker`
  - Parallel processing automatically used for batch calculations
- **Method-Specific Batch Functions**:
  - `exact_ci_blaker_batch()` - Parallel Blaker CIs for multiple tables
  - `exact_ci_midp_batch()` - Parallel Mid-P CIs for multiple tables
  - `parallel_compute_ci()` - Generic parallel CI computation utility
- **Core Parallelization**:
  - Unconditional method uses ProcessPoolExecutor for grid point calculations
  - `parallel_map()` utility provides robust parallel processing with fallback
  - Automatic worker count optimization based on available CPU cores
- **Batch Utilities**:
  - `batch_validate_counts()` - Validate multiple tables efficiently
  - `batch_calculate_odds_ratios()` - Calculate multiple odds ratios in batch
  - `optimize_core_cache_for_batch()` - Optimize caching for batch scenarios

### Code Style Rules
- Use uv for all package operations (per .windsurfrules)
- No hardcoded values in tests - use parameterized fixtures
- Type hints required for all function parameters and return values
- Comprehensive logging for debugging complex numerical calculations

### Test Markers
- `fast`: Quick-running tests (default)
- `slow`: Computationally intensive tests (use `--run-slow` to include)
- `unconditional`: Tests specific to Barnard's method
- `integration`: End-to-end tests comparing methods
- `edge`: Edge case tests with extreme table configurations