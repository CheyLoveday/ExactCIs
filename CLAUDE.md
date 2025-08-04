# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactCIs is a Python package that computes exact confidence intervals for odds ratios of 2×2 contingency tables. It implements five different methods: conditional (Fisher), mid-P adjusted, Blaker's exact, unconditional (Barnard's), and Wald-Haldane.

## Development Commands

### Testing
- `uv run pytest` - Run all tests, skipping slow tests by default
- `uv run pytest --run-slow` - Run all tests including slow ones
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest tests/test_methods/test_blaker.py::test_specific_function` - Run specific test
- `make test` - Alternative test command
- `make coverage` - Run tests with coverage report

### Code Quality
- `make lint` - Run pre-commit hooks (black, ruff, mypy)
- `ruff check src/` - Check code style with ruff
- `black src/` - Format code with black
- `mypy src/` - Type checking

### Building and Documentation
- `make docs` - Generate Sphinx documentation
- `make dist` - Build source and wheel distributions
- `uv build` - Alternative build command using uv

### Environment Setup
- `uv pip install -e ".[dev]"` - Install in development mode
- `scripts/setup_dev_env.sh` - Set up complete development environment

## Architecture

### Core Module (`src/exactcis/core.py`)
The central module containing:
- **Validation functions**: `validate_counts()` for input validation
- **PMF calculations**: Noncentral hypergeometric distribution functions (`log_nchg_pmf`, `log_nchg_cdf`, `pmf_weights`)
- **Numerical methods**: Root finding (`find_root_log`, `find_plateau_edge`, `find_smallest_theta`) and stable log-space computations
- **Support calculation**: `support()` function that determines valid range for contingency table cell values
- **Utility functions**: Odds ratio calculation, Haldane correction, batch processing functions

### Method Implementations (`src/exactcis/methods/`)
Each CI method is implemented in its own module:
- `conditional.py` - Fisher's exact method using noncentral hypergeometric CDF inversion
- `midp.py` - Mid-P correction reducing Fisher's conservatism
- `blaker.py` - Blaker's exact method using acceptability functions and plateau detection
- `unconditional.py` - Barnard's method with grid search over nuisance parameters
- `wald.py` - Haldane-Anscombe correction with asymptotic normal approximation

### Utilities (`src/exactcis/utils/`)
Support modules for:
- `root_finding.py` - Functional programming implementations of numerical methods
- `pmf_functions.py` - Optimized PMF calculation implementations
- `validators.py` - Input validation utilities
- `stats.py` - Statistical calculation helpers
- `optimization.py` - Performance optimization utilities

### Main API (`src/exactcis/__init__.py`)
The `compute_all_cis()` function provides a unified interface that runs all methods on the same 2×2 table for comparison.

## Key Design Patterns

### Functional Programming Approach
The codebase emphasizes pure functions, especially in numerical computations. Core functions like `find_root_log` delegate to functional implementations in utils modules when available, with fallbacks to original implementations.

### Log-Space Computations
Most probability calculations use log space to avoid numerical overflow/underflow, particularly in `log_nchg_pmf`, `logsumexp`, and root-finding functions.

### Timeout and Progress Handling
Long-running methods (especially unconditional) support timeout checking and progress callbacks. Functions accept optional `timeout_checker` and `progress_callback` parameters.

### Caching Strategy
Expensive computations like binomial coefficients (`log_binom_coeff`) and PMF weights use LRU caching. Cache sizes can be optimized for batch processing scenarios.

## Testing Strategy

### Test Structure
- Method-specific tests in `tests/test_methods/`
- Integration tests comparing against R implementations
- Edge case testing for boundary conditions and rare events
- Performance tests marked with `@pytest.mark.slow`

### Test Markers
- `slow` - Long-running tests (skipped by default)
- `unconditional` - Tests specific to Barnard's method
- `integration` - Cross-method validation tests
- `edge` - Boundary condition tests

### Running Subsets
Use pytest markers to run specific test categories:
- `pytest -m "not slow"` - Skip slow tests
- `pytest -m "unconditional"` - Only unconditional method tests

## Performance Considerations

### Unconditional Method Optimization
The unconditional method supports NumPy vectorization when available. Grid search can be parallelized, and timeout mechanisms prevent excessive computation time.

### Batch Processing
Core functions provide batch variants (`batch_calculate_odds_ratios`, `batch_log_nchg_pmf`) for processing multiple tables efficiently.

### Root Finding Precision
Root finding functions use adaptive precision and plateau detection to handle flat p-value functions common in exact methods.