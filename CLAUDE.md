# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactCIs is a Python package that provides five methods to compute exact confidence intervals for the odds ratio of a 2×2 contingency table. The package implements Fisher's conditional, Mid-P adjusted, Blaker's exact, Barnard's unconditional exact, and Wald-Haldane methods.

## Development Commands

### Testing
- `uv run pytest` - Run fast tests (default, skips slow tests)
- `uv run pytest --run-slow` - Run all tests including slow ones
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest --cov=src/exactcis tests/` - Run tests with coverage
- `pytest tests/test_methods/test_blaker.py -v` - Run specific test file
- `pytest tests/test_methods/test_blaker.py::test_specific_function -v` - Run specific test

### Code Quality
- `pre-commit run --all-files` - Run linting with pre-commit
- `make lint` - Alternative lint command
- `ruff check .` - Run ruff linter
- `black .` - Format code with black
- `mypy src/exactcis` - Type checking

### Development Setup
- `uv pip install -e ".[dev]"` - Install in development mode with dev dependencies
- `make dev-install` - Alternative development install
- `make clean` - Clean build artifacts

### Documentation
- `make docs` - Generate Sphinx documentation
- Documentation lives in `docs/source/` with RST files

### Building & Distribution
- `make dist` - Build source and wheel packages
- `twine check dist/*` - Check package before upload
- `make release` - Full release process (check + upload)

## Architecture Overview

The package follows a modular architecture with clear separation of concerns:

### Core Components
- `src/exactcis/core.py` - Core statistical functions, PMF calculations, root-finding algorithms
- `src/exactcis/__init__.py` - Main public API with `compute_all_cis()` orchestrator function

### Method Implementations (`src/exactcis/methods/`)
- `conditional.py` - Fisher's exact conditional method (noncentral hypergeometric CDF inversion)
- `midp.py` - Mid-P adjusted method (half-weight for observed table)
- `blaker.py` - Blaker's exact method (acceptability function inversion)
- `unconditional.py` - Barnard's unconditional exact method (grid/NumPy optimization over nuisance parameter)
- `wald.py` - Wald-Haldane approximation (normal approximation with Haldane correction)
- `clopper_pearson.py` - Clopper-Pearson method for binomial proportions

### Utilities (`src/exactcis/utils/`)
- `stats.py` - Statistical utility functions, corrections
- `validators.py` - Input validation functions
- `optimization.py` - Optimization algorithms
- `parallel.py` - Parallel processing utilities
- `root_finding.py` - Root-finding algorithms
- `pmf_functions.py` - Probability mass function calculations
- `shared_cache.py` - Caching utilities

## Key Implementation Details

### Numerical Stability
- All PMF calculations done in log-space to prevent overflow/underflow
- Uses `logsumexp` pattern for stable probability normalization
- Extensive use of `@lru_cache` for expensive computations

### Root Finding Strategy
- Primary: `find_root_log()` for bisection in log-space
- Fallback: `find_plateau_edge()` for detecting flat p-value regions
- Combined approach in `find_smallest_theta()` for robust CI bound detection

### Performance Features
- Optional NumPy vectorization for unconditional method grid search
- Timeout protection via `timeout_checker` callbacks
- Caching at multiple levels (binomial coefficients, PMF weights, etc.)
- Progress reporting callbacks for long-running calculations

### Method Selection Guidance
- **conditional** (Fisher): Guaranteed ≥1-α coverage, conservative, use for small samples
- **midp**: Less conservative than Fisher, good for epidemiology
- **blaker**: Exact coverage with minimal over-coverage, standard in genomics
- **unconditional** (Barnard): Narrowest exact CI, requires optimization, good for unfixed margins
- **wald_haldane**: Fast approximation for large samples

## Testing Strategy

### Test Organization
- `tests/test_methods/` - Method-specific tests
- `tests/test_utils/` - Utility function tests
- Tests use pytest markers: `@pytest.mark.slow`, `@pytest.mark.unconditional`

### Test Data Patterns
- Small contingency tables: `(12, 5, 8, 10)` commonly used
- Edge cases: zero cells, extreme ratios
- Comparative tests against R's exact2x2 package
- Performance benchmarks in `profiling/` directory

## Common Development Patterns

### Adding New Methods
1. Create new file in `src/exactcis/methods/`
2. Implement function following signature: `exact_ci_<method>(a, b, c, d, alpha=0.05, **kwargs)`
3. Add to `methods/__init__.py` exports
4. Add to `compute_all_cis()` in main `__init__.py`
5. Create corresponding test file in `tests/test_methods/`

### Debugging Numerical Issues
- Check log-space calculations in `core.py`
- Verify support calculations with `support()` function
- Use `logger` for intermediate values (already configured)
- Test with edge cases like zero cells

### Performance Optimization
- Profile with `line_profiler` (already configured in dev deps)
- Use `profiling/` directory for benchmark scripts
- Consider caching for expensive repeated calculations
- Implement timeout checks for long-running operations

## File Navigation Helpers

- CI method implementations: `src/exactcis/methods/<method>.py`
- Core algorithms: `src/exactcis/core.py:` (line numbers for key functions):
  - PMF calculations: ~100-400
  - Root finding: ~400-700
  - Support calculations: ~280-320
- Main orchestrator: `src/exactcis/__init__.py:40` (`compute_all_cis`)
- Test patterns: `tests/test_methods/test_<method>.py`