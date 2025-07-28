# Contributing to ExactCIs

Thank you for your interest in contributing to ExactCIs! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Set up the development environment** as described below

## Development Environment

ExactCIs uses `uv` for dependency management. To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/exactcis.git
cd exactcis

# Create and activate a virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Install development dependencies
uv pip install -e ".[dev]"
```

## Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow the coding style**:
   - Use [Black](https://black.readthedocs.io/) for code formatting
   - Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
   - Use type hints consistently
   - Write docstrings for all functions, classes, and modules

3. **Keep changes focused**:
   - Each pull request should address a single concern
   - Break large changes into smaller, logical commits

## Testing

ExactCIs uses a unified testing strategy with a single, feature-rich test runner. All changes must include appropriate tests:

### Running Tests Locally

**Use the unified test runner** for all local test execution:

```bash
# Run fast tests only (default, recommended for development)
python scripts/run_tests_unified.py

# Run all tests including slow tests (for comprehensive validation)
python scripts/run_tests_unified.py --all

# Run tests with coverage reporting
python scripts/run_tests_unified.py --coverage

# Run tests with progress visualization (requires rich/tqdm)
python scripts/run_tests_unified.py --progress

# Run tests in parallel for faster execution
python scripts/run_tests_unified.py --parallel

# Run a specific test module
python scripts/run_tests_unified.py --module=methods/blaker

# Run a specific test function
python scripts/run_tests_unified.py --test=test_blaker_ci
```

### Test Categories

- **Fast tests** (`@pytest.mark.fast`): Quick unit tests that run in under 1 second
- **Slow tests** (`@pytest.mark.slow`): Comprehensive tests that may take longer
- **Integration tests**: Tests that verify interactions between components

### Adding New Tests

1. **Add new tests** for new functionality:
   - Unit tests for individual functions
   - Integration tests for interactions between components
   - Edge case tests for boundary conditions

2. **Mark your tests appropriately**:
   ```python
   import pytest
   
   @pytest.mark.fast
   def test_quick_function():
       # Fast test that runs quickly
       pass
   
   @pytest.mark.slow
   def test_comprehensive_analysis():
       # Slower, more comprehensive test
       pass
   ```

### Alternative Test Commands

You can also run tests directly with pytest if needed:

```bash
# Basic pytest command (matches CI)
uv run pytest

# Run with coverage (matches CI)
uv run pytest --cov=src/exactcis --cov-report=xml

# Run slow tests
uv run pytest --run-slow
```

### Development Utilities

The `analysis/analysis_scripts/` directory contains development utilities that are NOT part of the formal test suite:

- `blaker_validation.py`: Manual validation tool for the Blaker method implementation
- Other analysis scripts for debugging and validation

These scripts can be run directly for development purposes but are not included in the standard test execution.

## Documentation

Documentation is a critical part of ExactCIs:

1. **Update docstrings** for any modified functions or classes
2. **Update user documentation** in the `docs/` directory
3. **Add examples** for new functionality
4. **Update the API reference** if you've added or changed public interfaces

## Submitting Changes

1. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: brief description of what was added"
   ```

2. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request** from your fork to the main repository
4. **Fill out the pull request template** with details about your changes

## Review Process

After submitting a pull request:

1. **Automated tests** will run to verify your changes
2. **Maintainers will review** your code
3. **Address any feedback** from the review
4. Once approved, your changes will be merged

## Statistical Validation

For changes to statistical methods:

1. **Validate against established implementations** (R, SciPy, etc.)
2. **Document any differences** in the validation summary
3. **Include benchmark results** for performance changes

## Performance Considerations

When making changes that might affect performance:

1. **Profile your changes** using the provided profiling tools
2. **Document performance characteristics** in the performance documentation
3. **Consider both speed and memory usage**

Thank you for contributing to ExactCIs!