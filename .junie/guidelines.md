# ExactCIs Development Guidelines

This document provides concise guidance for developers working on the ExactCIs package.

## 1. Project Organization

- **Package Structure**:
  - Organize code into logical modules under `src/exactcis/`
  - Move core functionality from `code.py` into appropriate modules:
    - `src/exactcis/core.py` - Core calculations and utilities
    - `src/exactcis/methods/` - Individual CI methods
    - `src/exactcis/utils/` - Helper functions
  - Create `__init__.py` files to expose public API

- **Directory Layout**:
  ```
  exactcis/
  ├── src/
  │   └── exactcis/
  │       ├── __init__.py
  │       ├── core.py
  │       ├── methods/
  │       │   ├── __init__.py
  │       │   ├── conditional.py
  │       │   ├── midp.py
  │       │   └── ...
  │       └── utils/
  │           ├── __init__.py
  │           └── ...
  ├── tests/
  │   ├── test_core.py
  │   └── test_methods/
  ├── docs/
  ├── examples/
  ├── pyproject.toml
  └── README.md
  ```

## 2. Dependency Management

- **Core Dependencies**:
  - Keep core functionality dependency-free for maximum portability
  - Use NumPy as an optional dependency for performance optimization

- **pyproject.toml**:
  - Define all dependencies in `pyproject.toml`
  - Use optional dependencies for features that require additional packages:
    ```toml
    [project]
    # ...
    dependencies = []
    
    [project.optional-dependencies]
    numpy = ["numpy>=1.20.0"]
    dev = [
        "pytest>=7.0.0",
        "black>=23.0.0",
        "ruff>=0.0.270"
    ]
    ```

- **Using uv**:
  - Install dependencies with: `uv pip install -e ".[dev]"`
  - Update dependencies with: `uv pip install -U -e ".[dev]"`
  - Create reproducible environments with: `uv pip freeze > requirements.lock`

## 3. Running Code & Tests

- **ALL commands MUST use uv**:
  - Run scripts: `uv run python -m exactcis.examples.demo`
  - Run tests: `uv run pytest tests/`
  - Run linters: `uv run ruff check .`
  - Run formatter: `uv run black .`

- **Development Workflow**:
  - Create virtual environment: `uv venv`
  - Activate environment: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
  - Install in development mode: `uv pip install -e ".[dev]"`
  - Run tests before committing: `uv run pytest`

## 4. Code Style & Architecture

- **Functional Approach**:
  - Write pure functions with clear inputs and outputs
  - Avoid mutable state and side effects
  - Follow the "never nesting" rule - limit nesting to 2-3 levels maximum

- **Code Organization**:
  - Break complex functions into smaller, single-purpose functions
  - Use function composition rather than deep nesting
  - Extract helper functions for reusable logic

- **Pythonic Practices**:
  - Use type hints consistently
  - Leverage Python's standard library effectively
  - Follow PEP 8 style guidelines
  - Use docstrings for all public functions and classes

- **Flat Structure Example**:
  ```python
  # Instead of:
  def complex_function(data):
      if condition1:
          for item in data:
              if condition2(item):
                  # Deep nesting...
  
  # Prefer:
  def meets_criteria(item):
      return condition2(item)
  
  def process_valid_items(data):
      return [process(item) for item in data if meets_criteria(item)]
  
  def complex_function(data):
      if not condition1:
          return []
      return process_valid_items(data)
  ```

## 5. Best Practices

- **Error Handling**:
  - Validate inputs early with descriptive error messages
  - Use appropriate exception types
  - Fail fast and explicitly

- **Performance**:
  - Use `@lru_cache` for expensive calculations
  - Provide NumPy-accelerated implementations where beneficial
  - Include fallbacks for when optional dependencies aren't available

- **Testing**:
  - Write unit tests for all functions
  - Include edge cases and numerical stability tests
  - Test with and without optional dependencies

- **Documentation**:
  - Document parameter constraints and assumptions
  - Include examples in docstrings
  - Maintain a changelog for version updates

- **Versioning**:
  - Follow semantic versioning (MAJOR.MINOR.PATCH)
  - Document breaking changes clearly

## 6. Test Data Generation

- **2x2 Contingency Tables for Testing**:
  - Use standardized test tables with known properties to verify implementation consistency
  - Test with tables having the same odds ratio but varying precision to validate CI calculations
  - Include edge cases and tables with zeros to test robustness
  - Monitor performance with progress markers to identify slow calculations

- **Standard Test Tables with Fixed Odds Ratio (OR = 3.0)**:
  ```
  # Table 1: Small sample size
  a=3, b=1, c=1, d=1
  OR = (3*1)/(1*1) = 3.0
  Total sample size = 6

  # Table 2: Medium sample size
  a=6, b=2, c=1, d=1
  OR = (6*1)/(2*1) = 3.0
  Total sample size = 10

  # Table 3: Larger sample size
  a=15, b=5, c=1, d=1
  OR = (15*1)/(5*1) = 3.0
  Total sample size = 22

  # Table 4: Even larger sample size
  a=30, b=10, c=1, d=1
  OR = (30*1)/(10*1) = 3.0
  Total sample size = 42

  # Table 5: Very large sample size
  a=90, b=30, c=1, d=1
  OR = (90*1)/(30*1) = 3.0
  Total sample size = 122
  ```

- **Logical Comparison Tables with Fixed Odds Ratio (OR = 2.0)**:
  - Located in `analysis/logical_comparison/logical_comparison_tables.py`
  - Uses 1000 cases and 1000 controls for all tables
  - Varies exposure proportion to create tables with increasing precision
  - Automatically calculates CIs using all available methods
  - Includes progress markers to monitor performance of each calculation
  
  ```python
  # Run the script with:
  uv run python analysis/logical_comparison/logical_comparison_tables.py
  
  # The script generates 5 tables with:
  # - Fixed OR of 2.0
  # - 1000 cases and 1000 controls
  # - Increasing precision (tightening CIs)
  # - Progress markers for performance monitoring
  ```

- **Performance Monitoring**:
  - The logical comparison script includes timestamps and execution times for each calculation
  - Use these markers to identify which methods or tables are causing performance issues
  - Compare execution times across different CI methods for the same table
  - Monitor how execution time changes as table properties change

- **Using Test Tables**:
  - Compare confidence interval widths across tables with the same OR
  - Verify that CIs get tighter as sample size increases
  - Check that all methods produce logically consistent results
  - Use these tables to validate new implementations against existing ones
  - Ensure that performance remains acceptable across all test cases