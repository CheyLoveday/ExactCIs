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