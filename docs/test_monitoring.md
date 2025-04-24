# Test Monitoring in ExactCIs

This document explains how to run and monitor tests in the ExactCIs package, including strategies for managing slow or computationally intensive tests.

## Test Suite Structure

The ExactCIs test suite is organized as follows:

- **Core Tests**: Basic functionality testing of core components (`test_core.py`)
- **Main Package Tests**: Testing the main package interface (`test_exactcis.py`)
- **Integration Tests**: Testing the interaction between components (`test_integration.py`)
- **Method-Specific Tests**: In the `test_methods/` directory:
  - `test_conditional.py`: Conditional (Fisher) confidence interval tests
  - `test_midp.py`: Mid-P adjusted confidence interval tests
  - `test_blaker.py`: Blaker's exact confidence interval tests
  - `test_unconditional.py`: Barnard's unconditional exact confidence interval tests
  - `test_wald.py`: Haldane-Anscombe Wald confidence interval tests
- **Utility Tests**: Testing statistical utility functions (`test_utils/test_stats.py`)

## Running Tests

### Basic Test Run

To run all tests with basic output:

```bash
uv run pytest
```

### Verbose Output

To see detailed progress of each test as it runs:

```bash
uv run pytest -v
```

### Running Only Fast Tests

By default, tests marked as "slow" are skipped. This is ideal for quick development cycles:

```bash
uv run pytest -v
```

### Running All Tests Including Slow Ones

To run all tests, including those marked as slow:

```bash
uv run pytest -v --run-slow
```

### Debugging with Detailed Output

To stop testing after the first failure and see print statements and logging output:

```bash
uv run pytest -xvs
```

The flags used are:
- `-x`: Stop after first failure
- `-v`: Verbose output
- `-s`: Allow print statements and logging output

## Test Categories and Selective Testing

Tests are categorized with markers to help organize and selectively run them:

- `fast`: Fast-running tests
- `slow`: Slow-running tests that might take longer to execute
- `unconditional`: Tests for the unconditional method
- `integration`: Integration tests

To run only tests with a specific marker:

```bash
uv run pytest -m fast
uv run pytest -m unconditional
uv run pytest -m "not slow"
```

To run a specific test file:

```bash
uv run pytest tests/test_methods/test_midp.py
```

To run a specific test function:

```bash
uv run pytest tests/test_integration.py::test_readme_example
```

## Test Parallelization

For faster test execution, especially for computationally intensive tests, you can run tests in parallel:

```bash
uv run pytest -n auto
```

This uses the pytest-xdist plugin to run tests in parallel across multiple CPU cores.

## Timeouts

Several tests, particularly those for the unconditional method, are computationally intensive and could run for a long time. To prevent tests from running indefinitely, timeouts are configured:

```bash
uv run pytest --timeout=300
```

This will terminate any test that runs longer than 300 seconds.

Some slow tests have specific timeout markers, for example:

```python
@pytest.mark.timeout(300)  # 5-minute timeout
```

## Handling Edge Cases and Large Computations

The test suite includes several strategies for handling edge cases and computationally intensive scenarios:

1. **Reduced Grid Sizes**: For unconditional tests, smaller grid sizes are used in tests to speed up execution
2. **Skip Markers**: Very intensive tests are skipped by default:
   ```python
   @pytest.mark.skip(reason="Too computationally intensive for regular testing")
   ```
3. **Exception Handling**: Some edge cases might legitimately raise exceptions, which the tests accommodate:
   ```python
   try:
       result = compute_function()
   except RuntimeError:
       # This is acceptable for this edge case
       pass
   ```
4. **Mock-Based Testing**: Some tests use mocking to test the interface without running the full algorithm

## Monitoring Progress in Slow Functions

Several improvements have been made to monitor progress in slow functions:

1. **Logging**: Detailed logging has been added to slow functions like `_pvalue_barnard` and `find_smallest_theta`.

2. **Progress Bars**: The grid search in `_pvalue_barnard` now displays a progress bar using tqdm.

3. **Periodic Updates**: Long-running calculations now log periodic updates to show they're still making progress.

## Example: Monitoring a Specific Slow Test

To run and monitor a specific slow test with full logging:

```bash
uv run pytest -xvs tests/test_integration.py::test_large_imbalance --run-slow
```

This will run only the `test_large_imbalance` test with verbose output and logging, and will stop after completion.
