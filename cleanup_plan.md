# ExactCIs Package Cleanup Plan

## 1. Redundant Files to Remove

### Backup Files ✓
- `/Users/chey/Analytical_Projects/ExactCIs/src/exactcis/methods/unconditional.py.bak`
- `/Users/chey/Analytical_Projects/ExactCIs/src/exactcis/methods/blaker.py.bak`
- `/Users/chey/Analytical_Projects/ExactCIs/src/exactcis/core.py.bak`

### Redundant Test Files
These test files in the root directory should be consolidated or removed:
- `ci_test.py`
- `ci_test_improved.py`
- `standalone_ci_test.py`
- `improved_ci_test.py`
- `test_fixed.py`
- `test_original.py`
- `test_improved_ci.py`
- `test_profiler.py`

### Analysis Scripts to Move to a Separate Directory
These scripts analyze performance but aren't necessary for the package itself:
- `compare_all_methods.py`
- `compare_methods.py`
- `comprehensive_comparison.py`
- `direct_comparison.py`
- `exact_methods_comparison.py`
- `implementation_comparison.py`
- `scipy_comparison.py`
- `simplified_comparison.py`
- `surgical_comparison.py`
- `midp_diagnostic.py`
- `optimize_slow_functions.py`
- `profile_slow_functions.py`
- `profile_with_timeout.py`
- `optimize_unconditional.py`
- `run_tests.py`
- `run_tests_with_progress.py`
- `thorough_testing.py`

## 2. Test Failures to Address

The test failures in `tests/test_methods/test_unconditional.py` need to be fixed. The main issues appear to be:
- Expected values in tests may not match current implementation
- Tests may be dependent on random number generation that's not seeded consistently
- Tests may have timeout issues

## 3. Package Structure Issues

- Ensure proper `__init__.py` files are present in all directories
- Move test functions to proper test modules
- Update imports to reflect proper package structure
- Make sure all dependencies are correctly specified in pyproject.toml

## 4. Documentation Completion

- Verify all API documentation is complete
- Update README with proper installation and usage instructions
- Ensure examples are working correctly

## 5. Build and Publish Steps

1. Fix test failures
2. Clean up project structure 
3. Update version number if necessary
4. Verify package builds correctly with uv
5. Run tests to confirm functionality
6. Create distribution artifacts for PyPI
7. Publish package to PyPI
