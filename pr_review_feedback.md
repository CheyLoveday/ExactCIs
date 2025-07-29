# PR #2 Review: Cognitive Load Reduction

## Overall Assessment

This PR makes significant progress in reducing cognitive load in the ExactCIs codebase, particularly for some of the most complex functions identified in the cognitive load audit. The approach of using functional programming principles with pure functions and immutable data structures is well-aligned with the recommendations from the audit report.

### Key Strengths

- **Excellent refactoring of `exact_ci_unconditional`**: The most problematic function has been transformed into a clean, well-structured implementation with clear separation of concerns.
- **Effective use of delegation pattern**: Functions like `_pmf_weights_impl` and `find_root_log` now delegate to functional implementations in separate modules.
- **Improved parameter organization**: The use of parameter objects (e.g., `TableData`, `UnconditionalConfig`) reduces cognitive load by grouping related parameters.
- **Enhanced documentation**: Docstrings and comments are clear and informative, explaining the purpose and approach of each function.

### Areas for Improvement

- **Incomplete refactoring**: Several functions identified in the audit report still contain complex implementations directly rather than delegating to functional implementations.
- **Inconsistent application of guard clauses**: Some functions could benefit from more aggressive use of guard clauses to reduce nesting.
- **Remaining complex functions**: Functions like `find_smallest_theta`, `fisher_lower_bound`, and `exact_ci_blaker` still have high cognitive load.

## Function-by-Function Analysis

### 1. `exact_ci_unconditional` (unconditional.py)

**Before**: 379 lines, 4 nesting levels, 12 parameters  
**After**: Significantly reduced complexity through delegation and functional decomposition

**Strengths**:
- Excellent separation of concerns (validation, caching, calculation)
- Use of parameter objects (`TableData`, `UnconditionalConfig`)
- Clear guard clauses for early returns
- Delegation to smaller, focused functions in utility modules

**Suggestions**:
- Consider further extracting the timeout checker setup into a separate function

### 2. `_pmf_weights_impl` (core.py)

**Before**: 120 lines, 4 nesting levels  
**After**: 10 lines, simple delegation to functional implementation

**Strengths**:
- Complete refactoring to delegate to functional implementation
- Clear error handling for import failures
- Maintains backward compatibility

**Suggestions**:
- Implement the fallback to original implementation that's currently commented out

### 3. `find_root_log` (core.py)

**Before**: 121 lines, 4 nesting levels, 7 parameters  
**After**: 10 lines, simple delegation to functional implementation

**Strengths**:
- Complete refactoring to delegate to functional implementation
- Clear error handling for import failures
- Maintains backward compatibility

**Suggestions**:
- Implement the fallback to original implementation that's currently commented out

### 4. `find_smallest_theta` (core.py)

**Before**: 112 lines, 3 nesting levels, 11 parameters  
**After**: Still approximately 112 lines with complex implementation

**Strengths**:
- Improved documentation with detailed comments
- Some guard clauses for early returns

**Suggestions**:
- Refactor to delegate to a functional implementation in a utility module
- Use parameter objects to reduce the number of parameters
- Extract the complex error handling and logging into separate functions
- Apply more guard clauses to reduce nesting

### 5. `fisher_lower_bound` (conditional.py)

**Before**: 112 lines, 4 nesting levels, 7 parameters  
**After**: Still approximately 112 lines with complex implementation

**Strengths**:
- Improved documentation with detailed comments

**Suggestions**:
- Refactor to delegate to a functional implementation in a utility module
- Extract the bracket expansion logic into a separate function
- Extract the root finding logic into a separate function
- Apply more guard clauses to reduce nesting

### 6. `exact_ci_conditional` (conditional.py)

**Before**: 106 lines, 3 nesting levels, 5 parameters  
**After**: Still approximately 106 lines with complex implementation

**Strengths**:
- Improved documentation with detailed comments
- Some early returns for special cases

**Suggestions**:
- Refactor to delegate to a functional implementation in a utility module
- Extract the special case handling for zero cells into separate functions
- Apply more guard clauses to reduce nesting

### 7. `exact_ci_blaker` (blaker.py)

**Before**: 122 lines, 3 nesting levels  
**After**: Still approximately 122 lines with complex implementation

**Strengths**:
- Improved documentation with detailed comments
- Some guard clauses for early returns

**Suggestions**:
- Refactor to delegate to a functional implementation in a utility module
- Extract the lower bound and upper bound calculations into separate functions
- Apply more guard clauses to reduce nesting

## Suggestions for Improvement

### 1. Complete the Refactoring Approach

The delegation pattern used for `exact_ci_unconditional`, `_pmf_weights_impl`, and `find_root_log` is excellent and should be extended to the remaining complex functions:

```python
def find_smallest_theta(...):
    """
    Finds the smallest theta such that func(theta) is close to alpha.
    
    This refactored function uses functional programming principles with pure functions
    for improved maintainability and testability.
    """
    try:
        from .utils.optimization import find_smallest_theta_impl_functional
        return find_smallest_theta_impl_functional(...)
    except ImportError:
        logger.warning("Functional implementation not available, using original")
        # Fallback to original implementation
```

### 2. Consistently Apply Parameter Objects

The use of parameter objects in `exact_ci_unconditional` is excellent and should be extended to other functions with many parameters:

```python
@dataclass
class SearchParams:
    alpha: float
    lo: float = 1e-8
    hi: float = 1.0
    two_sided: bool = True
    increasing: bool = False
    xtol: float = 1e-7
    max_iter: int = 100

def find_smallest_theta(func: Callable[[float], float], params: SearchParams, 
                       progress_callback=None, timeout_checker=None):
    # Implementation using params.alpha, params.lo, etc.
```

### 3. More Aggressive Use of Guard Clauses

Functions like `exact_ci_conditional` and `exact_ci_blaker` would benefit from more aggressive use of guard clauses:

```python
# Instead of:
if a == 0:
    # Special case handling
    # ...
elif c == 0:
    # Another special case
    # ...
else:
    # Normal case
    # ...

# Use:
if a == 0:
    return handle_zero_cell_a(b, c, d, alpha)
if c == 0:
    return handle_zero_cell_c(a, b, d, alpha)
# Normal case continues without nesting
```

### 4. Extract Pure Functions for Complex Logic

Complex logic blocks should be extracted into pure functions with clear names:

```python
# Instead of embedding complex bracket expansion logic in fisher_lower_bound:
def expand_brackets_for_lower_bound(lo, hi, p_value_func, max_attempts=40):
    # Implementation of bracket expansion logic
    return lo, hi, lo_val, hi_val

# Then in fisher_lower_bound:
lo, hi, lo_val, hi_val = expand_brackets_for_lower_bound(lo, hi, p_value_func)
```

## Positive Highlights

1. The refactoring of `exact_ci_unconditional` is exemplary and demonstrates a clear understanding of functional programming principles and cognitive load reduction.

2. The use of parameter objects like `TableData` and `UnconditionalConfig` is an excellent approach to reducing parameter count and improving code readability.

3. The delegation pattern used for `_pmf_weights_impl` and `find_root_log` is a clean and effective way to refactor complex functions while maintaining backward compatibility.

4. The improved documentation throughout the codebase makes the code more accessible and easier to understand.

## Critical Issues

While the PR makes significant progress in reducing cognitive load, there are several critical issues that need to be addressed:

### 1. Import Error Risk

The refactored functions use try-except blocks to import new functional modules but fail to provide proper fallback implementations:

```python
# In _pmf_weights_impl (core.py)
try:
    from .utils.pmf_functions import pmf_weights_impl_functional
    return pmf_weights_impl_functional(n1, n2, m, theta)
except ImportError:
    logger.warning("Functional PMF implementation not available, using original")
    # Fallback to original implementation would go here
    # For now, raising an error to ensure we catch import issues
    raise
```

```python
# In find_root_log (core.py)
try:
    from .utils.root_finding import find_root_log_impl_functional
    return find_root_log_impl_functional(...)
except ImportError:
    logger.warning("Functional root finding implementation not available, using original")
    # Fallback to original implementation would go here
    # For now, raising an error to ensure we catch import issues
    raise
```

**Recommendation**: Implement the fallback to the original implementation instead of raising exceptions. This will ensure backward compatibility and prevent runtime errors if the new modules have issues.

### 2. Performance Concern

The `enumerate_all_possible_tables` function in `utils/calculators.py` creates all possible 2x2 tables in memory, which could cause memory and performance issues for large datasets:

```python
def enumerate_all_possible_tables(n1: int, n2: int) -> List[TableData]:
    """Enumerate all possible 2x2 tables with given marginals."""
    tables = []
    for a in range(n1 + 1):
        for c in range(n2 + 1):
            b = n1 - a
            d = n2 - c
            tables.append(TableData(a, b, c, d))
    return tables
```

For large values of n1 and n2 (e.g., n1=1000, n2=1000), this would create over 1 million TableData objects in memory.

**Recommendation**: Refactor to use a generator pattern that yields tables one at a time instead of storing them all in memory:

```python
def enumerate_all_possible_tables(n1: int, n2: int):
    """Generator that yields all possible 2x2 tables with given marginals."""
    for a in range(n1 + 1):
        for c in range(n2 + 1):
            b = n1 - a
            d = n2 - c
            yield TableData(a, b, c, d)
```

### 3. Logic Gap

While the refactored `exact_ci_unconditional` function preserves many aspects of the original implementation, there are potential gaps in the logic:

1. **Cache Lookup**: The refactored function does check the cache, but it's not clear if the cache key construction is identical to the original.

2. **Search Parameter Derivation**: The `determine_theta_range` function appears to handle the basic cases, but may not cover all the edge cases that the original implementation handled.

3. **Refinement Steps**: The refinement logic in the refactored function appears to cover the essential steps, but a detailed comparison with the original implementation is needed to ensure all edge cases are handled.

**Recommendation**: Conduct a line-by-line comparison of the original and refactored functions to ensure that all logic is preserved, especially for edge cases and error handling.

## Completeness Assessment

Of the 7 high-priority functions identified in the cognitive load audit report, 3 have been refactored in this PR:

✅ **exact_ci_unconditional** (unconditional.py) - Completely refactored with excellent separation of concerns
✅ **_pmf_weights_impl** (core.py) - Refactored to delegate to functional implementation
✅ **find_root_log** (core.py) - Refactored to delegate to functional implementation

The following 4 high-priority functions still need refactoring:

❌ **find_smallest_theta** (core.py) - Still complex with 11 parameters and 3 nesting levels
❌ **fisher_lower_bound/upper_bound** (conditional.py) - Still complex with 7 parameters and 4 nesting levels
❌ **exact_ci_conditional** (conditional.py) - Still complex with 5 parameters and 3 nesting levels
❌ **exact_ci_blaker** (blaker.py) - Still complex with 3 nesting levels

## Conclusion

This PR makes significant progress in reducing cognitive load in the ExactCIs codebase, particularly for some of the most complex functions. The approach of using functional programming principles with pure functions and immutable data structures is well-aligned with the recommendations from the audit report.

To fully address the issues identified in this review, I recommend:

1. Implementing proper fallback mechanisms for the imported functional modules
2. Refactoring the `enumerate_all_possible_tables` function to use a generator pattern
3. Ensuring all logic from the original implementation is preserved in the refactored functions
4. Completing the refactoring of the remaining complex functions using the same delegation pattern
5. Consistently applying parameter objects to reduce parameter count
6. More aggressively using guard clauses to reduce nesting
7. Extracting pure functions for complex logic blocks

Overall, this PR is a strong step forward in improving the maintainability and readability of the ExactCIs codebase, but the critical issues identified need to be addressed before it can be merged.