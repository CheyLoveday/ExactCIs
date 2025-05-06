# API Reference

This document provides detailed information about the functions, classes, and parameters available in the ExactCIs package.

## Table of Contents

1. [Core Functions](#core-functions)
2. [Utility Functions](#utility-functions)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling](#error-handling)

## Core Functions

### exact_ci_unconditional

```python
from exactcis.methods.unconditional import exact_ci_unconditional

exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, 
                      grid_size: int = 50,
                      theta_min: Optional[float] = None,
                      theta_max: Optional[float] = None,
                      haldane: bool = False,
                      refine: bool = True,
                      use_profile: bool = True,
                      custom_range: Optional[Tuple[float, float]] = None,
                      theta_factor: float = 10,
                      optimization_params: Optional[Dict[str, Any]] = None)
```

**Description**:  
Calculates Barnard's unconditional exact confidence interval for the odds ratio of a 2×2 contingency table.

**Parameters**:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |
| grid_size | int | Size of grid for numerical optimization | 50 |
| theta_min | float | Minimum theta value for search (auto if None) | None |
| theta_max | float | Maximum theta value for search (auto if None) | None |
| haldane | bool | Apply Haldane's correction for zero cells | False |
| refine | bool | Use adaptive grid refinement for precision | True |
| use_profile | bool | Use profile likelihood for zero counts | True |
| custom_range | Tuple[float, float] | Custom (min, max) for theta search | None |
| theta_factor | float | Factor for determining auto theta range | 10 |
| optimization_params | Dict[str, Any] | Additional optimization parameters | None |

**Returns**:  
A tuple of `(lower_bound, upper_bound)` representing the confidence interval for the odds ratio.

**Raises**:
- `ValueError`: If input parameters are invalid
- `RuntimeError`: If computation fails to converge

**Example**:
```python
lower, upper = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05)
print(f"95% CI: ({lower:.6f}, {upper:.6f})")
```

**Notes**:
- This is the original implementation of Barnard's unconditional exact test
- For better performance, consider using `improved_ci_unconditional`

### improved_ci_unconditional

```python
from exactcis.methods.unconditional import improved_ci_unconditional

improved_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, 
                        grid_size: int = 50,
                        theta_min: Optional[float] = None,
                        theta_max: Optional[float] = None,
                        adaptive_grid: bool = True,
                        use_cache: bool = True,
                        cache_instance: Optional['CICache'] = None)
```

**Description**:  
Calculates Barnard's unconditional exact confidence interval with improved performance using caching and adaptive search strategies.

**Parameters**:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |
| grid_size | int | Size of grid for numerical optimization | 50 |
| theta_min | float | Minimum theta value for search (auto if None) | None |
| theta_max | float | Maximum theta value for search (auto if None) | None |
| adaptive_grid | bool | Use adaptive grid refinement | True |
| use_cache | bool | Use caching for speedup | True |
| cache_instance | CICache | Cache instance to use (creates new if None) | None |

**Returns**:  
A tuple of `(lower_bound, upper_bound)` representing the confidence interval for the odds ratio.

**Raises**:
- `ValueError`: If input parameters are invalid
- `RuntimeError`: If computation fails to converge

**Example**:
```python
lower, upper = improved_ci_unconditional(7, 3, 2, 8, alpha=0.05)
print(f"95% CI: ({lower:.6f}, {upper:.6f})")
```

**Notes**:
- This implementation produces identical results to `exact_ci_unconditional` but with significantly improved performance
- Recommended for most use cases, especially when calculating multiple confidence intervals

## Utility Functions

### unconditional_log_pvalue

```python
from exactcis.methods.unconditional import unconditional_log_pvalue

unconditional_log_pvalue(a: int, b: int, c: int, d: int, 
                       theta: float = 1.0, 
                       p1_values: Optional[np.ndarray] = None,
                       refine: bool = True,
                       use_profile: bool = False,
                       progress_callback: Optional[Callable[[float], None]] = None,
                       timeout_checker: Optional[Callable[[], bool]] = None)
```

**Description**:  
Calculates the log p-value for Barnard's unconditional exact test at a given odds ratio (theta).

**Parameters**:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) | required |
| b | int | Count in cell (1,2) | required |
| c | int | Count in cell (2,1) | required |
| d | int | Count in cell (2,2) | required |
| theta | float | Odds ratio parameter | 1.0 |
| p1_values | np.ndarray | Optional array of p1 values to evaluate | None |
| refine | bool | Whether to use refinement for precision | True |
| use_profile | bool | Whether to use profile likelihood | False |
| progress_callback | Callable | Optional callback for progress reporting | None |
| timeout_checker | Callable | Optional callback for timeout checking | None |

**Returns**:  
The natural logarithm of the p-value.

## Performance Optimization

### CICache

```python
from exactcis.utils.optimization import CICache

cache = CICache(max_size=100)
```

**Description**:  
Cache class for storing and retrieving confidence interval calculations to improve performance.

**Parameters**:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| max_size | int | Maximum number of cache entries | 100 |

**Methods**:

| Method | Description |
|--------|-------------|
| `get_exact(a, b, c, d, alpha)` | Get exact cache hit for the given parameters |
| `get_similar(a, b, c, d, alpha)` | Get similar cache entry for approximation |
| `add(a, b, c, d, alpha, ci, metadata)` | Add entry to cache |
| `clear()` | Clear all cache entries |

**Example**:
```python
from exactcis.utils.optimization import CICache
from exactcis.methods.unconditional import improved_ci_unconditional

# Create a cache instance
cache = CICache(max_size=1000)

# Use the cache for multiple calculations
tables = [(7, 3, 2, 8), (8, 2, 3, 7), (10, 5, 3, 12)]
results = []

for a, b, c, d in tables:
    ci = improved_ci_unconditional(a, b, c, d, alpha=0.05, cache_instance=cache)
    results.append(ci)
```

## Error Handling

The ExactCIs package includes comprehensive error handling for various edge cases:

### Common Errors

| Error | Description | Solution |
|-------|-------------|----------|
| `ValueError: Invalid table: negative counts` | Negative values in table | Ensure all counts are non-negative |
| `ValueError: Invalid table: non-integer counts` | Non-integer values in table | Ensure all counts are integers |
| `RuntimeError: Failed to converge` | Algorithm failed to converge | Try increasing grid_size or set custom bounds |
| `Warning: Large table detected` | Table dimensions are large | Consider using normal approximation for large tables |
| `ValueError: Invalid alpha value` | Alpha outside (0,1) range | Ensure alpha is between 0 and 1 |

### Handling Zero Cells

When one or more cells contain zeros, consider:

1. Using `haldane=True` to apply Haldane's correction
2. Using `use_profile=True` for profile likelihood approach
3. Specifying custom bounds with `custom_range` if automatic bounds fail
