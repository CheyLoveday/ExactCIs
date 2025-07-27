# API Reference

This document provides detailed information about the functions, classes, and parameters available in the ExactCIs package.

## Table of Contents

1. [Main Interface](#main-interface)
2. [Method Functions](#method-functions)
3. [Core Utilities](#core-utilities)
4. [Performance Optimization](#performance-optimization)
5. [Error Handling](#error-handling)

## Main Interface

### compute_all_cis

```python
exactcis.compute_all_cis(a, b, c, d, alpha=0.05, grid_size=200, timeout=None)
```

Computes confidence intervals for odds ratios using all available methods.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |
| grid_size | int | Size of grid for unconditional method | 200 |
| timeout | float or None | Maximum computation time in seconds | None |

**Returns**

dict
    Dictionary with method names as keys and (lower, upper) tuples as values.

**Raises**

- `ValueError`: If input counts are invalid or if margins are zero
- `TimeoutError`: If computation exceeds the specified timeout (when provided)

**Examples**

```python
from exactcis import compute_all_cis

results = compute_all_cis(12, 5, 8, 10, alpha=0.05)
for method, (lower, upper) in results.items():
    print(f"{method:12s}: ({lower:.3f}, {upper:.3f})")
```

**Notes**

This function validates the input counts before calculation and handles zero cells appropriately for each method.

**See Also**

- `validate_counts`: For input validation
- Method-specific functions for individual confidence interval calculations

## Method Functions

### exact_ci_conditional

```python
exactcis.methods.exact_ci_conditional(a, b, c, d, alpha=0.05)
```

Calculates Fisher's exact conditional confidence interval for the odds ratio of a 2×2 contingency table.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |

**Returns**

tuple
    A tuple of (lower_bound, upper_bound) representing the confidence interval for the odds ratio.

**Raises**

- `ValueError`: If input parameters are invalid

**Examples**

```python
from exactcis.methods import exact_ci_conditional

lower, upper = exact_ci_conditional(12, 5, 8, 10, alpha=0.05)
print(f"95% CI: ({lower:.3f}, {upper:.3f})")
# Output: 95% CI: (1.059, 8.726)
```

**Notes**

This implementation uses the non-central hypergeometric distribution and is guaranteed to have coverage ≥ 1-α.

### exact_ci_midp

```python
exactcis.methods.exact_ci_midp(a, b, c, d, alpha=0.05)
```

Calculates mid-P adjusted confidence interval for the odds ratio.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |

**Returns**

tuple
    A tuple of (lower_bound, upper_bound) representing the confidence interval for the odds ratio.

**Raises**

- `ValueError`: If input parameters are invalid

**Examples**

```python
from exactcis.methods import exact_ci_midp

lower, upper = exact_ci_midp(12, 5, 8, 10, alpha=0.05)
print(f"95% CI: ({lower:.3f}, {upper:.3f})")
# Output: 95% CI: (1.205, 7.893)
```

**Notes**

The mid-P adjustment gives half-weight to the observed table, resulting in narrower intervals than the conditional method with slightly lower coverage.

### exact_ci_blaker

```python
exactcis.methods.exact_ci_blaker(a, b, c, d, alpha=0.05)
```

Calculates Blaker's exact confidence interval for the odds ratio.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |

**Returns**

tuple
    A tuple of (lower_bound, upper_bound) representing the confidence interval for the odds ratio.

**Raises**

- `ValueError`: If input parameters are invalid

**Examples**

```python
from exactcis.methods import exact_ci_blaker

lower, upper = exact_ci_blaker(12, 5, 8, 10, alpha=0.05)
print(f"95% CI: ({lower:.3f}, {upper:.3f})")
# Output: 95% CI: (1.114, 8.312)
```

**Notes**

Uses the acceptability function to create non-flip intervals that are typically narrower than Fisher's while maintaining exact coverage.

### exact_ci_unconditional

```python
exactcis.methods.exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=50, timeout=None)
```

Calculates Barnard's unconditional exact confidence interval for the odds ratio.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |
| grid_size | int | Size of grid for numerical optimization | 50 |
| timeout | float or None | Maximum computation time in seconds | None |

**Returns**

tuple
    A tuple of (lower_bound, upper_bound) representing the confidence interval for the odds ratio.

**Raises**

- `ValueError`: If input parameters are invalid
- `TimeoutError`: If computation exceeds the specified timeout (when provided)

**Examples**

```python
from exactcis.methods import exact_ci_unconditional

lower, upper = exact_ci_unconditional(12, 5, 8, 10, alpha=0.05, grid_size=200)
print(f"95% CI: ({lower:.3f}, {upper:.3f})")
# Output: 95% CI: (1.132, 8.204)
```

**Notes**

This method treats both margins as independent binomials and maximizes over nuisance parameters. It is computationally intensive but provides the narrowest exact intervals.

### ci_wald_haldane

```python
exactcis.methods.ci_wald_haldane(a, b, c, d, alpha=0.05)
```

Calculates Wald-Haldane confidence interval for the odds ratio.

**Parameters**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| a | int | Count in cell (1,1) - successes in group 1 | required |
| b | int | Count in cell (1,2) - failures in group 1 | required |
| c | int | Count in cell (2,1) - successes in group 2 | required |
| d | int | Count in cell (2,2) - failures in group 2 | required |
| alpha | float | Significance level (1-confidence level) | 0.05 |

**Returns**

tuple
    A tuple of (lower_bound, upper_bound) representing the confidence interval for the odds ratio.

**Raises**

- `ValueError`: If input parameters are invalid

**Examples**

```python
from exactcis.methods import ci_wald_haldane

lower, upper = ci_wald_haldane(12, 5, 8, 10, alpha=0.05)
print(f"95% CI: ({lower:.3f}, {upper:.3f})")
# Output: 95% CI: (1.024, 8.658)
```

**Notes**

This method adds 0.5 to each cell count (Haldane-Anscombe correction) and applies the standard log-OR ± z·SE formula. It is very fast but provides only approximate coverage.

## Core Utilities

### validate_counts

```python
exactcis.core.validate_counts(a, b, c, d)
```

Validates the counts in a 2×2 contingency table.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| a | int or float | Count in cell (1,1) |
| b | int or float | Count in cell (1,2) |
| c | int or float | Count in cell (2,1) |
| d | int or float | Count in cell (2,2) |

**Returns**

None

**Raises**

- `ValueError`: If any count is negative or if any margin is zero

**Examples**

```python
from exactcis.core import validate_counts

# Valid counts
validate_counts(12, 5, 8, 10)  # No error

# Invalid counts - will raise ValueError
try:
    validate_counts(0, 0, 8, 10)
except ValueError as e:
    print(e)
    # Output: Cannot compute odds ratio with empty margins
```

## Performance Optimization

### CICache

A cache class for storing and retrieving confidence interval calculations to improve performance.

**Methods**

- `get`: Get cached result for specified parameters
- `add`: Add result to cache
- `clear`: Clear all cached results

**Examples**

```python
from exactcis.utils import CICache
from exactcis.methods import exact_ci_unconditional

# Create a cache instance
cache = CICache(max_size=100)

# Use the cache for multiple calculations
tables = [(12, 5, 8, 10), (7, 3, 2, 8), (10, 5, 3, 12)]
results = []

for a, b, c, d in tables:
    ci = exact_ci_unconditional(a, b, c, d, alpha=0.05, cache_instance=cache)
    results.append(ci)
```

### create_timeout_checker

```python
exactcis.utils.create_timeout_checker(timeout)
```

Creates a function that checks if a computation has exceeded its time limit.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| timeout | float or None | Maximum allowed computation time in seconds |

**Returns**

function
    A function that returns True if the timeout has been exceeded

**Examples**

```python
from exactcis.utils import create_timeout_checker
import time

checker = create_timeout_checker(5.0)  # 5-second timeout

start = time.time()
while not checker():
    # Do computation
    time.sleep(0.1)
    if time.time() - start > 10:  # Artificial break for example
        break

print("Done or timed out")
```

## Error Handling

### TimeoutError

Exception raised when a computation exceeds the specified timeout.

**Attributes**

- `message`: Error message
- `elapsed`: Time elapsed before timeout (seconds)

**Example**

```python
from exactcis.exceptions import TimeoutError

try:
    # Code that might timeout
    raise TimeoutError("Computation exceeded time limit", 30.5)
except TimeoutError as e:
    print(f"{e.message} (ran for {e.elapsed:.1f}s)")
```
