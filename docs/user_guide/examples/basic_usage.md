# Basic Usage

This guide demonstrates basic usage patterns for the ExactCIs package, providing examples for common scenarios and tasks.

## Prerequisites

Ensure you have ExactCIs installed:

```bash
pip install exactcis
```

## Basic Example

The most common use case is calculating a confidence interval for the odds ratio of a 2×2 contingency table:

```python
from exactcis.methods.unconditional import improved_ci_unconditional

# Example 2×2 table:
#      Success   Failure
# Grp1    7         3
# Grp2    2         8

a, b, c, d = 7, 3, 2, 8  # Cell counts

# Calculate 95% confidence interval
lower, upper = improved_ci_unconditional(a, b, c, d, alpha=0.05)
print(f"95% CI for odds ratio: ({lower:.4f}, {upper:.4f})")
# Output: 95% CI for odds ratio: (1.0472, 104.7200)

# Calculate 90% confidence interval
lower, upper = improved_ci_unconditional(a, b, c, d, alpha=0.10)
print(f"90% CI for odds ratio: ({lower:.4f}, {upper:.4f})")
# Output: 90% CI for odds ratio: (1.3810, 41.4300)
```

## Comparing Different Methods

You can compare the results from different methods:

```python
import scipy.stats as stats
from exactcis.methods.unconditional import improved_ci_unconditional, exact_ci_unconditional
import numpy as np

def normal_approx_ci(a, b, c, d, alpha=0.05):
    """Calculate CI using normal approximation"""
    # Add 0.5 to each cell (Haldane's correction)
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    
    # Calculate odds ratio and log odds ratio
    or_est = (a*d)/(b*c)
    log_or = np.log(or_est)
    
    # Standard error of log odds ratio
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    # Critical value
    z = stats.norm.ppf(1 - alpha/2)
    
    # Confidence interval for log odds ratio
    log_lower = log_or - z*se
    log_upper = log_or + z*se
    
    # Convert back to odds ratio scale
    return np.exp(log_lower), np.exp(log_upper)

# Example table
a, b, c, d = 7, 3, 2, 8

# Calculate using different methods
unconditional_ci = improved_ci_unconditional(a, b, c, d)
original_unconditional_ci = exact_ci_unconditional(a, b, c, d)
normal_ci = normal_approx_ci(a, b, c, d)
_, fisher_p = stats.fisher_exact([[a, b], [c, d]])

print("Method Comparison:")
print(f"Improved Unconditional: ({unconditional_ci[0]:.4f}, {unconditional_ci[1]:.4f})")
print(f"Original Unconditional: ({original_unconditional_ci[0]:.4f}, {original_unconditional_ci[1]:.4f})")
print(f"Normal Approximation: ({normal_ci[0]:.4f}, {normal_ci[1]:.4f})")
print(f"Fisher's Exact p-value: {fisher_p:.6f}")
```

## Handling Multiple Tables

When processing multiple tables, using caching can significantly improve performance:

```python
from exactcis.utils.optimization import CICache
from exactcis.methods.unconditional import improved_ci_unconditional
import time

# Create a cache instance
cache = CICache(max_size=1000)

# Define a set of tables
tables = [
    (7, 3, 2, 8),
    (8, 2, 3, 7),
    (10, 5, 3, 12),
    (15, 5, 7, 13),
    (20, 10, 10, 20),
    # Add more tables as needed
]

# Without cache
start_time = time.time()
results_no_cache = []
for a, b, c, d in tables:
    ci = improved_ci_unconditional(a, b, c, d, alpha=0.05, use_cache=False)
    results_no_cache.append(ci)
time_no_cache = time.time() - start_time

# With cache
start_time = time.time()
results_with_cache = []
for a, b, c, d in tables:
    ci = improved_ci_unconditional(a, b, c, d, alpha=0.05, cache_instance=cache)
    results_with_cache.append(ci)
time_with_cache = time.time() - start_time

print(f"Time without cache: {time_no_cache:.6f} seconds")
print(f"Time with cache: {time_with_cache:.6f} seconds")
print(f"Speedup: {time_no_cache/time_with_cache:.2f}x")
```

## Customizing Grid Size

For more precision or faster calculation, you can adjust the grid size:

```python
from exactcis.methods.unconditional import improved_ci_unconditional

a, b, c, d = 7, 3, 2, 8

# Default grid size
ci_default = improved_ci_unconditional(a, b, c, d)

# Larger grid for more precision
ci_precise = improved_ci_unconditional(a, b, c, d, grid_size=100)

# Smaller grid for faster calculation
ci_fast = improved_ci_unconditional(a, b, c, d, grid_size=20)

print(f"Default grid (50): ({ci_default[0]:.6f}, {ci_default[1]:.6f})")
print(f"Precise grid (100): ({ci_precise[0]:.6f}, {ci_precise[1]:.6f})")
print(f"Fast grid (20): ({ci_fast[0]:.6f}, {ci_fast[1]:.6f})")
```

## Specifying Custom Bounds

In some cases, particularly with extreme tables, you may want to specify custom bounds:

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example with rare events
a, b, c, d = 1, 1000, 10, 1000

# With auto bounds
try:
    ci_auto = exact_ci_unconditional(a, b, c, d)
    print(f"Auto bounds CI: ({ci_auto[0]:.6f}, {ci_auto[1]:.6f})")
except Exception as e:
    print(f"Auto bounds failed: {e}")

# With custom bounds
ci_custom = exact_ci_unconditional(a, b, c, d, theta_min=0.001, theta_max=1.0)
print(f"Custom bounds CI: ({ci_custom[0]:.6f}, {ci_custom[1]:.6f})")
```

## Error Handling

It's important to handle potential errors, especially for edge cases:

```python
from exactcis.methods.unconditional import improved_ci_unconditional

def safe_ci_calculation(a, b, c, d, alpha=0.05):
    """Safely calculate CI with fallback options"""
    try:
        # Try standard calculation
        lower, upper = improved_ci_unconditional(a, b, c, d, alpha=alpha)
        return lower, upper, "standard"
    except RuntimeError:
        try:
            # Try with larger grid
            lower, upper = improved_ci_unconditional(a, b, c, d, alpha=alpha, grid_size=100)
            return lower, upper, "large_grid"
        except RuntimeError:
            try:
                # Try with custom bounds
                if a*d == 0 or b*c == 0:  # Perfect separation
                    theta_min, theta_max = 0.0001, 10000
                else:
                    or_est = (a*d)/(b*c)
                    theta_min, theta_max = or_est/100, or_est*100
                
                lower, upper = improved_ci_unconditional(
                    a, b, c, d, alpha=alpha, 
                    theta_min=theta_min, theta_max=theta_max
                )
                return lower, upper, "custom_bounds"
            except Exception as e:
                # If all else fails, return None with error message
                return None, None, f"failed: {str(e)}"

# Example tables, including edge cases
tables = [
    (7, 3, 2, 8),      # Standard case
    (0, 10, 5, 5),     # Zero cell
    (100, 0, 0, 100)   # Perfect separation
]

for a, b, c, d in tables:
    lower, upper, method = safe_ci_calculation(a, b, c, d)
    print(f"Table ({a}, {b}, {c}, {d}):")
    if lower is not None:
        print(f"  CI: ({lower:.6f}, {upper:.6f}) [Method: {method}]")
    else:
        print(f"  CI calculation failed: {method}")
```

These examples cover the most common use cases for ExactCIs. For more advanced usage, please refer to the API Reference and Methodology documentation.
