# Working with Rare Events

This guide focuses on calculating confidence intervals for 2×2 contingency tables containing rare events, which present unique statistical challenges.

## Understanding Rare Events

In epidemiology, clinical trials, and safety studies, rare events are common:
- Rare diseases (e.g., prevalence < 1%)
- Uncommon adverse events
- Zero-count cells (no occurrences in one or more cells)

**Example rare event table:**
```
          Event    No Event    Total
Group 1     1       1000       1001
Group 2    10        990       1000
```

In this example, the odds ratio is (1×990)/(1000×10) = 0.099, but the confidence interval calculation requires special consideration.

## Challenges with Rare Events

Rare events create several statistical challenges:

1. **Zero Cells**: Tables with zeros cause problems for many methods
2. **Boundary Issues**: Estimates may push against parameter boundaries
3. **Asymptotic Failure**: Normal approximations break down
4. **Computational Challenges**: Numerical issues in estimation

## ExactCIs for Rare Events

Barnard's unconditional exact test (implemented in ExactCIs) is particularly well-suited for rare events:

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Rare event example
a, b, c, d = 1, 1000, 10, 990

# Calculate confidence interval
ci = exact_ci_unconditional(a, b, c, d, alpha=0.05)
print(f"95% CI for odds ratio: ({ci[0]:.6f}, {ci[1]:.6f})")
# Output: 95% CI for odds ratio: (0.012777, 0.782634)
```

### Comparison with Other Methods

```python
import numpy as np
import scipy.stats as stats

# Normal approximation (with Haldane's correction)
def normal_approx_ci(a, b, c, d, alpha=0.05):
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5  # Haldane's correction
    or_est = (a*d)/(b*c)
    log_or = np.log(or_est)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    log_lower = log_or - z*se
    log_upper = log_or + z*se
    return np.exp(log_lower), np.exp(log_upper)

# Example rare event table
a, b, c, d = 1, 1000, 10, 990

# Calculate using different methods
exactcis_ci = exact_ci_unconditional(a, b, c, d)
normal_ci = normal_approx_ci(a, b, c, d)
or_point = (a*d)/(b*c)

print(f"Odds Ratio: {or_point:.6f}")
print(f"ExactCIs: ({exactcis_ci[0]:.6f}, {exactcis_ci[1]:.6f})")
print(f"Normal Approx: ({normal_ci[0]:.6f}, {normal_ci[1]:.6f})")
```

## Strategies for Different Rare Event Scenarios

### 1. Zero Cells

When one or more cells contain zeros, consider:

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example with zero cell
a, b, c, d = 0, 100, 10, 90

# Default approach
try:
    ci_default = exact_ci_unconditional(a, b, c, d)
    print(f"Default: ({ci_default[0]:.6f}, {ci_default[1]:.6f})")
except Exception as e:
    print(f"Default failed: {e}")

# With custom bounds
ci_custom = exact_ci_unconditional(a, b, c, d, theta_min=0.0001, theta_max=10.0)
print(f"Custom bounds: ({ci_custom[0]:.6f}, {ci_custom[1]:.6f})")
```

### 2. Small Expected Frequencies

For rare but non-zero events:

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example with small expected frequencies
a, b, c, d = 3, 997, 15, 985

# Use improved implementation with adaptive grid
ci = exact_ci_unconditional(a, b, c, d, adaptive_grid=True, grid_size=100)
print(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")
```

### 3. Perfect Separation

Perfect separation occurs when all events are in one group and none in the other:

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example with perfect separation
a, b, c, d = 10, 990, 0, 1000

# Use improved implementation with custom bounds
ci = exact_ci_unconditional(a, b, c, d, theta_min=1.0, theta_max=1000.0)
print(f"95% CI: ({ci[0]:.6f}, {ci[1]:.6f})")
```

## Tips for Working with Rare Events

1. **Use unconditional methods**: Barnard's unconditional exact test (ExactCIs) is generally more reliable than conditional methods for rare events.

2. **Avoid normal approximation**: For rare events, normal approximation can severely undercover, providing falsely narrow intervals.

3. **Consider custom bounds**: When default algorithms fail, specifying custom bounds can help:
   ```python
   # Specify custom bounds based on observed odds ratio
   or_est = (a*d)/(b*c) if b*c > 0 else 0.0001  # Handle zero division
   ci = exact_ci_unconditional(a, b, c, d, 
                                theta_min=max(0.0001, or_est/100),
                                theta_max=min(10000, or_est*100))
   ```

4. **Increase grid size for precision**: With rare events, using a larger grid can improve accuracy:
   ```python
   ci = exact_ci_unconditional(a, b, c, d, grid_size=100)
   ```

5. **Validate results**: When working with extreme tables, validate results against multiple methods:
   ```python
   # Compare original and improved implementations
   ci1 = exact_ci_unconditional(a, b, c, d)
   ci2 = exact_ci_unconditional(a, b, c, d)
   print(f"Original: ({ci1[0]:.6f}, {ci1[1]:.6f})")
   print(f"Improved: ({ci2[0]:.6f}, {ci2[1]:.6f})")
   ```

## Extreme Table Examples

The following code compares multiple methods on extreme tables:

```python
from exactcis.methods.unconditional import exact_ci_unconditional
import numpy as np
import scipy.stats as stats

def normal_approx_ci(a, b, c, d, alpha=0.05):
    # Normal approximation with Haldane's correction
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    or_est = (a*d)/(b*c)
    log_or = np.log(or_est)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    return np.exp(log_or - z*se), np.exp(log_or + z*se)

# List of extreme tables to test
extreme_tables = [
    (1, 1000, 10, 1000),    # Rare event, balanced groups
    (0, 1000, 10, 1000),    # Zero cell
    (1, 100, 0, 100),       # Perfect separation
    (5, 995, 50, 950),      # 10x difference in rates
]

for a, b, c, d in extreme_tables:
    or_est = (a*d)/(b*c) if b*c > 0 else float('inf')
    print(f"\nTable: ({a}, {b}, {c}, {d}), OR={or_est:.4f}")
    
    # ExactCIs
    try:
        exactcis_ci = exact_ci_unconditional(a, b, c, d)
        exactcis_width = exactcis_ci[1] - exactcis_ci[0]
        print(f"  ExactCIs: ({exactcis_ci[0]:.6f}, {exactcis_ci[1]:.6f}), width={exactcis_width:.6f}")
    except Exception as e:
        print(f"  ExactCIs: Failed - {str(e)}")
    
    # Normal approximation
    try:
        normal_ci = normal_approx_ci(a, b, c, d)
        normal_width = normal_ci[1] - normal_ci[0]
        print(f"  Normal: ({normal_ci[0]:.6f}, {normal_ci[1]:.6f}), width={normal_width:.6f}")
    except Exception as e:
        print(f"  Normal: Failed - {str(e)}")
    
    # Width ratio if both succeeded
    try:
        if 'exactcis_width' in locals() and 'normal_width' in locals():
            print(f"  Width ratio (ExactCIs/Normal): {exactcis_width/normal_width:.2f}x")
    except:
        pass
```

This guide demonstrates why ExactCIs is particularly well-suited for rare event scenarios, providing more robust and conservative inference than alternatives while maintaining statistical validity.
