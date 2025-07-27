# ExactCIs User Guide

## Overview

ExactCIs is a Python package that provides methods for calculating exact confidence intervals for odds ratios in 2×2 contingency tables. It implements five different methods, each with specific statistical properties and use cases.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Format and Input](#data-format-and-input)
4. [Available Methods](#available-methods)
5. [Function Reference](#function-reference)
6. [Method Selection Guide](#method-selection-guide)
7. [Package Architecture](#package-architecture)
8. [Implementation Details](#implementation-details)
9. [Performance Considerations](#performance-considerations)
10. [Examples](#examples)
11. [Comparison with Other Packages](#comparison-with-other-packages)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

## Installation

### Basic Installation

```bash
pip install exactcis
```

### With NumPy Acceleration (Recommended)

```bash
pip install "exactcis[numpy]"
```

NumPy acceleration significantly improves performance for the unconditional method with large grid sizes.

### Development Installation

```bash
git clone https://github.com/yourusername/exactcis.git
cd exactcis
uv pip install -e ".[dev]"
```

## Quick Start

```python
from exactcis import compute_all_cis

# Define a 2×2 table:
#       Cases  Controls
# Exposed    a=12    b=5
# Unexposed  c=8     d=10

# Calculate confidence intervals using all methods
results = compute_all_cis(12, 5, 8, 10, alpha=0.05)

# Print results
for method, (lower, upper) in results.items():
    print(f"{method:12s} CI: ({lower:.3f}, {upper:.3f})")
```

Output:
```
conditional   CI: (1.059, 8.726)
midp          CI: (1.205, 7.893)
blaker        CI: (1.114, 8.312)
unconditional CI: (1.132, 8.204)
wald_haldane  CI: (1.024, 8.658)
```

## Data Format and Input

ExactCIs works with 2×2 contingency tables of the form:

```
      Success   Failure
Group1    a        b
Group2    c        d
```

Where:
- `a` is the number of successes in Group 1
- `b` is the number of failures in Group 1
- `c` is the number of successes in Group 2
- `d` is the number of failures in Group 2

All functions that calculate confidence intervals take these four counts as their first four arguments.

## Available Methods

The package implements five methods for calculating confidence intervals:

1. **Conditional (Fisher's exact)**: Based on the conditional hypergeometric distribution, these intervals are guaranteed to have coverage ≥ 1-α.

2. **Mid-P-adjusted**: A modification of Fisher's exact method that gives half-weight to the observed table, reducing conservatism.

3. **Blaker's exact**: Uses the acceptability function to create non-flip intervals that are typically narrower than Fisher's.

4. **Unconditional (Barnard's)**: Treats both margins as independent binomials and maximizes over nuisance parameters.

5. **Wald-Haldane**: An approximation method that adds 0.5 to each cell and applies the standard log-OR ± z·SE formula.

For a visual comparison of these methods, see the [method comparison diagram](img/method_comparison_diagram.md).

## Function Reference

### Primary Functions

#### compute_all_cis

```python
def compute_all_cis(a, b, c, d, alpha=0.05, grid_size=50, timeout=None)
```

Computes confidence intervals using all available methods.

**Parameters:**
- `a`, `b`, `c`, `d`: Cell counts in the 2×2 table
- `alpha`: Significance level (default: 0.05)
- `grid_size`: Grid size for unconditional method (default: 50)
- `timeout`: Maximum time in seconds for computation (default: None)

**Returns:**
- Dictionary with method names as keys and (lower, upper) tuples as values

### Method-Specific Functions

#### exact_ci_conditional

```python
def exact_ci_conditional(a, b, c, d, alpha=0.05)
```

Computes Fisher's exact conditional confidence interval.

#### exact_ci_midp

```python
def exact_ci_midp(a, b, c, d, alpha=0.05)
```

Computes mid-P adjusted confidence interval.

#### exact_ci_blaker

```python
def exact_ci_blaker(a, b, c, d, alpha=0.05)
```

Computes Blaker's exact confidence interval.

#### exact_ci_unconditional

```python
def exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=50, timeout=None)
```

Computes Barnard's unconditional exact confidence interval.

#### exact_ci_wald_haldane

```python
def exact_ci_wald_haldane(a, b, c, d, alpha=0.05)
```

Computes Wald-Haldane confidence interval (with 0.5 added to each cell).

## Method Selection Guide

| Method | When to Use | Computational Cost | Conservative? |
|--------|-------------|-------------------|---------------|
| **Conditional** | Small samples, regulatory settings, fixed margins | Moderate | Very |
| **Mid-P** | When strict coverage isn't required, epidemiological studies | Moderate | Less |
| **Blaker** | Need exact intervals with minimal over-coverage | Moderate-High | Moderate |
| **Unconditional** | More power needed, unfixed margins | High | Moderate |
| **Wald-Haldane** | Large samples, quick approximations | Very Low | No |

### Performance Benchmarks

For more detailed information about method performance across different sample sizes, see the [performance benchmarks diagram](img/performance_benchmarks.md).

### Method Selection Decision Tree

To help you choose the most appropriate method for your specific use case, we've created a decision tree:

![Method Selection Decision Tree](img/method_selection.md)

The decision tree considers factors such as:
- Sample size
- Presence of zeros or small cell counts
- Need for guaranteed coverage
- Computational constraints
- Study design (fixed margins vs. unfixed margins)

## Package Architecture

ExactCIs is organized into a modular structure that separates core functionality, method implementations, and utilities. This design enables easy extension and maintenance.

### Component Structure

The package consists of the following main components:
- Public API (`__init__.py`): Entry point for users
- Core module (`core.py`): Core statistical functions and algorithms
- Method implementations (`methods/`): Individual CI methods
- Utilities (`utils/`): Support functions

For a detailed visual representation of the package architecture, see the [architecture documentation](architecture.md) and the [package structure diagram](img/package_structure.md).

### Data Flow

The typical data flow through ExactCIs follows these steps:
1. Input validation
2. Method selection
3. p-value function calculation
4. Root finding to determine CI bounds
5. Result compilation

For a visual representation of this process, see the [data flow diagram](img/data_flow.md) and the [CI calculation diagram](img/ci_calculation.md).

## Implementation Details

ExactCIs uses several optimized algorithms:

- **Root finding**: Robust bisection methods to invert p-value functions
- **Grid search**: Adaptive grid search for unconditional method
- **NumPy acceleration**: Vectorized operations for improved performance
- **Numerical stability**: Special handling for edge cases and small counts

### Key Algorithms

The core of ExactCIs relies on several key algorithms:

1. **p-value calculation**: Different for each method
   - Conditional: Uses non-central hypergeometric distribution
   - Mid-P: Modified version of conditional with half-weight for observed table
   - Blaker: Uses acceptability function based on hypergeometric distribution
   - Unconditional: Maximizes p-value over nuisance parameter
   - Wald-Haldane: Uses normal approximation with continuity correction

2. **Root finding**:
   - Bisection method for most CI bounds
   - Log-space search for wide-ranging odds ratios
   - Edge detection for flat p-value regions

3. **Optimizations**:
   - Caching frequently used calculations
   - Early stopping when precision goals are met
   - Timeout protection for long-running calculations

## Performance Considerations

- **Unconditional method** is the most computationally intensive
  - Increasing `grid_size` improves precision but increases computation time
  - NumPy acceleration improves performance dramatically
- **Timeout protection** prevents excessively long computations
- **Caching techniques** minimize redundant calculations

## Examples

### Basic Contingency Table Analysis

```python
from exactcis import compute_all_cis

# Example: Clinical trial with treatment (Group 1) vs. control (Group 2)
# Treatment: 15 responses out of 40 patients
# Control: 7 responses out of 35 patients
results = compute_all_cis(15, 25, 7, 28, alpha=0.05)

print(f"Odds Ratio: {(15*28)/(25*7):.2f}")
for method, (lower, upper) in results.items():
    print(f"{method:12s} CI: ({lower:.3f}, {upper:.3f})")
```

### Working with Zero Cells

```python
from exactcis import compute_all_cis

# Example with a zero cell
# Group 1: 10 successes, 20 failures
# Group 2: 0 successes, 15 failures
results = compute_all_cis(10, 20, 0, 15, alpha=0.05)

# Note: Some methods handle zeros better than others
for method, ci in results.items():
    if ci is None:
        print(f"{method:12s} CI: Method failed for this data")
    else:
        lower, upper = ci
        if upper == float('inf'):
            print(f"{method:12s} CI: ({lower:.3f}, Infinity)")
        else:
            print(f"{method:12s} CI: ({lower:.3f}, {upper:.3f})")
```

### Using Method-Specific Functions

```python
from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_unconditional
)

# Define table
a, b, c, d = 12, 8, 5, 10

# Use specific methods with custom parameters
ci_conditional = exact_ci_conditional(a, b, c, d, alpha=0.01)  # 99% CI
ci_midp = exact_ci_midp(a, b, c, d, alpha=0.05)  # 95% CI
ci_unconditional = exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=100)  # More precise grid

print(f"99% Conditional CI: ({ci_conditional[0]:.3f}, {ci_conditional[1]:.3f})")
print(f"95% Mid-P CI: ({ci_midp[0]:.3f}, {ci_midp[1]:.3f})")
print(f"95% Unconditional CI: ({ci_unconditional[0]:.3f}, {ci_unconditional[1]:.3f})")
```

## Comparison with Other Packages

ExactCIs provides results that are generally consistent with other implementations like R's exact2x2 package, with some differences in numerical approaches:

- **Conditional method** aligns well with R's exact2x2 and SciPy's fisher_exact for most cases
- **Mid-P** implementations can vary slightly between packages due to different handling of the observed table
- **Unconditional method** shows the largest variation between implementations, especially for edge cases

See the [implementation comparison document](implementation_comparison.md) for detailed benchmarks.

## Troubleshooting

### Common Issues

1. **Long computation times**:
   - For unconditional method, decrease `grid_size` or use `timeout`
   - Use NumPy acceleration with `pip install "exactcis[numpy]"`

2. **Memory errors**:
   - Reduce `grid_size` for large tables
   - Use method-specific functions instead of `compute_all_cis`

3. **NaN or Inf results**:
   - Check for zeros or very small counts in your table
   - Some methods handle edge cases better than others

### Error Messages

- `"Error in log-p calculation"`: Numerical issue, try a different method
- `"Timeout reached"`: Computation exceeded timeout limit, increase timeout or use a different method
- `"Invalid table"`: Check that all table entries are non-negative integers

## References

1. Fay MP. (2010). Confidence intervals that match Fisher's exact or Blaker's exact tests. Biostatistics, 11(2):373-374.

2. Barnard GA. (1945). A new test for 2×2 tables. Nature, 156:177.

3. Blaker H. (2000). Confidence curves and improved exact confidence intervals for discrete distributions. Canadian Journal of Statistics, 28:783-798.

4. Agresti A, Coull BA. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. The American Statistician, 52:119-126.

5. Lydersen S, Fagerland MW, Laake P. (2009). Recommended tests for association in 2×2 tables. Statistics in Medicine, 28:1159-1175.
