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
def compute_all_cis(a, b, c, d, alpha=0.05, grid_size=200)
```

Computes confidence intervals using all available methods.

**Parameters:**
- `a`, `b`, `c`, `d`: Cell counts in the 2×2 table
- `alpha`: Significance level (default: 0.05)
- `grid_size`: Grid size for unconditional method (default: 200)

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
def exact_ci_midp(a, b, c, d, alpha=0.05, progress_callback=None)
```

Computes mid-P adjusted confidence interval.

**Parameters:**
- `a`, `b`, `c`, `d`: Cell counts in the 2×2 table
- `alpha`: Significance level (default: 0.05)
- `progress_callback`: Optional callback function to report progress (0-100)

#### exact_ci_blaker

```python
def exact_ci_blaker(a, b, c, d, alpha=0.05)
```

Computes Blaker's exact confidence interval.

#### exact_ci_unconditional

```python
def exact_ci_unconditional(a, b, c, d, alpha=0.05, **kwargs)
```

Computes Barnard's unconditional exact confidence interval.

**Parameters:**
- `a`, `b`, `c`, `d`: Cell counts in the 2×2 table
- `alpha`: Significance level (default: 0.05)
- `**kwargs`: Additional parameters including:
  - `grid_size`: Size of grid for numerical optimization (default: 15)
  - `theta_min`, `theta_max`: Theta range bounds
  - `custom_range`: Custom range for theta search
  - `theta_factor`: Factor for automatic theta range (default: 100)
  - `haldane`: Apply Haldane's correction (default: False)
  - `timeout`: Optional timeout in seconds
  - `use_cache`: Whether to use caching (default: True)
  - `use_profile`: Use profile likelihood approach (default: False)
  - `progress_callback`: Optional callback function to report progress

#### ci_wald_haldane

```python
def ci_wald_haldane(a, b, c, d, alpha=0.05)
```

Computes Wald-Haldane confidence interval (with 0.5 added to each cell).

### Batch Processing Functions

#### exact_ci_blaker_batch

```python
def exact_ci_blaker_batch(tables, alpha=0.05, max_workers=None, backend=None)
```

Computes Blaker's exact confidence intervals for multiple 2×2 tables in parallel.

**Parameters:**
- `tables`: List of 2×2 tables, each as (a, b, c, d)
- `alpha`: Significance level (default: 0.05)
- `max_workers`: Maximum number of parallel workers (default: None, auto-detected)
- `backend`: Parallelization backend ('thread', 'process', or None for auto)

**Returns:**
- List of (lower, upper) tuples representing the confidence intervals for each table

#### exact_ci_conditional_batch

```python
def exact_ci_conditional_batch(tables, alpha=0.05, max_workers=None, backend=None)
```

Computes Fisher's exact conditional confidence intervals for multiple 2×2 tables in parallel.

**Parameters:**
- `tables`: List of 2×2 tables, each as (a, b, c, d)
- `alpha`: Significance level (default: 0.05)
- `max_workers`: Maximum number of parallel workers (default: None, auto-detected)
- `backend`: Parallelization backend ('thread', 'process', or None for auto)

**Returns:**
- List of (lower, upper) tuples representing the confidence intervals for each table

#### exact_ci_midp_batch

```python
def exact_ci_midp_batch(tables, alpha=0.05, max_workers=None, backend=None)
```

Computes mid-P adjusted confidence intervals for multiple 2×2 tables in parallel.

**Parameters:**
- `tables`: List of 2×2 tables, each as (a, b, c, d)
- `alpha`: Significance level (default: 0.05)
- `max_workers`: Maximum number of parallel workers (default: None, auto-detected)
- `backend`: Parallelization backend ('thread', 'process', or None for auto)

**Returns:**
- List of (lower, upper) tuples representing the confidence intervals for each table

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

## Implementation Details

### Statistical Foundations

Each method is based on specific statistical principles:

1. **Conditional (Fisher's)**: Uses the non-central hypergeometric distribution conditioned on fixed margins.
2. **Mid-P**: Modifies Fisher's method by giving half-weight to the observed table.
3. **Blaker's**: Uses the acceptability function to find the smallest confidence set.
4. **Unconditional**: Maximizes over nuisance parameters (p1, p2) for each fixed odds ratio.
5. **Wald-Haldane**: Uses the normal approximation with continuity correction.

### Numerical Methods

The package employs several numerical techniques:

- **Root finding**: For boundary determination in conditional and mid-P methods
- **Grid search**: For optimization in the unconditional method
- **Caching**: To avoid redundant calculations
- **Parallel processing**: For batch operations

## Performance Considerations

### Computational Complexity

The methods vary significantly in computational cost:

1. **Wald-Haldane**: O(1) - Constant time
2. **Conditional/Mid-P/Blaker**: O(n) - Linear in table size
3. **Unconditional**: O(n²) - Quadratic in grid size

### Memory Usage

Memory usage is generally low except for the unconditional method with large grid sizes.

### Optimization Strategies

Several strategies are employed to improve performance:

1. **Caching**: Memoization of intermediate results
2. **Parallel processing**: For batch operations
3. **Early termination**: For search algorithms
4. **Numerical stability**: Log-space calculations for extreme values

## Examples

### Basic Usage

```python
from exactcis import compute_all_cis

# Define a 2×2 table
a, b, c, d = 12, 5, 8, 10

# Calculate confidence intervals
results = compute_all_cis(a, b, c, d, alpha=0.05)

# Print results
for method, (lower, upper) in results.items():
    print(f"{method:12s} CI: ({lower:.3f}, {upper:.3f})")
```

### Method-Specific Usage

```python
from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_blaker,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Define a 2×2 table
a, b, c, d = 12, 5, 8, 10

# Calculate confidence intervals using specific methods
ci_conditional = exact_ci_conditional(a, b, c, d, alpha=0.05)
ci_midp = exact_ci_midp(a, b, c, d, alpha=0.05)
ci_blaker = exact_ci_blaker(a, b, c, d, alpha=0.05)
ci_unconditional = exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=200)
ci_wald = ci_wald_haldane(a, b, c, d, alpha=0.05)

# Print results
print(f"Conditional:   ({ci_conditional[0]:.3f}, {ci_conditional[1]:.3f})")
print(f"Mid-P:         ({ci_midp[0]:.3f}, {ci_midp[1]:.3f})")
print(f"Blaker:        ({ci_blaker[0]:.3f}, {ci_blaker[1]:.3f})")
print(f"Unconditional: ({ci_unconditional[0]:.3f}, {ci_unconditional[1]:.3f})")
print(f"Wald-Haldane:  ({ci_wald[0]:.3f}, {ci_wald[1]:.3f})")
```

### Batch Processing

```python
from exactcis.methods.conditional import exact_ci_conditional_batch
from exactcis.methods.midp import exact_ci_midp_batch
from exactcis.methods.blaker import exact_ci_blaker_batch

# Define multiple 2×2 tables
tables = [
    (12, 5, 8, 10),
    (7, 3, 2, 8),
    (10, 5, 3, 12)
]

# Calculate confidence intervals for all tables in parallel
conditional_results = exact_ci_conditional_batch(tables, alpha=0.05)
midp_results = exact_ci_midp_batch(tables, alpha=0.05)
blaker_results = exact_ci_blaker_batch(tables, alpha=0.05)

# Print results
for i, ((lower_c, upper_c), (lower_m, upper_m), (lower_b, upper_b)) in enumerate(
    zip(conditional_results, midp_results, blaker_results)
):
    print(f"Table {i+1}:")
    print(f"  Conditional: ({lower_c:.3f}, {upper_c:.3f})")
    print(f"  Mid-P:       ({lower_m:.3f}, {upper_m:.3f})")
    print(f"  Blaker:      ({lower_b:.3f}, {upper_b:.3f})")
```

### Advanced Usage

```python
from exactcis.methods import exact_ci_unconditional

# Define a 2×2 table
a, b, c, d = 12, 5, 8, 10

# Calculate confidence interval with custom parameters
ci = exact_ci_unconditional(
    a, b, c, d,
    alpha=0.05,
    grid_size=300,
    theta_min=0.01,
    theta_max=100,
    use_profile=True
)

print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
```

## Comparison with Other Packages

ExactCIs has been validated against other established packages:

| Package | Language | Methods | Performance | Notes |
|---------|----------|---------|------------|-------|
| **ExactCIs** | Python | 5 methods | Optimized | Comprehensive, parallel batch processing |
| **exact2x2** | R | 4 methods | Good | Well-established, widely used |
| **statsmodels** | Python | 1 method | Basic | Limited to Fisher's exact |
| **scipy.stats** | Python | None | N/A | No CI methods for 2×2 tables |

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for solutions to common issues.

## References

1. Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley.
2. Blaker, H. (2000). Confidence curves and improved exact confidence intervals for discrete distributions. Canadian Journal of Statistics, 28(4), 783-798.
3. Fagerland, M. W., Lydersen, S., & Laake, P. (2015). Recommended confidence intervals for two independent binomial proportions. Statistical Methods in Medical Research, 24(2), 224-254.
4. Fisher, R. A. (1935). The logic of inductive inference. Journal of the Royal Statistical Society, 98(1), 39-82.
5. Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion: comparison of seven methods. Statistics in Medicine, 17(8), 857-872.