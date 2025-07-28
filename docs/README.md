# ExactCIs Documentation

## Overview

ExactCIs is a Python package for calculating exact confidence intervals for 2×2 contingency tables, with a focus on providing robust and valid statistical inference even in challenging scenarios such as small sample sizes and rare events.

This documentation provides comprehensive information about the package, its methodology, and guidance on when to use different approaches.

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Documentation](#core-documentation)
   - [API Reference](api_reference.md)
   - [Methodology](methodology.md)
   - [Method Comparison Guide](method_comparison.md)
5. [Examples & Tutorials](#examples--tutorials)
   - [Basic Usage](examples/basic_usage.md)
   - [Working with Rare Events](examples/rare_events.md)
   - [Method Selection Guide](examples/method_selection.md)
6. [Performance Considerations](performance.md)
7. [References & Citations](references.md)

## Introduction

ExactCIs implements multiple methods for calculating confidence intervals for odds ratios and relative risks in 2×2 contingency tables, with a primary focus on Barnard's unconditional exact test. This method is particularly valuable for small sample sizes, sparse tables, and situations where the most conservative (but statistically valid) inference is required.

The package offers both the standard implementation and an improved implementation with enhanced performance characteristics through adaptive search strategies and caching.

## Installation

```bash
pip install exactcis
```

## Quick Start

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example 2×2 table
#      Success   Failure
# Grp1    7         3
# Grp2    2         8

# Calculate 95% confidence interval for the odds ratio
lower, upper = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05)
print(f"95% CI for odds ratio: ({lower:.6f}, {upper:.6f})")
```

## Core Documentation

- **[API Reference](api_reference.md)**: Detailed documentation of all functions, classes, and parameters.
- **[Methodology](methodology.md)**: Mathematical foundations and algorithmic details of implemented methods.
- **[Method Comparison Guide](method_comparison.md)**: Comprehensive comparison of different confidence interval methods and when to use each.

## Examples & Tutorials

- **[Basic Usage](examples/basic_usage.md)**: Step-by-step guide to common use cases.
- **[Working with Rare Events](examples/rare_events.md)**: Special considerations for tables with rare events.
- **[Method Selection Guide](examples/method_selection.md)**: Decision tree for selecting the appropriate method.

## Performance Considerations

See [Performance Considerations](performance.md) for detailed information on optimizing computation time and memory usage.

## References & Citations

See [References & Citations](references.md) for academic references and how to cite this package.
