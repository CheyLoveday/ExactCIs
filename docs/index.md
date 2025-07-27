# ExactCIs Documentation

Welcome to the ExactCIs documentation. This package provides methods for calculating exact confidence intervals for odds ratios in 2×2 contingency tables.

## Key Features

- **Multiple Exact Methods**: Implements several exact methods (conditional, mid-P, Blaker, unconditional) to handle different statistical needs
- **Robust Edge Case Handling**: Carefully handles zero cells and sparse tables with specialized algorithms
- **Performance Optimizations**: Uses adaptive grid search and caching for efficient computation
- **Timeout Control**: Prevents computational hangs with configurable timeout functionality
- **Clear Method Selection Guidance**: Helps users choose the most appropriate method for their data
- **Comprehensive Documentation**: Includes detailed methodology, examples, and visual aids

## Documentation Structure

- [User Guide](user_guide.md) - Start here for an introduction to ExactCIs
- [API Reference](api_reference.md) - Detailed function and parameter documentation
- [Architecture](architecture.md) - Package design and implementation details
- [Methodology](methodology.md) - Statistical foundations and implementation details
- [Implementation Comparison](implementation_comparison.md) - Comparison with other packages
- [Development Guide](development_guide.md) - For contributors
- [Troubleshooting](troubleshooting.md) - Solutions to common issues

## Examples

- [Quick Start Example](../_temporary/examples/quick_start.ipynb)
- [Method Comparison Example](../_temporary/examples/method_comparison.ipynb)
- [Handling Edge Cases](../_temporary/examples/edge_cases.ipynb)
- [Using Timeout Functionality](../_temporary/examples/timeout_example.ipynb)

## Visual Documentation

- [Package Structure](img/package_structure.md)
- [Data Flow](img/data_flow.md)
- [CI Calculation Process](img/ci_calculation.md)
- [Method Comparison](img/method_comparison_diagram.md)
- [Method Selection Guide](img/method_selection.md)
- [Performance Benchmarks](img/performance_benchmarks.md)

## Quick Installation

```bash
# Basic installation
pip install exactcis

# With NumPy acceleration for better performance
pip install "exactcis[numpy]"

# Development installation (including test dependencies)
pip install "exactcis[dev]"
```

## Quick Example

```python
from exactcis import compute_all_cis

# 2×2 table:   Cases   Controls
#   Exposed      a=12     b=5
#   Unexposed    c=8      d=10

results = compute_all_cis(12, 5, 8, 10, alpha=0.05)
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

## Glossary

- **Odds Ratio (OR)**: A measure of association between exposure and outcome. The ratio of the odds of the outcome in an exposed group to the odds in an unexposed group.
- **Confidence Interval (CI)**: A range of values that is likely to contain the true parameter value with a specified probability.
- **Exact Method**: A statistical method that uses the exact sampling distribution rather than asymptotic approximations.
- **Conditional Method**: An approach that conditions on both margins of the 2×2 table, using the noncentral hypergeometric distribution.
- **Unconditional Method**: An approach that treats row or column totals as random variables rather than fixed.
- **Mid-P Method**: A modification of the exact method that counts only half the probability of the observed outcome, often giving CIs with coverage closer to the nominal level.
- **Coverage Probability**: The probability that the confidence interval contains the true parameter value.
- **2×2 Contingency Table**: A four-cell table representing counts of two binary variables.
