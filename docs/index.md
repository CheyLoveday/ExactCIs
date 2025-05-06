# ExactCIs Documentation

Welcome to the ExactCIs documentation. This package provides methods for calculating exact confidence intervals for odds ratios in 2×2 contingency tables.

## Documentation Structure

- [User Guide](user_guide.md) - Start here for an introduction to ExactCIs
- [API Reference](api_reference.md) - Detailed function and parameter documentation
- [Architecture](architecture.md) - Package design and implementation details
- [Methodology](methodology.md) - Statistical foundations and implementation details
- [Implementation Comparison](implementation_comparison.md) - Comparison with other packages
- [Development Guide](development_guide.md) - For contributors

## Examples

- [Quick Start Example](../examples/quick_start.ipynb)
- [Method Comparison Example](../examples/method_comparison.ipynb)

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
