# ExactCIs: Exact Confidence Intervals for Odds Ratios

Provides five methods to compute confidence intervals for the odds ratio of a 2×2 table \(\bigl[\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\bigr]\). It validates inputs, computes exact‑conditional probabilities via the noncentral hypergeometric distribution, and inverts p‑value functions by robust root‑finding. Small, single‑purpose functions are chained by an orchestrator `compute_all_cis`.

## Documentation

- [User Guide](docs/user_guide.md): Comprehensive guide to using ExactCIs
- [API Reference](docs/api_reference.md): Detailed function and parameter documentation
- [Architecture](docs/architecture.md): Package design, data flow, and implementation details
- [Examples](examples/): Jupyter notebooks with practical examples
- [Method Comparison](docs/method_comparison.md): Detailed analysis of different CI methods
- [Performance](docs/performance.md): Performance considerations and optimization

## Installation

### Using pip

```bash
pip install exactcis
```

### Optional dependencies

```bash
# For data manipulation with pandas
pip install "exactcis[pandas]"

# For symbolic mathematics
pip install "exactcis[sympy]"

# For visualization
pip install "exactcis[viz]"

# For all optional dependencies
pip install "exactcis[full]"
```

### Development installation

```bash
# Clone the repository
git clone https://github.com/yourusername/exactcis.git
cd exactcis

# Install with development dependencies using uv
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run all tests including slow ones
uv run pytest --run-slow
```

## Command Line Interface

ExactCIs provides a command-line interface for quick calculations:

```bash
# Basic usage
exactcis-cli 12 5 8 10 --method blaker

# With additional options
exactcis-cli 12 5 8 10 --method unconditional --alpha 0.01 --grid-size 30 --apply-haldane --verbose
```

For help and all available options:

```bash
exactcis-cli --help
```

---

## Methods & When to Use Them

| Method          | How It Works                                                                                                            | When to Use                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **conditional**<br/>(Fisher) | Inverts the noncentral hypergeometric CDF at \(\alpha/2\) in each tail.                                                  | • Very small samples or rare events<br/>• Regulatory/safety‑critical studies requiring guaranteed ≥ 1–α coverage<br/>• Fixed‑margin designs (case–control) |
| **midp**<br/>(Mid‑P adjusted) | Same inversion but gives half‑weight to the observed table in the tail p‑value, reducing conservatism.                 | • Epidemiology or surveillance where conservative Fisher intervals are too wide<br/>• Moderate samples where slight undercoverage is tolerable for tighter intervals |
| **blaker**<br/>(Blaker's exact) | Uses the acceptability function \(f(k)=\min[P(K≤k),P(K≥k)]\) and inverts it exactly for monotonic, non‑flip intervals. | • Routine exact inference when Fisher is overly conservative<br/>• Fields standardized on Blaker (e.g. genomics, toxicology)<br/>• Exact coverage with minimal over‑coverage |
| **unconditional**<br/>(Barnard's) | Treats both margins as independent binomials, optimizes over nuisance \(p_1\) via grid (or NumPy) search, and inverts the worst‑case p‑value. | • Small clinical trials or pilot studies with unfixed margins<br/>• Need maximum power and narrowest exact CI<br/>• Compute budget allows optimization or vectorized acceleration |
| **wald_haldane**<br/>(Haldane–Anscombe) | Adds 0.5 to each cell and applies the standard log‑OR ± z·SE formula; includes a pure‑Python normal quantile fallback if SciPy is absent. | • Large samples where asymptotic Wald is reasonable<br/>• Quick, approximate intervals for routine reporting<br/>• When speed and convenience outweigh strict exactness |

For a detailed guide on selecting the appropriate method, see the [Method Selection Guide](docs/img/method_selection.md).

---

## Quick Start Examples

```python
from exactcis import compute_all_cis

# 2×2 table:   Cases   Controls
#   Exposed      a=12     b=5
#   Unexposed    c=8      d=10

results = compute_all_cis(12, 5, 8, 10, alpha=0.05, grid_size=500)
for method, (lo, hi) in results.items():
    print(f"{method:12s} CI: ({lo:.3f}, {hi:.3f})")
```

This prints:

```
conditional  CI: (1.059, 8.726)
midp         CI: (1.205, 7.893)
blaker       CI: (1.114, 8.312)
unconditional CI: (1.132, 8.204)
wald_haldane CI: (1.024, 8.658)
```

For more detailed examples, see the [Quick Start Notebook](examples/quick_start.ipynb) and [Method Comparison Notebook](examples/method_comparison.ipynb).

## Method-Specific Usage

You can also use each method individually:

```python
from exactcis.methods import (
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_blaker,
    exact_ci_unconditional,
    ci_wald_haldane
)

# Example: Compute a 99% confidence interval using Fisher's method
lower, upper = exact_ci_conditional(12, 5, 8, 10, alpha=0.01)
print(f"99% CI: ({lower:.3f}, {upper:.3f})")
```

## Timeout Functionality

The `exact_ci_unconditional` method (Barnard's) includes timeout protection to prevent long-running calculations:

```python
from exactcis.methods import exact_ci_unconditional

# With a 30-second timeout
result = exact_ci_unconditional(12, 5, 8, 10, alpha=0.05, timeout=30)

# If the calculation exceeds the timeout, it will return None or raise an exception
# depending on the context
```

This timeout feature is particularly useful for large or imbalanced tables that might otherwise cause 
computationally intensive calculations to run indefinitely.

## Running Tests

The package includes a comprehensive test suite. By default, tests marked as "slow" are skipped:

```bash
uv run pytest -v
```

To run all tests, including slow ones:

```bash
uv run pytest --run-slow -v
```

## Performance Considerations

For optimal performance with the unconditional method:
- Install with NumPy acceleration: `pip install "exactcis[numpy]"`
- Use appropriate grid sizes based on your precision requirements
- Set reasonable timeouts for large or imbalanced tables

For detailed performance information, see the [Performance Documentation](docs/performance.md).

## Comparison with Other Implementations

ExactCIs has been benchmarked against:
- R's exact2x2 package
- SciPy's fisher_exact function
- StatXact

For benchmark results, see the [Implementation Comparison](docs/implementation_comparison.md).

## Citation

If you use ExactCIs in your research, please cite:

```
ExactCIs: A Python package for exact confidence intervals for odds ratios.
https://github.com/yourusername/exactcis
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
