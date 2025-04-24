# ExactCIs: Exact Confidence Intervals for Odds Ratios

Provides five methods to compute confidence intervals for the odds ratio of a 2×2 table \(\bigl[\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\bigr]\). It validates inputs, computes exact‑conditional probabilities via the noncentral hypergeometric distribution, and inverts p‑value functions by robust root‑finding. Small, single‑purpose functions are chained by an orchestrator `compute_all_cis`.

## Installation

### Using pip

```bash
pip install exactcis
```

### Optional NumPy acceleration

For faster computation of Barnard's unconditional CI:

```bash
pip install "exactcis[numpy]"
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

---

## Methods & When to Use Them

| Method          | How It Works                                                                                                            | When to Use                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **conditional**<br/>(Fisher) | Inverts the noncentral hypergeometric CDF at \(\alpha/2\) in each tail.                                                  | • Very small samples or rare events<br/>• Regulatory/safety‑critical studies requiring guaranteed ≥ 1–α coverage<br/>• Fixed‑margin designs (case–control) |
| **midp**<br/>(Mid‑P adjusted) | Same inversion but gives half‑weight to the observed table in the tail p‑value, reducing conservatism.                 | • Epidemiology or surveillance where conservative Fisher intervals are too wide<br/>• Moderate samples where slight undercoverage is tolerable for tighter intervals |
| **blaker**<br/>(Blaker's exact) | Uses the acceptability function \(f(k)=\min[P(K≤k),P(K≥k)]\) and inverts it exactly for monotonic, non‑flip intervals. | • Routine exact inference when Fisher is overly conservative<br/>• Fields standardized on Blaker (e.g. genomics, toxicology)<br/>• Exact coverage with minimal over‑coverage |
| **unconditional**<br/>(Barnard's) | Treats both margins as independent binomials, optimizes over nuisance \(p_1\) via grid (or NumPy) search, and inverts the worst‑case p‑value. | • Small clinical trials or pilot studies with unfixed margins<br/>• Need maximum power and narrowest exact CI<br/>• Compute budget allows optimization or vectorized acceleration |
| **wald_haldane**<br/>(Haldane–Anscombe) | Adds 0.5 to each cell and applies the standard log‑OR ± z·SE formula; includes a pure‑Python normal quantile fallback if SciPy is absent. | • Large samples where asymptotic Wald is reasonable<br/>• Quick, approximate intervals for routine reporting<br/>• When speed and convenience outweigh strict exactness |

---

## Example Usage

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

—compare widths and choose the method whose balance of exactness, conservatism, interval length, and computational cost best fits your study.

## Running Tests

The package includes a comprehensive test suite. By default, tests marked as "slow" are skipped:

```bash
uv run pytest -v
```

To run all tests including computationally intensive ones:

```bash
uv run pytest -v --run-slow
```

For more details on testing, see the [test monitoring documentation](docs/test_monitoring.md).
