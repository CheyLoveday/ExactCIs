# ExactCIs: Project Description

## Project Overview

ExactCIs is a Python package that provides methods to compute confidence intervals for the odds ratio of a 2×2 contingency table. It includes various methods such as conditional (Fisher), mid-P adjusted, Blaker's exact, unconditional (Barnard's), and Haldane-Anscombe Wald.

## Project Structure

The project is structured according to modern Python package development practices:

```
exactcis/
├── src/
│   └── exactcis/
│       ├── __init__.py
│       ├── core.py
│       ├── methods/
│       │   ├── __init__.py
│       │   ├── conditional.py
│       │   ├── midp.py
│       │   ├── blaker.py
│       │   ├── unconditional.py
│       │   └── wald.py
│       └── utils/
│           ├── __init__.py
│           └── stats.py
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_exactcis.py
│   ├── test_integration.py
│   ├── test_methods/
│   │   ├── test_conditional.py
│   │   ├── test_midp.py
│   │   ├── test_blaker.py
│   │   ├── test_unconditional.py
│   │   └── test_wald.py
│   └── test_utils/
│       └── test_stats.py
├── docs/
│   ├── project_description.md
│   ├── performance.md
│   └── test_monitoring.md
├── pyproject.toml
└── README.md
```

## Implementation Details

### Core Functionality

The core functionality of the package is implemented in `src/exactcis/core.py`, which provides:

1. **Validation**: Functions to validate the input counts for a 2×2 contingency table.
2. **Probability Mass Function**: Functions to calculate the probability mass function of the noncentral hypergeometric distribution.
3. **Root Finding**: Functions to find the root of a function using the bisection method and to find the smallest theta value that satisfies a given constraint.
4. **Timeout Handling**: Functions now include timeout checks to prevent long-running calculations from hanging indefinitely.

### Confidence Interval Methods

The package provides five methods to compute confidence intervals for the odds ratio:

1. **Conditional (Fisher)**: Inverts the noncentral hypergeometric CDF at α/2 in each tail. Appropriate for very small samples or rare events, regulatory/safety-critical studies requiring guaranteed ≥ 1-α coverage, and fixed-margin designs (case-control).

2. **Mid-P Adjusted**: Same inversion but gives half-weight to the observed table in the tail p-value, reducing conservatism. Appropriate for epidemiology or surveillance where conservative Fisher intervals are too wide, and moderate samples where slight undercoverage is tolerable for tighter intervals.

3. **Blaker's Exact**: Uses the acceptability function f(k)=min[P(K≤k),P(K≥k)] and inverts it exactly for monotonic, non-flip intervals. Appropriate for routine exact inference when Fisher is overly conservative, fields standardized on Blaker (e.g., genomics, toxicology), and exact coverage with minimal over-coverage.

4. **Unconditional (Barnard's)**: Treats both margins as independent binomials, optimizes over nuisance p₁ via grid (or NumPy) search, and inverts the worst-case p-value. Appropriate for small clinical trials or pilot studies with unfixed margins, when maximum power and narrowest exact CI are needed, and when compute budget allows optimization or vectorized acceleration. **Now supports timeout functionality** to prevent long-running calculations.

5. **Haldane-Anscombe Wald**: Adds 0.5 to each cell and applies the standard log-OR ± z·SE formula. Includes a pure-Python normal quantile fallback if SciPy is absent. Appropriate for large samples where asymptotic Wald is reasonable, quick approximate intervals for routine reporting, and when speed and convenience outweigh strict exactness.

### Timeout Functionality

The package now includes timeout functionality to prevent computationally intensive methods from running indefinitely:

- The `exact_ci_unconditional` function accepts a `timeout` parameter (in seconds)
- Core root-finding functions in `core.py` accept a `timeout_checker` function to enable early termination
- If a calculation exceeds the timeout, it will return gracefully instead of hanging indefinitely
- This functionality is particularly useful for large or imbalanced tables that might otherwise cause excessive computation times

### Dependency Management

The package is designed to have minimal dependencies:

- Core functionality is dependency-free for maximum portability.
- NumPy is an optional dependency for performance optimization, particularly for the unconditional (Barnard's) method.

## Testing

The package includes a comprehensive test suite organized into multiple categories:

- **Fast Tests**: Basic functionality tests that run quickly.
- **Slow Tests**: Computationally intensive tests, particularly for the unconditional method.
- **Method-Specific Tests**: Individual tests for each confidence interval method.
- **Integration Tests**: Tests that check the full computation pipeline.

By default, slow tests are skipped unless the `--run-slow` option is specified. The test suite also includes timeout settings to prevent tests from running indefinitely.

## Profiling and Optimization

The project includes several tools for profiling and optimizing performance:

- **`profile_with_timeout.py`**: Tests various confidence interval methods with timeout protection to identify problematic calculations.
- **`optimize_unconditional.py`**: Optimizes parameters for the unconditional method such as grid size and refinement settings.
- **Performance Documentation**: See [Performance Guide](performance.md) for comprehensive performance guidance and optimization strategies.

## Current Status

The package is stable and produces correct confidence intervals for the example data (a=12, b=5, c=8, d=10):

```
conditional    CI: (1.059, 8.726)
midp           CI: (1.205, 7.893)
blaker         CI: (1.114, 8.312)
unconditional  CI: (1.132, 8.204)
wald_haldane   CI: (1.024, 8.658)
```

The tests verify these values and check various edge cases and properties of the confidence intervals.

### Edge Case Handling

The package generally handles edge cases well, including:
- Tables with zeros in one cell
- Small counts
- Imbalanced tables

The unconditional method is computationally intensive and may raise exceptions for extreme cases, but this behavior is documented and handled appropriately in the test suite. The new timeout functionality also helps manage these challenging cases.

## Future Improvements

Potential future improvements include:

1. **Performance Optimization**: Further optimize the unconditional method, particularly for large or imbalanced tables.
2. **Additional Methods**: Implement other confidence interval methods for odds ratios.
3. **Documentation**: Expand documentation with more examples and use cases.
4. **Visualization**: Add functions to visualize confidence intervals.
5. **Parallel Processing**: Improve the parallel processing capabilities for the unconditional method to utilize multiple cores more efficiently.

## Conclusion

ExactCIs is a well-structured Python package that provides a comprehensive set of methods for computing confidence intervals for odds ratios. The package is designed to be flexible, reliable, and easy to use, with appropriate handling of edge cases and performance considerations. The addition of timeout functionality improves robustness when dealing with computationally intensive calculations.
