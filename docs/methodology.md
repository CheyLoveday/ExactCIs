# Methodology

## Statistical Foundation

The ExactCIs package primarily implements Barnard's unconditional exact test for calculating confidence intervals for odds ratios in 2×2 contingency tables. This document provides a detailed explanation of the underlying methodology, its implementation, and the mathematical principles behind it.

## Table of Contents

1. [2×2 Contingency Tables](#2×2-contingency-tables)
2. [Confidence Intervals for Odds Ratios](#confidence-intervals-for-odds-ratios)
3. [Barnard's Unconditional Exact Test](#barnards-unconditional-exact-test)
4. [Comparison to Other Methods](#comparison-to-other-methods)
5. [Numerical Implementation](#numerical-implementation)
6. [Edge Cases and Special Handling](#edge-cases-and-special-handling)

## 2×2 Contingency Tables

A 2×2 contingency table represents counts for two binary variables:

```
      Success   Failure   Total
Group1    a        b      a+b
Group2    c        d      c+d
Total    a+c      b+d    a+b+c+d
```

The parameters of interest derived from such tables include:

- **Odds Ratio (OR)**: (a×d)/(b×c)
- **Relative Risk (RR)**: [a/(a+b)]/[c/(c+d)]

## Confidence Intervals for Odds Ratios

Confidence intervals provide a range of plausible values for the true parameter (odds ratio) given the observed data and a specified confidence level (typically 95%, corresponding to α=0.05).

For a confidence level of (1-α)×100%, we find bounds (L, U) such that:
- P(odds ratio < L | observed data) = α/2
- P(odds ratio > U | observed data) = α/2

## Barnard's Unconditional Exact Test

### Conceptual Foundation

Barnard's unconditional exact test is fundamentally different from conditional approaches like Fisher's exact test:

1. **Unconditional Nature**: Unlike Fisher's test which conditions on marginal totals, Barnard's test considers all possible tables with the given sample sizes, without conditioning on row or column totals.

2. **Maximizing P-values**: For a given odds ratio (θ), the test computes p-values over all possible nuisance parameters (p₁, p₂) and takes the maximum, providing the most conservative inference.

3. **Finding Confidence Bounds**: The confidence interval bounds are the values of θ where the maximum p-value equals α/2.

### Mathematical Formulation

For a 2×2 table with cell counts `a`, `b`, `c`, `d`, and an odds ratio θ:

1. For each possible combination of success probabilities p₁ and p₂:
   - Calculate the probability of observing the given table
   - Calculate the probability of observing tables more extreme than the observed one

2. The p-value is the maximum of these probabilities over all (p₁, p₂) pairs that satisfy:
   - θ = [p₁(1-p₂)]/[p₂(1-p₁)]

3. The confidence interval bounds are the values of θ where this maximum p-value equals α/2.

## Comparison to Other Methods

Barnard's unconditional exact test has several distinguishing features compared to other methods:

1. **More Conservative than Fisher's Exact Test**: The unconditional approach generally produces wider confidence intervals than Fisher's conditional approach, especially for small sample sizes.

2. **No Conditioning Assumption**: Unlike Fisher's test, Barnard's test does not assume fixed marginal totals, which can be more appropriate in certain experimental designs.

3. **Computationally Intensive**: The unconditional approach requires maximizing over nuisance parameters, making it more computationally intensive.

## Numerical Implementation

The ExactCIs implementation uses several numerical techniques to efficiently compute confidence intervals:

### Grid-Based Search

1. **Adaptive Grid**: The algorithm uses an adaptive grid for p₁ values, with more points concentrated around the maximum likelihood estimate.

2. **Grid Size Optimization**: Grid size is automatically determined based on table dimensions to balance precision and computational efficiency.

3. **Parallel Processing**: When available, the implementation uses parallel processing for evaluating grid points.

### Bound Finding Algorithm

1. **Initial Bounds**: The algorithm starts with wide initial bounds (often based on normal approximation).

2. **Refining Bounds**: Binary search is used to refine the bounds until the desired precision is achieved.

3. **Caching Strategy**: The improved implementation uses caching to avoid redundant calculations for similar tables.

### Pseudo-code for the Core Algorithm

```
function calculate_CI(a, b, c, d, alpha):
    # Initial bounds
    lower_bound = initial_lower_guess()
    upper_bound = initial_upper_guess()

    # Refine lower bound
    while not converged(lower_bound):
        p_value = max_p_value_over_nuisance_parameters(a, b, c, d, lower_bound)
        if p_value < alpha/2:
            decrease lower_bound
        else:
            increase lower_bound

    # Refine upper bound
    while not converged(upper_bound):
        p_value = max_p_value_over_nuisance_parameters(a, b, c, d, upper_bound)
        if p_value < alpha/2:
            increase upper_bound
        else:
            decrease upper_bound

    return (lower_bound, upper_bound)
```

## Edge Cases and Special Handling

The implementation includes special handling for various edge cases (building on the [Numerical Implementation](#numerical-implementation) techniques described above):

1. **Zero Cells**: When any cell contains zero, special approaches are used to avoid undefined odds ratios.

2. **Large Tables**: For large tables, approximations are used to improve computational efficiency.

3. **Perfect Separation**: Special handling for cases of perfect or quasi-complete separation.

4. **Numerical Stability**: Log-space calculations to maintain numerical stability with very small probabilities.

These methodological considerations ensure that ExactCIs provides valid and robust statistical inference across a wide range of scenarios, including challenging cases with small sample sizes and rare events.
