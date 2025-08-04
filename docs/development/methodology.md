# Methodology

## Statistical Foundation

The ExactCIs package implements five different methods for calculating confidence intervals for odds ratios in 2×2 contingency tables: conditional (Fisher's exact), mid-P adjusted, Blaker's exact, unconditional (Barnard's exact), and Wald-Haldane. This document provides a detailed explanation of the underlying methodology, mathematical principles, and implementation details for each method.

## Table of Contents

1. [2×2 Contingency Tables](#2×2-contingency-tables)
2. [Confidence Intervals for Odds Ratios](#confidence-intervals-for-odds-ratios)
3. [Conditional (Fisher's Exact) Method](#conditional-fishers-exact-method)
4. [Mid-P Adjusted Method](#mid-p-adjusted-method) 
5. [Blaker's Exact Method](#blakers-exact-method)
6. [Unconditional (Barnard's) Exact Method](#unconditional-barnards-exact-method)
7. [Wald-Haldane Method](#wald-haldane-method)
8. [Numerical Implementation Considerations](#numerical-implementation-considerations)
9. [Edge Cases and Special Handling](#edge-cases-and-special-handling)

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

## Conditional (Fisher's Exact) Method

### Conceptual Foundation

Fisher's exact test conditions on the observed marginal totals and treats the cell count `a` as following a noncentral hypergeometric distribution:

1. **Conditional Nature**: The test conditions on the observed row and column totals as fixed.

2. **Hypergeometric Distribution**: Given marginals, `a` follows a noncentral hypergeometric distribution with parameter θ (odds ratio).

3. **Confidence Bounds**: Find θ values where P(X ≥ a | θ) = α/2 (lower bound) and P(X ≤ a | θ) = α/2 (upper bound).

### Mathematical Formulation

For a 2×2 table with cell counts `a`, `b`, `c`, `d`:
- Row totals: r₁ = a + b, r₂ = c + d  
- Column totals: c₁ = a + c, c₂ = b + d
- Total: N = a + b + c + d

The support for `a` is: max(0, r₁ - c₂) ≤ a ≤ min(r₁, c₁)

Confidence bounds are found by solving:
- Lower bound: P(X ≥ a | θ, r₁, c₁, N) = α/2
- Upper bound: P(X ≤ a | θ, r₁, c₁, N) = α/2

## Mid-P Adjusted Method

### Conceptual Foundation

The Mid-P method reduces the conservatism of Fisher's exact test by giving half-weight to the probability of the observed table:

1. **Half-Weight Adjustment**: Instead of including the full probability of the observed outcome, includes only half of it in the tail probability calculation.

2. **Reduced Conservatism**: This adjustment typically results in shorter confidence intervals while maintaining the exact calculation framework.

3. **Two-Sided P-Value**: Uses min(lower tail, upper tail) approach with the half-weight adjustment.

### Mathematical Formulation

For observed count `a`, the Mid-P p-value is:
- Lower tail: P(X < a) + 0.5 × P(X = a)  
- Upper tail: P(X > a) + 0.5 × P(X = a)
- Two-sided p-value: 2 × min(lower tail, upper tail)

Confidence bounds are θ values where this p-value equals α.

## Blaker's Exact Method

### Conceptual Foundation

Blaker's method uses acceptability functions to determine which tables are "as extreme or more extreme" than the observed table:

1. **Acceptability Criterion**: A table with count k is included if P(X = k | θ) ≤ P(X = a | θ) × (1 + ε), where ε is a small tolerance.

2. **Plateau Detection**: The method handles flat regions in the p-value function through careful numerical implementation.

3. **Non-Conditional**: Does not require conditioning on marginal totals like Fisher's method.

### Mathematical Formulation

For each θ, the Blaker p-value is:
p-value(θ) = Σ P(X = k | θ) for all k where P(X = k | θ) ≤ P(X = a | θ) × (1 + ε)

Confidence bounds are θ values where p-value(θ) = α.

## Unconditional (Barnard's) Exact Method

### Conceptual Foundation

Barnard's unconditional exact test is the most general exact approach:

1. **Unconditional Nature**: Does not condition on marginal totals, considering all possible tables with the given sample sizes.

2. **Maximizing P-values**: For a given odds ratio (θ), computes p-values over all possible nuisance parameters (p₁, p₂) and takes the maximum.

3. **Most Conservative**: Provides the most conservative exact inference, appropriate for critical applications.

### Mathematical Formulation

For a 2×2 table with cell counts `a`, `b`, `c`, `d`, and odds ratio θ:

1. For each grid point (p₁, p₂) satisfying θ = [p₁(1-p₂)]/[p₂(1-p₁)]:
   - Calculate binomial probabilities for both rows
   - Sum probabilities of tables "as extreme or more extreme"

2. The p-value for θ is the maximum over all valid (p₁, p₂) pairs

3. Confidence bounds are θ values where this maximum p-value equals α/2

## Wald-Haldane Method

### Conceptual Foundation

The Wald-Haldane method provides a simple asymptotic approximation:

1. **Haldane Correction**: Adds 0.5 to each cell to handle zero counts

2. **Log-Normal Approximation**: Uses asymptotic normality of log(OR) 

3. **Closed Form**: Provides immediate results without iterative computation

### Mathematical Formulation

After adding 0.5 to each cell (a', b', c', d'):
- Point estimate: OR̂ = (a' × d')/(b' × c')
- Standard error: SE = √(1/a' + 1/b' + 1/c' + 1/d')  
- Confidence interval: exp(log(OR̂) ± z₁₋α/₂ × SE)

## Numerical Implementation Considerations

The ExactCIs implementation uses several numerical techniques to efficiently and robustly compute confidence intervals across all methods:

### Log-Space Computations

1. **Numerical Stability**: Most probability calculations use log space to avoid overflow/underflow, implemented in functions like `log_nchg_pmf`, `logsumexp`, and binomial coefficient calculations.

2. **Stable Root Finding**: Root-finding algorithms work in log-probability space when possible to maintain precision.

3. **Underflow Protection**: Automatic detection and handling of numerical underflow with conservative fallbacks.

### Root Finding and Optimization

1. **Adaptive Bracketing**: Robust bracket expansion for root-finding across all exact methods, with multiple fallback strategies.

2. **Multiple Solvers**: Primary use of Brent's method with bisection fallbacks for improved reliability.

3. **Convergence Tolerance**: Adaptive precision control based on problem characteristics and numerical stability.

### Caching and Performance Optimization

1. **Multi-Level Caching**: LRU caches for expensive computations like binomial coefficients and PMF values.

2. **Parallel Processing**: Batch processing support across all methods with automatic worker optimization.

3. **JIT Compilation**: Numba acceleration for computationally intensive operations (especially unconditional method).

### Memory Management

1. **Cache Size Control**: Adaptive cache sizing based on available memory and problem complexity.

2. **Cleanup Mechanisms**: Automatic cache clearing for long-running computations to prevent memory issues.

### Error Handling Strategy

1. **Conservative Fallbacks**: When numerical methods fail, all methods return conservative intervals (0, ∞) with appropriate logging.

2. **Validation Layers**: Multi-level input validation with clear error messages for invalid table configurations.

3. **Graceful Degradation**: Automatic fallback to simpler methods when advanced optimizations fail.

### Method-Specific Optimizations

**Conditional Method:**
- Specialized zero-cell handling matching R's fisher.test implementation
- Fisher-Tippett fallback approaches for edge cases

**Mid-P Method:**  
- Efficient support range calculations with offset-based indexing
- Adaptive search range expansion for extreme cases

**Blaker's Method:**
- PMF caching during root-finding iterations to avoid redundant calculations
- Epsilon tolerance handling for plateau detection

**Unconditional Method:**
- Table-size-dependent grid optimization
- MLE-centered adaptive grids for improved convergence
- Early termination based on probability thresholds

**Wald-Haldane Method:**
- Pure Python normal quantile implementation as scipy fallback
- Automatic Haldane correction detection and application

## Edge Cases and Special Handling

The implementation includes comprehensive special handling for various edge cases across all methods:

### Zero Cell Handling

**Conditional Method:**
- Specialized algorithms for each type of zero cell (a=0, b=0, c=0, d=0)
- Fisher-Tippett correction fallbacks when standard approaches fail
- Matches R's fisher.test behavior for consistency

**Mid-P and Blaker Methods:**
- Automatic Haldane correction application for zero cells
- Support range validation to ensure observed counts are feasible
- Conservative fallbacks when support validation fails

**Unconditional Method:**
- Grid point exclusion for invalid probability combinations
- Automatic handling of boundary cases in grid search
- Conservative bounds when numerical issues arise

**Wald-Haldane Method:**
- Automatic 0.5 addition to all cells (Haldane correction)
- No special handling needed due to correction

### Boundary and Extreme Cases

1. **Perfect Separation**: When odds ratio approaches 0 or ∞, methods use appropriate limiting behaviors

2. **Small Samples**: Enhanced precision and validation for tables with very small marginals

3. **Large Tables**: Computational optimizations and approximations for efficiency without sacrificing accuracy

4. **Flat P-Value Functions**: Plateau detection and handling, especially important for Blaker's method

### Numerical Stability Measures

1. **Underflow Protection**: Automatic detection and handling of probability underflow across all methods

2. **Overflow Prevention**: Log-space computations and scaling to prevent numerical overflow

3. **Convergence Issues**: Multiple fallback strategies when root-finding fails to converge

4. **Invalid Bounds**: Validation and correction of crossed or invalid confidence bounds

These comprehensive methodological considerations ensure that ExactCIs provides valid and robust statistical inference across all five implemented methods, handling challenging cases with small sample sizes, rare events, and numerical edge cases gracefully.
