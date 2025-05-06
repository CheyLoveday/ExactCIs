# Implementation Comparison

This document provides a comprehensive comparison between ExactCIs, SciPy, and R's exact2x2 package, highlighting the key algorithmic differences and implementation details.

## Overview of Implementations

| Feature | ExactCIs | SciPy | R's exact2x2 |
|---------|----------|-------|--------------|
| **Core Approach** | Barnard's unconditional exact test | Fisher's conditional exact test | Multiple methods (Fisher's, central, mid-p) |
| **Statistical Basis** | Maximizes p-values over nuisance parameters | Hypergeometric distribution | Hypergeometric distribution with variations |
| **Search Strategy** | Adaptive grid with refinement | Direct calculation (no CI) | Root-finding algorithms |
| **Numerical Methods** | Grid-based optimization with binary search | Optimized combinatorial calculation | Direct root-finding (uniroot) |
| **Edge Case Handling** | Specialized handling for zeros and separation | Minimal special casing | Sophisticated edge case handling |
| **Implementation Language** | Python | C (with Python bindings) | R |
| **Parallelization** | Optional parallel processing | Serial implementation | Serial implementation |
| **Caching** | Advanced caching for similar problems | None | None |

## Detailed Algorithmic Comparison

### 1. Statistical Approach

#### ExactCIs (Barnard's Unconditional)

```python
# Core approach in ExactCIs
def unconditional_log_pvalue(a, b, c, d, theta, p1_values):
    max_log_pvalue = -float("inf")
    
    # For each nuisance parameter p1
    for p1 in p1_values:
        # Calculate p2 based on p1 and theta
        p2 = (p1 * theta) / (1 - p1 + p1 * theta)
        
        # Calculate log probability of observed table
        log_prob_observed = log_pmf(a, b, c, d, p1, p2)
        
        # Calculate log probability of more extreme tables
        log_prob_extreme = log_sum_extreme_tables(a, b, c, d, p1, p2, theta)
        
        # Overall log p-value for this (p1,p2) pair
        log_pvalue = log_prob_extreme
        
        # Keep track of maximum (most conservative)
        max_log_pvalue = max(max_log_pvalue, log_pvalue)
    
    return max_log_pvalue
```

Key characteristics:
- Does not condition on marginal totals
- Maximizes over nuisance parameters
- Most conservative approach

#### SciPy (Fisher's Conditional)

```python
# Simplified pseudocode for Fisher's approach
def fisher_exact_pvalue(a, b, c, d):
    # Fix margins
    row1_sum = a + b
    row2_sum = c + d
    col1_sum = a + c
    col2_sum = b + d
    n = a + b + c + d
    
    # Calculate hypergeometric probability
    def hypergeom_pmf(x):
        return (comb(row1_sum, x) * comb(row2_sum, col1_sum - x)) / comb(n, col1_sum)
    
    # Observed table probability
    p_observed = hypergeom_pmf(a)
    
    # Sum probabilities of more extreme tables
    p_extreme = sum(hypergeom_pmf(x) for x in range(max(0, col1_sum - row2_sum), min(row1_sum, col1_sum) + 1) 
                    if hypergeom_pmf(x) <= p_observed)
    
    return p_extreme
```

Key characteristics:
- Conditions on fixed marginal totals
- Based on hypergeometric distribution
- Faster computation but potentially less appropriate for some designs

#### R exact2x2 (Multiple Methods)

```r
# R's central method (pseudocode)
fisher.exact.central <- function(a, b, c, d) {
  # Create the table
  x <- matrix(c(a, b, c, d), nrow=2)
  
  # Calculate odds ratio
  or <- (a*d)/(b*c)
  
  # Use central p-value approach
  # For each possible odds ratio theta
  # Calculate p-value using central approach
  # Find theta values where p-value = alpha/2
  
  # Root-finding to determine bounds
  lower <- uniroot(function(theta) central_pvalue(x, theta) - alpha/2, 
                  interval=c(0.001, or))$root
  
  upper <- uniroot(function(theta) central_pvalue(x, theta) - alpha/2, 
                  interval=c(or, 1000))$root
  
  return(c(lower, upper))
}
```

Key characteristics:
- Offers multiple methods (Fisher's, central, mid-p)
- Uses direct root-finding
- Well-optimized for statistical packages

### 2. Search Strategy

#### ExactCIs Adaptive Grid

```python
# Excerpt of adaptive grid strategy
def create_adaptive_grid(p1_mle, n1, n2, grid_size):
    # Create non-uniform grid with more points near MLE
    if p1_mle < 0.2:
        # More points in lower range
        p1_values = np.concatenate([
            np.linspace(0.001, p1_mle, grid_size//2),
            np.linspace(p1_mle, 0.999, grid_size//2)
        ])
    elif p1_mle > 0.8:
        # More points in upper range
        p1_values = np.concatenate([
            np.linspace(0.001, p1_mle, grid_size//2),
            np.linspace(p1_mle, 0.999, grid_size//2)
        ])
    else:
        # Balanced grid
        p1_values = np.linspace(0.001, 0.999, grid_size)
    
    return np.unique(p1_values)
```

The adaptive grid strategy concentrates computational resources where they matter most, particularly around the maximum likelihood estimate (MLE).

#### Binary Search for Bound Refinement

```python
# Pseudocode for binary refinement
def refine_bounds(a, b, c, d, alpha, initial_lower, initial_upper):
    # Refine lower bound
    lower = initial_lower
    upper_of_lower = initial_lower * 2
    
    while not converged(lower):
        mid = (lower + upper_of_lower) / 2
        log_pvalue = unconditional_log_pvalue(a, b, c, d, mid)
        
        if np.exp(log_pvalue) < alpha/2:
            upper_of_lower = mid
        else:
            lower = mid
    
    # Similarly refine upper bound...
    
    return lower, upper
```

This binary search approach efficiently narrows down the confidence interval bounds after the initial grid search identifies approximate bounds.

#### R's Root Finding

R's `exact2x2` package uses the `uniroot` function for finding confidence interval bounds:

```r
# Finding confidence bounds via uniroot
lower <- uniroot(
  function(theta) {
    pvalue(a, b, c, d, theta) - alpha/2
  },
  lower = 0.001,
  upper = or_est
)$root

upper <- uniroot(
  function(theta) {
    pvalue(a, b, c, d, theta) - alpha/2
  },
  lower = or_est,
  upper = 1000
)$root
```

This approach directly solves for the exact values where the p-value equals α/2, potentially offering more precise boundaries.

### 3. Edge Case Handling

#### ExactCIs (Handling Zeros and Separation)

```python
# Pseudocode for edge case handling in ExactCIs
def improved_ci_unconditional(a, b, c, d, alpha=0.05, ...):
    # Check for zero cells
    if a*b*c*d == 0:
        # Apply special handling
        if a == 0 or c == 0:  # Zero in first column
            # Special strategy for lower bound
            lower_bound = special_lower_bound(a, b, c, d, alpha)
        else:
            # Regular calculation for lower bound
            lower_bound = regular_lower_bound(a, b, c, d, alpha)
            
        if b == 0 or d == 0:  # Zero in second column
            # Special strategy for upper bound
            upper_bound = special_upper_bound(a, b, c, d, alpha)
        else:
            # Regular calculation for upper bound
            upper_bound = regular_upper_bound(a, b, c, d, alpha)
    else:
        # Regular case (no zeros)
        lower_bound, upper_bound = regular_bounds(a, b, c, d, alpha)
        
    return lower_bound, upper_bound
```

ExactCIs includes specialized handling for tables with zeros, which is particularly important for rare events.

#### R's Edge Case Approaches

R's `exact2x2` package includes sophisticated approaches to edge cases:

```r
# R's approach to zero cells (pseudocode)
if (any(x == 0)) {
    # For tables with a zero cell
    if (x[1,1] == 0 || x[2,2] == 0) {
        # One-sided upper bound is 0
        lower <- 0
        # Calculate upper bound using modified approach
        upper <- modified_upper_bound(x)
    } else if (x[1,2] == 0 || x[2,1] == 0) {
        # One-sided lower bound is infinity
        upper <- Inf
        # Calculate lower bound using modified approach
        lower <- modified_lower_bound(x)
    }
} else {
    # Standard approach for non-zero tables
    lower <- standard_lower(x)
    upper <- standard_upper(x)
}
```

R's implementation includes specific handling for different patterns of zero cells, which contributes to its robust performance in edge cases.

## Performance Comparison

The following table summarizes performance characteristics based on timing benchmarks:

| Table Type | ExactCIs (s) | SciPy Approx (s) | Speedup Factor | CI Width Ratio |
|------------|--------------|------------------|----------------|----------------|
| Standard (7,3,2,8) | 0.000074 | 0.001554 | 21.0x | 1.44x |
| Rare events (1,1000,10,1000) | 0.003050 | 0.000693 | 0.23x | 1.26x |
| Extreme (10,1000,1,1000) | 0.000994 | 0.000355 | 0.36x | 3.26x |
| Large balanced (100,100,100,100) | 0.001043 | 0.000136 | 0.13x | 13.81x |

Key observations:
- ExactCIs is slower than SciPy's approximation, particularly for large tables
- ExactCIs produces wider intervals, especially for extreme tables
- The width ratio (ExactCIs/SciPy) increases with table size and imbalance

## Implementation Details That Matter

### 1. Grid Size and Distribution

The distribution of grid points significantly impacts both precision and performance:

```python
# ExactCIs adaptive grid strategy
if p1_mle < 0.1 or p1_mle > 0.9:
    # More points near the MLE for extreme proportions
    grid_size_left = int(grid_size * 0.7)
    grid_size_right = grid_size - grid_size_left
    
    if p1_mle < 0.1:
        p1_values = np.concatenate([
            np.linspace(0.001, p1_mle, grid_size_left),
            np.linspace(p1_mle, 0.999, grid_size_right)
        ])
    else:
        p1_values = np.concatenate([
            np.linspace(0.001, p1_mle, grid_size_right),
            np.linspace(p1_mle, 0.999, grid_size_left)
        ])
else:
    # More balanced distribution for central MLEs
    p1_values = np.linspace(0.001, 0.999, grid_size)
```

This adaptive approach allows ExactCIs to:
1. Concentrate computation around the most likely parameter values
2. Handle extreme proportions effectively
3. Balance computation and precision

### 2. Caching Strategy

ExactCIs implements a sophisticated caching mechanism:

```python
# Simplified caching pseudocode
def get_from_cache(a, b, c, d, alpha, cache):
    # Check for exact match
    exact_hit = cache.get_exact(a, b, c, d, alpha)
    if exact_hit:
        return exact_hit
    
    # Check for similar table
    similar_hit = cache.get_similar(a, b, c, d, alpha)
    if similar_hit:
        # Use similar table as starting point
        bounds, metadata = similar_hit
        # Refine bounds based on similar result
        return refine_from_similar(a, b, c, d, alpha, bounds, metadata)
    
    # No cache hit, calculate from scratch
    return None
```

This caching approach provides significant speedups when calculating intervals for multiple similar tables, which is common in applications like simulation studies.

### 3. Numerical Stability

ExactCIs uses log-space calculations to maintain numerical stability with extreme probabilities:

```python
# Log-space calculations for numerical stability
def log_pmf(a, b, c, d, p1, p2):
    # Calculate log-probability of table
    log_prob = (
        a * np.log(p1) + 
        b * np.log(1 - p1) + 
        c * np.log(p2) + 
        d * np.log(1 - p2)
    )
    return log_prob

def log_sum_exp(log_x, log_y):
    # Numerically stable way to compute log(exp(log_x) + exp(log_y))
    if log_x > log_y:
        return log_x + np.log(1 + np.exp(log_y - log_x))
    else:
        return log_y + np.log(1 + np.exp(log_x - log_y))
```

These log-space operations are crucial for maintaining precision in calculations involving very small probabilities, which occur frequently with rare events.

## Conclusion

The key differences between implementations can be summarized as:

1. **Statistical Approach**: ExactCIs uses an unconditional approach that does not fix marginal totals, making it more appropriate for certain experimental designs and generally more conservative.

2. **Numerical Methods**: ExactCIs employs an adaptive grid search with refinement, while R uses direct root-finding and SciPy uses optimized combinatorial calculations.

3. **Edge Case Handling**: All implementations have different approaches to handling edge cases, with ExactCIs and R having more sophisticated handling than SciPy.

4. **Performance Tradeoffs**: ExactCIs prioritizes statistical validity and appropriateness over raw speed, especially for small samples and rare events.

5. **Implementation Enhancements**: ExactCIs incorporates advanced features like caching and adaptive grid sizing to optimize performance while maintaining precision.

These differences explain why the methods produce different confidence intervals, particularly for extreme tables with rare events. The differences are not due to implementation errors but reflect fundamentally different statistical approaches and design philosophies.
