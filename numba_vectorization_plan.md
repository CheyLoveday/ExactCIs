# Numba Vectorization Implementation Plan for ExactCIs

## Overview

This document outlines a plan to extend Numba JIT compilation across the ExactCIs project to improve performance through vectorization. Currently, Numba is only used in the `unconditional.py` module, but there are significant opportunities to apply similar optimizations to other computationally intensive parts of the codebase.

## Current Numba Implementation

The existing implementation in `unconditional.py` provides a good template:

1. Graceful fallbacks when Numba is not available
2. JIT compilation with `nopython=True` and `cache=True` for optimal performance
3. Pattern of core JIT-compiled functions with wrapper functions for interface handling

## Vectorization Opportunities by Module

### 1. Core Module (`core.py`)

#### Mathematical Operations
- `logsumexp`: Vectorize summation of exponentials in log space
- `log_binom_coeff`: JIT-compile binomial coefficient calculation

#### Probability Calculations
- `log_nchg_pmf`: Vectorize noncentral hypergeometric PMF calculation
- `pmf_weights`: JIT-compile weight calculations
- `nchg_pdf`: Vectorize PDF calculations

#### Optimization Functions
- `find_root_log`: JIT-compile root finding algorithm
- `find_smallest_theta`: Vectorize theta search operations
- `find_plateau_edge`: JIT-compile plateau detection

### 2. Conditional Module (`methods/conditional.py`)

- `_cdf_cached` and `_sf_cached`: JIT-compile CDF/SF calculations
- `fisher_lower_bound` and `fisher_upper_bound`: Vectorize bound calculations
- `_find_better_bracket`: JIT-compile bracket optimization

### 3. Blaker Module (`methods/blaker.py`)

- `blaker_acceptability`: Vectorize acceptability calculations
- `blaker_p_value`: JIT-compile p-value calculations
- `BlakerPMFCache.get_pmf_values`: Optimize cache operations with Numba

### 4. PMF Functions (`utils/pmf_functions.py`)

- `calculate_log_binomial_terms`: Vectorize binomial term calculations
- `normalize_weights_numerically_stable`: JIT-compile normalization
- `pmf_weights_normal_case`: Vectorize weight calculations

## Implementation Approach

### Phase 1: Core Mathematical Functions

1. Add Numba import handling to each module (following `unconditional.py` pattern)
2. Implement JIT-compiled versions of core mathematical functions:
   - `logsumexp_numba`
   - `log_binom_coeff_numba`
   - `log_nchg_pmf_numba`

### Phase 2: Probability Distribution Functions

1. Implement JIT-compiled versions of probability functions:
   - `pmf_weights_numba`
   - `nchg_pdf_numba`
   - `_cdf_cached_numba` and `_sf_cached_numba`

### Phase 3: Optimization Algorithms

1. Implement JIT-compiled versions of optimization functions:
   - `find_root_log_numba`
   - `find_smallest_theta_numba`
   - `find_plateau_edge_numba`

### Phase 4: Method-Specific Functions

1. Implement JIT-compiled versions of method-specific functions:
   - `fisher_lower_bound_numba` and `fisher_upper_bound_numba`
   - `blaker_acceptability_numba` and `blaker_p_value_numba`

## Implementation Pattern

For each function to be vectorized:

1. Create a JIT-compiled version with the `_numba` suffix
2. Use the pattern:
   ```python
   if has_numba:
       @jit(nopython=True, cache=True)
       def function_name_numba(...):
           # JIT-compiled implementation
   else:
       # Fallback implementation or alias to original
       function_name_numba = function_name
   ```
3. Modify the original function to use the JIT-compiled version when appropriate:
   ```python
   def function_name(...):
       if has_numba and <condition for using numba>:
           return function_name_numba(...)
       else:
           # Original implementation
   ```

## Testing Strategy

1. Create unit tests comparing results from original and JIT-compiled functions
2. Implement performance benchmarks to measure speedup
3. Test with and without Numba available to ensure fallbacks work correctly

## Performance Monitoring

1. Add timing instrumentation to measure impact of JIT compilation
2. Create benchmarks for key operations before and after vectorization
3. Document performance improvements in the project documentation

## Prioritization

Based on the analysis, the following functions should be prioritized for vectorization due to their computational intensity and frequency of use:

1. `logsumexp` and `log_binom_coeff` (core mathematical operations)
2. `pmf_weights` and related functions (probability calculations)
3. `find_smallest_theta` and `find_plateau_edge` (optimization algorithms)
4. `blaker_acceptability` and `blaker_p_value` (method-specific calculations)

## Conclusion

Extending Numba vectorization across the ExactCIs project will significantly improve performance, especially for large tables and batch operations. The existing implementation in `unconditional.py` provides a solid template to follow, and the modular structure of the codebase makes it well-suited for incremental vectorization.