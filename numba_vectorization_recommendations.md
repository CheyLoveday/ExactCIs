# Numba Vectorization Recommendations for ExactCIs

## Executive Summary

After a thorough review of the ExactCIs codebase, we've identified significant opportunities to improve performance through Numba vectorization. Currently, Numba is only used in the `unconditional.py` module, but extending this approach to other computationally intensive parts of the codebase could yield substantial performance improvements, especially for large tables and batch operations.

## Key Findings

1. **Limited Current Usage**: Numba is currently only used in `unconditional.py` for a few specific functions.
2. **Consistent Implementation Pattern**: The existing implementation provides a good template with graceful fallbacks.
3. **Many Vectorization Opportunities**: Several computationally intensive functions across multiple modules could benefit from JIT compilation.

## High-Priority Vectorization Targets

### Core Mathematical Operations

| Function | Module | Reason for Prioritization |
|----------|--------|---------------------------|
| `logsumexp` | `core.py` | Used extensively throughout the codebase for numerical stability in probability calculations |
| `log_binom_coeff` | `core.py` | Fundamental operation for many statistical calculations |
| `log_nchg_pmf` | `core.py` | Core probability calculation used in confidence interval methods |

### Probability Calculations

| Function | Module | Reason for Prioritization |
|----------|--------|---------------------------|
| `pmf_weights` | `core.py` | Intensive calculation used repeatedly during optimization |
| `nchg_pdf` | `core.py` | Vectorizable PDF calculation used in multiple methods |
| `_cdf_cached` & `_sf_cached` | `conditional.py` | Frequently called during bound calculations |

### Optimization Algorithms

| Function | Module | Reason for Prioritization |
|----------|--------|---------------------------|
| `find_root_log` | `core.py` | Complex algorithm with nested loops that would benefit from JIT |
| `find_smallest_theta` | `core.py` | Intensive search operation used in confidence interval calculations |
| `find_plateau_edge` | `core.py` | Contains loops that could be optimized with Numba |

### Method-Specific Functions

| Function | Module | Reason for Prioritization |
|----------|--------|---------------------------|
| `fisher_lower_bound` & `fisher_upper_bound` | `conditional.py` | Complex bound calculations with nested operations |
| `blaker_acceptability` & `blaker_p_value` | `blaker.py` | Intensive probability calculations that are called repeatedly |
| `BlakerPMFCache.get_pmf_values` | `blaker.py` | Cache operations that could be optimized |

## Implementation Recommendations

1. **Follow Existing Pattern**: Use the implementation pattern from `unconditional.py` with graceful fallbacks.
2. **Phased Approach**: Implement vectorization in phases, starting with core mathematical operations.
3. **Consistent Naming**: Use the `_numba` suffix for JIT-compiled versions of functions.
4. **Thorough Testing**: Ensure numerical equivalence between original and vectorized implementations.
5. **Performance Benchmarking**: Measure and document performance improvements.

## Expected Benefits

1. **Performance Improvements**: Significant speedup for computationally intensive operations.
2. **Scalability**: Better handling of large tables and batch operations.
3. **Resource Efficiency**: Reduced CPU usage and memory consumption.
4. **Consistency**: More uniform performance characteristics across different methods.

## Implementation Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Numba compatibility issues | Implement robust fallback mechanisms |
| Numerical precision differences | Thorough testing comparing results with original implementations |
| Maintenance complexity | Clear documentation and consistent implementation patterns |
| Learning curve for contributors | Provide examples and documentation for Numba usage |

## Conclusion

Extending Numba vectorization across the ExactCIs project represents a high-value opportunity for performance improvement with relatively low implementation risk. The existing codebase structure is well-suited for incremental vectorization, and the current implementation in `unconditional.py` provides a solid template to follow.

We recommend proceeding with the implementation plan outlined in the accompanying document, starting with the highest-priority functions identified in this analysis.