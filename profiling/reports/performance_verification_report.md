# ExactCIs Conditional Method Performance Verification Report

## Executive Summary

This report investigates the claimed performance improvements for the conditional method optimization in the ExactCIs library. The investigation reveals significant discrepancies between claimed and actual performance metrics, but confirms that the optimizations are correctly implemented and effective.

Key findings:
- The optimizations (memoization, pre-bracketing) are correctly implemented
- The shared cache system works effectively, providing significant speedup after warm-up
- Performance claims in the optimization summary (92% improvement, 5.20ms → 0.42ms) could not be verified
- Actual performance shows ~8-9ms execution time for the very_large test case after warm-up
- Initial execution is much slower (~230ms) due to cache initialization

## Investigation Methodology

1. **Code Analysis**: Examined the implementation of optimizations in conditional.py
2. **Baseline Verification**: Checked timing_results.json for baseline measurements
3. **Direct Measurement**: Ran direct benchmarks to measure current performance
4. **Cache Analysis**: Investigated the shared cache implementation
5. **Scaling Behavior**: Tested performance across different table sizes

## Detailed Findings

### 1. Optimization Implementation

The code correctly implements the claimed optimizations:
- **Memoization**: Uses `@cached_cdf_function` and `@cached_sf_function` decorators
- **Logging Guards**: Includes `logger.isEnabledFor` checks
- **Hoisted Calculations**: Moved fixed calculations outside closures
- **Pre-bracketing**: Implemented 3-point evaluation for better root-finding

### 2. Performance Discrepancies

Multiple inconsistent baseline claims were found:
- **Optimization Summary**: Claims 5.20ms baseline → 0.42ms optimized (92% improvement)
- **Optimization Plan**: States ~9.3ms average execution time baseline
- **timing_results.json**: Shows 19.06ms for the very_large case

Our measurements show:
- **First run**: ~230-236ms (includes cache initialization)
- **Subsequent runs**: ~8-9ms (with warm cache)

### 3. Cache Effectiveness

The shared cache system is highly effective:
- First execution is slow due to cache initialization (~230ms)
- Subsequent executions are much faster (~8-9ms)
- Cache persists between function calls but is reset between benchmarks
- Implementation uses a process-safe shared dictionary with proper locking

### 4. Scaling Behavior

Performance scales well with table size:
- very_small (N=14): 5.30ms
- medium_balanced (N=55): 6.28ms
- large_balanced (N=275): 6.80ms
- very_large (N=750): 8.61ms

This shows approximately linear scaling with table size, which is excellent.

## Explaining the Discrepancies

Several factors could explain the performance discrepancies:

1. **Measurement Methodology**:
   - The claimed 5.20ms baseline may have been measured differently
   - Our measurements include function call overhead
   - The optimization summary may be reporting only the core calculation time

2. **Environment Differences**:
   - Different hardware (CPU, memory)
   - Different Python/NumPy/SciPy versions
   - Different operating systems

3. **Cache Warm-up**:
   - The claimed 0.42ms may be from a fully warmed cache
   - Our measurements reset the cache between benchmarks

4. **Implementation Changes**:
   - The code may have changed since the optimization summary was written
   - Additional features or error handling may have been added

## Conclusions

1. **Optimizations are Effective**: The shared cache and pre-bracketing optimizations work as intended and provide significant performance improvements after warm-up.

2. **Performance Claims are Misleading**: The claimed 92% improvement (5.20ms → 0.42ms) could not be verified. The actual performance is ~8-9ms after warm-up.

3. **Scaling is Excellent**: The method scales well with table size, showing only modest increases in execution time as tables grow larger.

4. **Cache Initialization is Expensive**: The first execution is much slower due to cache initialization, which should be noted in documentation.

## Recommendations

1. **Update Documentation**: Revise performance claims to reflect actual measurements.

2. **Clarify Measurement Methodology**: Document how performance is measured, including warm-up runs and cache state.

3. **Add Cache Pre-warming**: Consider adding an option to pre-warm the cache for batch processing.

4. **Standardize Benchmarking**: Implement a consistent benchmarking methodology across all methods.

5. **Include Environment Information**: Document the environment used for performance measurements.

## Date: August 4, 2025