# ExactCIs Comprehensive Profiling Report

## Executive Summary

This report presents the findings from a comprehensive profiling analysis of the ExactCIs package. The analysis identified several performance bottlenecks and evaluated the effectiveness of implemented optimizations.

### Key Findings:

1. **Performance Ranking**: Methods range from microseconds to hundreds of milliseconds in execution time:
   - Blaker's Method: Slowest (18.8ms average, up to 270ms for large tables)
   - Conditional Method: Moderate (5.2ms average)
   - Unconditional Method: Fast (34μs average)
   - Mid-P Method: Very fast (12μs average)
   - Wald-Haldane Method: Extremely fast (2μs average)

2. **Critical Bottlenecks**:
   - Redundant PMF calculations in Blaker method
   - Inefficient root-finding algorithms
   - Lack of effective caching for repeated calculations
   - Suboptimal parallel processing implementation

3. **Optimization Effectiveness**:
   - Blaker Method: 69.7% reduction in execution time
   - Conditional Method: Significant improvement, though actual measurements (8-9ms) differ from claimed (0.42ms)
   - Vectorized Bracket Expansion: Modest 4.79% overall improvement with mixed results
   - Shared Inter-Process Caching: Limited performance impact due to implementation challenges

## Detailed Findings

### 1. Method Performance Analysis

#### Blaker's Method
- **Baseline**: 18.8ms average, 270ms for very large tables
- **Bottlenecks**: Redundant PMF calculations during root-finding
- **Optimizations Implemented**:
  - Guard logging calls (5-10% improvement)
  - Inline boundary checks (3-5% improvement)
  - Primitive cache keys (15-20% improvement)
  - 3-Point pre-bracketing (marginal impact)
- **Results**: 69.7% reduction in execution time (5.69ms average)

#### Conditional Method
- **Baseline**: 5.2ms average, 19ms for very large tables
- **Bottlenecks**: Conservative bracketing algorithms, repeated CDF/SF calculations
- **Optimizations Implemented**:
  - Memoize CDF/SF calls (28% improvement)
  - Guard debug logging (minimal impact)
  - Hoist fixed marginals (slight performance decrease)
  - 3-Point pre-bracketing (32% additional improvement)
  - Vectorized bracket expansion (4.79% overall improvement)
- **Results**: Significant improvement, though actual measurements (8-9ms) differ from claimed (0.42ms)

#### Unconditional Method
- **Performance**: 34μs average
- **Strengths**: Effective caching, grid optimization, early termination
- **No Critical Bottlenecks**: Already highly optimized

#### Mid-P Method
- **Performance**: 12μs average
- **Strengths**: Extensive caching, optimized support range calculations
- **No Critical Bottlenecks**: Already highly optimized

#### Wald-Haldane Method
- **Performance**: 2μs average
- **Strengths**: Closed-form calculation, no iterative computations
- **No Critical Bottlenecks**: Already optimal

### 2. Optimization Effectiveness

#### Caching Strategies
- **Primitive Cache Keys**: Most effective optimization for Blaker method (15-20% improvement)
- **Memoization**: Most effective optimization for Conditional method (28% improvement)
- **Shared Inter-Process Caching**: Limited impact due to implementation challenges
  - Cache hit rates: 0% (each worker creates isolated cache)
  - Performance overhead: 2-4x slower than sequential for small datasets

#### Algorithm Improvements
- **3-Point Pre-Bracketing**: Effective for Conditional method (32% improvement), marginal for Blaker method
- **Vectorized Bracket Expansion**: Mixed results (4.79% overall improvement)
  - Significant improvements for some cases (up to 25.44% faster)
  - Performance degradation for other cases (up to 23.18% slower)

#### Code Optimizations
- **Guard Logging Calls**: 5-10% improvement for Blaker method
- **Inline Boundary Checks**: 3-5% improvement for Blaker method
- **Hoist Fixed Marginals**: Slight performance decrease for Conditional method

### 3. Parallel Processing Analysis

- **Infrastructure**: Robust parallel processing with proper error handling
- **Limitations**:
  - Process overhead exceeds computation time for small tables
  - Cache isolation (each worker process creates its own Manager instance)
  - Too many workers for small datasets
- **Recommendations**:
  - Process-level initialization
  - Shared memory approach using `multiprocessing.shared_memory`
  - Optimize worker count based on dataset size

## Performance Verification

### Measurement Discrepancies

The performance verification revealed discrepancies between claimed and actual performance metrics:

- **Conditional Method**:
  - Claimed: 5.20ms baseline → 0.42ms optimized (92% improvement)
  - Actual: ~8-9ms execution time for the very_large test case after warm-up
  - Initial execution is much slower (~230ms) due to cache initialization

### Cache Effectiveness

- **First Run**: ~230-236ms (includes cache initialization)
- **Subsequent Runs**: ~8-9ms (with warm cache)
- **Scaling**: Approximately linear scaling with table size

## Recommendations

### 1. High-Priority Optimizations

1. **Deploy Blaker Method Optimizations**:
   - Primitive cache keys
   - Guard logging calls
   - Inline boundary checks
   - Expected impact: 69.7% reduction in execution time

2. **Deploy Conditional Method Optimizations**:
   - Memoize CDF/SF calls
   - 3-Point pre-bracketing
   - Expected impact: Significant improvement in execution time

3. **Improve Cache Initialization**:
   - Add cache pre-warming option for batch processing
   - Document cache initialization overhead
   - Expected impact: Reduced initial execution time

### 2. Medium-Priority Optimizations

4. **Refine Vectorized Bracket Expansion**:
   - Implement hybrid approach that selects between sequential and vectorized expansion
   - Apply selectively based on table configurations
   - Expected impact: Consistent performance improvement across all scenarios

5. **Enhance Parallel Processing**:
   - Implement process-level cache initialization
   - Use shared memory for large datasets
   - Optimize worker count based on dataset size
   - Expected impact: Improved parallel processing performance

### 3. Low-Priority Optimizations

6. **Standardize Benchmarking**:
   - Implement consistent benchmarking methodology
   - Include warm-up runs in measurements
   - Document environment information
   - Expected impact: More accurate performance claims

7. **Add Performance Regression Testing**:
   - Add benchmarking to CI pipeline
   - Set performance thresholds
   - Alert on performance regressions
   - Expected impact: Maintained performance over time

## Conclusion

The ExactCIs package has undergone significant optimization efforts that have substantially improved performance, particularly for the slowest methods (Blaker and Conditional). The implemented optimizations have been effective, with caching strategies providing the most significant performance gains.

While there are discrepancies between claimed and actual performance improvements, the overall trend is positive, with the Blaker method showing a 69.7% reduction in execution time and the Conditional method showing significant improvement.

The recommendations in this report provide a roadmap for further optimizing the package, with a focus on deploying the most effective optimizations, improving cache initialization, and enhancing parallel processing.

Date: August 5, 2025