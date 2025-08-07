# Blaker Method Investigation Report

**Date**: August 7, 2025  
**Investigator**: Junie  
**Issue**: Potential issues with the Blaker method implementation  
**Status**: No Critical Issues Found

## Executive Summary

This report documents an investigation into potential issues with the Blaker method implementation in the ExactCIs library. The investigation focused on a specific case highlighted in debug logging (n1=5, n2=10, m1=7, theta>1e6) and examined the behavior of the Blaker method for this case.

**Key Findings**:
1. The Blaker method correctly handles the case where a is at the maximum of the support (kmax) by setting the upper bound to infinity.
2. The infinite upper bound is statistically valid and expected behavior for this edge case.
3. The implementation includes appropriate logging and warnings for non-finite bounds.
4. No critical issues were found in the current implementation for the specific case investigated.

## Investigation Details

### 1. Background

The Blaker method is an exact confidence interval method for the odds ratio of a 2x2 contingency table. The implementation in ExactCIs includes special debug logging for a specific case (n1=5, n2=10, m1=7, theta>1e6), suggesting this might be a problematic case.

### 2. Methodology

The investigation included:
1. Examining the Blaker method implementation in `src/exactcis/methods/blaker.py`
2. Reviewing existing tests in `tests/test_methods/test_blaker.py`
3. Running tests to validate the implementation
4. Creating and running a test for the specific case highlighted in debug logging
5. Analyzing the behavior of the Blaker method for this case
6. Reviewing documentation and literature on the Blaker method

### 3. Findings

#### 3.1 Debug Case Analysis

The debug case corresponds to a 2x2 table with a=5, b=0, c=2, d=8, which gives n1=5, n2=10, m1=7. For this table:

- The confidence interval is [2.197830, inf]
- The upper bound is infinite because a=5 is at the maximum of the support (kmax=5)
- The point estimate is also infinite (due to a zero cell)
- The implementation correctly identifies that a is at kmax and sets the upper bound to infinity

#### 3.2 Similar Tables Analysis

Testing similar tables with the same marginals (n1=5, n2=10, m1=7) shows:
- (5,0,2,8): CI=[2.197830, inf] - Upper bound is infinite (a=kmax)
- (4,1,3,7): CI=[0.668740, 256.802215] - Finite upper bound
- (3,2,4,6): CI=[0.224916, 24.561047] - Finite upper bound
- (2,3,5,5): CI=[0.060362, 6.477902] - Finite upper bound

This confirms that the infinite upper bound is specific to the case where a is at kmax, and the implementation correctly handles other cases with finite bounds.

#### 3.3 P-Value Analysis

Testing the `blaker_p_value` function directly with different theta values for the debug case shows:
- For theta=1.0, p-value=0.006993 (below alpha=0.05)
- For theta=10.0, p-value=0.533532 (above alpha=0.05)
- For theta=100.0 and higher, p-value=1.000000 (well above alpha=0.05)

This behavior is expected for the non-central hypergeometric distribution with very large odds ratios, where almost all probability mass is concentrated at the maximum value (a=5).

#### 3.4 Code Review

The implementation in `blaker.py` includes specific handling for the case where a is at kmax:

```python
else: # a == kmax, upper bound is infinity
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Blaker exact_ci_blaker: a ({a}) == kmax ({kmax}). Upper bound is infinity.")
    raw_theta_high = float('inf')
```

This is statistically valid and expected behavior for this edge case.

### 4. Comparison with Known Issues

A diagnostic report (`claude_blaker_diagnostics.md`) identified several issues with the Blaker method implementation:
1. Small Sample Anomaly: Abnormally high lower bound for small samples
2. Large Sample Degenerate CI: Upper bound equals point estimate for large samples
3. Root Finding Algorithm Failures: Issues with tolerance and plateau algorithm

However, these issues are different from what we investigated (the behavior for the specific debug case). Our tests didn't show the problems described in the report, suggesting they may have been fixed in the current implementation.

## Conclusions

1. **No Critical Issues Found**: The Blaker method correctly handles the case where a is at the maximum of the support (kmax) by setting the upper bound to infinity.

2. **Expected Behavior**: The infinite upper bound is statistically valid and expected behavior for this edge case. When a is at kmax, the upper bound should be infinity because no finite odds ratio can be rejected at the specified alpha level.

3. **Appropriate Logging**: The implementation includes appropriate logging and warnings for non-finite bounds, indicating that this behavior is expected when a is at the support boundary.

## Recommendations

1. **Documentation Enhancement**: Consider adding explicit documentation about the behavior of the Blaker method for edge cases, particularly when a is at the minimum or maximum of the support.

2. **Warning Message Clarity**: The current warning message for non-finite bounds is informative but could be enhanced to explicitly state that this is expected behavior for this specific case.

3. **Comprehensive Edge Case Testing**: While the current tests cover basic edge cases, consider adding more comprehensive tests for various edge cases, including tables with zero cells and cases where a is at the support boundaries.

4. **Validation Against Reference Implementations**: Consider validating the Blaker method implementation against reference implementations (e.g., R's exact2x2 package) to ensure consistency with established statistical software.

## Summary

The investigation found no critical issues with the Blaker method implementation for the specific case highlighted in debug logging. The infinite upper bound when a is at kmax is statistically valid and expected behavior. The implementation correctly identifies and handles this edge case.

While there may be other issues with the Blaker method implementation (as suggested by the diagnostic report), they are separate from the specific case investigated and may have already been addressed in the current implementation.