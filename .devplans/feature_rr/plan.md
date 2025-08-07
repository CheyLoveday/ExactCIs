# Relative Risk Confidence Intervals Implementation Plan

This document outlines the plan for implementing confidence intervals for relative risk in the ExactCIs package.

## 1. Overview

Relative risk (RR) is a key measure in epidemiological studies, representing the ratio of the probability of an outcome in an exposed group to the probability of an outcome in an unexposed group. While the ExactCIs package currently calculates point estimates for relative risk, it lacks methods for computing confidence intervals.

This implementation will add several methods for calculating confidence intervals for relative risk, following the research outlined in `research.md`.

## 2. API Design

### 2.1 Core Functions

We will implement the following core functions for calculating confidence intervals for relative risk:

```python
from typing import Tuple

# Score-based methods
def ci_score_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the asymptotic score confidence interval for relative risk."""
    
def ci_score_cc_rr(a: int, b: int, c: int, d: int, delta: float = 4.0, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the continuity-corrected score confidence interval for relative risk."""

# Nonparametric methods
def ci_ustat_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the U-statistic-based confidence interval for relative risk."""

# Classical methods
def ci_wald_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the Wald confidence interval for relative risk (log scale)."""
    
def ci_wald_katz_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the Katz-adjusted Wald confidence interval for relative risk (independent proportions)."""
    
def ci_wald_correlated_rr(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate the Wald confidence interval for relative risk with correlation adjustment."""
```

### 2.2 Batch Processing Functions

For each core function, we will implement a corresponding batch processing function:

```python
from typing import List, Tuple, Optional, Callable

def ci_score_rr_batch(tables: List[Tuple[int, int, int, int]], alpha: float = 0.05, 
                      max_workers: Optional[int] = None, backend: Optional[str] = None, 
                      progress_callback: Optional[Callable] = None) -> List[Tuple[float, float]]:
    """Calculate asymptotic score confidence intervals for relative risk in batch."""
    
# Similar batch functions for other methods
```

### 2.3 Convenience Function

We will implement a convenience function to calculate confidence intervals using all available methods:

```python
from typing import Dict, Tuple

def compute_all_rr_cis(a: int, b: int, c: int, d: int, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for the relative risk using all available methods.
    
    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary mapping method names to confidence intervals (lower_bound, upper_bound)
    """
```

### 2.4 Batch Calculation of Relative Risk

We will also implement a batch function for calculating point estimates of relative risk:

```python
from typing import List, Tuple

def batch_calculate_relative_risks(tables: List[Tuple[int, int, int, int]]) -> List[float]:
    """
    Calculate relative risks for multiple 2x2 tables in batch.
    
    Args:
        tables: List of (a, b, c, d) tuples representing 2x2 contingency tables
        
    Returns:
        List of relative risk values
    """
```

## 3. Implementation Details

### 3.1 File Structure

We will create the following new files:

1. `src/exactcis/methods/relative_risk.py`: Main implementation of relative risk confidence interval methods
2. `tests/test_methods/test_relative_risk.py`: Tests for the relative risk methods

We will also update the following existing files:

1. `src/exactcis/__init__.py`: Expose the new functions
2. `src/exactcis/core.py`: Add the batch calculation function for relative risk

### 3.2 Method Implementation Details

#### 3.2.1 Score-Based Methods

The score-based methods will be implemented following the mathematical framework described in the research document:

1. **Asymptotic Score Interval**:
   - Implement the score test statistic: `S(θ₀) = (x₁₁ + x₁₂) - (x₁₁ + x₂₁)θ₀ / sqrt(...)`
   - Use root-finding to determine the confidence bounds
   - Handle edge cases (zero cells, perfect separation)

2. **Modified Score Interval with Continuity Correction**:
   - Implement the corrected score statistic with parameter δ
   - Default to δ = 4 (ASCC-M) as recommended
   - Allow customization of the correction strength

#### 3.2.2 Nonparametric Methods

The U-statistic method will be implemented as follows:

1. **U-Statistic-Based Interval**:
   - Implement the variance estimation for log RR using rank-based approach
   - Use t-distribution with n-1 degrees of freedom for improved small-sample coverage
   - Handle edge cases appropriately

#### 3.2.3 Classical Methods

The classical methods will be implemented as follows:

1. **Wald Intervals**:
   - Implement the simple normal-theory approach on log scale
   - Add appropriate continuity corrections for small samples

2. **Independent Proportions (Katz Method)**:
   - Implement the variance formula: `Var(log θ̂ᵣᵣ) = (1-p₁)/x₁₁ + (1-p₂)/x₂₁`
   - Use for case-control designs

3. **Correlated Proportions**:
   - Implement the modified variance accounting for covariance structure
   - Calculate the critical covariance term: `Cov(p̂₁, p̂₂) = (x₁₁x₂₂ - x₁₂x₂₁) / (n(n-1))`

### 3.3 Utility Functions

We will implement the following utility functions:

1. **Constrained MLE**: Compute p̃₂₁ using quadratic formula solution
2. **Ferrari's Method**: Implement Ferrari's method for solving quartic equations
3. **Root Selection**: Implement logic for choosing biologically meaningful positive roots

### 3.4 Integration with Existing Code

The new functions will be integrated with the existing codebase as follows:

1. Add the batch calculation function for relative risk to `core.py`
2. Update `__init__.py` to expose the new functions
3. Ensure consistent error handling and validation with the existing code
4. Maintain the same API design pattern as the existing methods

## 4. Testing Strategy

### 4.1 Unit Tests

We will implement comprehensive unit tests for each method:

1. **Basic Functionality Tests**:
   - Test each method with typical 2×2 tables
   - Verify that the confidence intervals contain the point estimate
   - Check that the confidence intervals have the expected width

2. **Edge Case Tests**:
   - Zero cells (a=0, b=0, c=0, or d=0)
   - Small counts (a, b, c, d < 5)
   - Large imbalances (a >> c or c >> a)
   - Perfect separation (a>0, d>0, b=0, c=0)

3. **Parameter Tests**:
   - Different alpha levels (0.01, 0.05, 0.1)
   - Different continuity correction parameters for ASCC

### 4.2 Integration Tests

We will implement integration tests to verify that the new functions work correctly with the existing codebase:

1. Test the convenience function `compute_all_rr_cis`
2. Test batch processing functions with various table sizes
3. Test integration with the parallel processing utilities

### 4.3 Validation Tests

We will implement validation tests to compare our results with established implementations:

1. Compare with R's `epiR` package
2. Compare with SciPy's implementations where applicable
3. Verify that the methods meet the expected coverage properties

### 4.4 Performance Tests

We will implement performance tests to ensure that the methods are efficient:

1. Test the performance of each method with various table sizes
2. Test the performance of batch processing functions with various numbers of tables
3. Compare the performance of different methods

## 5. Implementation Phases

### 5.1 Phase 1: Core Implementation

1. Implement the batch calculation function for relative risk in `core.py`
2. Implement the Wald method for relative risk (simplest method)
3. Write tests for the Wald method
4. Update `__init__.py` to expose the new functions

### 5.2 Phase 2: Score-Based Methods

1. Implement the asymptotic score method
2. Implement the continuity-corrected score method
3. Write tests for the score-based methods
4. Update the convenience function to include the new methods

### 5.3 Phase 3: Nonparametric and Classical Methods

1. Implement the U-statistic method
2. Implement the Katz method for independent proportions
3. Implement the correlated proportions method
4. Write tests for the nonparametric and classical methods
5. Update the convenience function to include the new methods

### 5.4 Phase 4: Batch Processing and Optimization

1. Implement batch processing functions for all methods
2. Optimize the implementation for performance
3. Write tests for the batch processing functions
4. Finalize the documentation

## 6. Dependencies and Prerequisites

The implementation will depend on the following existing components:

1. Core validation functions (`validate_counts`)
2. Optimization utilities (`find_root`, `find_sign_change`)
3. Statistical utilities (`normal_quantile`, `t_quantile`)
4. Parallel processing utilities

No new external dependencies are required.

## 7. Documentation

We will provide comprehensive documentation for the new functions:

1. **Docstrings**: Detailed docstrings for all functions, following the Google style used in the project
2. **User Guide**: Examples and explanations for using the relative risk confidence interval methods
3. **API Reference**: Technical documentation for the new functions
4. **Method Selection Guide**: Guidance on selecting the appropriate method based on sample size and study design

## 8. Validation Approach

We will validate the implementation using the following approach:

1. **Comparison with Established Implementations**:
   - Compare with R's `epiR` package
   - Compare with SciPy's implementations where applicable
   - Compare with published results in the literature

2. **Coverage Analysis**:
   - Simulate data to verify that the methods achieve the nominal coverage
   - Test with various sample sizes and parameter values
   - Verify that the methods handle edge cases correctly

3. **Numerical Stability**:
   - Test with extreme values to ensure numerical stability
   - Verify that the methods handle underflow and overflow correctly
   - Ensure that the methods are robust to rounding errors

## 9. Timeline

The implementation will be completed in the following timeline:

1. **Phase 1**: 1 week
2. **Phase 2**: 2 weeks
3. **Phase 3**: 2 weeks
4. **Phase 4**: 1 week

Total estimated time: 6 weeks

## 10. Conclusion

This implementation plan outlines a comprehensive approach to adding confidence intervals for relative risk to the ExactCIs package. The implementation will follow the existing API design patterns and coding standards, and will be thoroughly tested and documented. The new functions will provide users with a range of methods for calculating confidence intervals for relative risk, with guidance on selecting the appropriate method based on their specific needs.