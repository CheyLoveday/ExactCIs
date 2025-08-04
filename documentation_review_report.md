# ExactCIs Documentation Review Report

## Overview

This report summarizes the findings from a comprehensive review of the ExactCIs library documentation compared to its actual implementation. The review focused on verifying that the documentation accurately reflects the code implementation, with particular attention to API consistency, parameter documentation, and example correctness.

## Summary of Findings

The ExactCIs library documentation is generally well-structured and comprehensive, with detailed API references, user guides, and examples. However, several discrepancies were identified between the documentation and the actual implementation, which could lead to confusion for users.

### Documentation Strengths

1. **Comprehensive API Reference**: The API reference provides detailed information about the main functions, including parameters, return values, exceptions, and examples.
2. **Well-Structured Quick Start Guide**: The quick start notebook provides clear examples of how to use the library for various use cases.
3. **Detailed Method Descriptions**: Each confidence interval method is well-described with its statistical properties and appropriate use cases.
4. **Troubleshooting Guide**: The troubleshooting guide addresses common issues and provides solutions.

### Documentation Gaps and Inconsistencies

1. **Undocumented Batch Processing Functions**: Several batch processing functions are implemented but not documented in the API reference:
   - `exact_ci_blaker_batch`
   - `exact_ci_conditional_batch`
   - `exact_ci_midp_batch`

2. **Parameter Inconsistencies**:
   - **Timeout Parameter**: The API documentation for `compute_all_cis` mentions a `timeout` parameter, but it's not implemented in the function.
   - **Grid Size Default**: For `exact_ci_unconditional`, the API documentation specifies a default `grid_size` of 50, but the implementation in the quick start guide uses a default of 15.
   - **Progress Callback Parameter**: For `exact_ci_midp`, the implementation includes a `progress_callback` parameter that's not mentioned in the API documentation.

3. **Undocumented Parameters**:
   - For `exact_ci_unconditional`, several parameters accepted through `**kwargs` are not documented in the API reference:
     - `theta_min`, `theta_max`: Theta range bounds
     - `custom_range`: Custom range for theta search
     - `theta_factor`: Factor for automatic theta range
     - `haldane`: Apply Haldane's correction
     - `use_cache`: Whether to use caching
     - `use_profile`: Mentioned in troubleshooting guide but not in API reference

4. **Inconsistent Examples**:
   - The troubleshooting guide mentions using a timeout parameter with `compute_all_cis`, but this functionality doesn't exist in the implementation.

## Detailed Findings by Component

### 1. Main Interface (`compute_all_cis`)

**Documentation vs. Implementation**:
- **Function Signature**:
  - Documentation: `exactcis.compute_all_cis(a, b, c, d, alpha=0.05, grid_size=200, timeout=None)`
  - Implementation: `def compute_all_cis(a: int, b: int, c: int, d: int, alpha: float = 0.05, grid_size: int = 200) -> Dict[str, Tuple[float, float]]:`
  - **Discrepancy**: The `timeout` parameter is documented but not implemented.

- **Exceptions**:
  - Documentation: "ValueError: If input counts are invalid or if margins are zero" and "TimeoutError: If computation exceeds the specified timeout (when provided)"
  - Implementation: Only ValueError is raised through `validate_counts`
  - **Discrepancy**: The TimeoutError is documented but not implemented.

### 2. Method Functions

#### 2.1 Blaker's Method (`exact_ci_blaker`)

**Documentation vs. Implementation**:
- No significant discrepancies found. The function signature, parameters, return values, and exceptions match the documentation.

#### 2.2 Conditional Method (`exact_ci_conditional`)

**Documentation vs. Implementation**:
- No significant discrepancies found. The function signature, parameters, return values, and exceptions match the documentation.

#### 2.3 Mid-P Method (`exact_ci_midp`)

**Documentation vs. Implementation**:
- **Function Signature**:
  - Documentation: `exactcis.methods.exact_ci_midp(a, b, c, d, alpha=0.05)`
  - Implementation: `def exact_ci_midp(a: int, b: int, c: int, d: int, alpha: float = 0.05, progress_callback: Optional[Callable] = None) -> Tuple[float, float]:`
  - **Discrepancy**: The `progress_callback` parameter is not documented.

#### 2.4 Unconditional Method (`exact_ci_unconditional`)

**Documentation vs. Implementation**:
- **Function Signature**:
  - Documentation: `exactcis.methods.exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=50, timeout=None)`
  - Implementation: `def exact_ci_unconditional(a: int, b: int, c: int, d: int, alpha: float = 0.05, **kwargs) -> Tuple[float, float]:`
  - **Discrepancy**: The implementation uses `**kwargs` instead of explicitly listing parameters.

- **Parameters**:
  - Documentation: `grid_size` default is 50
  - Implementation: `grid_size` default is 15 (from docstring)
  - **Discrepancy**: Different default values.

- **Undocumented Parameters**:
  - Several parameters accepted through `**kwargs` are not documented in the API reference:
    - `theta_min`, `theta_max`: Theta range bounds
    - `custom_range`: Custom range for theta search
    - `theta_factor`: Factor for automatic theta range
    - `haldane`: Apply Haldane's correction
    - `use_cache`: Whether to use caching
    - `use_profile`: Mentioned in troubleshooting guide

#### 2.5 Wald-Haldane Method (`ci_wald_haldane`)

**Documentation vs. Implementation**:
- No significant discrepancies found. The function signature, parameters, return values, and exceptions match the documentation.

### 3. Batch Processing Functions

**Documentation vs. Implementation**:
- **Discrepancy**: The batch processing functions are implemented but not documented in the API reference:
  - `exact_ci_blaker_batch`
  - `exact_ci_conditional_batch`
  - `exact_ci_midp_batch`

## Recommendations

Based on the findings, the following recommendations are made to improve the documentation:

1. **Document Batch Processing Functions**:
   - Add API reference entries for all batch processing functions
   - Include examples of how to use these functions for processing multiple tables efficiently

2. **Fix Parameter Inconsistencies**:
   - Remove the `timeout` parameter from the `compute_all_cis` documentation or implement it in the function
   - Update the default `grid_size` value in either the documentation or implementation to ensure consistency
   - Add documentation for the `progress_callback` parameter in `exact_ci_midp`

3. **Document Additional Parameters**:
   - Add documentation for all parameters accepted through `**kwargs` in `exact_ci_unconditional`
   - Clarify the purpose and usage of each parameter

4. **Update Examples**:
   - Ensure that all examples in the documentation use parameters that are actually implemented
   - Add examples for batch processing functions

5. **Add Advanced Usage Documentation**:
   - Create a dedicated section for advanced usage, including:
     - Batch processing
     - Progress tracking
     - Caching strategies
     - Custom parameter tuning

6. **Improve Troubleshooting Guide**:
   - Update the troubleshooting guide to reflect the actual implementation
   - Remove references to non-existent parameters or functionality

## Conclusion

The ExactCIs library has comprehensive documentation, but there are several discrepancies between the documentation and the actual implementation. Addressing these issues will improve the user experience and reduce confusion for users of the library.

The most critical issues to address are:
1. Documenting the batch processing functions
2. Fixing the parameter inconsistencies, especially for `compute_all_cis` and `exact_ci_unconditional`
3. Updating examples to use only implemented functionality

By aligning the documentation with the implementation, the ExactCIs library will be more accessible and easier to use for statistical analysis.