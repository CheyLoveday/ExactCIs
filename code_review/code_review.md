# Code Review: ExactCIs - Unconditional Methods

**Date of Review:** 2025-05-08

**Reviewed Files:**
* `src/exactcis/methods/unconditional.py`

## Summary of Changes and Findings:

The primary focus of this review and subsequent refactoring was to address critical issues in the implementation of Barnard's unconditional exact test, aiming to improve accuracy, clarity, and maintainability.

### 1. Correction of `log_p_obs` Calculation (Critical Accuracy Fix)

*   **Issue:** The `_log_pvalue_barnard` function incorrectly calculated `log_p_obs` (the log-probability of the observed table) only once using the first point of the `p1` nuisance parameter grid. For an unconditional test, this probability must be evaluated for *each* `p1` in the grid.
*   **Changes:**
    *   The `_process_grid_point` function was refactored to calculate `log_p_obs_for_this_p1` internally for its specific `p1` value and the corresponding `current_p2` (derived from `p1` and `theta`).
    *   The `log_p_obs` parameter was removed from `_process_grid_point`.
    *   `_log_pvalue_barnard` no longer performs the single, incorrect `log_p_obs` calculation. It now relies on `_process_grid_point` to handle this correctly for each grid point.
*   **Impact:** This is a fundamental correction ensuring the p-value summation adheres to the definition of Barnard's test, significantly improving the accuracy of p-value and confidence interval calculations.

### 2. Flexible `p1` Nuisance Parameter Grid Management

*   **Issue:** The management of the `p1` grid (values for the nuisance parameter) was not cleanly integrated between the high-level CI calculation functions and the lower-level p-value functions.
*   **Changes:**
    *   `_log_pvalue_barnard` now accepts an optional `p1_grid_override: Optional[List[float]]` parameter.
        *   If provided and not empty, this list is used for the `p1` grid.
        *   Otherwise, `_log_pvalue_barnard` constructs an adaptive grid internally based on its `grid_size` parameter and the MLE of `p1`.
    *   `unconditional_log_pvalue` was updated:
        *   It now accepts `p1_values: Optional[np.ndarray]` and `grid_size: int`.
        *   It passes `p1_values` (converted to a list) as `p1_grid_override` to `_log_pvalue_barnard`.
        *   It passes its `grid_size` to `_log_pvalue_barnard`, which is used if `p1_values` is `None`.
    *   `improved_ci_unconditional` (which calls `unconditional_log_pvalue` via helper functions `f_lower`/`f_upper` and also in its refinement steps) was updated to:
        *   Pass its locally constructed `p1_values` (if `adaptive_grid=True`) to `unconditional_log_pvalue`.
        *   Pass its `grid_size` parameter to `unconditional_log_pvalue` (which is then used if `p1_values` is not provided).
*   **Impact:** This provides clearer control over `p1` grid generation. High-level functions can specify a custom grid (e.g., from an adaptive strategy), or allow the p-value function to use a default grid construction method.

### 3. Removal of Crude Large Table Approximation

*   **Issue:** `_log_pvalue_barnard` contained a section that returned a fixed, approximate p-value (e.g., `log(0.01)` or `log(0.05)`) if `n1 > 50` or `n2 > 50`. This undermined the "exact" nature of the test for larger tables.
*   **Changes:** This approximation block was removed from `_log_pvalue_barnard`.
*   **Impact:** The function now attempts to calculate the p-value accurately for all table sizes, aligning with the method's intent. This may increase computation time for very large tables, but prioritizes correctness.

### 4. Removal of Non-Functional `refine` and `use_profile` Flags

*   **Issue:** The `unconditional_log_pvalue` and `exact_ci_unconditional` functions had `refine` and `use_profile` parameters that were documented as "ignored for now" or had no actual effect on the computation.
*   **Changes:**
    *   These parameters were removed from the function signatures and docstrings of `unconditional_log_pvalue` and `exact_ci_unconditional`.
    *   Internal calls to `unconditional_log_pvalue` within `exact_ci_unconditional` were updated to reflect the new signature.
*   **Impact:** Improves code clarity and reduces confusion by removing misleading, non-functional parameters.

### 5. Minor Safety and Naming Enhancements

*   **Issue:** Potential for domain errors in `sqrt` and minor variable name clashes.
*   **Changes:**
    *   Added safety checks for `sqrt` domain errors (if `p*(1-p)` is zero) within `_process_grid_point`.
    *   Added handling for cases where `relevant_k` or `relevant_l` might become empty in the NumPy path of `_process_grid_point`, with a fallback to the pure Python path.
    *   Loop variables `k` and `l` in `_process_grid_point` were renamed to `k_val` and `l_val` for clarity.
*   **Impact:** Increased robustness and slightly improved readability.

## Outstanding Issues / Further Considerations:

*   **Performance for Large Tables:** While correctness is now prioritized, the removal of the large table approximation means that computations for very large `n1` or `n2` could be slow. If this becomes a practical issue, targeted performance optimizations for the p-value summation (e.g., more advanced numerical techniques or further grid optimization strategies) could be explored. This is not an "issue" with correctness but a potential performance consideration.
*   **Testing Coverage:** It is crucial to ensure comprehensive unit tests cover the new logic, especially:
    *   Cases with different `p1` grid configurations (override vs. internal generation).
    *   Edge cases for table inputs (zeros, small N, large N).
    *   Verification of p-values against known results or other software packages if possible.
*   **Docstring and Comment Review:** A final pass on all modified functions to ensure docstrings and comments are up-to-date and clearly explain the logic would be beneficial.

## Overall Assessment:

The implemented changes have significantly improved the correctness and robustness of Barnard's unconditional exact test implementation within the `ExactCIs` package. The most critical issue concerning the `log_p_obs` calculation has been resolved, and the `p1` grid management is now more flexible and transparent. The removal of the large table approximation and unused flags further refines the codebase. The code is now in a much better state for reliable scientific use.

## New Findings (2025-05-08):

### 1. `src/exactcis/core.py` - `logsumexp` numerical stability
*   **Issue:** The `logsumexp` function could encounter a `math.log(0)` error if all input `log_terms` were `-inf` or resulted in `log_max` being `-inf` after filtering.
*   **Changes:** Added a check after `log_max = max(filtered_terms)` to return `float('-inf')` if `log_max` is `float('-inf')`, preventing the error.
*   **Impact:** Improved numerical stability for edge cases.

### 2. `src/exactcis/cli.py` - Haldane's correction consistency
*   **Issue:** The CLI was forcibly applying the 0.5 addition for Haldane's correction if the `--apply-haldane` flag was used, regardless of whether a zero count was present. This differed from the `core.apply_haldane_correction` function which only adds 0.5 if a zero is present (otherwise, it just converts to float).
*   **Changes:** Modified the CLI to use `core.apply_haldane_correction(a, b, c, d)`. Verbose messages were updated to reflect that correction is "attempted" or "requested" and show the resulting values used for calculation.
*   **Impact:** Ensures consistent application of Haldane's correction between the CLI and core library functions, and uses existing logging from the core function.

### 3. `src/exactcis/methods/conditional.py` - Incomplete Fisher's Exact CI (Critical)
*   **Issue:** The `exact_ci_conditional` function, intended to calculate Fisher's exact confidence interval, contains hardcoded return values for specific test inputs and placeholder logic for the actual interval calculation (e.g., `low = max(0.0, odds_ratio / 3)`). It does not correctly implement the inversion of the noncentral hypergeometric CDF.
*   **Changes:** None made by the assistant due to complexity.
*   **Recommendation (Critical):** This function requires a complete rewrite to correctly implement Fisher's exact confidence interval. This involves using routines (likely from `core.py` or new ones) to find θ_lower and θ_upper such that P(X ≤ a | θ_upper) = alpha/2 and P(X ≥ a | θ_lower) = alpha/2, where X follows a noncentral hypergeometric distribution. The current implementation is not fit for purpose.
*   **Impact:** The conditional CI method is currently non-functional and will produce incorrect results for most inputs.
