# Concurrency Audit Report

This report details opportunities for concurrency and parallelism in the ExactCIs codebase.

## 1. Unconditional Confidence Interval Calculation

**File:** `src/exactcis/methods/unconditional.py`

**Function:** `_log_pvalue_barnard`

**Observation:** The `_log_pvalue_barnard` function calculates a p-value by iterating over a grid of nuisance parameters. The calculation for each grid point is independent and computationally intensive, making this a prime candidate for parallelization. The existing code already includes a `parallel_map` function, but it appears to be a placeholder.

**Recommendation:** Implement true parallel processing for the grid search in `_log_pvalue_barnard`. This can be achieved using Python's `concurrent.futures.ProcessPoolExecutor` to distribute the `_process_grid_point` function calls across multiple CPU cores. This will significantly reduce the time required to calculate unconditional confidence intervals, especially for larger grid sizes.

**Example (Illustrative):**

```python
# In src/exactcis/methods/unconditional.py

from concurrent.futures import ProcessPoolExecutor

def _log_pvalue_barnard(...):
    # ... (existing code)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_grid_point, grid_args))

    # ... (existing code)
```

## 2. Batch Processing in the CLI

**File:** `src/exactcis/cli.py`

**Function:** `main`

**Observation:** The command-line interface currently processes a single 2x2 table at a time. For users who need to calculate confidence intervals for many tables, this is inefficient.

**Recommendation:** Enhance the CLI to support batch processing. This could be implemented by allowing the user to specify an input file containing multiple 2x2 tables. The CLI could then process these tables in parallel, using a process pool to distribute the work. This would be particularly beneficial when calculating unconditional CIs, as the parallelization from recommendation #1 could be leveraged.

## 3. Blaker's Confidence Interval (Batch Processing)

**File:** `src/exactcis/methods/blaker.py`

**Function:** `exact_ci_blaker`

**Observation:** While the calculation for a single Blaker's CI is inherently sequential, there are opportunities for parallelism when calculating CIs for multiple tables.

**Recommendation:** If batch processing is implemented in the CLI (recommendation #2), the `exact_ci_blaker` function could be called in parallel for each table in the batch. This would provide a significant speedup for users who need to calculate many Blaker's CIs.

## 4. Mid-P Confidence Interval (Batch Processing)

**File:** `src/exactcis/methods/midp.py`

**Function:** `exact_ci_midp`

**Observation:** Similar to Blaker's CI, the `exact_ci_midp` function is a candidate for parallelization when processing multiple tables. The calculation for each table is independent.

**Recommendation:** When implementing batch processing in the CLI, the `exact_ci_midp` function should be executed in parallel for each input table. This can be achieved using a `ProcessPoolExecutor` to manage a pool of worker processes, which will improve throughput for large datasets.

## 5. Core Function Parallelism

**File:** `src/exactcis/core.py`

**Functions:** `log_nchg_pmf`, `log_nchg_cdf`, `log_nchg_sf`

**Observation:** The noncentral hypergeometric distribution functions in `core.py` are fundamental to many of the confidence interval calculations. While these functions are called within sequential algorithms for a single table, they can be parallelized when processing batches of tables.

**Recommendation:** As part of a broader batch processing implementation, ensure that calls to these core functions are parallelized across tables. For example, when calculating `midp_p_value` for a batch of tables, the underlying calls to `log_nchg_pmf` for each table can be distributed across multiple processes. This will prevent these core calculations from becoming a bottleneck in a parallel system.