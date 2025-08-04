Basic Usage
==========

This guide demonstrates basic usage patterns for the ExactCIs package, providing examples for common scenarios and tasks.

Prerequisites
------------

Ensure you have ExactCIs installed:

.. code-block:: bash

   pip install exactcis

Basic Example
------------

The most common use case is calculating a confidence interval for the odds ratio of a 2×2 contingency table:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Example 2×2 table:
   #      Success   Failure
   # Grp1    7         3
   # Grp2    2         8

   a, b, c, d = 7, 3, 2, 8  # Cell counts

   # Calculate 95% confidence interval
   lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05)
   print(f"95% CI for odds ratio: ({lower:.4f}, {upper:.4f})")
   # Output: 95% CI for odds ratio: (1.0472, 104.7200)

   # Calculate 90% confidence interval
   lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.10)
   print(f"90% CI for odds ratio: ({lower:.4f}, {upper:.4f})")
   # Output: 90% CI for odds ratio: (1.3810, 41.4300)

Comparing Different Methods
-------------------------

You can compare the results from different methods:

.. code-block:: python

   import scipy.stats as stats
   from exactcis.methods.unconditional import exact_ci_unconditional
   import numpy as np

   def normal_approx_ci(a, b, c, d, alpha=0.05):
       """Calculate CI using normal approximation"""
       # Add 0.5 to each cell (Haldane's correction)
       a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
       
       # Calculate odds ratio and log odds ratio
       or_est = (a*d)/(b*c)
       log_or = np.log(or_est)
       
       # Standard error of log odds ratio
       se = np.sqrt(1/a + 1/b + 1/c + 1/d)
       
       # Critical value
       z = stats.norm.ppf(1 - alpha/2)
       
       # Confidence interval for log odds ratio
       log_lower = log_or - z*se
       log_upper = log_or + z*se
       
       # Convert back to odds ratio scale
       return np.exp(log_lower), np.exp(log_upper)

   # Example table
   a, b, c, d = 7, 3, 2, 8

   # Calculate using different methods
   unconditional_ci = exact_ci_unconditional(a, b, c, d)
   original_unconditional_ci = exact_ci_unconditional(a, b, c, d)
   normal_ci = normal_approx_ci(a, b, c, d)
   _, fisher_p = stats.fisher_exact([[a, b], [c, d]])

   print("Method Comparison:")
   print(f"Improved Unconditional: ({unconditional_ci[0]:.4f}, {unconditional_ci[1]:.4f})")
   print(f"Original Unconditional: ({original_unconditional_ci[0]:.4f}, {original_unconditional_ci[1]:.4f})")
   print(f"Normal Approximation: ({normal_ci[0]:.4f}, {normal_ci[1]:.4f})")
   print(f"Fisher's Exact p-value: {fisher_p:.6f}")

Calculating Confidence Intervals for Proportions
--------------------------------------------

For estimating confidence intervals for proportions rather than odds ratios, use the Clopper-Pearson method:

.. code-block:: python

   from exactcis.methods import exact_ci_clopper_pearson
   
   # Example 2×2 table:
   #      Success   Failure
   # Grp1    7         3
   # Grp2    2         8
   
   a, b, c, d = 7, 3, 2, 8  # Cell counts
   
   # Calculate 95% confidence interval for proportion in group 1
   lower1, upper1 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=1)
   p1 = a / (a + b)  # Point estimate for group 1
   print(f"Group 1 proportion: {p1:.4f}")
   print(f"95% CI for group 1 proportion: ({lower1:.4f}, {upper1:.4f})")
   
   # Calculate 95% confidence interval for proportion in group 2
   lower2, upper2 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=2)
   p2 = c / (c + d)  # Point estimate for group 2
   print(f"Group 2 proportion: {p2:.4f}")
   print(f"95% CI for group 2 proportion: ({lower2:.4f}, {upper2:.4f})")
   
   # Compare with other proportion estimation methods
   from scipy.stats import binom
   
   # Wilson score interval (another common method for proportions)
   def wilson_score_interval(x, n, alpha=0.05):
       """Calculate Wilson score interval for a proportion"""
       z = stats.norm.ppf(1 - alpha/2)
       p_hat = x / n
       
       # Wilson score formula
       denominator = 1 + z**2/n
       center = (p_hat + z**2/(2*n)) / denominator
       halfwidth = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
       
       return max(0, center - halfwidth), min(1, center + halfwidth)
   
   # Calculate Wilson score intervals
   wilson_lower1, wilson_upper1 = wilson_score_interval(a, a+b)
   wilson_lower2, wilson_upper2 = wilson_score_interval(c, c+d)
   
   print("\nComparison with Wilson score interval:")
   print(f"Group 1: Clopper-Pearson ({lower1:.4f}, {upper1:.4f}), Wilson ({wilson_lower1:.4f}, {wilson_upper1:.4f})")
   print(f"Group 2: Clopper-Pearson ({lower2:.4f}, {upper2:.4f}), Wilson ({wilson_lower2:.4f}, {wilson_upper2:.4f})")
   print("\nNote: Clopper-Pearson intervals are typically wider (more conservative) than Wilson score intervals")

Handling Multiple Tables
---------------------

When processing multiple tables, using caching can significantly improve performance:

.. code-block:: python

   from exactcis.utils.optimization import CICache
   from exactcis.methods.unconditional import exact_ci_unconditional
   import time

   # Create a cache instance
   cache = CICache(max_size=1000)

   # Define a set of tables
   tables = [
       (7, 3, 2, 8),
       (8, 2, 3, 7),
       (10, 5, 3, 12),
       (15, 5, 7, 13),
       (20, 10, 10, 20),
       # Add more tables as needed
   ]

   # Without cache
   start_time = time.time()
   results_no_cache = []
   for a, b, c, d in tables:
       ci = exact_ci_unconditional(a, b, c, d, alpha=0.05, use_cache=False)
       results_no_cache.append(ci)
   time_no_cache = time.time() - start_time
   print(f"Time without cache: {time_no_cache:.4f} seconds")

   # With cache
   start_time = time.time()
   results_with_cache = []
   for a, b, c, d in tables:
       ci = exact_ci_unconditional(a, b, c, d, alpha=0.05, use_cache=True, cache=cache)
       results_with_cache.append(ci)
   time_with_cache = time.time() - start_time
   print(f"Time with cache: {time_with_cache:.4f} seconds")
   print(f"Speedup: {time_no_cache/time_with_cache:.2f}x")

Parallel Processing
----------------

For large batches of tables, parallel processing can be used:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional_parallel
   import time

   # Define a large set of tables
   large_batch = [
       (7, 3, 2, 8),
       (8, 2, 3, 7),
       (10, 5, 3, 12),
       (15, 5, 7, 13),
       (20, 10, 10, 20),
       (25, 15, 12, 28),
       (30, 20, 15, 35),
       (40, 25, 20, 45),
       (50, 30, 25, 55),
       (60, 35, 30, 65),
   ]

   # Sequential processing
   start_time = time.time()
   sequential_results = []
   for a, b, c, d in large_batch:
       ci = exact_ci_unconditional(a, b, c, d)
       sequential_results.append(ci)
   sequential_time = time.time() - start_time
   print(f"Sequential processing time: {sequential_time:.4f} seconds")

   # Parallel processing
   start_time = time.time()
   parallel_results = exact_ci_unconditional_parallel(large_batch)
   parallel_time = time.time() - start_time
   print(f"Parallel processing time: {parallel_time:.4f} seconds")
   print(f"Speedup: {sequential_time/parallel_time:.2f}x")

Handling Zero Cells
----------------

ExactCIs handles tables with zero cells gracefully:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Table with a zero cell
   a, b, c, d = 10, 5, 0, 15
   lower, upper = exact_ci_unconditional(a, b, c, d)
   print(f"Table with zero: 95% CI for odds ratio: ({lower:.4f}, {upper:.4f})")

   # Table with multiple zeros
   a, b, c, d = 10, 0, 0, 15
   lower, upper = exact_ci_unconditional(a, b, c, d)
   print(f"Table with multiple zeros: 95% CI for odds ratio: ({lower:.4f}, {upper:.4f})")

Custom Confidence Levels
---------------------

You can specify custom confidence levels:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   a, b, c, d = 7, 3, 2, 8

   # 99% confidence interval
   lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.01)
   print(f"99% CI: ({lower:.4f}, {upper:.4f})")

   # 90% confidence interval
   lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.10)
   print(f"90% CI: ({lower:.4f}, {upper:.4f})")

   # 80% confidence interval
   lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.20)
   print(f"80% CI: ({lower:.4f}, {upper:.4f})")

Working with Pandas DataFrames
---------------------------

ExactCIs can be easily integrated with pandas:

.. code-block:: python

   import pandas as pd
   from exactcis.methods.unconditional import exact_ci_unconditional

   # Create a DataFrame with study results
   data = {
       'Study': ['Study A', 'Study B', 'Study C', 'Study D'],
       'Treatment_Success': [15, 12, 25, 30],
       'Treatment_Failure': [5, 8, 15, 10],
       'Control_Success': [10, 8, 20, 15],
       'Control_Failure': [10, 12, 20, 25]
   }
   df = pd.DataFrame(data)

   # Calculate confidence intervals for each study
   results = []
   for _, row in df.iterrows():
       a = row['Treatment_Success']
       b = row['Treatment_Failure']
       c = row['Control_Success']
       d = row['Control_Failure']
       
       # Calculate odds ratio
       or_point = (a*d)/(b*c) if b*c > 0 else float('inf')
       
       # Calculate 95% CI
       lower, upper = exact_ci_unconditional(a, b, c, d)
       
       results.append({
           'Study': row['Study'],
           'OR': or_point,
           'Lower_95CI': lower,
           'Upper_95CI': upper
       })

   # Create results DataFrame
   results_df = pd.DataFrame(results)
   print(results_df)

Command-Line Interface
-------------------

ExactCIs provides a command-line interface for quick calculations:

.. code-block:: bash

   # Basic usage
   exactcis --a 7 --b 3 --c 2 --d 8
   
   # Specify confidence level
   exactcis --a 7 --b 3 --c 2 --d 8 --alpha 0.01
   
   # Choose method
   exactcis --a 7 --b 3 --c 2 --d 8 --method midp
   
   # Get detailed output
   exactcis --a 7 --b 3 --c 2 --d 8 --verbose