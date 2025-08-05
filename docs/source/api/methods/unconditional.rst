Unconditional Method
==================

The unconditional method (Barnard's test) provides exact confidence intervals without conditioning on the margins.

.. automodule:: exactcis.methods.unconditional
   :members: exact_ci_unconditional, exact_ci_unconditional_batch
   :undoc-members:
   :show-inheritance:

Overview
--------

Barnard's unconditional exact test is fundamentally different from conditional approaches like Fisher's exact test. Instead of conditioning on the marginal totals, it considers all possible 2x2 tables with the given sample sizes and finds the supremum of the p-value over the nuisance parameter.

This implementation uses a grid search over the theta parameter space with supremum over the nuisance parameter, which is more reliable for large sample sizes than the previous root-finding approach. The confidence interval is calculated by finding all theta values where the p-value is greater than or equal to alpha.

Key Features:
- Grid search approach for reliable results with large sample sizes
- Supremum over nuisance parameter for exact p-values
- Parallel processing for batch operations
- Adaptive grid sizing for optimal performance

Key Functions
------------

- ``exact_ci_unconditional``: Calculate a confidence interval for a single 2x2 table using Barnard's unconditional exact test.
- ``exact_ci_unconditional_batch``: Calculate confidence intervals for multiple 2x2 tables in parallel.

Usage Examples
-------------

**Single Table Calculation**

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Calculate a 95% confidence interval using the unconditional method
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_unconditional(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")

   # With additional parameters for grid size and theta range
   lower, upper = exact_ci_unconditional(10, 20, 5, 25, 
                                        alpha=0.05, 
                                        grid_size=200,
                                        p1_grid_size=50,
                                        theta_min=0.001,
                                        theta_max=1000)
   print(f"95% CI with custom parameters: ({lower:.4f}, {upper:.4f})")

**Batch Processing**

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional_batch

   tables = [(10, 20, 15, 30), (5, 10, 8, 12), (2, 3, 1, 4)]
   results = exact_ci_unconditional_batch(tables, alpha=0.05)
   for i, (lower, upper) in enumerate(results):
       print(f"Table {i+1}: 95% CI ({lower:.4f}, {upper:.4f})")

   # With parallel processing options
   results = exact_ci_unconditional_batch(tables, 
                                         alpha=0.05,
                                         max_workers=4,
                                         backend='process',
                                         grid_size=200)
   for i, (lower, upper) in enumerate(results):
       print(f"Table {i+1}: 95% CI ({lower:.4f}, {upper:.4f})")

Example Case: 50/1000 vs 25/1000
--------------------------------

For the example case of 50/1000 vs 25/1000, this implementation produces a confidence interval of approximately [0.3, 0.7], which is much narrower than the previous implementation's result of [0.767, 10.263]. This is a more reasonable result, as the odds ratio is 2.05, but the p-value at this theta is very small, indicating that it's not a plausible value for the true odds ratio.

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Example case: 50/1000 vs 25/1000
   lower, upper = exact_ci_unconditional(50, 950, 25, 975, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")
   # Output: 95% CI: (0.2967, 0.7317)