Mid-P Method
===========

The Mid-P method provides a less conservative alternative to Fisher's exact test by using the mid-p-value.

.. automodule:: exactcis.methods.midp
   :members: exact_ci_midp, exact_ci_midp_batch
   :undoc-members:
   :show-inheritance:

Overview
--------

The Mid-P method adjusts the p-value calculation by using the mid-p-value, which is the usual p-value minus half the probability of the observed outcome. This adjustment makes the test less conservative than Fisher's exact test, resulting in narrower confidence intervals that still maintain good coverage properties.

This implementation uses a grid search approach with confidence interval inversion, which is more reliable for large sample sizes than root-finding methods. It also supports parallel processing for batch operations, significantly speeding up calculations for large datasets.

Key Functions
------------

- ``exact_ci_midp``: Calculate a confidence interval for a single 2x2 table using the Mid-P method.
- ``exact_ci_midp_batch``: Calculate confidence intervals for multiple 2x2 tables in parallel.

Usage Examples
-------------

**Single Table Calculation**

.. code-block:: python

   from exactcis.methods.midp import exact_ci_midp

   # Calculate a 95% confidence interval using the Mid-P method
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_midp(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")

**Batch Processing**

.. code-block:: python

   from exactcis.methods.midp import exact_ci_midp_batch

   tables = [(10, 20, 15, 30), (5, 10, 8, 12), (2, 3, 1, 4)]
   results = exact_ci_midp_batch(tables, alpha=0.05)
   for i, (lower, upper) in enumerate(results):
       print(f"Table {i+1}: 95% CI ({lower:.4f}, {upper:.4f})")
