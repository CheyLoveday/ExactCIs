Clopper-Pearson Method
==================

The Clopper-Pearson method provides exact confidence intervals for binomial proportions using the binomial cumulative distribution function.

.. automodule:: exactcis.methods.clopper_pearson
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Clopper-Pearson method calculates exact confidence intervals for binomial proportions using the binomial cumulative distribution function. It is sometimes called the "exact" method because it uses the binomial distribution directly rather than approximations.

For a binomial proportion p with x successes in n trials, the Clopper-Pearson interval is calculated as follows:

1. Lower bound: Find p_L such that P(X ≥ x | n, p_L) = α/2
   - This is equivalent to finding p_L such that the cumulative distribution function F(x-1; n, p_L) = 1 - α/2
   - If x = 0, the lower bound is 0

2. Upper bound: Find p_U such that P(X ≤ x | n, p_U) = α/2
   - This is equivalent to finding p_U such that the cumulative distribution function F(x; n, p_U) = α/2
   - If x = n, the upper bound is 1

In the context of a 2x2 contingency table, the method can calculate the confidence interval for either the proportion in group 1 (p1 = a/(a+b)) or the proportion in group 2 (p2 = c/(c+d)).

Key Functions
------------

- ``exact_ci_clopper_pearson`` - Calculate confidence intervals for binomial proportions using the Clopper-Pearson method
- ``exact_ci_clopper_pearson_batch`` - Calculate confidence intervals for multiple 2x2 tables in parallel

Usage Example
------------

.. code-block:: python

   from exactcis.methods.clopper_pearson import exact_ci_clopper_pearson
   
   # Calculate a 95% confidence interval for the proportion in group 1
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_clopper_pearson(10, 20, 5, 25, alpha=0.05, group=1)
   print(f"95% CI for p1: ({lower:.4f}, {upper:.4f})")
   
   # Calculate a 95% confidence interval for the proportion in group 2
   lower, upper = exact_ci_clopper_pearson(10, 20, 5, 25, alpha=0.05, group=2)
   print(f"95% CI for p2: ({lower:.4f}, {upper:.4f})")