Wald Method
==========

The Wald method provides asymptotic confidence intervals based on the normal approximation.

.. automodule:: exactcis.methods.wald
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Wald method calculates confidence intervals using the normal approximation to the sampling distribution of the log odds ratio. This method is computationally efficient but may not provide accurate intervals for small sample sizes or rare events.

Key Functions
------------

- ``exact_ci_wald`` - Calculate confidence intervals using the Wald method
- ``wald_test`` - Perform a hypothesis test using the Wald method

Usage Example
------------

.. code-block:: python

   from exactcis.methods.wald import exact_ci_wald
   
   # Calculate a 95% confidence interval using the Wald method
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_wald(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")

Limitations
----------

The Wald method has several limitations:

1. It may not perform well for small sample sizes
2. It can produce inaccurate intervals for rare events
3. It may produce confidence intervals that include impossible values (e.g., negative odds ratios)