Mid-P Method
===========

The Mid-P method provides a less conservative alternative to Fisher's exact test by using the mid-p-value.

.. automodule:: exactcis.methods.midp
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Mid-P method adjusts the p-value calculation by using the mid-p-value, which is the usual p-value minus half the probability of the observed outcome. This adjustment makes the test less conservative than Fisher's exact test while still maintaining good coverage properties.

Key Functions
------------

- ``exact_ci_midp`` - Calculate confidence intervals using the Mid-P method
- ``midp_test`` - Perform a hypothesis test using the Mid-P method

Usage Example
------------

.. code-block:: python

   from exactcis.methods.midp import exact_ci_midp
   
   # Calculate a 95% confidence interval using the Mid-P method
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_midp(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")