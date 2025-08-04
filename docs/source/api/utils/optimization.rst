Optimization
============

The optimization module provides utility functions for numerical optimization.

.. automodule:: exactcis.utils.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module includes functions for numerical optimization, particularly for finding confidence interval bounds through root-finding and optimization algorithms.

Key Functions
------------

- Root-finding algorithms for p-value functions
- Bracket expansion algorithms for finding initial bounds
- Optimization algorithms for finding confidence interval bounds
- Utility functions for numerical optimization

Usage Example
------------

.. code-block:: python

   from exactcis.utils.optimization import find_root
   
   # Example of using the find_root function
   def f(x):
       return x**2 - 4  # Find the root of x^2 - 4 = 0
   
   root = find_root(f, 1, 3)  # Should find x = 2
   print(f"Root: {root:.6f}")

Implementation Details
--------------------

The optimization module implements several numerical algorithms:

1. **Bisection Method**: A simple root-finding method that repeatedly bisects an interval and selects the subinterval containing the root.

2. **Brent's Method**: A more efficient root-finding algorithm that combines the bisection method, the secant method, and inverse quadratic interpolation.

3. **Bracket Expansion**: An algorithm for finding an interval that contains a root by expanding the initial bracket.

4. **Adaptive Search**: An algorithm that adaptively adjusts the search strategy based on the function behavior.