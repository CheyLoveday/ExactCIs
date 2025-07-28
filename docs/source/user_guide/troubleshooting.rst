Troubleshooting
==============

This guide covers common issues that might arise when using the ExactCIs package and provides solutions.

Performance Issues
----------------

Slow Computation
^^^^^^^^^^^^^

**Issue**: The confidence interval calculation is taking too long, especially with the unconditional method.

**Solutions**:

1. **Use a smaller grid size**:

   .. code-block:: python

      # Reduce grid size from the default 200 to 100
      results = exact_ci_unconditional(a, b, c, d, grid_size=100)

2. **Set a timeout**:

   .. code-block:: python

      # Set a 30-second timeout
      try:
          results = compute_all_cis(a, b, c, d, timeout=30)
      except TimeoutError:
          print("Computation timed out. Consider using another method.")

3. **Use a different method**:
   
   - For large tables, the conditional or mid-P methods are often much faster
   - The Wald method with Haldane correction is the fastest but less precise

4. **Use caching**:

   .. code-block:: python

      from exactcis.utils import CICache
      
      cache = CICache(max_size=100)
      
      # Make multiple calculations with the cache
      for table in tables:
          a, b, c, d = table
          ci = exact_ci_unconditional(a, b, c, d, cache_instance=cache)

Memory Usage
^^^^^^^^^^

**Issue**: High memory consumption during calculations.

**Solutions**:

1. **Reduce grid size**
2. **Clear the cache between batches**:

   .. code-block:: python

      cache.clear()  # Free up memory

Numerical Issues
--------------

Convergence Problems
^^^^^^^^^^^^^^^^^

**Issue**: Calculations fail to converge or produce errors like ``RuntimeError: Failed to converge``.

**Solutions**:

1. **Try custom bounds**:

   .. code-block:: python

      # Specify custom bounds for theta search
      ci = exact_ci_unconditional(a, b, c, d, theta_min=0.01, theta_max=100)

2. **Increase grid density**:

   .. code-block:: python

      ci = exact_ci_unconditional(a, b, c, d, grid_size=300)

Zero Cells
^^^^^^^^

**Issue**: Tables with zero cells cause errors or unusual results.

**Solutions**:

1. **Use the Haldane correction**:

   .. code-block:: python

      ci = ci_wald_haldane(a, b, c, d)  # Automatically adds 0.5 to each cell

2. **Use profile likelihood approach**:

   .. code-block:: python

      ci = exact_ci_unconditional(a, b, c, d, use_profile=True)

3. **Consider the scientific context** - Sometimes a zero cell represents a structural zero (impossibility) rather than a sampling zero

Error Messages
-----------

"Invalid table: negative counts"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution**: Ensure all counts are non-negative integers. Check data preprocessing steps.

"Invalid table: non-integer counts"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution**: Round your data or consider whether non-integer counts make sense in your context.

"Invalid alpha value"
^^^^^^^^^^^^^^^^^

**Solution**: The significance level alpha must be between 0 and 1, typically 0.05 (for 95% confidence).

"Cannot compute odds ratio with empty margins"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solution**: This occurs when an entire row or column is zero. The odds ratio is undefined in this case. Consider using a different measure of association.

Integration Issues
---------------

Import Errors
^^^^^^^^^^^

**Solution**: Ensure you've installed the package with the correct dependencies:

.. code-block:: bash

   # Basic installation
   pip install exactcis

   # With NumPy acceleration 
   pip install "exactcis[numpy]"

Version Compatibility
^^^^^^^^^^^^^^^^^

**Issue**: Code examples don't work with your installed version.

**Solutions**:

1. Check the version with ``import exactcis; print(exactcis.__version__)``
2. Upgrade to the latest version: ``pip install --upgrade exactcis``

Getting Additional Help
--------------------

If your issue is not covered here, you can:

1. Check the `GitHub issues <https://github.com/your-username/ExactCIs/issues>`_ for similar problems
2. Open a new issue with:
   
   - A minimal reproducible example
   - Your package version
   - Complete error message and stack trace