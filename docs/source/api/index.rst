API Reference
============

This section provides detailed API documentation for the ExactCIs package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core
   methods/index
   utils/index

Core Module
----------

The core module provides the fundamental functionality for calculating confidence intervals.

Methods
-------

The methods modules implement various confidence interval calculation methods:

* :doc:`methods/unconditional` - Unconditional exact confidence intervals
* :doc:`methods/conditional` - Conditional (Fisher's exact) confidence intervals
* :doc:`methods/blaker` - Blaker's confidence interval method

Utilities
---------

The utilities modules provide supporting functionality:

* :doc:`utils/parallel` - Parallel processing utilities
* :doc:`utils/shared_cache` - Shared inter-process cache implementation