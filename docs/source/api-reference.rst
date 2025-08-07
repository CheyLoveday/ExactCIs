API Reference
=============

This section provides detailed documentation for the ExactCIs package API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Core Module
-----------

The core module provides the fundamental functionality for calculating confidence intervals.

:doc:`api/core`

Methods
-------

ExactCIs implements various methods for calculating confidence intervals:

* :doc:`api/methods/blaker` - Blaker's confidence interval method
* :doc:`api/methods/conditional` - Conditional (Fisher's exact) confidence intervals
* :doc:`api/methods/midp` - Mid-P confidence intervals
* :doc:`api/methods/unconditional` - Unconditional exact confidence intervals
* :doc:`api/methods/wald` - Wald confidence intervals
* :doc:`api/methods/relative_risk` - Relative risk confidence intervals

Utilities
---------

The utilities modules provide supporting functionality:

* :doc:`api/utils/parallel` - Parallel processing utilities
* :doc:`api/utils/shared_cache` - Shared inter-process cache implementation
* :doc:`api/utils/stats` - Statistical utility functions
* :doc:`api/utils/optimization` - Optimization algorithms

Command Line Interface
--------------------

ExactCIs provides a command-line interface for quick calculations:

:doc:`api/cli`
