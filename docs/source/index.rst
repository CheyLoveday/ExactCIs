ExactCIs: Exact Confidence Intervals for Odds Ratios
====================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide/index
   api-reference
   contributing
   changelog
   development/index

ExactCIs is a Python package that provides methods for calculating exact confidence intervals for odds ratios in 2x2 contingency tables. It implements several statistical methods including Blaker's exact method, conditional exact method, mid-P method, and unconditional exact method.

Features
--------

* Multiple methods for computing exact confidence intervals
* Includes corrections for zero-cells (Haldane's correction)
* Command-line interface for quick calculations
* Comprehensive test suite and validation
* Fully typed API with detailed documentation

Quick Install
------------

.. code-block:: bash

   pip install exactcis

Simple Example
-------------

.. code-block:: python

   from exactcis.methods import exact_ci_blaker
   
   # Calculate a 95% confidence interval using Blaker's method
   # for a 2x2 table: [[10, 20], [5, 25]]
   lower, upper = exact_ci_blaker(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI: ({lower:.4f}, {upper:.4f})")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
