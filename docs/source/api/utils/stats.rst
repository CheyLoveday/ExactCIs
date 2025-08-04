Statistics
==========

The statistics module provides utility functions for statistical calculations.

.. automodule:: exactcis.utils.stats
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module includes functions for calculating various statistical measures and probabilities related to 2x2 contingency tables.

Key Functions
------------

- Functions for calculating odds ratios and relative risks
- Functions for calculating p-values
- Functions for calculating confidence intervals
- Utility functions for statistical calculations

Usage Example
------------

.. code-block:: python

   from exactcis.utils.stats import odds_ratio
   
   # Calculate the odds ratio for a 2x2 table: [[10, 20], [5, 25]]
   or_value = odds_ratio(10, 20, 5, 25)
   print(f"Odds Ratio: {or_value:.4f}")