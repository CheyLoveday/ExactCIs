Usage Guide
===========

This guide provides examples and best practices for using the ExactCIs package.

Basic Usage
----------

ExactCIs provides several methods for calculating confidence intervals for odds ratios in 2x2 contingency tables.

The basic inputs for all methods are:

* ``a``: Count in cell (1,1) - exposed cases
* ``b``: Count in cell (1,2) - exposed controls
* ``c``: Count in cell (2,1) - unexposed cases
* ``d``: Count in cell (2,2) - unexposed controls
* ``alpha``: Significance level (default 0.05 for 95% confidence interval)

Example
~~~~~~~

.. code-block:: python

    from exactcis.methods import exact_ci_blaker

    # Calculate 95% confidence interval using Blaker's method
    a, b, c, d = 10, 15, 5, 20  # 2x2 table: [[10, 15], [5, 20]]
    lower, upper = exact_ci_blaker(a, b, c, d, alpha=0.05)
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"95% CI: ({lower:.4f}, {upper:.4f})")

Available Methods
----------------

ExactCIs implements the following confidence interval methods:

Blaker's Exact Method
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from exactcis.methods import exact_ci_blaker
    
    lower, upper = exact_ci_blaker(a, b, c, d, alpha=0.05)

Conditional Exact Method
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from exactcis.methods import exact_ci_conditional
    
    lower, upper = exact_ci_conditional(a, b, c, d, alpha=0.05)

Mid-P Method
~~~~~~~~~~~

.. code-block:: python

    from exactcis.methods import exact_ci_midp
    
    lower, upper = exact_ci_midp(a, b, c, d, alpha=0.05)

Unconditional Exact Method
~~~~~~~~~~~~~~~~~~~~~~~~

The unconditional method accepts an additional parameter, ``grid_size``, which controls the precision of the grid search:

.. code-block:: python

    from exactcis.methods import exact_ci_unconditional
    
    lower, upper = exact_ci_unconditional(a, b, c, d, alpha=0.05, grid_size=20)

Wald Method with Haldane's Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from exactcis.methods import ci_wald_haldane
    
    lower, upper = ci_wald_haldane(a, b, c, d, alpha=0.05)

Handling Zero Cells
------------------

When one or more cells in the 2x2 table contain zeros, you may want to apply Haldane's correction, which adds 0.5 to each cell:

.. code-block:: python

    from exactcis.core import apply_haldane_correction
    from exactcis.methods import exact_ci_blaker
    
    # Original table with a zero cell
    a, b, c, d = 0, 10, 5, 15
    
    # Apply Haldane's correction
    a_corr, b_corr, c_corr, d_corr = apply_haldane_correction(a, b, c, d)
    
    # Calculate CI using corrected values
    lower, upper = exact_ci_blaker(a_corr, b_corr, c_corr, d_corr, alpha=0.05)

Command-Line Interface
--------------------

ExactCIs also provides a command-line interface for quick calculations:

.. code-block:: bash

    # Basic usage
    exactcis-cli 10 15 5 20 --method blaker
    
    # With additional options
    exactcis-cli 10 15 5 20 --method unconditional --alpha 0.01 --grid-size 30 --apply-haldane --verbose
    
For more information on the CLI options:

.. code-block:: bash

    exactcis-cli --help

Method Selection Guide
--------------------

Which method should you choose? Here's a brief guide:

* **Blaker's method**: Generally preferred for most cases. Produces narrower intervals than the conditional method while maintaining the correct coverage.
* **Conditional method**: The most conservative approach, guaranteeing at least the nominal coverage level.
* **Mid-P method**: Often produces intervals with better average performance but may undercover in some cases.
* **Unconditional method**: Computationally intensive but can produce more accurate intervals in certain scenarios.
* **Wald method with Haldane's correction**: A simple approximation, useful mainly for comparison or when computational resources are limited.
