Command Line Interface
=====================

ExactCIs provides a command-line interface for quick calculations without writing Python code.

.. automodule:: exactcis.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The command-line interface allows users to calculate confidence intervals for 2x2 contingency tables directly from the terminal. It supports all the methods implemented in ExactCIs and provides options for customizing the calculation.

Usage
-----

.. code-block:: bash

   # Basic usage
   exactcis --a 10 --b 20 --c 5 --d 25 --method blaker

   # Specify confidence level (default is 0.95)
   exactcis --a 10 --b 20 --c 5 --d 25 --method conditional --alpha 0.01

   # Get help
   exactcis --help

Options
-------

- ``--a``, ``--b``, ``--c``, ``--d``: The four cells of the 2x2 contingency table
- ``--method``: The method to use (blaker, conditional, midp, unconditional, wald)
- ``--alpha``: The significance level (default: 0.05)
- ``--alternative``: The alternative hypothesis (two-sided, less, greater)
- ``--statistic``: The statistic to calculate (odds-ratio, relative-risk)
- ``--format``: The output format (text, json, csv)
- ``--verbose``: Enable verbose output
- ``--help``: Show help message

Examples
--------

Calculate a 95% confidence interval for the odds ratio using Blaker's method:

.. code-block:: bash

   exactcis --a 10 --b 20 --c 5 --d 25 --method blaker

Calculate a 99% confidence interval for the relative risk using the conditional method:

.. code-block:: bash

   exactcis --a 10 --b 20 --c 5 --d 25 --method conditional --alpha 0.01 --statistic relative-risk

Output the results in JSON format:

.. code-block:: bash

   exactcis --a 10 --b 20 --c 5 --d 25 --method unconditional --format json