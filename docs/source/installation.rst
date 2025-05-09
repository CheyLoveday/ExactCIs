Installation
============

Requirements
-----------

ExactCIs requires Python 3.8 or newer. The package has the following core dependencies:

* numpy (>=1.20.0)
* scipy (>=1.7.0)

Install from PyPI
----------------

The recommended way to install ExactCIs is via pip:

.. code-block:: bash

   pip install exactcis

This will install the core package with all required dependencies.

Optional Dependencies
-------------------

ExactCIs has several optional dependency sets that can be installed depending on your needs:

.. code-block:: bash

   # Install with pandas support
   pip install exactcis[pandas]

   # Install with visualization support
   pip install exactcis[viz]

   # Install with all optional dependencies
   pip install exactcis[full]

Development Installation
-----------------------

For development purposes, you may want to install ExactCIs with all development dependencies:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/exactcis.git
   cd exactcis

   # Install in development mode with dev dependencies
   pip install -e ".[dev]"
