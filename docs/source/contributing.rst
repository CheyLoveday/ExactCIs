Contributing
============

Thank you for your interest in contributing to ExactCIs! This document provides guidelines and instructions for contributing to the project.

Table of Contents
----------------

#. Code of Conduct
#. Getting Started
#. Development Environment
#. Making Changes
#. Testing
#. Documentation
#. Submitting Changes
#. Review Process

Code of Conduct
--------------

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when contributing to the project.

Getting Started
-------------

#. **Fork the repository** on GitHub
#. **Clone your fork** to your local machine
#. **Set up the development environment** as described below

Development Environment
---------------------

ExactCIs uses ``uv`` for dependency management. To set up your development environment:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/exactcis.git
   cd exactcis

   # Create and activate a virtual environment
   uv venv

   # Activate the environment
   source .venv/bin/activate  # Unix/macOS
   # or
   .venv\Scripts\activate  # Windows

   # Install development and documentation dependencies
   uv pip install -e ".[dev,docs]"

Making Changes
------------

#. **Create a new branch** for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

#. **Follow the coding style**:

   * Use `Black <https://black.readthedocs.io/>`_ for code formatting
   * Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines
   * Use type hints consistently
   * Write docstrings for all functions, classes, and modules

#. **Keep changes focused**:

   * Each pull request should address a single concern
   * Break large changes into smaller, logical commits

Testing
------

All changes must include appropriate tests:

#. **Run existing tests** to ensure your changes don't break existing functionality:

   .. code-block:: bash

      uv run pytest

#. **Add new tests** for new functionality:

   * Unit tests for individual functions
   * Integration tests for interactions between components
   * Edge case tests for boundary conditions

#. **Run the full test suite** including slow tests:

   .. code-block:: bash

      uv run pytest --run-slow

#. **Check test coverage**:

   .. code-block:: bash

      uv run pytest --cov=src/exactcis

Documentation
-----------

Documentation is a critical part of ExactCIs:

#. **Update docstrings** for any modified functions or classes
#. **Update user documentation** in the ``docs/`` directory
#. **Add examples** for new functionality
#. **Update the API reference** if you've added or changed public interfaces

Docstring Format
^^^^^^^^^^^^^

ExactCIs follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_. All public functions and classes should include:

- Short summary
- Extended description (if needed)
- Parameters section
- Returns section
- Raises section (if applicable)
- See Also section (if applicable)
- Notes section (if applicable)
- Examples section with runnable code

Example:

.. code-block:: python

   def exact_ci_conditional(a, b, c, d, alpha=0.05):
       """
       Calculate Fisher's exact conditional confidence interval for the odds ratio.
       
       Parameters
       ----------
       a : int
           Count in cell (1,1) - successes in group 1
       b : int
           Count in cell (1,2) - failures in group 1
       c : int
           Count in cell (2,1) - successes in group 2
       d : int
           Count in cell (2,2) - failures in group 2
       alpha : float, default=0.05
           Significance level (1-confidence level)
       
       Returns
       -------
       tuple
           Lower and upper bounds of the confidence interval
       
       Raises
       ------
       ValueError
           If any count is negative or if any margin is zero
       
       Examples
       --------
       >>> from exactcis.methods import exact_ci_conditional
       >>> lower, upper = exact_ci_conditional(12, 5, 8, 10, alpha=0.05)
       >>> print(f"95% CI: ({lower:.3f}, {upper:.3f})")
       95% CI: (1.059, 8.726)
       """
       
Building Documentation
^^^^^^^^^^^^^^^^^

The documentation is built using Sphinx with the numpydoc extension:

.. code-block:: bash

   cd docs
   make html

Submitting Changes
----------------

#. **Commit your changes** with clear, descriptive commit messages:

   .. code-block:: bash

      git commit -m "Add feature: brief description of what was added"

#. **Push your changes** to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

#. **Submit a pull request** from your fork to the main repository
#. **Fill out the pull request template** with all relevant information

Review Process
------------

All pull requests will go through a review process:

#. Automated checks for code style, test coverage, and documentation
#. Code review by at least one maintainer
#. Feedback and requested changes
#. Merge once all issues are resolved

Version Numbering
--------------

ExactCIs follows `Semantic Versioning <https://semver.org/>`_:

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

Release Process
------------

When preparing a new release:

#. Update version number in ``pyproject.toml``
#. Update changelog (``CHANGELOG.md``)
#. Create distribution packages: ``python -m build``
#. Upload to PyPI: ``python -m twine upload dist/*``
#. Create a new GitHub release with release notes
