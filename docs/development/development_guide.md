# Development Guide

This document provides guidelines for contributors to the ExactCIs package.

## Code Style

ExactCIs follows standard Python coding conventions:

- We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Line length limited to 88 characters (compatible with Black)
- Use 4 spaces for indentation (no tabs)
- Use docstrings for all public modules, functions, classes, and methods

## Documentation

### Docstring Format

ExactCIs follows the [NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html). All public functions and classes should include:

- Short summary
- Extended description (if needed)
- Parameters section
- Returns section
- Raises section (if applicable)
- See Also section (if applicable)
- Notes section (if applicable)
- Examples section with runnable code

Example:

```python
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
```

### Documentation Structure

The documentation follows this overall structure, similar to SciPy:

- **User Guide**: High-level overview and introduction
- **API Reference**: Detailed function documentation
- **Examples**: Executable code examples
- **Architecture**: Package design and internals
- **Development Guide**: For contributors

## Testing

- All new features should include unit tests
- Run tests with `pytest`
- Include both fast unit tests and slower integration tests
- Use `@pytest.mark.slow` to mark tests that take longer to run

## Version Numbering

ExactCIs follows [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation as needed
6. Submit a pull request

## Recommended Development Tools

- **Black**: For code formatting
- **isort**: For import sorting
- **pylint** or **flake8**: For code quality checks
- **mypy**: For optional static type checking
- **pytest**: For testing
- **pytest-cov**: For test coverage

## Building Documentation

The documentation is built using Sphinx with the numpydoc extension:

```bash
cd docs
make html
```

## Release Process

1. Update version number in `setup.py`
2. Update changelog (`CHANGELOG.md`)
3. Create distribution packages: `python -m build`
4. Upload to PyPI: `python -m twine upload dist/*`
5. Create a new GitHub release with release notes
