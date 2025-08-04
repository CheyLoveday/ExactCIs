# ExactCIs Documentation

## Overview

ExactCIs is a Python package for calculating exact confidence intervals for 2×2 contingency tables, with a focus on providing robust and valid statistical inference even in challenging scenarios such as small sample sizes and rare events.

This documentation provides comprehensive information about the package, its methodology, and guidance on when to use different approaches.

## Documentation Structure

The documentation is organized using Sphinx, a popular documentation generator for Python projects. The structure follows standard Python documentation practices:

```
docs/
├── Makefile                  # Build script for Sphinx
├── build/                    # Built documentation (output)
├── source/                   # Source files for Sphinx
│   ├── _static/              # Static files (CSS, images, etc.)
│   ├── _templates/           # Custom Sphinx templates
│   ├── api/                  # API reference documentation
│   │   ├── core.rst          # Core module API
│   │   ├── methods/          # Methods API documentation
│   │   └── utils/            # Utils API documentation
│   ├── user_guide/           # User guide documentation
│   ├── development/          # Development documentation
│   ├── conf.py               # Sphinx configuration
│   └── index.rst             # Main index file
└── README.md                 # This file
```

## Building the Documentation

To build the documentation:

1. Install the required dependencies:
   ```bash
   pip install sphinx sphinx_rtd_theme m2r2
   ```

2. Build the HTML documentation:
   ```bash
   cd docs
   make html
   ```

3. The built documentation will be available in `docs/build/html/index.html`.

## Contributing to the Documentation

When contributing to the documentation, please follow these guidelines:

1. **API Documentation**: API documentation is automatically generated from docstrings. Update the docstrings in the source code rather than editing the RST files directly.

2. **User Guide**: The user guide is written in reStructuredText (.rst) format. Edit the files in `docs/source/user_guide/`.

3. **Development Documentation**: Documentation for developers is in `docs/source/development/`.

4. **Examples**: Add examples to `docs/source/user_guide/examples/`.

5. **Images**: Place images in `docs/source/_static/images/`.

## Documentation Standards

- Use NumPy-style docstrings in the source code.
- Write clear, concise documentation with examples.
- Include mathematical formulas using LaTeX syntax where appropriate.
- Cross-reference related documentation using Sphinx's cross-referencing syntax.
- Test code examples to ensure they work as expected.

## Quick Start

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example 2×2 table
#      Success   Failure
# Grp1    7         3
# Grp2    2         8

# Calculate 95% confidence interval for the odds ratio
lower, upper = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05)
print(f"95% CI for odds ratio: ({lower:.6f}, {upper:.6f})")
```

## Documentation Sections

- **API Reference**: Detailed documentation of all functions, classes, and parameters.
- **User Guide**: Step-by-step guides for using ExactCIs.
- **Development**: Information for developers contributing to ExactCIs.
- **Examples**: Practical examples demonstrating ExactCIs usage.

## References & Citations

See the References section in the built documentation for academic references and how to cite this package.
