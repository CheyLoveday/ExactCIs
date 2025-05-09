#!/bin/bash
# Script to set up the development environment with Python 3.11

set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Activate the virtual environment
source .venv/bin/activate

# Install the package in development mode with all dependencies
uv pip install -e ".[dev]"

# Install additional dependencies for testing and building docs
uv pip install pytest pytest-cov sphinx sphinx-rtd-theme build twine

echo "Development environment is ready!"
echo ""
echo "To activate the environment in your shell, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To build documentation:"
echo "  cd docs && make html"
