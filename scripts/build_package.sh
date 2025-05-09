#!/bin/bash
# Simple script to build the package for distribution

set -e

# Create the directory if it doesn't exist
mkdir -p scripts

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Clean previous builds
rm -rf dist build *.egg-info

# Install build dependencies
uv pip install -U build twine

# Build the package
echo "Building package..."
uv run python -m build

# Check the distribution
echo "Checking distribution..."
uv run twine check dist/*

echo "Build complete! Distribution files in ./dist/"
echo ""
echo "To install locally:"
echo "  uv pip install dist/*.whl"
echo ""
echo "To upload to PyPI:"
echo "  uv run twine upload dist/*"
echo ""
echo "To upload to TestPyPI (recommended for testing first):"
echo "  uv run twine upload --repository testpypi dist/*"
