#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define input and output directories
HTML_DIR="docs/build/html"
MARKDOWN_DIR="markdown_output"

# Ensure the output directory exists and is empty
rm -rf "$MARKDOWN_DIR"
mkdir -p "$MARKDOWN_DIR"

echo "Starting HTML to Markdown conversion..."

# Find all HTML files and convert them
# The -p option to mkdir ensures parent directories are created as needed
find "$HTML_DIR" -name "*.html" | while read -r html_file; do
  # Calculate the relative path from the HTML_DIR base
  relative_path="${html_file#$HTML_DIR/}"
  # Create the corresponding markdown file path
  markdown_file="$MARKDOWN_DIR/$relative_path"
  # Change the extension from .html to .md
  markdown_file="${markdown_file%.html}.md"

  # Create the target directory for the markdown file
  mkdir -p "$(dirname "$markdown_file")"

  echo "Converting $html_file to $markdown_file"

  # Run Pandoc to convert the file.
  # --from html: Specifies the input format.
  # --to gfm: Specifies the output format as GitHub Flavored Markdown, which is a good baseline.
  # --wrap=none: Prevents Pandoc from wrapping lines, which can interfere with formatting.
  # --strip-comments: Removes HTML comments.
  pandoc "$html_file" --from html --to gfm --wrap=none --strip-comments -o "$markdown_file"
done

# Copy static assets like images
# Check if the _static directory exists in the Sphinx output
if [ -d "$HTML_DIR/_static" ]; then
  echo "Copying static assets..."
  # Create a corresponding directory in the markdown output
  mkdir -p "$MARKDOWN_DIR/_static"
  cp -r "$HTML_DIR/_static/." "$MARKDOWN_DIR/_static/"
else
  echo "No _static directory found. Skipping asset copy."
fi

# Generate SUMMARY.md for GitBook navigation
echo "Generating SUMMARY.md for GitBook navigation..."

# Create SUMMARY.md file with title
cat > "$MARKDOWN_DIR/SUMMARY.md" << EOF
# Summary

* [Introduction](README.md)
EOF

# Function to add a file to SUMMARY.md with proper indentation
# Arguments:
#   $1: File path relative to MARKDOWN_DIR
#   $2: Indentation level (number of spaces)
#   $3: Title override (optional)
add_to_summary() {
  local file_path="$1"
  local indent="$2"
  local title="$3"
  
  # Skip if file doesn't exist
  if [ ! -f "$MARKDOWN_DIR/$file_path" ]; then
    return
  fi
  
  # Extract title from the first heading in the file if not provided
  if [ -z "$title" ]; then
    title=$(head -n 1 "$MARKDOWN_DIR/$file_path" | sed 's/^# //')
  fi
  
  # Add entry to SUMMARY.md with proper indentation
  printf "%*s* [%s](%s)\n" "$indent" "" "$title" "$file_path" >> "$MARKDOWN_DIR/SUMMARY.md"
}

# Add main sections following Diátaxis framework
echo "" >> "$MARKDOWN_DIR/SUMMARY.md"
echo "## Getting Started" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "installation.md" 0 "Installation"

echo "" >> "$MARKDOWN_DIR/SUMMARY.md"
echo "## User Guide" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "user_guide/index.md" 0 "User Guide Overview"
add_to_summary "user_guide/quick_reference.md" 2
add_to_summary "user_guide/examples/basic_usage.md" 2
add_to_summary "user_guide/examples/method_selection.md" 2
add_to_summary "user_guide/examples/rare_events.md" 2
add_to_summary "user_guide/troubleshooting.md" 2

echo "" >> "$MARKDOWN_DIR/SUMMARY.md"
echo "## API Reference" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "api-reference.md" 0 "API Overview"
add_to_summary "api/core.md" 2 "Core Module"

echo "### Methods" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "api/methods/blaker.md" 2 "Blaker's Method"
add_to_summary "api/methods/conditional.md" 2 "Conditional Method"
add_to_summary "api/methods/midp.md" 2 "Mid-P Method"
add_to_summary "api/methods/unconditional.md" 2 "Unconditional Method"
add_to_summary "api/methods/wald.md" 2 "Wald Method"

echo "### Utilities" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "api/utils/parallel.md" 2 "Parallel Processing"
add_to_summary "api/utils/shared_cache.md" 2 "Shared Cache"
add_to_summary "api/utils/stats.md" 2 "Statistics Utilities"
add_to_summary "api/utils/optimization.md" 2 "Optimization"
add_to_summary "api/cli.md" 2 "Command Line Interface"

echo "" >> "$MARKDOWN_DIR/SUMMARY.md"
echo "## Development" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "development/index.md" 0 "Development Overview"
add_to_summary "development/architecture.md" 2
add_to_summary "development/methodology.md" 2
add_to_summary "development/blaker_correction_guide.md" 2
add_to_summary "development/performance_optimization.md" 2

echo "### Analysis & Validation" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "development/method_comparison.md" 2
add_to_summary "development/implementation_comparison.md" 2
add_to_summary "development/validation_summary.md" 2
add_to_summary "development/test_monitoring.md" 2

echo "" >> "$MARKDOWN_DIR/SUMMARY.md"
echo "## Project Info" >> "$MARKDOWN_DIR/SUMMARY.md"
add_to_summary "contributing.md" 0 "Contributing"
add_to_summary "changelog.md" 0 "Changelog"

# Ensure the main index.md is copied to README.md for GitBook
if [ -f "$MARKDOWN_DIR/index.md" ]; then
  echo "Creating README.md from index.md..."
  cp "$MARKDOWN_DIR/index.md" "$MARKDOWN_DIR/README.md"
else
  echo "Warning: index.md not found. Creating a basic README.md..."
  echo "# ExactCIs: Exact Confidence Intervals for Odds Ratios" > "$MARKDOWN_DIR/README.md"
  echo "" >> "$MARKDOWN_DIR/README.md"
  echo "This is the documentation for the ExactCIs package." >> "$MARKDOWN_DIR/README.md"
fi

echo "SUMMARY.md generation complete."
echo "Conversion complete. Markdown files are in $MARKDOWN_DIR"