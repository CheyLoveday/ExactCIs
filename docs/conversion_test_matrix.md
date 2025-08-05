# Conversion Fidelity Test Matrix

This document provides a structured approach for testing the fidelity of documentation conversion from Sphinx (reStructuredText) to GitBook (Markdown).

## Test Matrix Structure

The test matrix is organized with documentation pages in rows and element types in columns. Each cell indicates whether a specific element type appears on a specific page and should be tested.

### Element Types to Test

Based on the guidelines in `conversion_fidelity_testing.md`, the following element types require special attention:

1. **Cross-References**
   - References to other documents (`:doc:`)
   - References to specific sections (`:ref:`)
   - References to API elements (`:py:class:`, `:py:func:`, etc.)

2. **Admonitions**
   - Notes
   - Warnings
   - Tips
   - Important notices

3. **Tables**
   - Simple tables
   - Grid tables
   - Tables with complex headers
   - Tables with list items or code blocks

4. **Code Blocks**
   - Different programming languages
   - Line numbers
   - Emphasized lines
   - Code blocks within lists or tables

5. **Math Equations**
   - Inline math
   - Display math
   - Complex equations with special symbols

6. **Images and Figures**
   - Simple images
   - Figures with captions
   - Images with size specifications
   - SVG or other vector graphics

7. **Custom Sphinx Extensions and Directives**
   - Any custom extensions used in the documentation

## Documentation Pages

Based on the `convert_html_to_md.sh` script, the following pages are included in the documentation:

### Getting Started
- installation.md

### User Guide
- user_guide/index.md
- user_guide/quick_reference.md
- user_guide/examples/basic_usage.md
- user_guide/examples/method_selection.md
- user_guide/examples/rare_events.md
- user_guide/troubleshooting.md

### API Reference
- api-reference.md
- api/core.md
- api/methods/blaker.md
- api/methods/conditional.md
- api/methods/midp.md
- api/methods/unconditional.md
- api/methods/wald.md
- api/utils/parallel.md
- api/utils/shared_cache.md
- api/utils/stats.md
- api/utils/optimization.md
- api/cli.md

### Development
- development/index.md
- development/architecture.md
- development/methodology.md
- development/blaker_correction_guide.md
- development/performance_optimization.md
- development/method_comparison.md
- development/implementation_comparison.md
- development/validation_summary.md
- development/test_monitoring.md

### Project Info
- contributing.md
- changelog.md

## Test Matrix

| Page | Cross-References | Admonitions | Tables | Code Blocks | Math Equations | Images | Custom Directives |
|------|------------------|-------------|--------|-------------|----------------|--------|-------------------|
| installation.md | | | | | | | |
| user_guide/index.md | | | | | | | |
| user_guide/quick_reference.md | | | | | | | |
| user_guide/examples/basic_usage.md | | | | | | | |
| user_guide/examples/method_selection.md | | | | | | | |
| user_guide/examples/rare_events.md | | | | | | | |
| user_guide/troubleshooting.md | | | | | | | |
| api-reference.md | | | | | | | |
| api/core.md | | | | | | | |
| api/methods/blaker.md | | | | | | | |
| api/methods/conditional.md | | | | | | | |
| api/methods/midp.md | | | | | | | |
| api/methods/unconditional.md | | | | | | | |
| api/methods/wald.md | | | | | | | |
| api/utils/parallel.md | | | | | | | |
| api/utils/shared_cache.md | | | | | | | |
| api/utils/stats.md | | | | | | | |
| api/utils/optimization.md | | | | | | | |
| api/cli.md | | | | | | | |
| development/index.md | | | | | | | |
| development/architecture.md | | | | | | | |
| development/methodology.md | | | | | | | |
| development/blaker_correction_guide.md | | | | | | | |
| development/performance_optimization.md | | | | | | | |
| development/method_comparison.md | | | | | | | |
| development/implementation_comparison.md | | | | | | | |
| development/validation_summary.md | | | | | | | |
| development/test_monitoring.md | | | | | | | |
| contributing.md | | | | | | | |
| changelog.md | | | | | | | |

## Testing Procedure

1. For each page in the documentation:
   - Build the Sphinx documentation
   - Convert to Markdown using the conversion script
   - View the page in GitBook (or locally)
   - Check each element type that appears on the page
   - Document any issues found

2. For each issue found:
   - Document the page and element type
   - Provide before/after screenshots
   - Describe the issue
   - Suggest a solution

## Issue Tracking Template

For each issue found, use the following template:

```markdown
### Issue: [Brief Description]

**Page:** [Page Path]
**Element Type:** [Element Type]
**Severity:** [High/Medium/Low]

**Description:**
[Detailed description of the issue]

**Before (Sphinx):**
[Screenshot or description of how it appears in Sphinx]

**After (GitBook):**
[Screenshot or description of how it appears in GitBook]

**Proposed Solution:**
[Suggested fix for the issue]

**Status:** [Open/In Progress/Resolved]
```

## Prioritization

Testing should be prioritized based on:

1. **Page Importance**:
   - User Guide pages (highest priority)
   - API Reference pages (high priority)
   - Development pages (medium priority)
   - Project Info pages (lower priority)

2. **Element Complexity**:
   - Complex elements (tables, math equations, custom directives)
   - Elements that are known to have conversion issues

## Continuous Improvement

This test matrix should be updated as:

1. New pages are added to the documentation
2. New element types are used in the documentation
3. Issues are found and resolved
4. The conversion process is improved

Regular testing using this matrix will ensure that the documentation maintains high quality across both Sphinx and GitBook platforms.