# Documentation Cleanup Guide

This document outlines the documentation consolidation that has been performed and identifies redundant files that could be removed in a future cleanup.

## Consolidation Summary

The documentation has been consolidated into the `docs/source` directory, which is now the single source of truth for the project's documentation. All Markdown files from other directories have been converted to reStructuredText (RST) format and placed in the appropriate location in the `docs/source` directory.

## Redundant Files

The following files have been consolidated and could be removed in a future cleanup:

### User Guide

- `docs/user_guide/quick_reference.md` → `docs/source/user_guide/quick_reference.rst`
- `docs/user_guide/examples/basic_usage.md` → `docs/source/user_guide/examples/basic_usage.rst`
- `docs/user_guide/examples/method_selection.md` → `docs/source/user_guide/examples/method_selection.rst`
- `docs/user_guide/examples/rare_events.md` → `docs/source/user_guide/examples/rare_events.rst`
- `docs/user_guide/troubleshooting.md` → `docs/source/user_guide/troubleshooting.rst`
- `docs/user_guide/user_guide.md` → Content integrated into various files in `docs/source/user_guide/`

### Development Documentation

- `docs/development/performance_optimization.md` → `docs/source/development/performance_optimization.rst`
- `docs/development/development_guide.md` → Content merged into `docs/source/contributing.rst`
- `docs/development/architecture.md` → `docs/source/development/architecture.rst` (already existed)
- `docs/development/methodology.md` → `docs/source/development/methodology.rst` (already existed)
- `docs/development/blaker_correction_guide.md` → `docs/source/development/blaker_correction_guide.rst` (already existed)
- `docs/development/method_comparison.md` → `docs/source/development/method_comparison.rst` (already existed)
- `docs/development/implementation_comparison.md` → `docs/source/development/implementation_comparison.rst` (already existed)
- `docs/development/validation_summary.md` → `docs/source/development/validation_summary.rst` (already existed)
- `docs/development/test_monitoring.md` → `docs/source/development/test_monitoring.rst` (already existed)
- `docs/development/Literature.md` → `docs/source/development/literature/literature.rst` (already existed)
- `docs/development/method_comparison_analysis.md` → `docs/source/development/literature/method_comparison_analysis.rst` (already existed)
- `docs/development/comparison_analysis_summary.md` → `docs/source/development/analysis/comparison_analysis_summary.rst` (already existed)

### API Reference

- `docs/api/api_reference.md` → Not converted, as `docs/source/api-reference.rst` uses autodoc to generate API documentation from docstrings

### Analysis/Literature

- `analysis/literature/Literature.md` → `docs/source/development/literature/literature.rst` (already existed)
- `analysis/literature/blaker_correction_guide.md` → `docs/source/development/blaker_correction_guide.rst` (already existed)
- `analysis/literature/comparison_analysis_summary.md` → `docs/source/development/analysis/comparison_analysis_summary.rst` (already existed)
- `analysis/literature/implementation_comparison.md` → `docs/source/development/implementation_comparison.rst` (already existed)
- `analysis/literature/method_comparison.md` → `docs/source/development/method_comparison.rst` (already existed)
- `analysis/literature/method_comparison_analysis.md` → `docs/source/development/literature/method_comparison_analysis.rst` (already existed)
- `analysis/literature/methodology.md` → `docs/source/development/methodology.rst` (already existed)
- `analysis/literature/performance_optimization.md` → `docs/source/development/performance_optimization.rst`
- `analysis/literature/validation_summary.md` → `docs/source/development/validation_summary.rst` (already existed)

## Cleanup Process

To clean up the redundant files:

1. Verify that all content has been properly consolidated and that the documentation builds successfully
2. Create a backup of the redundant files
3. Remove the redundant files
4. Update any references to the removed files in the codebase

This cleanup should be done as a separate task after the consolidation has been reviewed and approved.

## Benefits of Consolidation

The consolidation of documentation into the `docs/source` directory provides several benefits:

1. **Single Source of Truth**: All documentation is now in one place, making it easier to find and maintain
2. **Consistent Format**: All documentation is now in reStructuredText format, which is the standard for Sphinx documentation
3. **Automated API Documentation**: The API reference is now generated automatically from docstrings, ensuring it stays in sync with the code
4. **Improved Navigation**: The documentation is now organized into a clear hierarchy with proper cross-references
5. **Reduced Redundancy**: Duplicate content has been eliminated, reducing the risk of inconsistencies
6. **Better Maintainability**: The documentation is now easier to maintain and update