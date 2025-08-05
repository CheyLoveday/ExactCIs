# Documentation Maintenance Guide

This guide documents the implementation of the documentation engineering tasks for the ExactCIs project and provides instructions for future maintenance.

## Overview

The ExactCIs documentation system has been enhanced with several features to improve quality, maintainability, and user experience:

1. **Conversion Fidelity Testing**: A structured approach for testing and ensuring the fidelity of documentation conversion from Sphinx to GitBook.
2. **Automated Quality Checks**: Automated checks for broken links and other quality issues in the documentation.
3. **Version Management**: A strategy for maintaining versioned documentation for different releases of the library.

This guide explains how these features work and how to maintain them.

## 1. Conversion Fidelity Testing

### Implementation

A comprehensive test matrix has been created to systematically test the conversion from Sphinx (reStructuredText) to GitBook (Markdown):

- **File**: `docs/conversion_test_matrix.md`
- **Purpose**: Provides a structured approach for testing the fidelity of documentation conversion.
- **Features**:
  - Lists all documentation pages
  - Identifies element types that require special attention (cross-references, admonitions, tables, etc.)
  - Includes a testing procedure and issue tracking template
  - Provides prioritization guidelines

### Maintenance Instructions

To maintain the conversion fidelity testing process:

1. **Regular Testing**:
   - After significant documentation updates, run through the test matrix for affected pages
   - Focus on pages with complex elements (tables, math equations, etc.)
   - Document any issues found using the issue tracking template

2. **Updating the Test Matrix**:
   - When new pages are added to the documentation, add them to the test matrix
   - When new element types are used, add them to the list of elements to test
   - Update the test matrix as issues are found and resolved

3. **Continuous Improvement**:
   - Apply lessons learned to improve both the conversion process and source content
   - Update the testing guidelines as new issues are discovered

## 2. Automated Quality Checks

### Implementation

Automated checks have been implemented to ensure the quality of the documentation:

1. **Link Checker**:
   - **File**: `scripts/check_links.py`
   - **Purpose**: Checks for broken links in the converted Markdown files
   - **Features**:
     - Identifies internal links that point to non-existent files or anchors
     - Optionally checks external links
     - Generates a detailed report of broken links

2. **GitHub Actions Workflow**:
   - **File**: `.github/workflows/docs-to-gitbook.yml`
   - **Purpose**: Automates the documentation build, conversion, and deployment process
   - **Features**:
     - Runs the link checker after conversion
     - Creates issues for broken links
     - Monitors GitBook sync status
     - Includes a scheduled job for regular link validation

### Maintenance Instructions

To maintain the automated quality checks:

1. **Link Checker**:
   - Run the link checker locally before committing significant documentation changes:
     ```bash
     python scripts/check_links.py --markdown-dir ./markdown_output
     ```
   - Review and fix any broken links found

2. **GitHub Actions Workflow**:
   - Monitor the workflow runs in the GitHub Actions tab
   - Review any issues created by the workflow for broken links
   - Check the workflow logs for warnings about GitBook sync status

3. **Scheduled Checks**:
   - The workflow includes a scheduled job that runs weekly to check all links
   - Review the artifacts and issues from these scheduled runs
   - Fix any broken links found, especially external links that may break over time

## 3. Version Management

### Implementation

A version management strategy has been implemented to maintain documentation for different versions of the library:

1. **Strategy Document**:
   - **File**: `docs/version_management_strategy.md`
   - **Purpose**: Outlines the approach for managing versioned documentation
   - **Features**:
     - Explains GitBook's versioning capabilities
     - Defines the Git branch and tag structure
     - Describes the documentation build and deployment process for different versions

2. **Version Release Workflow**:
   - **File**: `.github/workflows/release-docs-version.yml`
   - **Purpose**: Automates the process of releasing documentation for a specific version
   - **Features**:
     - Creates a version-specific documentation branch
     - Builds and converts the documentation
     - Adds version information to the README
     - Deploys to a version-specific GitBook sync branch

3. **Main Workflow Update**:
   - **File**: `.github/workflows/docs-to-gitbook.yml`
   - **Purpose**: Includes version information in the main documentation
   - **Features**:
     - Adds a "Versions" section to the README
     - Informs users about the version selector in GitBook

### Maintenance Instructions

To maintain the version management system:

1. **Releasing a New Version**:
   - When releasing a new version of ExactCIs:
     1. Complete the code changes and testing for the new version
     2. Update the version number in the project files
     3. Create a Git tag for the new version
     4. Run the "Release Documentation Version" workflow with the new version number:
        - Go to the Actions tab in GitHub
        - Select the "Release Documentation Version" workflow
        - Click "Run workflow"
        - Enter the version number (e.g., v1.0.0)
        - Click "Run workflow"
     5. Configure a new GitBook variant for the version:
        - Go to the GitBook space settings
        - Navigate to the "Variants" section
        - Click "Add variant"
        - Name it according to the version (e.g., "v1.0.0")
        - Configure Git Sync to use the `gitbook-sync-vX.Y.Z` branch
     6. Update the version selector in GitBook to include the new version

2. **Maintaining Versions**:
   - Regularly review the list of versions in GitBook
   - Consider archiving or removing very old versions if they are no longer relevant
   - Ensure that the version selector is prominently displayed in the GitBook interface

3. **Documentation Consistency**:
   - Ensure that the documentation structure remains consistent across versions
   - When making structural changes, consider whether they should be backported to older versions

## 4. Future Improvements

Consider the following improvements to further enhance the documentation system:

1. **Automated Testing**:
   - Develop automated tests for conversion fidelity
   - Implement visual regression testing for the GitBook output

2. **Enhanced Monitoring**:
   - Add more detailed monitoring of the GitBook sync process
   - Implement alerts for documentation quality issues

3. **User Feedback**:
   - Add a feedback mechanism to the documentation
   - Collect and analyze user feedback to identify areas for improvement

4. **Documentation Analytics**:
   - Implement analytics to track documentation usage
   - Use analytics data to prioritize documentation improvements

## Conclusion

The documentation system for ExactCIs has been significantly enhanced with features for testing, quality assurance, and version management. By following the maintenance instructions in this guide, you can ensure that the documentation remains high-quality, up-to-date, and accessible to users of all versions of the library.

Remember that documentation is an ongoing process. Regularly review and update the documentation as the library evolves, and continue to improve the documentation system based on user feedback and changing requirements.