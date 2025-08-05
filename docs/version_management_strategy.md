# Documentation Version Management Strategy

This document outlines the strategy for managing versioned documentation for the ExactCIs project using GitBook and Git tags.

## Overview

As the ExactCIs library evolves, it's important to maintain documentation for different versions of the library. Users may be using older versions of the library and need access to the corresponding documentation. This strategy ensures that:

1. Documentation for each released version is preserved and accessible
2. Users can easily find documentation for the specific version they're using
3. The documentation build and deployment process remains automated
4. The maintenance burden for documentation maintainers is minimized

## GitBook Versioning Capabilities

GitBook provides built-in support for versioned documentation through its "Variants" feature:

1. **Variants**: GitBook allows creating multiple variants of the same documentation space, each with its own content.
2. **Version Selector**: A dropdown menu can be added to the documentation site, allowing users to switch between versions.
3. **Default Version**: One version can be designated as the default, which users see when they first visit the documentation.
4. **Branch-Based Sync**: Each variant can sync with a different branch in the Git repository.

## Version Management Strategy

### 1. Git Branch and Tag Structure

We will use the following branch and tag structure to manage documentation versions:

1. **`main` branch**: Contains the latest development version of the documentation.
2. **`gitbook-sync` branch**: Contains the converted Markdown files for the latest development version.
3. **Version branches**: For each released version, we'll create a branch named `docs-vX.Y.Z` (e.g., `docs-v1.0.0`).
4. **Version tags**: Each release will have a tag in the format `vX.Y.Z` (e.g., `v1.0.0`).

### 2. Documentation Build and Deployment Process

The documentation build and deployment process will be enhanced to support versioning:

1. **For the latest development version**:
   - The existing GitHub Actions workflow will continue to build and deploy documentation from the `main` branch to the `gitbook-sync` branch.
   - This will be synced with the "Latest" variant in GitBook.

2. **For released versions**:
   - When a new version is released, a new GitHub Actions workflow will be triggered.
   - This workflow will:
     - Create a new branch `docs-vX.Y.Z` from the current state of `main`
     - Build the documentation and convert it to Markdown
     - Push the Markdown files to a branch named `gitbook-sync-vX.Y.Z`
     - This branch will be synced with a new variant in GitBook for the specific version

### 3. Implementation Plan

#### Step 1: Create a Version Release Workflow

Create a new GitHub Actions workflow file `.github/workflows/release-docs-version.yml`:

```yaml
name: Release Documentation Version

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.0.0)'
        required: true

permissions:
  contents: write

jobs:
  release-docs-version:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Validate version format
        run: |
          if ! [[ ${{ github.event.inputs.version }} =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Invalid version format. Must be in the format vX.Y.Z (e.g., v1.0.0)"
            exit 1
          fi
      
      - name: Create documentation branch
        run: |
          VERSION=${{ github.event.inputs.version }}
          git checkout -b docs-$VERSION
          git push origin docs-$VERSION
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,docs]"
          pip install requests
      
      - name: Install Pandoc
        run: |
          sudo apt-get update
          sudo apt-get install -y pandoc
      
      - name: Build Sphinx docs
        run: |
          cd docs
          make html
      
      - name: Convert HTML to Markdown
        run: |
          chmod +x scripts/convert_html_to_md.sh
          ./scripts/convert_html_to_md.sh
      
      - name: Check for broken links
        id: link_check
        run: |
          chmod +x scripts/check_links.py
          python scripts/check_links.py --markdown-dir ./markdown_output
          echo "link_check_status=$?" >> $GITHUB_OUTPUT
        continue-on-error: true
      
      - name: Upload link check report
        uses: actions/upload-artifact@v3
        with:
          name: link-check-report-${{ github.event.inputs.version }}
          path: |
            link_check_report.md
            link_check_results.log
      
      - name: Create issue for broken links
        if: steps.link_check.outputs.link_check_status != '0'
        uses: peter-evans/create-issue-from-file@v4
        with:
          title: "Broken links found in documentation for ${{ github.event.inputs.version }}"
          content-filepath: ./link_check_report.md
          labels: documentation, bug
      
      - name: Deploy to versioned GitBook sync branch
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./markdown_output
          publish_branch: gitbook-sync-${{ github.event.inputs.version }}
          force_orphan: true
          user_name: 'github-actions[bot]'
          user_email: 'github-actions[bot]@users.noreply.github.com'
          commit_message: "Deploy: Documentation for ${{ github.event.inputs.version }}"
```

#### Step 2: Update the Main Documentation Workflow

Update the existing `.github/workflows/docs-to-gitbook.yml` to include version information:

```yaml
# Add to the Convert HTML to Markdown step
- name: Convert HTML to Markdown
  run: |
    chmod +x scripts/convert_html_to_md.sh
    ./scripts/convert_html_to_md.sh
    
    # Add version information to README.md
    echo "" >> ./markdown_output/README.md
    echo "## Versions" >> ./markdown_output/README.md
    echo "" >> ./markdown_output/README.md
    echo "This is the documentation for the latest development version of ExactCIs." >> ./markdown_output/README.md
    echo "For released versions, use the version selector in the GitBook interface." >> ./markdown_output/README.md
```

#### Step 3: Configure GitBook Variants

For each released version:

1. Create a new variant in the GitBook space:
   - Go to the GitBook space settings
   - Navigate to the "Variants" section
   - Click "Add variant"
   - Name it according to the version (e.g., "v1.0.0")
   - Set the appropriate visibility and access settings

2. Configure Git Sync for the variant:
   - In the variant settings, go to the "Git Sync" section
   - Connect to the GitHub repository
   - Select the corresponding `gitbook-sync-vX.Y.Z` branch
   - Configure the sync settings (bidirectional sync, sync frequency, etc.)

3. Set up the version selector:
   - In the GitBook space settings, enable the version selector
   - Arrange versions in the desired order (typically latest first, followed by older versions)
   - Set the default version (usually the latest stable release)

### 4. Release Process

When releasing a new version of ExactCIs:

1. Complete the code changes and testing for the new version
2. Update the version number in the project files
3. Create a Git tag for the new version
4. Run the "Release Documentation Version" workflow with the new version number
5. Configure a new GitBook variant for the version
6. Update the version selector in GitBook to include the new version

### 5. Maintenance Considerations

1. **Storage and Performance**:
   - Each version creates a new branch in the repository, which increases storage requirements
   - Consider archiving or removing very old versions if they are no longer relevant

2. **Consistency Across Versions**:
   - Ensure that the documentation structure remains consistent across versions
   - When making structural changes, consider whether they should be backported to older versions

3. **Version Selector Visibility**:
   - Ensure that the version selector is prominently displayed in the GitBook interface
   - Include clear information about which version corresponds to which library version

4. **Documentation for Documentation**:
   - Maintain clear instructions for documentation maintainers on how to release new versions
   - Document the relationship between code versions and documentation versions

## Conclusion

This version management strategy provides a comprehensive approach to maintaining versioned documentation for the ExactCIs project. By leveraging GitBook's variants feature and Git's branching capabilities, we can provide users with access to documentation for any version of the library while maintaining an automated build and deployment process.

The strategy is designed to be sustainable and maintainable, with clear processes for releasing new versions and maintaining existing ones. It balances the needs of users (access to version-specific documentation) with the needs of maintainers (automation and ease of maintenance).