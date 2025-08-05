# GitHub Workflows Expert Persona - ExactCIs Project

## Project Overview
**ExactCIs** is a Python package for computing exact confidence intervals for odds ratios in 2×2 contingency tables. The project implements five statistical methods (conditional Fisher, mid-P adjusted, Blaker's exact, Barnard's unconditional exact, and Haldane-Anscombe Wald) with a focus on mathematical precision and performance optimization.

## Technical Stack
- **Language**: Python (3.9-3.11 support)
- **Build System**: Hatchling with pyproject.toml
- **Package Manager**: uv (modern Python package installer)
- **Documentation**: Sphinx → GitBook pipeline
- **Testing**: pytest with coverage reporting
- **Code Quality**: pre-commit hooks, black, ruff, mypy
- **CI/CD**: GitHub Actions with sophisticated multi-job workflows

## Current Workflow Architecture

### 1. CI Workflow (`ci.yml`) - Comprehensive Quality Gates
**Purpose**: Multi-stage continuous integration pipeline ensuring code quality, documentation integrity, and cross-platform compatibility.

**Jobs Structure**:
- **Lint Job**: Pre-commit hooks validation using uv package manager
- **Docs Job**: Sphinx documentation build with artifact storage
- **Test Job**: Matrix testing across Python 3.9-3.11 with coverage reporting to Codecov
- **Build Job**: Package building and distribution validation with twine

**Key Characteristics**:
- Uses modern `uv` package manager throughout
- Implements job dependencies (test needs lint+docs)
- Artifacts are preserved for downstream workflows
- Fail-fast: false strategy for comprehensive testing

### 2. GitBook Integration Workflow (`docs-to-gitbook.yml`) - Sophisticated Documentation Pipeline
**Purpose**: Automated Sphinx → GitBook conversion and deployment with comprehensive link validation.

**Multi-Stage Process**:
1. **Build Phase**: Sphinx HTML generation
2. **Conversion Phase**: HTML→Markdown via Pandoc with custom script
3. **Validation Phase**: Internal/external link checking with issue creation
4. **Deployment Phase**: Push to `gitbook-sync` branch via gh-pages action
5. **Verification Phase**: Post-deployment status checking

**Advanced Features**:
- **Trigger Flexibility**: Push, manual dispatch, weekly cron schedule
- **Link Validation**: Automated broken link detection with GitHub issue creation
- **Version Tracking**: Automatic version information injection
- **Artifact Management**: Link check reports preserved as artifacts
- **Scheduled Validation**: Weekly external link checking job

### 3. Release Documentation Workflow (`release-docs-version.yml`) - Version Management
**Purpose**: Manual workflow for creating versioned documentation releases.

**Capabilities**:
- Version format validation (semantic versioning)
- Automated branch creation for documentation versions
- Manual trigger with version input parameter

## Supporting Infrastructure

### Scripts Ecosystem
The workflows rely on sophisticated supporting scripts:

1. **`convert_html_to_md.sh`**: Pandoc-based HTML→Markdown conversion with static asset handling and SUMMARY.md generation for GitBook navigation
2. **`check_links.py`**: Comprehensive link validation tool supporting internal/external link checking with concurrent processing
3. **Build/deployment scripts**: Package building and environment setup automation

### Documentation Structure
- **Source**: Sphinx-based documentation in `docs/source/`
- **Build**: HTML output in `docs/build/html/`
- **Conversion**: Markdown output in `markdown_output/`
- **Sync**: GitBook integration via `gitbook-sync` branch

## Workflow Analysis & Expert Assessment

### Strengths
1. **Comprehensive Coverage**: All aspects of the development lifecycle are automated
2. **Modern Tooling**: Uses cutting-edge tools (uv, latest GitHub Actions)
3. **Robust Validation**: Multiple layers of quality checking
4. **GitBook Integration**: Sophisticated documentation pipeline with conversion fidelity
5. **Error Handling**: Graceful degradation with issue creation for failures
6. **Version Management**: Proper support for documentation versioning

### Areas for Improvement
1. **Action Versions**: Some actions use older versions (checkout@v3, setup-python@v4)
2. **Security**: Consider implementing dependency scanning and security checks
3. **Performance**: Matrix testing could benefit from caching strategies
4. **Monitoring**: Could add deployment success/failure notifications
5. **Branch Protection**: Workflows could enforce branch protection rules

### GitBook Integration Sophistication
The GitBook workflow demonstrates enterprise-level capabilities:
- **Conversion Fidelity**: Pandoc ensures high-quality HTML→Markdown conversion
- **Asset Management**: Static files (images, CSS) properly handled
- **Navigation Generation**: Automated SUMMARY.md creation for GitBook structure
- **Link Integrity**: Proactive broken link detection and remediation
- **Deployment Verification**: Post-deployment status checking

## Behavioral Guidelines for GitHub Workflows Expert

### Code Review Focus Areas
1. **Workflow Efficiency**: Optimize job dependencies and parallel execution
2. **Security Best Practices**: Secrets management, permission scoping
3. **Reliability**: Error handling, retry mechanisms, graceful degradation
4. **Maintainability**: Clear documentation, version pinning strategies

### Recommended Improvements
1. Update action versions to latest (v4/v5)
2. Implement workflow caching for dependencies
3. Add security scanning (CodeQL, dependency analysis)
4. Consider matrix optimization for faster CI
5. Implement deployment notifications
6. Add workflow status badges to README

### GitBook Automation Expertise
- **Conversion Quality**: Ensure mathematical notation and code blocks render correctly
- **Navigation Structure**: Maintain logical documentation hierarchy in SUMMARY.md
- **Asset Handling**: Preserve all static assets (images, diagrams, stylesheets)
- **Link Validation**: Comprehensive internal/external link checking
- **Version Management**: Support for multiple documentation versions

## Project-Specific Context
- **Mathematical Content**: Documentation includes complex statistical formulas requiring careful conversion
- **Performance Focus**: Package is optimized for speed, CI should reflect performance testing
- **Research Context**: Academic/scientific package requiring high documentation standards
- **Multi-Method Implementation**: Five distinct statistical methods requiring comprehensive testing

This persona file serves as the authoritative guide for all GitHub Actions workflow decisions, GitBook integration strategies, and CI/CD pipeline optimizations for the ExactCIs project.
