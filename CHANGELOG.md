# Changelog

All notable changes to the ExactCIs project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2023-12-15

### Added
- Initial release of ExactCIs
- Five confidence interval methods for 2×2 contingency tables:
  - Conditional (Fisher's exact)
  - Mid-P adjusted
  - Blaker's exact
  - Barnard's unconditional exact
  - Haldane-Anscombe Wald
- Comprehensive documentation including:
  - User guide
  - API reference
  - Method comparison guide
  - Performance documentation
  - Validation summary
- Extensive test suite with unit tests, edge case tests, and performance tests
- Performance optimization features:
  - Optional NumPy acceleration
  - Timeout protection for long-running calculations
  - Caching for repeated calculations
- R comparison tools for validation against established implementations

### Fixed
- Numerical stability issues in unconditional method for extreme tables
- Edge case handling for tables with zero cells
- Timeout handling in long-running calculations

## [Unreleased]

### Added
- Validation summary document
- CHANGELOG.md for tracking version changes
- LICENSE file with MIT License
- CONTRIBUTING.md with guidelines for contributors

### Changed
- Improved documentation for method selection
- Enhanced performance documentation with optimization guidance
- Updated package metadata for publication readiness

### Fixed
- Minor documentation inconsistencies
- Edge case handling improvements