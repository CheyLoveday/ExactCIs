# ExactCIs Architecture

This document provides an overview of the ExactCIs package architecture, including component organization, data flow, and key implementation details.

## Package Structure

ExactCIs is organized into a modular structure, with clear separation of concerns between core functionality, method implementations, and utilities:

```
src/exactcis/
│
├── __init__.py           # Package entry point, compute_all_cis API
├── _version.py           # Version information
├── cli.py                # Command-line interface
├── core.py               # Core statistical functions and algorithms
│
├── methods/              # Implementation of different CI methods
│   ├── __init__.py       # Exports all method implementations
│   ├── conditional.py    # Fisher's exact conditional method
│   ├── midp.py           # Mid-P adjusted method
│   ├── blaker.py         # Blaker's exact method
│   ├── unconditional.py  # Barnard's unconditional exact method
│   └── wald.py           # Wald-Haldane approximation method
│
└── utils/                # Utility functions and optimizations
    ├── __init__.py
    ├── calculators.py    # Statistical calculation utilities
    ├── data_models.py    # Data structures and models
    ├── optimization.py   # Performance optimization utilities
    ├── parallel.py       # Parallel processing support
    ├── pmf_functions.py  # Probability mass function implementations
    ├── root_finding.py   # Root-finding algorithm implementations
    ├── stats.py          # Statistical utility functions
    ├── transformers.py   # Data transformation utilities
    └── validators.py     # Input validation utilities
```

A visual representation of the package structure is available in the [package structure diagram](img/package_structure.md).

## Component Relationships

The following diagram illustrates the relationships between the main components of the ExactCIs package:

```
                    ┌─────────────────────────────┐
                    │       Public API            │
                    │   (compute_all_cis, CLI)    │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Module                              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Validation  │  │ PMF & Stats  │  │  Root Finding &      │   │
│  │ Functions   │  │ Calculations │  │  Numerical Methods   │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Method Implementations                      │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │Conditional  │ │   Mid-P     │ │  Blaker's   │ │Unconditional│ │
│ │  (Fisher)   │ │  Adjusted   │ │   Exact     │ │ (Barnard's) │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                                 │
│               ┌─────────────────────────────────┐               │
│               │       Wald-Haldane             │               │
│               │     (Asymptotic)               │               │
│               └─────────────────────────────────┘               │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Utility Modules                            │
│                                                                 │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│ │ Optimization │ │   Parallel   │ │ PMF Functions│ │ Stats & │ │
│ │  & Caching   │ │  Processing  │ │ & Root Finding│ │Validators│ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

The typical data flow through the ExactCIs package is illustrated below:

![Data Flow Diagram](img/data_flow.md)

For a more detailed view of the calculation process, see the [confidence interval calculation diagram](img/ci_calculation.md).

## Method Comparison

Each confidence interval method has different characteristics that make it suitable for different scenarios:

![Method Comparison Diagram](img/method_comparison_diagram.md)

For help selecting the appropriate method for your specific use case, refer to the [method selection guide](img/method_selection.md).

## Performance Benchmarks

The performance of each method varies depending on the sample size and other factors:

![Performance Benchmarks](img/performance_benchmarks.md)

## Key Algorithms

### Root Finding

The package implements several numerical algorithms for finding confidence interval bounds:

1. **Bisection Method**: Used for most methods to find roots of p-value functions
2. **Log-Space Search**: Enhanced stability for wide-ranging odds ratios
3. **Plateau Edge Detection**: Specialized algorithm for detecting edges of flat p-value regions

### P-value Calculation

Each method calculates p-values differently:

* **Conditional (Fisher)**: Uses noncentral hypergeometric distribution conditioning on margins
* **Mid-P**: Fisher's method with half-weight for observed table probability
* **Blaker**: Uses acceptability functions with tolerance-based comparison criteria
* **Unconditional (Barnard)**: Maximizes p-value over grid of nuisance parameters (p₁, p₂)
* **Wald-Haldane**: Uses log-normal approximation with Haldane correction (0.5 added to each cell)

## Performance Optimizations

ExactCIs incorporates several performance optimizations:

1. **Multi-Level Caching**: 
   - LRU caches for binomial coefficients, PMF calculations
   - Method-specific caches (e.g., Blaker PMF cache during root-finding)
   - Cache size optimization based on problem complexity

2. **Numerical Stability**: 
   - Log-space calculations for all probability computations
   - Stable logsumexp implementation
   - Underflow/overflow protection with automatic fallbacks

3. **Parallel Processing**:
   - Batch processing support for all methods
   - Automatic worker count optimization
   - Process-based parallelization for CPU-bound tasks

4. **JIT Compilation**:
   - Numba acceleration for unconditional method grid computations
   - Automatic fallback to pure Python when Numba unavailable

5. **Algorithmic Optimizations**:
   - Adaptive grid sizing based on table dimensions
   - Early termination based on convergence criteria
   - MLE-centered grids for faster convergence
   - Timeout protection with graceful degradation

## Error Handling

The package implements comprehensive error handling across all methods:

* **Input Validation**: 
  - Multi-level validation for contingency table counts
  - Margin validation (no empty margins allowed)
  - Alpha parameter validation (must be in (0,1))

* **Numerical Safeguards**: 
  - Automatic detection of numerical underflow/overflow
  - Conservative fallbacks when root-finding fails
  - Crossed bounds detection and correction

* **Method-Specific Handling**:
  - Zero cell handling with specialized algorithms per method
  - Support range validation for Blaker and Mid-P methods
  - Grid boundary handling for unconditional method

* **Graceful Degradation**: 
  - Conservative interval fallbacks (0, ∞) when computation fails
  - Comprehensive logging of issues for debugging
  - Timeout handling with partial result reporting

## Extension Points

ExactCIs is designed with several extension points for future enhancement:

1. **New Methods**: 
   - Additional CI methods can be added to the methods/ directory
   - Consistent interface pattern for integration with compute_all_cis()
   - Automatic batch processing support through established patterns

2. **Optimization Strategies**: 
   - Pluggable root-finding algorithms in utils/root_finding.py
   - Custom caching strategies through utils/optimization.py
   - Alternative PMF implementations in utils/pmf_functions.py

3. **Parallel Processing**: 
   - Extensible parallel processing framework in utils/parallel.py
   - Support for different execution models (threads vs processes)
   - Custom worker optimization strategies

4. **Performance Enhancements**:
   - Additional JIT compilation targets beyond Numba
   - GPU acceleration hooks for large-scale computations
   - Alternative numerical libraries integration

## Testing Framework

The testing approach provides comprehensive coverage across multiple dimensions:

* **Unit Tests**: 
  - Individual core functions (PMF calculations, root finding, validation)
  - Utility module functions (parallel processing, optimization, stats)
  - Numerical algorithm correctness and stability

* **Method Tests**: 
  - Each CI method tested independently with known cases
  - Batch processing functionality for all methods
  - Error handling and edge case behavior validation

* **Integration Tests**: 
  - Cross-method comparisons and consistency checks
  - R implementation comparisons for validation
  - End-to-end workflow testing through compute_all_cis()

* **Edge Case Tests**: 
  - Zero cell handling across all methods
  - Extreme odds ratios and small/large sample sizes
  - Numerical boundary conditions and stability

* **Performance Tests**: 
  - Computational efficiency monitoring with timeout testing
  - Memory usage validation for batch processing
  - Parallel processing scalability verification

* **Property-Based Testing**:
  - Coverage properties (confidence intervals contain true parameter)
  - Monotonicity and consistency properties across parameter ranges
  - Robustness testing with randomly generated valid tables
