# ExactCIs Architecture

This document provides an overview of the ExactCIs package architecture, including component organization, data flow, and key implementation details.

## Package Structure

ExactCIs is organized into a modular structure, with clear separation of concerns between core functionality, method implementations, and utilities:

```
src/exactcis/
│
├── __init__.py           # Package entry point, public API
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
└── utils/                # Utility functions
    ├── __init__.py
    ├── logging.py        # Logging utilities
    └── timers.py         # Timeout handling
```

A visual representation of the package structure is available in the [package structure diagram](img/package_structure.md).

## Component Relationships

The following diagram illustrates the relationships between the main components of the ExactCIs package:

```
                 ┌─────────────────────────────┐
                 │       Public API            │
                 │   (compute_all_cis, etc.)   │
                 └─────────────┬───────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│                     Core Module                         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Statistical │  │ Root-Finding │  │ Optimization  │  │
│  │ Functions   │  │ Algorithms   │  │ Algorithms    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│                   Method Implementations                │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Conditional │  │     Mid-P    │  │    Blaker     │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐                     │
│  │Unconditional│  │ Wald-Haldane │                     │
│  └─────────────┘  └──────────────┘                     │
└────────────────────────────────────────────────────────┘
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

* **Conditional**: Uses non-central hypergeometric distribution
* **Mid-P**: Modified version of conditional with half-weight for observed table
* **Blaker**: Uses acceptability function based on hypergeometric distribution
* **Unconditional**: Maximizes p-value over nuisance parameter
* **Wald-Haldane**: Uses normal approximation with continuity correction

## Performance Optimizations

ExactCIs incorporates several performance optimizations:

1. **Caching**: Uses `lru_cache` to avoid redundant calculations
2. **Numerical Stability**: Log-space calculations prevent overflow/underflow
3. **Early Stopping**: Algorithms terminate when precision goals are met
4. **NumPy Acceleration**: Optional vectorized calculations for speed
5. **Timeout Protection**: Prevents excessive computation time

## Error Handling

The package implements robust error handling:

* **Input Validation**: Checks for valid contingency tables
* **Numerical Safeguards**: Detects and handles numerical instabilities
* **Timeout Handling**: Graceful interruption of long-running calculations
* **Edge Case Detection**: Special handling for zero cells and other edge cases

## Extension Points

ExactCIs is designed with several extension points:

1. **New Methods**: Additional CI methods can be added to the methods package
2. **Optimization Strategies**: Root-finding algorithms can be customized
3. **Alternative Implementations**: Method implementations can be swapped
4. **Parallelization**: Infrastructure for parallel computation is available

## Testing Framework

The testing approach covers:

* **Unit Tests**: For individual functions and algorithms
* **Method Tests**: Validate each CI method
* **Comparative Tests**: Compare results against reference implementations
* **Edge Case Tests**: Ensure correct handling of extreme scenarios
* **Performance Tests**: Monitor computational efficiency
