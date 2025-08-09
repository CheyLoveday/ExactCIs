# Phase 3: Mathematical Operations Consolidation Plan

**Status: Planned for Post-Phase 2 Implementation**

## Overview

Consolidate all mathematical functions into a centralized `utils/math_core.py` module to eliminate code duplication and provide unified caching strategy for optimal performance.

## Current State Analysis

### Scattered Mathematical Functions
- **Core math functions**: `support`, `log_binom_coeff`, `logsumexp` in `core.py`
- **Safe operations**: `utils/mathops.py` 
- **Statistical utilities**: `utils/stats.py`
- **PMF calculations**: `utils/pmf_functions.py`
- **Probability functions**: `utils/probabilities.py`

### Problems to Solve
- **Cache fragmentation**: Multiple `@lru_cache` decorators with inconsistent sizes
- **Code duplication**: Similar mathematical operations in different modules
- **Import confusion**: Mathematical functions scattered across modules
- **Maintenance burden**: Updates require changes to multiple files

## Proposed Solution

### New Module: `utils/math_core.py`

```python
"""
Centralized mathematical operations for ExactCIs.

This module serves as the single source of truth for all mathematical
operations, providing unified caching and optimized algorithms.
"""

from functools import lru_cache
from typing import Union, List, Tuple
import math
import numpy as np

# Centralized cache configuration
MATH_CACHE_SIZE = 2048
SUPPORT_CACHE_SIZE = 1024

@lru_cache(maxsize=MATH_CACHE_SIZE)
def log_binom_coeff(n: Union[int, float], k: Union[int, float]) -> float:
    """Calculate log of binomial coefficient in numerically stable way."""
    # Move from core.py with enhanced error handling

@lru_cache(maxsize=SUPPORT_CACHE_SIZE) 
def hypergeometric_support(n1: int, n2: int, m1: int) -> SupportData:
    """Calculate support for hypergeometric distribution."""
    # Move from core.py with consistent naming

def logsumexp(log_terms: List[float]) -> float:
    """Compute log(sum(exp(log_terms))) in numerically stable way."""
    # Enhanced version from core.py

def log_nchg_pmf(k: int, n1: int, n2: int, m1: int, theta: float) -> float:
    """Calculate log noncentral hypergeometric PMF."""
    # Consolidated PMF calculations

def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """Safe division with configurable default for zero denominator."""
    # From mathops.py

def exp_safe(x: float, max_exp: float = 700.0) -> float:
    """Safe exponential function preventing overflow."""
    # From mathops.py

# Cache management utilities
def clear_math_caches() -> None:
    """Clear all mathematical function caches."""
    
def get_cache_info() -> dict:
    """Get cache statistics for monitoring."""

def configure_cache_sizes(math_cache: int = MATH_CACHE_SIZE,
                         support_cache: int = SUPPORT_CACHE_SIZE) -> None:
    """Reconfigure cache sizes for different use cases."""
```

## Implementation Plan

### Phase 3.1: Create Core Infrastructure ✅
- [ ] Create `utils/math_core.py` with centralized functions
- [ ] Implement unified caching strategy with configuration options
- [ ] Add comprehensive unit tests for all mathematical operations
- [ ] Create cache monitoring and management utilities

### Phase 3.2: Migration Strategy 📋
- [ ] **Update imports systematically**:
  - `utils/pmf_functions.py` → import from `math_core`
  - `methods/` files → import from `math_core`
  - `utils/estimates.py` → import from `math_core`
  - Other utility modules as needed

- [ ] **Legacy compatibility wrappers**:
  - Keep forwarding functions in `core.py` with deprecation warnings
  - Maintain existing APIs during transition period
  - Gradual phase-out of legacy imports

### Phase 3.3: Optimization and Enhancement 🚀
- [ ] **Performance improvements**:
  - Profile cache hit rates and adjust sizes
  - Optimize frequently-used mathematical operations
  - Add vectorized versions where beneficial

- [ ] **Enhanced functionality**:
  - Add mathematical constants and tolerances
  - Implement specialized functions for edge cases
  - Provide debug/verbose modes for troubleshooting

## Benefits

### Performance Gains
- **Unified caching**: Single, optimized caching strategy across all math operations
- **Reduced redundancy**: Eliminate duplicate computations across modules
- **Cache efficiency**: Better hit rates with consolidated cache management

### Maintainability Improvements  
- **Single source**: One location for all mathematical operations
- **Consistent APIs**: Unified function signatures and error handling
- **Easier testing**: Centralized unit tests for mathematical correctness
- **Clear dependencies**: Simplified import structure

### Development Benefits
- **Easier debugging**: Single location to add logging/tracing
- **Enhanced reliability**: Comprehensive error handling and edge case management
- **Future extensibility**: Clean foundation for additional mathematical operations

## Migration Timeline

### Week 1: Foundation
- Create `math_core.py` with core functions
- Implement caching infrastructure
- Add comprehensive unit tests

### Week 2: Core Migration
- Update `pmf_functions.py` and `estimates.py`
- Migrate method implementations one by one
- Ensure golden parity throughout

### Week 3: Optimization
- Profile performance and optimize cache sizes  
- Add monitoring and management utilities
- Complete documentation and examples

### Week 4: Legacy Cleanup
- Add deprecation warnings to old locations
- Plan timeline for removing legacy wrappers
- Update documentation to reflect new structure

## Success Metrics

### Quality Gates
- [ ] **Golden parity**: All tests pass with identical numerical results
- [ ] **Performance**: No regression in computation times
- [ ] **Cache efficiency**: >90% hit rate for frequently used functions
- [ ] **Coverage**: 100% test coverage for mathematical operations

### Deliverables
- [ ] **Consolidated module**: Single `math_core.py` with all functions
- [ ] **Migration guide**: Clear instructions for updating imports
- [ ] **Performance report**: Before/after benchmarking results  
- [ ] **Updated documentation**: Architecture guide reflecting new structure

## Dependencies

### Prerequisites
- Phase 0-2 refactoring must be complete
- Golden parity testing framework operational
- All existing mathematical functions identified and documented

### Risks and Mitigation
- **Import cycle risk**: Careful dependency analysis before migration
- **Performance regression**: Comprehensive benchmarking at each step
- **Cache tuning**: A/B testing with different cache configurations

This phase will establish ExactCIs as having a truly unified mathematical foundation, setting the stage for Phase 4's method registry and enhanced API development.