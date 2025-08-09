# Phase 3: CI Inversion Engine Unification

**Status: Critical Refactor - Highest Value Post Phase 2**

## Overview

Centralize all confidence interval inversion/search logic into a single, robust engine (`utils.inversion` + `utils.solvers`) and eliminate the fragmented, method-specific root-finding implementations across all OR methods.

## Problem Analysis

### **Current State: Fragmented CI Inversion** 🔴

Each method currently implements its own CI bound-finding logic:

**`conditional.py`**:
- Bespoke bracketing with `_expand_bracket_vectorized()` 
- Custom expansion helpers `_find_better_bracket()`
- Falls back to `scipy.optimize.brentq`/`bisect`
- Custom zero-cell fallback paths

**`midp.py`**:
- Custom adaptive grid search implementation
- Method-specific plateau handling
- Independent bracketing logic

**`blaker.py`**:
- Ad-hoc root finding for "bump" p-value functions
- Custom plateau detection for acceptability regions

**`core.py`** (Legacy):
- `find_root()` and `find_root_log()` with basic bisection
- Inconsistent tolerance and domain handling
- No unified infinite-bound detection

### **Problems This Creates** ❌

**1. Inconsistent Behavior**
- Different methods handle infinite bounds differently
- Varying tolerance levels and convergence criteria
- Inconsistent zero-cell and edge-case handling
- Divergent plateau detection strategies

**2. Code Duplication**
- Multiple implementations of bracket expansion
- Repeated root-finding algorithms with subtle differences
- Fragmented error handling and diagnostics
- Duplicated infinite-bound logic

**3. Maintenance Burden**
- Bug fixes must be applied to multiple locations
- Performance improvements don't propagate across methods
- Inconsistent caching strategies
- Difficult to reason about overall numerical behavior

**4. Robustness Issues**
- Some methods more fragile than others on edge cases
- Inconsistent handling of plateau regions (critical for exact methods)
- Different failure modes across methods

## Solution: Unified CI Inversion Engine

### **Core Architecture**

```python
# Single engine combining existing robust infrastructure
from utils.inversion import invert_two_sided_ci
from utils.solvers import bracket_log_space, bisection_safe
from utils.ci_search import find_confidence_interval_adaptive_grid

def unified_ci_engine(p_value_func: Callable[[float], float],
                     alpha: float = 0.05,
                     domain: Tuple[float, float] = (1e-12, 1e12),
                     method_hints: Dict[str, Any] = None) -> Tuple[float, float]:
    """
    Unified confidence interval inversion engine.
    
    Automatically selects optimal strategy:
    1. Try monotonic inversion with robust solvers
    2. Fall back to adaptive grid search for plateau/non-monotone cases
    3. Handle infinite bounds and edge cases consistently
    
    Args:
        p_value_func: Function returning p-value for given parameter
        alpha: Significance level
        domain: Parameter search domain
        method_hints: Method-specific optimization hints
        
    Returns:
        (lower_bound, upper_bound) with consistent infinite-bound handling
    """
```

### **Method Integration Pattern**

**Before (conditional.py)**:
```python
def ci_conditional(a, b, c, d, alpha=0.05):
    # Custom bracketing logic
    bounds = _find_better_bracket(...)
    lower = scipy.optimize.brentq(lambda x: pval_lower(x) - alpha, ...)
    upper = scipy.optimize.brentq(lambda x: pval_upper(x) - alpha, ...)
    # Custom infinite bound handling
    return lower, upper
```

**After (conditional.py)**:
```python
def ci_conditional(a, b, c, d, alpha=0.05):
    # Quick zero-cell returns (preserved)
    if _has_zero_cells_requiring_special_handling(a, b, c, d):
        return _handle_zero_cell_bounds(a, b, c, d, alpha)
    
    # Unified p-value function
    def conditional_p_value(theta):
        return calculate_fisher_exact_p_value(a, b, c, d, theta)
    
    # Single engine handles all complexity
    return unified_ci_engine(
        conditional_p_value,
        alpha=alpha,
        domain=(1e-12, 1e12),
        method_hints={'plateau_detection': True, 'exact_method': True}
    )
```

## Implementation Plan

### **Phase 3.1: Engine Enhancement** 🔧

**Extend `utils.inversion.invert_two_sided_ci`**:
```python
def invert_two_sided_ci(p_value_func, alpha=0.05, 
                       domain=(1e-12, 1e12),
                       fallback_to_grid=True,
                       plateau_detection=True,
                       **solver_kwargs) -> Tuple[float, float]:
    """
    Enhanced inversion with automatic fallback strategies.
    
    Strategy:
    1. Try bracket_log_space + bisection_safe (fast, robust for monotonic)
    2. If no sign change or plateau detected → adaptive grid search
    3. Unified infinite bound detection and handling
    4. Comprehensive diagnostics for troubleshooting
    """
    
    # Step 1: Attempt monotonic inversion
    try:
        bracket, diag = bracket_log_space(p_value_func, domain=domain, target=alpha)
        if diag.converged:
            lower, _ = bisection_safe(lambda x: p_value_func(x) - alpha, 
                                    bracket[0], domain[1])
            upper, _ = bisection_safe(lambda x: p_value_func(x) - alpha,
                                    domain[0], bracket[1]) 
            return lower, upper
    except BracketingFailure:
        pass
    
    # Step 2: Fallback to adaptive grid search
    if fallback_to_grid:
        return find_confidence_interval_adaptive_grid(
            p_value_func, alpha, domain, **solver_kwargs
        )
    
    # Step 3: Last resort - domain bounds with warning
    logger.warning("CI inversion failed, returning domain bounds")
    return domain[0], domain[1]
```

### **Phase 3.2: Method Refactoring** ⚙️

**Conditional Method Migration**:
- [ ] Extract p-value calculation into pure function
- [ ] Replace custom bracketing with unified engine
- [ ] Preserve zero-cell fast paths where beneficial
- [ ] Validate golden parity throughout

**Mid-P Method Migration**:
- [ ] Wrap `calculate_midp_pvalue()` for engine compatibility
- [ ] Leverage adaptive grid fallback (mid-P often non-monotonic)
- [ ] Remove custom grid search implementation
- [ ] Validate against existing fixtures

**Blaker Method Migration**:
- [ ] Adapt acceptability function for engine interface
- [ ] Utilize plateau detection for "bump" p-value regions
- [ ] Remove method-specific root finding
- [ ] Ensure optimal coverage properties maintained

### **Phase 3.3: Legacy Cleanup** 🧹

**Deprecate Core Legacy Functions**:
- [ ] Add deprecation warnings to `find_root()` and `find_root_log()`
- [ ] Migrate any remaining callers to unified engine
- [ ] Remove after confirmation no dependencies remain

**Remove Method-Specific Helpers**:
- [ ] `conditional.py`: Remove `_expand_bracket_vectorized()`, `_find_better_bracket()`
- [ ] `midp.py`: Remove custom adaptive grid implementation
- [ ] `blaker.py`: Remove ad-hoc root finding helpers

**Consolidate Error Handling**:
- [ ] Unified exception types for inversion failures
- [ ] Consistent warning messages across methods
- [ ] Standardized diagnostic information

### **Phase 3.4: Validation & Optimization** 🧪

**Golden Parity Validation**:
- [ ] Run comprehensive golden parity tests with `EXACTCIS_STRICT_PARITY=1`
- [ ] Accept minor numerical differences due to improved algorithms
- [ ] Document any changes in precision or convergence behavior

**Performance Analysis**:
- [ ] Benchmark function call overhead of unified engine
- [ ] Profile cache hit rates for shared components
- [ ] Measure overall speed improvement from reduced duplication

**Robustness Testing**:
- [ ] Test edge cases: zeros, extreme ratios, tiny/huge sample sizes
- [ ] Validate infinite bound detection across all methods
- [ ] Stress test plateau detection and adaptive fallbacks

## Benefits Analysis

### **Correctness & Consistency** ✅
- **Unified infinite bound handling**: All methods handle `(0, ∞)` cases identically
- **Consistent tolerances**: Same convergence criteria across methods  
- **Standardized edge cases**: Zero cells, extreme ratios handled uniformly
- **Robust plateau handling**: Advanced detection prevents degenerate solutions

### **Code Quality Improvements** 📈
- **Reduced duplication**: ~60% reduction in root-finding code
- **Single source of truth**: One place to fix bugs and add features
- **Cleaner method implementations**: Focus on statistical logic vs numerical details
- **Better testability**: Isolated engine can be thoroughly unit tested

### **Performance Gains** ⚡
- **Shared caching**: Bracket hints and evaluations cached across methods
- **Optimized algorithms**: Best-in-class solvers used consistently
- **Reduced overhead**: Eliminate redundant bracketing attempts
- **Intelligent fallbacks**: Automatic selection of optimal strategy

### **Maintainability** 🔧
- **Centralized improvements**: Algorithm enhancements benefit all methods
- **Easier debugging**: Single place to add logging and diagnostics  
- **Policy consistency**: Tolerance changes applied uniformly
- **Future extensibility**: New methods automatically inherit robust inversion

## Risk Assessment & Mitigation

### **🟡 Medium Risk: Golden Parity Drift**
- **Risk**: Unified algorithms may produce slightly different numerical results
- **Mitigation**: 
  - Implement method-level compatibility flags
  - Keep old paths behind debug switches initially
  - Use relaxed golden parity tolerances during transition
  - Document legitimate precision improvements

### **🟡 Medium Risk: Performance Regression**
- **Risk**: Engine overhead might slow down simple cases
- **Mitigation**:
  - Profile all methods before/after migration
  - Cache expensive computations in engine
  - Provide fast paths for common scenarios
  - Monitor function call counts via diagnostics

### **🟢 Low Risk: Plateau Detection Issues**
- **Risk**: Automatic fallback might not work for all p-value shapes  
- **Mitigation**:
  - Extensive testing on known plateau cases (mid-P, Blaker)
  - Method hints allow tuning detection sensitivity
  - Manual override options for edge cases
  - Comprehensive diagnostic logging

### **🔴 High Risk: Breaking Zero-Cell Handling**
- **Risk**: Unified engine might not preserve method-specific zero-cell logic
- **Mitigation**:
  - Preserve existing fast paths where they provide value
  - Extensive testing on zero-cell fixtures
  - Method-specific hints guide engine behavior
  - Fallback to method-specific handling if needed

## Success Criteria

### **Quality Gates**
- [ ] **Golden parity**: 95%+ of existing test cases produce identical results  
- [ ] **Edge case robustness**: All zero-cell and extreme-ratio cases handled correctly
- [ ] **Performance**: No >5% regression in computation times
- [ ] **Code reduction**: >50% elimination of duplicate root-finding code

### **Technical Deliverables**
- [ ] **Enhanced unified engine**: `utils.inversion` with intelligent fallbacks
- [ ] **Migrated methods**: All OR methods using consistent inversion
- [ ] **Deprecated legacy**: Clear deprecation path for old root-finding functions
- [ ] **Comprehensive tests**: Engine and method integration fully validated

### **Documentation & Examples**
- [ ] **Architecture guide**: Clear explanation of inversion engine design
- [ ] **Migration notes**: Changes in behavior and precision improvements
- [ ] **Troubleshooting guide**: Using diagnostics to debug inversion issues
- [ ] **Performance analysis**: Benchmarking results and optimization insights

## Timeline & Dependencies

### **Prerequisites**
- Phase 0-2 infrastructure must be stable and well-tested
- Golden parity testing framework operational  
- All existing CI methods have comprehensive test coverage

### **Estimated Timeline: 4-6 Weeks**

**Week 1-2**: Engine enhancement and design validation
**Week 3-4**: Method-by-method migration with parity testing
**Week 5**: Legacy cleanup and performance optimization  
**Week 6**: Documentation and final validation

### **Critical Path**
1. `utils.inversion` enhancement → method migration → legacy removal
2. Golden parity validation gates each migration step
3. Performance benchmarking prevents regressions

## Long-Term Impact

This refactoring establishes ExactCIs as having a **truly unified, robust numerical foundation**. It eliminates the last major source of inconsistency in the codebase and provides a clean platform for:

- **Future method development**: New CI methods automatically inherit robust inversion
- **Performance optimization**: Centralized caching and algorithm improvements
- **Research applications**: Consistent, reliable numerical behavior for publications
- **Advanced features**: Foundation for likelihood ratio and other sophisticated methods

**This is the refactoring that transforms ExactCIs from a collection of methods into a cohesive, production-grade statistical engine.**