# Legacy Code Cleanup Roadmap

**Status: Active - Post Phase 2 Refactoring Cleanup**

## Overview

This document provides a systematic roadmap for removing outdated and superfluous code identified during the Phase 0-2 refactoring. All marked sections have been tagged with TODO comments for review and removal.

## Identified Legacy Code Categories

### 🔄 **Legacy Wrappers and Compatibility Functions**

#### In `src/exactcis/core.py`

**1. `validate_counts()` - Lines 20-33**
```python
# TODO: REVIEW FOR REMOVAL - Legacy validation wrapper
# This validation function is now centralized in utils/validation.py
def validate_counts(a, b, c, d) -> None:
```
- **Status**: Superseded by `utils/validation.py`
- **Dependencies**: Check for direct imports in method files
- **Action**: Replace with direct imports from `utils.validation`

**2. `apply_haldane_correction()` - Lines 36-52**
```python
# TODO: REVIEW FOR REMOVAL - Legacy correction wrapper  
# Correction functionality is now centralized in utils/continuity.py and utils/corrections.py
def apply_haldane_correction(a, b, c, d) -> Tuple[float, float, float, float]:
```
- **Status**: Superseded by centralized correction policies
- **Dependencies**: Used by legacy method implementations
- **Action**: Migrate to `utils.continuity.get_corrected_counts()`

**3. `find_root()` - Lines 367-404**
```python
# TODO: REVIEW FOR REMOVAL - Legacy root finding algorithm
# Root finding is now centralized in utils/solvers.py with more robust algorithms
def find_root(f, lo=1e-8, hi=1.0, tol=1e-8, maxiter=60) -> float:
```
- **Status**: Superseded by `utils/solvers.bisection_safe()` and `find_root_robust()`
- **Dependencies**: May be used by exact methods not yet refactored
- **Action**: Complete method refactoring first, then remove

**4. `_pmf_weights_original_impl()` - Lines 1038+**
```python
# TODO: REVIEW FOR REMOVAL - Original PMF weights implementation
# This is a fallback implementation that exists only for compatibility
def _pmf_weights_original_impl(n1, n2, m, theta) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
```
- **Status**: Fallback for `utils/pmf_functions.py`
- **Dependencies**: Imported by functional implementation
- **Action**: Confirm functional version covers all cases, then remove

**5. `_find_root_log_original_impl()` - Lines 1065+**
```python
# TODO: REVIEW FOR REMOVAL - Original root finding implementation
# This is a fallback implementation that exists only for compatibility
def _find_root_log_original_impl(f, lo=1e-8, hi=1.0, **kwargs) -> Optional[float]:
```
- **Status**: Fallback for `utils/root_finding.py` and `utils/solvers.py`
- **Dependencies**: Used when functional imports fail
- **Action**: Ensure centralized solvers are robust, then remove

#### In `src/exactcis/utils/validators.py`

**6. `validate_alpha()` - Lines 27-38**
```python  
# TODO: REVIEW FOR REMOVAL - Deprecated alpha validation function
# This function is deprecated and replaced by exactcis.utils.validation.validate_alpha
def validate_alpha(alpha: float) -> bool:
```
- **Status**: Deprecated wrapper around `utils.validation.validate_alpha`
- **Dependencies**: Check for remaining imports
- **Action**: Direct replacement with `utils.validation.validate_alpha`

### 📋 **Duplicate Functionality**

#### Between `utils/corrections.py` and `utils/continuity.py`

**7. `detect_zero_cells()` in `corrections.py` - Lines 100-119**
```python
# TODO: REVIEW FOR REMOVAL - Duplicate zero cell detection
# This function is duplicated in utils/continuity.py with identical functionality  
def detect_zero_cells(a, b, c, d) -> Tuple[bool, int, list[str]]:
```
- **Status**: Identical implementation exists in `continuity.py`
- **Dependencies**: Check which modules import from which location
- **Action**: Consolidate under `utils.continuity.py`, update imports

#### Between `utils/root_finding.py` and `utils/solvers.py`

**8. Overlapping root finding functionality**
```python
# TODO: REVIEW FOR REMOVAL - Potential duplication with solvers.py
# This module contains root finding functionality that may overlap with
# the more comprehensive solver algorithms in utils/solvers.py.
```
- **Status**: Functional overlap needs analysis
- **Dependencies**: Check which functions are unique vs duplicated
- **Action**: Merge unique functions into `solvers.py`, remove duplicates

### ⚠️ **Mixed Legacy/Modern Imports**

#### In `src/exactcis/methods/relative_risk.py`

**9. Legacy core imports - Lines 13-20**
```python
# TODO: REVIEW FOR REMOVAL - Mixed legacy and centralized imports
# Some imports still reference legacy core functions (find_root_log, find_plateau_edge)
from exactcis.core import find_root_log, find_plateau_edge  # LEGACY: Replace with solvers
```
- **Status**: Uses both old and new infrastructure  
- **Dependencies**: Score-based methods rely on these functions
- **Action**: Replace with centralized solver functions

#### In `src/exactcis/utils/pmf_functions.py`

**10. Core module dependency - Lines 12-15**
```python
# TODO: REVIEW FOR REMOVAL - Import from legacy core module
# PMF functions import support and log_binom_coeff from core.py
from ..core import support, log_binom_coeff, SupportData
```
- **Status**: Mathematical functions should be in dedicated utilities
- **Dependencies**: Core PMF functionality relies on these
- **Action**: Move to `utils/math_core.py` in Phase 3

## Cleanup Timeline

### 🚀 **Immediate Actions (Phase 2+)**

**Week 1: Dependency Analysis**
- [ ] Audit all imports to identify remaining dependencies on legacy functions
- [ ] Create dependency graph showing which methods use which legacy code
- [ ] Prioritize removal order based on dependency chains

**Week 2: Simple Replacements**  
- [ ] Replace `validate_alpha()` wrapper with direct imports
- [ ] Consolidate `detect_zero_cells()` under single module
- [ ] Update import statements for validation functions

### 📋 **Medium-term Actions (Phase 3 Prerequisites)**

**Month 1: Method Refactoring Dependencies**
- [ ] Complete refactoring of exact methods still using `find_root()`
- [ ] Update relative risk methods to use centralized solvers  
- [ ] Remove mixed legacy/modern import patterns

**Month 2: Mathematical Consolidation**
- [ ] Move mathematical functions to `utils/math_core.py`
- [ ] Remove redundant PMF implementations
- [ ] Update all mathematical operation imports

### 🔄 **Long-term Actions (Post-Phase 3)**

**Quarter 1: Complete Legacy Removal**
- [ ] Remove all fallback `_*_original_impl()` functions
- [ ] Clean up `core.py` to contain only essential PMF and distribution functions
- [ ] Remove deprecated wrapper functions with deprecation warnings

**Quarter 2: Architecture Finalization**
- [ ] Finalize clean module boundaries
- [ ] Remove any remaining compatibility shims
- [ ] Complete import statement cleanup across codebase

## Risk Assessment & Mitigation

### 🔴 **High Risk: Method Dependencies**
- **Risk**: Removing `find_root_log` before exact methods are refactored
- **Mitigation**: Complete Phase 3 method refactoring before cleanup
- **Detection**: Run full test suite after each removal

### 🟡 **Medium Risk: Import Cycles**
- **Risk**: Moving functions between modules may create cycles
- **Mitigation**: Careful dependency analysis before moves
- **Detection**: Import testing in isolated environments

### 🟢 **Low Risk: Simple Wrappers**
- **Risk**: Breaking external code that imports deprecated functions  
- **Mitigation**: Deprecation warnings, gradual phase-out
- **Detection**: Monitor for deprecation warning usage

## Validation Strategy

### Golden Parity Testing
- [ ] Run full golden parity test suite after each removal
- [ ] Use `EXACTCIS_STRICT_PARITY=1` for exact matching
- [ ] Maintain comprehensive fixture coverage

### Import Testing
- [ ] Test import statements in isolated environments
- [ ] Verify no circular import issues introduced  
- [ ] Check that public API remains unchanged

### Performance Monitoring
- [ ] Benchmark before/after each major removal
- [ ] Monitor memory usage changes
- [ ] Verify cache performance maintained

## Success Metrics

### Code Quality
- [ ] **Duplication reduction**: >80% elimination of redundant functions
- [ ] **Import clarity**: Single source for each functionality type
- [ ] **Module boundaries**: Clear separation of concerns
- [ ] **Maintenance burden**: Reduced complexity for future changes

### Reliability  
- [ ] **Test coverage**: Maintained >95% throughout cleanup
- [ ] **Golden parity**: All numerical results unchanged
- [ ] **API stability**: No breaking changes to public interfaces
- [ ] **Performance**: No regressions in computation speed

This roadmap provides a systematic approach to removing technical debt while maintaining the stability and reliability achieved through the Phase 0-2 refactoring effort.