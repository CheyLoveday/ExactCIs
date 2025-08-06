# ExactCIs Implementation Roadmap

**Final Implementation Plan - August 5, 2025**

## Executive Summary

This document provides a comprehensive analysis of planned improvements against the current codebase and establishes a concrete implementation roadmap. After reviewing all proposed fixes from multiple AI reviewers and analyzing the current source code, this plan prioritizes changes that will provide maximum impact while preserving the excellent theoretical foundation already implemented.

## Current Implementation Analysis

### ✅ **Excellent Foundation Already Established**

The current codebase demonstrates several strengths that **should be preserved**:

1. **Strong Mathematical Foundation**:
   - Mid-P method uses theoretically sound grid search with CI inversion
   - Unconditional method implements profile likelihood approach (theoretically superior)
   - Robust numerical stability with log-space calculations

2. **Comprehensive Architecture**:
   - Well-structured utilities: `optimization.py`, `root_finding.py`, `parallel.py`, `shared_cache.py`
   - Extensive caching and optimization infrastructure already exists
   - Batch processing capabilities with parallel support
   - JIT acceleration via Numba where available

3. **Production-Ready Features**:
   - Comprehensive error handling and validation
   - Progress callbacks for long-running operations
   - Timeout protection and graceful degradation
   - Memory usage optimizations

### ⚠️ **Key Issues Identified**

1. **Performance Bottleneck**: Unconditional method's O(n₁ × n₂) complexity per theta evaluation
2. **Grid Dependency**: Both Mid-P and Unconditional methods show sensitivity to grid resolution
3. **Code Organization**: Some duplication in interval search patterns across methods
4. **Memory Usage**: Large sample sizes create significant memory pressure in unconditional method

## Planned Improvements vs. Current Reality

### 1. **Shared Search Logic Abstraction** 
**Status**: ⚠️ **NEEDS CAREFUL EVALUATION**

**Current Reality**:
- `src/exactcis/utils/optimization.py` already exists with `CICache` and search utilities
- `src/exactcis/utils/root_finding.py` provides sophisticated root-finding infrastructure
- Methods have significantly different algorithmic approaches:
  - Mid-P: Grid search with vectorized PMF calculations
  - Unconditional: Profile likelihood with MLE optimization per theta

**Revised Recommendation**:
- **DO NOT** create a generic `_search.py` module that forces methods into same pattern
- **INSTEAD**: Enhance existing `optimization.py` with method-specific optimizations
- **PRESERVE**: Each method's unique algorithmic strengths

### 2. **Root-Finding for Unconditional Method**
**Status**: ✅ **HIGHEST PRIORITY - READY FOR IMPLEMENTATION**

**Current Reality**:
- Profile likelihood approach is mathematically superior (correctly implemented)
- Grid search over theta is the performance bottleneck
- `root_finding.py` infrastructure already exists and is sophisticated

**Implementation Plan**:
- Replace theta grid search with `scipy.optimize.brentq` for interval bounds
- Retain profile likelihood MLE optimization for nuisance parameter  
- Ensure monotonicity verification and robust error handling
- Target: 10-50x performance improvement for large samples

### 3. **Mid-P Adaptive Grid Refinement**
**Status**: ✅ **MEDIUM PRIORITY - ENHANCEMENT READY**

**Current Reality**:
- Current grid search implementation is fast and reliable
- Some heuristic adjustments (`* 0.9`, `* 1.1`) could be eliminated
- `optimization.py` has foundation for adaptive strategies

**Implementation Plan**:
- Two-stage refinement: coarse grid → fine grid around preliminary bounds
- Remove heuristic corrections with better boundary detection
- Target: Improved precision without significant performance cost

### 4. **Memory Optimization for Large Samples**
**Status**: ✅ **HIGH PRIORITY - ARCHITECTURE ENHANCEMENT**

**Current Reality**:
- Joint probability matrix: `(n₁+1) × (n₂+1)` per theta evaluation
- For 1000×1000 samples: ~8MB per theta, significant memory pressure
- Early termination and caching already partially implemented

**Implementation Plan**:
- Sparse matrix operations for low-probability regions
- Chunked processing with streaming computation
- Enhanced early termination based on probability thresholds
- Memory usage warnings and automatic fallback strategies

## Implementation Priority Matrix

### **Phase 1: Performance Critical (Weeks 1-2)**

#### 1.1 Root-Finding for Unconditional Method ⭐ **HIGHEST IMPACT**
- **Files to modify**: `src/exactcis/methods/unconditional.py`
- **Dependencies**: `src/exactcis/utils/root_finding.py` (already exists)
- **Estimated effort**: 3-5 days
- **Expected improvement**: 10-50x performance for large samples
- **Risk**: Low (preserves existing profile likelihood approach)

#### 1.2 Memory Optimization for Unconditional Method ⭐ **HIGH IMPACT**
- **Files to modify**: `src/exactcis/methods/unconditional.py`, `src/exactcis/utils/optimization.py`
- **New capabilities**: Sparse matrix support, chunked processing
- **Estimated effort**: 4-6 days
- **Expected improvement**: Support for much larger sample sizes
- **Risk**: Medium (requires careful testing of numerical stability)

### **Phase 2: Precision Enhancement (Weeks 3-4)**

#### 2.1 Mid-P Adaptive Grid Refinement ⭐ **MEDIUM IMPACT**
- **Files to modify**: `src/exactcis/methods/midp.py`
- **Dependencies**: `src/exactcis/utils/optimization.py` (enhance existing)
- **Estimated effort**: 2-3 days  
- **Expected improvement**: Better precision, cleaner implementation
- **Risk**: Low (current method already works well)

#### 2.2 Enhanced Caching Strategy ⭐ **MEDIUM IMPACT**
- **Files to modify**: `src/exactcis/utils/optimization.py`, `src/exactcis/utils/shared_cache.py`
- **New capabilities**: Cross-theta caching, pattern recognition
- **Estimated effort**: 3-4 days
- **Expected improvement**: Faster repeated calculations
- **Risk**: Low (builds on existing infrastructure)

### **Phase 3: Architecture Cleanup (Week 5)**

#### 3.1 Enhanced Parameter Validation ⭐ **LOW IMPACT**
- **Files to modify**: `src/exactcis/utils/validators.py`
- **New capabilities**: Grid size recommendations, memory usage warnings
- **Estimated effort**: 1-2 days
- **Expected improvement**: Better user experience
- **Risk**: Very low

#### 3.2 Documentation and Testing Updates ⭐ **LOW IMPACT**  
- **Files to modify**: Documentation files, test suite
- **Dependencies**: Updated implementations from Phases 1-2
- **Estimated effort**: 2-3 days
- **Expected improvement**: Accuracy of documentation
- **Risk**: Very low

## Technical Implementation Specifications

### **Root-Finding Implementation Details**

```python
# Conceptual approach - NOT actual code
def find_confidence_bounds_with_root_finding(p_value_func, alpha, sample_or):
    """
    Replace grid search with bracketed root finding for theta bounds.
    
    Uses scipy.optimize.brentq to find theta values where:
    p_value_func(theta) - alpha/2 = 0
    """
    # Lower bound: find theta < sample_or where p-value crosses alpha/2
    lower_bound = brentq(
        lambda theta: p_value_func(theta) - alpha/2,
        lo=theta_min, hi=sample_or,
        xtol=1e-6
    )
    
    # Upper bound: find theta > sample_or where p-value crosses alpha/2  
    upper_bound = brentq(
        lambda theta: p_value_func(theta) - alpha/2,
        lo=sample_or, hi=theta_max,
        xtol=1e-6
    )
    
    return lower_bound, upper_bound
```

### **Memory Optimization Strategy**

```python
# Conceptual approach - NOT actual code
def compute_joint_probabilities_chunked(n1, n2, p1, p2, chunk_size=1000):
    """
    Stream-process joint probability matrix to reduce memory usage.
    
    Instead of creating full (n1+1) × (n2+1) matrix, process in chunks
    and early-terminate on low-probability regions.
    """
    total_log_prob = -np.inf
    
    for i_chunk in range(0, n1+1, chunk_size):
        for j_chunk in range(0, n2+1, chunk_size):
            # Process chunk with early termination
            chunk_contrib = process_probability_chunk(
                i_chunk, j_chunk, chunk_size, p1, p2
            )
            if chunk_contrib < threshold:
                continue  # Skip low-probability regions
            total_log_prob = logsumexp([total_log_prob, chunk_contrib])
    
    return total_log_prob
```

## Risk Assessment and Mitigation

### **High-Risk Areas**
1. **Numerical Stability**: Root-finding requires careful handling of edge cases
   - **Mitigation**: Extensive testing with boundary conditions, fallback to grid search
   
2. **Backwards Compatibility**: Performance optimizations might change results slightly
   - **Mitigation**: Comprehensive regression testing, optional parameters for old behavior

### **Medium-Risk Areas**
1. **Memory Optimization Complexity**: Chunking algorithms can be complex
   - **Mitigation**: Phased implementation with extensive memory profiling
   
2. **Performance Regression**: Optimization attempts might slow down common cases
   - **Mitigation**: Benchmarking at each step, performance regression tests

### **Low-Risk Areas**
1. **Mid-P Refinement**: Current implementation already works well
2. **Documentation Updates**: Non-functional changes

## Success Metrics

### **Performance Targets**
- **Unconditional Method**: 10-50x speedup for n > 500 samples
- **Memory Usage**: Support 2000×2000 tables without excessive memory pressure
- **Mid-P Precision**: Eliminate heuristic adjustments while maintaining speed

### **Quality Targets**
- **Zero Breaking Changes**: All existing functionality preserved
- **Comprehensive Testing**: >95% code coverage maintained
- **Documentation Accuracy**: All reviewed documents updated to reflect changes

## Implementation Team Recommendations

### **Phase 1 Focus** 
Prioritize unconditional method performance improvements - this addresses the most significant current limitation while preserving the excellent theoretical foundation.

### **Preserve What Works**
The current Mid-P implementation is excellent and should only receive precision enhancements, not algorithmic changes.

### **Build on Existing Infrastructure**
Leverage the sophisticated utilities already built rather than creating parallel systems.

---

**Roadmap prepared by Claude - August 5, 2025**  
**Status: Ready for Implementation | Risk Level: Low-Medium | Timeline: 5 weeks**