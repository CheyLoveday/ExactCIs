# Conditional Method Optimization Plan

**Target**: Significant performance improvement for Fisher's exact confidence intervals  
**Current baseline**: ~9.3ms average execution time (from profiling analysis)  
**Approach**: Staged implementation focusing on low-risk/high-ROI optimizations

## Current Performance Analysis

**Bottlenecks identified from code review:**
- Repeated `nchypergeom_fisher.cdf/sf` calls in tight root-finding loops (lines 165, 279)
- String interpolation in debug logging during bracket expansion (lines 182, 196, 213, 295, 309, 326)
- Repeated tuple unpacking and marginal calculations inside p_value_func closures
- Inefficient bracket expansion with sequential trials (lines 192-196, 208-213, 304-309, 321-326)
- No early exit conditions for extreme parameter values

**Files involved:**
- Primary: `src/exactcis/methods/conditional.py`
- Testing: `tests/test_methods/test_conditional.py`
- Profiling: `profiling/performance_benchmark.py`

## Staged Implementation Plan

### Stage 1: Memoize CDF/SF Calls ⚡
**Difficulty**: LOW  
**Expected improvement**: 20-30%  
**Risk**: Minimal

**Implementation:**
```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _cdf_cached(a, N, c1, r1, psi):
    return nchypergeom_fisher.cdf(a, N, c1, r1, psi)

@lru_cache(maxsize=512)
def _sf_cached(a, N, c1, r1, psi):
    return nchypergeom_fisher.sf(a, N, c1, r1, psi)
```

**Files to modify**: `src/exactcis/methods/conditional.py:165,279`

**Validation criteria:**
- All existing tests pass
- No change in numerical results
- Significant performance improvement in repeated CDF/SF evaluations

---

### Stage 2: Guard Debug Logging ⚡
**Difficulty**: LOW  
**Expected improvement**: 5-10%  
**Risk**: Minimal

**Implementation:**
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Lower bound initial bracket: lo={lo} (val={lo_val}), hi={hi} (val={hi_val})")
```

**Files to modify**: `src/exactcis/methods/conditional.py:182,196,213,295,309,326`

**Validation criteria:**
- All existing tests pass
- No change in numerical results
- Reduced string interpolation overhead

---

### Stage 3: Hoist Fixed Marginals ⚡
**Difficulty**: LOW  
**Expected improvement**: 5-10%  
**Risk**: Low

**Implementation:**
```python
def fisher_lower_bound(a, b, c, d, min_k, max_k, N, r1, c1, alpha):
    # Compute once, outside closure
    target_prob = alpha / 2.0
    
    # Create closure that captures precomputed values
    def p_value_func(psi):
        return _sf_cached(a-1, N, c1, r1, psi) - target_prob
```

**Files to modify**: `src/exactcis/methods/conditional.py:fisher_lower_bound,fisher_upper_bound`

**Validation criteria:**
- All existing tests pass
- Same numerical results
- Reduced tuple unpacking overhead

---

### Stage 4: Coarse-Grid 3-Point Pre-Bracket 🔧
**Difficulty**: MODERATE  
**Expected improvement**: 15-25%  
**Risk**: Medium

**Implementation:**
```python
def _find_better_bracket(p_value_func, lo, hi):
    """Use 3-point evaluation to narrow bracket before brentq."""
    mid = np.sqrt(lo * hi)  # Geometric mean
    
    try:
        lo_val = p_value_func(lo)
        mid_val = p_value_func(mid)
        hi_val = p_value_func(hi)
        
        # Find subinterval where sign change occurs
        if lo_val * mid_val < 0:
            return lo, mid
        elif mid_val * hi_val < 0:
            return mid, hi
        else:
            return lo, hi  # Fallback to original bracket
    except:
        return lo, hi
```

**Files to modify**: `src/exactcis/methods/conditional.py:fisher_lower_bound,fisher_upper_bound`

**Validation criteria:**
- All existing tests pass
- Same numerical accuracy
- Reduced brentq iterations

---

### Stage 5: Vectorized Bracket Expansion 🔧
**Difficulty**: MODERATE  
**Expected improvement**: 10-20%  
**Risk**: Medium

**Implementation:**
```python
def _expand_bracket_vectorized(p_value_func, initial_lo, initial_hi, target_sign_lo, target_sign_hi):
    """Vectorized bracket expansion instead of sequential loops."""
    
    # Generate candidate values
    lo_candidates = initial_lo / (5.0 ** np.arange(1, 21))  # 20 candidates
    hi_candidates = initial_hi * (5.0 ** np.arange(1, 21))  # 20 candidates
    
    try:
        # Vectorized evaluation where possible
        lo_vals = np.array([p_value_func(x) for x in lo_candidates])
        hi_vals = np.array([p_value_func(x) for x in hi_candidates])
        
        # Find first suitable bracket
        lo_idx = np.where((lo_vals > 0) == target_sign_lo)[0]
        hi_idx = np.where((hi_vals < 0) == target_sign_hi)[0]
        
        if len(lo_idx) > 0 and len(hi_idx) > 0:
            return lo_candidates[lo_idx[0]], hi_candidates[hi_idx[0]]
    except:
        pass
        
    return initial_lo, initial_hi  # Fallback
```

**Files to modify**: `src/exactcis/methods/conditional.py` (bracket expansion logic)

**Validation criteria:**
- All existing tests pass
- Same numerical results
- Faster bracket finding

---

## Implementation Priority and Decision Points

### Phase 1: Low-Risk Optimizations (Stages 1-3)
**Expected cumulative improvement**: 30-50%
- **Stage 1**: Memoization - highest ROI, lowest risk
- **Stage 2**: Logging guards - minimal risk, easy win
- **Stage 3**: Hoist marginals - simple refactoring

**Decision point after Phase 1:**
- If >40% improvement achieved: Continue to Phase 2
- If <40% improvement: Investigate why cache isn't helping

### Phase 2: Moderate-Risk Optimizations (Stages 4-5)
**Expected additional improvement**: 25-45%
- **Stage 4**: 3-point pre-bracketing - moderate complexity
- **Stage 5**: Vectorized expansion - higher complexity

**Decision point after Phase 2:**
- If cumulative >60% improvement: STOP - excellent result
- If <60%: Consider additional optimizations

## Implementation Protocol

**For each stage:**

1. **Create feature branch**: `git checkout -b feature/conditional-stage-N-description`
2. **Implement changes** with comprehensive documentation
3. **Validate correctness**: Compare results against baseline
4. **Profile performance**: 
   ```bash
   cd profiling/
   python performance_benchmark.py --method conditional --runs 100
   ```
5. **Run full test suite**: 
   ```bash
   uv run pytest tests/test_methods/test_conditional.py -v
   uv run pytest tests/test_integration.py -v
   ```
6. **Measure improvement**: Require measurable performance gain
7. **Code review**: Focus on numerical accuracy and edge cases

## Deferred Optimizations (Future Consideration)

### Medium Priority
- **Log-scale operations**: Use `logsf/logcdf` for numerical stability
- **Early exit conditions**: Skip computation when bounds are obvious

### Low Priority (Complex/High Risk)
- **JIT compilation**: Numba/Cython for tail-sum loops
- **Analytical bounds**: Cornfield approximation for initial brackets
- **Batch precomputation**: Full support CDF tables
- **GPU acceleration**: CUDA/OpenCL for hypergeometric sums

## Success Metrics

**Performance**: 50-70% improvement in average execution time  
**Reliability**: Zero test regressions  
**Maintainability**: Code complexity remains manageable  
**Numerical accuracy**: No degradation in precision

## Risk Mitigation

**Stage 1 (Memoization)**: 
- Cache size limits prevent memory issues
- LRU eviction handles long-running processes

**Stage 2 (Logging)**: 
- Minimal change, only guards existing code

**Stage 3 (Hoisting)**: 
- Simple refactoring, easy to validate

**Stages 4-5**: 
- Comprehensive fallbacks to original methods
- Extensive validation against baseline results

---

**Status**: Ready for implementation  
**Priority**: Stage 1 → Stage 2 → Stage 3 → Evaluate → Stage 4 → Stage 5  
**Next step**: Begin Stage 1 - Memoize CDF/SF Calls