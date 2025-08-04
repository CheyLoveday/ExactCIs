# Blaker Method Optimization Plan

**Target**: 80-90% performance improvement  
**Current baseline**: 18.8ms average execution time  
**Approach**: Staged implementation with validation at each step

## Current Performance Analysis

**Bottlenecks identified:**
- PMF recalculation overhead in tight loops
- MD5 hashing for cache keys (`BlakerPMFCache:57`)
- String interpolation in logging calls during hot paths
- Boundary validation inside p-value computation loops
- Linear-scale arithmetic causing numerical precision issues

**Files involved:**
- Primary: `src/exactcis/methods/blaker.py`
- Supporting: `src/exactcis/core.py` (for root-finding improvements)
- Testing: `tests/test_methods/test_blaker.py`
- Profiling: `profiling/performance_benchmark.py`

## Staged Implementation Plan

### Stage 1: Guard Logging Calls ⚡
**Difficulty**: LOW  
**Expected improvement**: 5-10%  
**Risk**: Minimal

**Implementation:**
```python
# Replace lines 106, 112, 131, 146, 157, 158 in blaker.py
if logger.isEnabledFor(logging.INFO):
    logger.info(f"[DEBUG_ACCEPTABILITY] n1=5,n2=10,m1=7,theta={theta:.2e}")
```

**Files to modify**: `src/exactcis/methods/blaker.py:106,112,131,146,157,158`

**Validation criteria:**
- All existing tests pass
- No change in numerical results
- Measurable performance improvement in verbose logging scenarios

---

### Stage 2: Inline Boundary Checks ⚡
**Difficulty**: LOW  
**Expected improvement**: 3-5%  
**Risk**: Low (requires careful validation logic)

**Implementation:**
- Move validation from `blaker_p_value():139-142` to `exact_ci_blaker():200`
- Pre-validate `a` range once per CI calculation
- Pass validation flag to `blaker_p_value()` to skip repeated checks

**Files to modify**: 
- `src/exactcis/methods/blaker.py:139-142` → `exact_ci_blaker():200`

**Validation criteria:**
- All existing tests pass
- Same error handling for invalid inputs
- Performance improvement in tight loops

---

### Stage 3: Primitive Cache Keys ⚡
**Difficulty**: LOW  
**Expected improvement**: 15-20%  
**Risk**: Low

**Implementation:**
- Replace `hashlib.md5(support_x.tobytes()).hexdigest()[:8]` with `tuple(support_x)` 
- Refactor `BlakerPMFCache` to use primitive types as keys
- Consider replacing custom cache with `@lru_cache` decorator

**Files to modify**: 
- `src/exactcis/methods/blaker.py:BlakerPMFCache:48-58`

**Validation criteria:**
- All existing tests pass
- Same caching behavior
- Significant reduction in MD5 computation overhead

---

### Stage 4: Log-Scale Operations 🔧
**Difficulty**: MODERATE  
**Expected improvement**: 30-35%  
**Risk**: Medium (numerical precision critical)

**Implementation:**
- Replace `nchg_pdf()` calls with existing `log_nchg_pmf()` from `core.py:19`
- Use `scipy.special.logsumexp()` for probability summation
- Implement early-exit summation when convergence achieved
- Maintain backward compatibility for edge cases

**Files to modify**:
- `src/exactcis/methods/blaker.py:blaker_acceptability():110`
- `src/exactcis/methods/blaker.py:blaker_p_value():150-155`

**Validation criteria:**
- All existing tests pass with numerical precision maintained
- Edge case handling preserved
- Substantial performance improvement
- Better numerical stability for extreme values

---

### Stage 5: 3-Point Pre-Bracketing 🔧
**Difficulty**: MODERATE  
**Expected improvement**: 25-40%  
**Risk**: Medium (algorithm complexity)

**Implementation:**
- Before calling `find_smallest_theta()`, evaluate p-values at 3 strategic θ points
- Use initial estimates to narrow search interval
- Reduce Brent's method iterations by providing better brackets

**Files to modify**:
- `src/exactcis/methods/blaker.py:exact_ci_blaker():214-229,235-251`
- Leverage existing `find_smallest_theta()` infrastructure

**Validation criteria:**
- All existing tests pass
- Same numerical accuracy as current implementation
- Significant reduction in root-finding iterations
- Robust handling of edge cases

---

## Cumulative Performance Targets

**After Stage 3**: ~30% improvement (multiplicative gains from low-effort optimizations)  
**After Stage 4**: ~60% cumulative improvement  
**After Stage 5**: **80-90% target achieved**

## Decision Points

### After Stage 3
- **Measure cumulative improvement**
- If >60%: Continue to Stage 4
- If <60%: Investigate why low-effort optimizations underperformed

### After Stage 5
- **Check if 80-90% target achieved**
- If YES: **STOP** - Mission accomplished
- If NO: Consider additional moderate-effort optimizations

## Implementation Protocol

**For each stage:**

1. **Create feature branch**: `git checkout -b feature/blaker-stage-N-description`
2. **Implement changes** with comprehensive documentation
3. **Create stage-specific tests**: Verify correctness against baseline
4. **Profile performance**: 
   ```bash
   cd profiling/
   python performance_benchmark.py --method blaker --runs 100
   ```
5. **Validate improvements**: Require measurable performance gain
6. **Run full test suite**: 
   ```bash
   uv run pytest tests/test_methods/test_blaker.py -v
   uv run pytest tests/test_integration.py -v
   ```
7. **Code review**: Focus on numerical accuracy and edge cases
8. **Merge criteria**: Tests pass AND performance improves AND zero regressions

## Abort Criteria

**Stop immediately if:**
- Any stage fails to improve performance meaningfully
- Introduces numerical regressions
- Breaks existing functionality
- Makes code significantly more complex without proportional gains

## Success Metrics

**Performance**: 80-90% improvement in average execution time  
**Reliability**: Zero test regressions  
**Maintainability**: Code complexity remains manageable  
**Numerical accuracy**: No degradation in precision or edge case handling

## Post-Implementation

**Documentation**: Update performance benchmarks in `profiling/README.md`  
**Validation**: Run extended test suite with large tables  
**Monitoring**: Establish performance regression testing in CI

---

**Status**: Ready for implementation  
**Next step**: Begin Stage 1 - Guard Logging Calls