# ExactCIs Optimization Quick Reference

## 🎯 Key Performance Issues

| Method | Current Speed | Main Bottleneck | Fix Priority |
|--------|-------------|-----------------|-------------|
| **Blaker's** | 0.1985s (64x slower) | `log_binom_coeff` calls | 🔴 **CRITICAL** |
| **Mid-P** | 0.1696s (55x slower) | Same as Blaker's | 🟡 High |
| **Unconditional** | 0.1098s (35x slower) | Grid search | 🟡 Medium |
| **Conditional** | 0.0031s | No issues | ✅ Good |

## 🔥 Critical Lines of Code

**File: `/src/exactcis/methods/blaker.py`**
- **Line 202**: `probs = nchg_pdf(...)` - **99.8% of Blaker's time**

**File: `/src/exactcis/core.py`**  
- **Lines 372-373**: `log_binom_coeff(n1, k)` calls - **41.9% of PMF time**
- **Line 99**: `log_binom_coeff` function - **Called 52,354 times**

## ⚡ Immediate Quick Fixes

### 1. Increase Cache Size (5 minutes)
```python
# In core.py, change:
@lru_cache(maxsize=128)  # OLD
@lru_cache(maxsize=2048)  # NEW - 16x larger cache
def log_binom_coeff(n, k):
```
**Expected improvement**: 20-30% faster

### 2. Cache PMF Weights (30 minutes)
```python
# Add to pmf_weights function:
@lru_cache(maxsize=512)
def pmf_weights(n1, n2, m, theta):
```
**Expected improvement**: 30-50% faster

### 3. Reduce Unconditional Grid Size (10 minutes)
```python
# Change default grid_size from 50 to 20 in unconditional.py
exact_ci_unconditional(..., grid_size: int = 20)  # Was 50
```
**Expected improvement**: 2-3x faster for unconditional

## 📊 Expected Combined Impact

**Conservative estimate** (implementing all quick fixes):
- **Blaker's**: 0.1985s → 0.040s (**5x faster**)
- **Mid-P**: 0.1696s → 0.050s (**3.4x faster**)  
- **Unconditional**: 0.1098s → 0.037s (**3x faster**)

## 🔍 Files to Modify

1. **`/src/exactcis/core.py`**
   - Line 99: Increase `log_binom_coeff` cache size
   - Line 302: Add caching to `pmf_weights`

2. **`/src/exactcis/methods/unconditional.py`**  
   - Reduce default grid_size parameter

3. **`/src/exactcis/methods/blaker.py`**
   - Consider adding PMF result caching

## 🧪 Testing Commands

```bash
# Before changes - baseline timing
uv run python profile_with_timeout.py --timeout 60 --num-cases 5

# After changes - measure improvement  
uv run python profile_with_timeout.py --timeout 60 --num-cases 5

# Comprehensive analysis
uv run python master_profiler.py --quick
```

## 🎯 Success Metrics

- **Blaker's method** < 0.050s average time
- **Mid-P method** < 0.060s average time  
- **Zero timeout failures** for standard test cases
- **Maintained accuracy** (results unchanged)

## 🚀 Implementation Order

1. **Day 1**: Increase `log_binom_coeff` cache size → Test → Measure
2. **Day 2**: Add `pmf_weights` caching → Test → Measure  
3. **Day 3**: Reduce unconditional grid size → Test → Measure
4. **Day 4**: Run comprehensive profiling → Document improvements

**Total effort**: ~4 days for 3-5x performance improvement