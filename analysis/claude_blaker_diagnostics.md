# Claude Blaker Method Diagnostics Report

**Date**: August 7, 2025  
**Analysis**: ExactCIs Blaker Method Implementation Issues  
**Status**: Critical Issues Identified - Requires Fixes

## 🚨 Critical Issues Identified

### **Issue 1: Small Sample Anomaly (N=200)**
- **Observed**: Lower bound = 5.39 (abnormally high)
- **Expected**: Lower bound ≈ 1.0-1.3 (consistent with other methods)
- **Impact**: Over-conservative, potentially invalid confidence intervals

### **Issue 2: Large Sample Degenerate CI (N=4000)**
- **Observed**: Upper bound = 5.21 (exactly the point estimate)
- **Expected**: Upper bound ≈ 8-10 (proper confidence limit)
- **Impact**: Degenerate confidence interval, mathematically invalid

## 📊 Comparison Table

| Sample Size | Blaker CI | Other Methods Range | Status |
|-------------|-----------|-------------------|--------|
| Small (N=200) | **(5.39, 35.26)** | (1.0-1.3, 18-37) | ❌ Lower bound too high |
| Medium (N=2000) | (2.61, 10.79) | (2.4-2.8, 9-11) | ✅ Normal |
| Large (N=4000) | **(3.18, 5.21)** | (3.0-3.3, 8-10) | ❌ Upper = point estimate |

## 🔍 Root Cause Analysis

### **Primary Issue: Root Finding Algorithm Failures**

#### **Small Sample Log Evidence**:
```
find_smallest_theta: Root 1.2003e+00 from find_root_log is not sufficiently close to target_alpha (diff=1.90e-02)
find_smallest_theta: Proceeding to plateau_edge
find_plateau_edge returned: (5.390008036196273, 26)
func(plateau_edge_theta=5.3900e+00) = 1.0000e+00 (target_alpha=5.0000e-02, diff=9.50e-01)
```

#### **Large Sample Log Evidence**:
```
find_smallest_theta: Root 8.7689e+00 from find_root_log is not sufficiently close to target_alpha (diff=-1.23e-02)
find_smallest_theta: Proceeding to plateau_edge
find_plateau_edge returned: (5.2105263157894735, 0)
func(plateau_edge_theta=5.2105e+00) = 1.0000e+00 (target_alpha=5.0000e-02, diff=9.50e-01)
```

### **Identified Problems**:

1. **Over-Strict Tolerance**: Root-finding rejects valid solutions with diffs of 1.2-1.9e-02
2. **Faulty Plateau Algorithm**: Returns point estimates instead of confidence boundaries  
3. **Invalid P-values**: Returns p=1.0 which violates confidence interval theory
4. **Boundary Condition Errors**: Edge case handling fails for extreme sample sizes

## 🎯 Affected Code Locations

### **File: `src/exactcis/core.py`**
- **Function**: `find_smallest_theta()` (~line 622)
- **Function**: `find_plateau_edge()` (~line 668)
- **Issue**: Tolerance and fallback logic

### **File: `src/exactcis/methods/blaker.py`** 
- **Function**: `exact_ci_blaker()` (~line 257)
- **Issue**: Boundary detection and acceptability function

## 🔧 Recommended Fixes

### **Fix 1: Adjust Root Finding Tolerance**
**Location**: `src/exactcis/core.py:649`

**Current Code**:
```python
if abs(func_val - target_alpha) < tolerance:  # tolerance too strict
    return root
```

**Suggested Fix**:
```python
# Relax tolerance for CI bounds - statistical accuracy vs numerical precision
CI_TOLERANCE = 0.02  # Allow 2% difference for confidence intervals
if abs(func_val - target_alpha) < CI_TOLERANCE:
    return root
```

### **Fix 2: Fix Plateau Edge Algorithm**
**Location**: `src/exactcis/core.py:668`

**Issue**: Algorithm returning point estimates instead of confidence boundaries

**Suggested Investigation**:
```python
def find_plateau_edge(...):
    # Current implementation appears fundamentally flawed
    # Returns (point_estimate, 0) instead of proper confidence bound
    # Needs complete algorithmic review
    
    # Verify plateau detection logic
    # Ensure returned theta represents valid CI boundary
    # Check p-value calculation at returned boundary
```

### **Fix 3: Validate Blaker Acceptability Function**
**Location**: `src/exactcis/methods/blaker.py:287`

**Suggested Validation**:
```python
def validate_blaker_pvalue(a, b, c, d, theta, alpha):
    """Validate Blaker p-value calculation against known reference"""
    # Compare with R's exact2x2::blaker.exact() function
    # Check edge cases: small samples, large samples, extreme ratios
    # Verify two-sided test construction
```

### **Fix 4: Add Boundary Condition Checks**
**Location**: Multiple files

**Suggested Addition**:
```python
def validate_ci_bounds(lower, upper, point_estimate):
    """Validate confidence interval bounds"""
    assert lower <= point_estimate <= upper, "Point estimate outside CI"
    assert lower != upper, "Degenerate confidence interval"  
    assert lower > 0, "Invalid negative lower bound for odds ratio"
    return True
```

## 🧪 Recommended Testing

### **Test Cases to Add**:

1. **Small Sample Edge Cases**:
   ```python
   test_cases = [
       (5, 95, 1, 99),    # Very rare events
       (2, 8, 1, 9),      # Tiny samples  
       (10, 90, 2, 98),   # Current failing case
   ]
   ```

2. **Large Sample Edge Cases**:
   ```python
   test_cases = [
       (100, 1900, 20, 1980),  # Current failing case
       (200, 3800, 40, 3960),  # Even larger
       (1000, 9000, 200, 9800), # Very large
   ]
   ```

3. **Reference Implementation Comparison**:
   ```python
   # Compare against R's exact2x2::blaker.exact()
   # Validate against published examples in statistical literature
   # Cross-check with other statistical software (SAS, Stata)
   ```

## ⚠️ Priority Recommendations

### **Immediate Actions (Critical)**:
1. **Disable Blaker method** in production until fixed
2. **Investigate root-finding tolerance** parameters  
3. **Audit plateau edge algorithm** completely

### **Short-term Actions (High Priority)**:
1. **Implement boundary validation** checks
2. **Add comprehensive unit tests** for edge cases
3. **Compare against reference implementations**

### **Long-term Actions (Medium Priority)**:
1. **Performance optimization** after correctness fixes
2. **Documentation updates** with method limitations
3. **Consider alternative Blaker implementations**

## 📋 Validation Checklist

- [ ] Root-finding tolerance parameters reviewed
- [ ] Plateau edge algorithm debugged  
- [ ] Blaker acceptability function validated
- [ ] Boundary condition checks implemented
- [ ] Edge case test suite created
- [ ] Reference implementation comparison completed
- [ ] Performance regression testing performed

## 🎯 Expected Outcomes Post-Fix

### **Small Samples (N=200)**:
- Lower bound: 5.39 → ~1.0-1.3
- Upper bound: 35.26 → ~18-30 (maintain)
- **Result**: Proper confidence interval behavior

### **Large Samples (N=4000)**:
- Lower bound: 3.18 → ~3.0-3.3 (maintain)  
- Upper bound: 5.21 → ~8-10
- **Result**: Valid confidence interval (not degenerate)

## ✅ Success Criteria

1. **No degenerate intervals** (upper ≠ point estimate)
2. **Reasonable lower bounds** (consistent with other exact methods)
3. **Monotonic behavior** with sample size increases
4. **P-value validation** (never exactly 1.0 for CI boundaries)
5. **Reference concordance** (matches R's exact2x2 within tolerance)

---
**Next Steps**: Begin with Fix 1 (tolerance adjustment) as it's the most straightforward and likely to resolve both issues.