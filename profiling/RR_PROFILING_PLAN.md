# Relative Risk Methods Performance Profiling Plan

## Overview

This plan extends ExactCIs profiling infrastructure to include the newly implemented and fixed relative risk methods. Building on the existing OR profiling framework, we'll add comprehensive performance analysis for all 6 RR methods.

## Current Status

✅ **Recently Fixed RR Methods (August 2025):**
- **Score Methods**: Fixed infinite bound issues via enhanced bracket expansion 
- **Parameter Ordering**: Fixed `ci_score_cc_rr` API compatibility 
- **Zero-Cell Handling**: Enhanced `ci_wald_correlated_rr` edge case behavior
- **Test Coverage**: All 57 RR tests now passing (50 passed, 7 skipped, 0 failed)

## Profiling Infrastructure Extensions

### 1. Enhanced Performance Profiler (`profiling/performance_profiler.py`)

**Add RR Methods Integration:**
```python
# Add to existing imports
from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr, 
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr
)

# Extend METHOD_FUNCTIONS dictionary
RR_METHODS = {
    'wald_rr': ci_wald_rr,
    'wald_katz_rr': ci_wald_katz_rr,
    'wald_corr_rr': ci_wald_correlated_rr,
    'score_rr': ci_score_rr,
    'score_cc_rr': ci_score_cc_rr,
    'ustat_rr': ci_ustat_rr
}

# Combined methods for comparative analysis
ALL_METHODS = {**OR_METHODS, **RR_METHODS}
```

**RR-Specific Test Scenarios:**
```python
RR_TEST_SCENARIOS = [
    # Standard epidemiological cases
    (20, 80, 10, 90, "epi_standard"),       # RR ≈ 2.0
    (90, 910, 10, 990, "epi_smoking"),     # RR ≈ 9.0 (smoking-lung cancer)
    (15, 85, 25, 75, "clinical_trial"),    # RR ≈ 0.6 (protective effect)
    
    # Edge cases for RR-specific algorithms  
    (0, 100, 5, 95, "zero_exposed"),       # RR = 0
    (5, 5, 0, 10, "zero_unexposed"),       # RR = ∞
    (3, 7, 2, 8, "small_sample"),          # Small n, finite bounds
    
    # Score method stress tests (recent fixes)
    (15, 5, 10, 10, "score_fix_case"),     # Previously returned (1.18, inf)
    (1, 1, 1, 1, "minimal_counts"),        # Extreme sparsity
    
    # Performance stress cases
    (150, 50, 100, 100, "moderate_large"), # n=400
    (500, 500, 300, 700, "large_balanced"), # n=2000
]
```

### 2. New RR-Specific Profiling Scripts

#### **A. `profiling/rr_performance_benchmark.py`**
```python
"""
Performance benchmark specifically for relative risk methods.
Focus on recent algorithmic improvements and edge cases.
"""

class RRPerformanceBenchmark:
    def __init__(self):
        self.rr_methods = RR_METHODS
        self.scenarios = RR_TEST_SCENARIOS
        
    def benchmark_score_methods_fixes(self):
        """
        Benchmark score methods focusing on recent bracket expansion fixes.
        Validates that finite bounds are returned in reasonable time.
        """
        
    def benchmark_zero_cell_handling(self):
        """
        Test performance of zero-cell detection and delegation logic.
        Particularly for wald_correlated_rr improvements.
        """
        
    def benchmark_parameter_ordering_impact(self):
        """
        Verify that parameter order fix didn't impact performance.
        Compare ci_score_cc_rr before/after API change.
        """
```

#### **B. `profiling/rr_accuracy_benchmark.py`**
```python
"""
Cross-validate RR methods against R packages and SciPy implementations.
Performance + accuracy combined analysis.
"""

def compare_with_scipy_relative_risk():
    """Compare with scipy.stats.contingency.relative_risk (≥1.11)"""
    
def compare_with_r_ratesci():
    """Compare with R ratesci package via rpy2 (if available)"""
    
def validate_recent_fixes():
    """Ensure fixes maintain accuracy while improving performance"""
```

#### **C. `profiling/rr_scaling_analysis.py`**
```python
"""
Analyze how RR methods scale with table size and sparsity.
"""

def test_score_bracket_expansion_scaling():
    """
    Test how enhanced bracket expansion performs across table sizes.
    Measure iterations needed for convergence.
    """
    
def test_zero_cell_delegation_overhead():
    """
    Measure overhead of zero-cell detection logic.
    """
```

### 3. Integration with Existing Infrastructure

#### **Enhanced Line Profiler (`profiling/line_profiler.py`)**
```python
# Add RR methods to detailed line-by-line profiling
RR_PROFILE_TARGETS = [
    ('ci_score_rr', 'Score method with bracket expansion fixes'),
    ('ci_score_cc_rr', 'Score CC with parameter order fix'),  
    ('ci_wald_correlated_rr', 'Wald correlated with zero-cell handling'),
    ('find_score_ci_bound', 'Enhanced root finding algorithm')
]

def profile_rr_methods():
    """Add RR methods to existing line profiler infrastructure"""
```

#### **Extended Timing Results (`profiling/timing_results.json`)**
```json
{
  "rr_methods": {
    "wald_rr": {"scenarios": {...}, "avg_time_ms": ...},
    "score_rr": {"scenarios": {...}, "convergence_stats": {...}},
    "score_cc_rr": {"scenarios": {...}, "parameter_fix_impact": {...}}
  },
  "or_vs_rr_comparison": {
    "similar_scenarios": {...},
    "algorithmic_differences": {...}
  }
}
```

### 4. New Performance Reports

#### **A. `profiling/reports/RR_PERFORMANCE_ANALYSIS.md`**
```markdown
# Relative Risk Methods Performance Analysis

## Executive Summary
- Performance impact of August 2025 fixes
- Comparison with OR methods  
- Scaling characteristics
- Recommendations for optimization

## Method-Specific Analysis
### Score Methods
- Bracket expansion algorithm performance
- Convergence rates across scenarios  
- Root finding efficiency metrics

### Zero-Cell Handling
- Delegation logic overhead
- Performance of fallback methods

### Parameter Order Fix Impact
- Benchmarks before/after API change
- Validation of no performance regression
```

#### **B. `profiling/reports/RR_SCALING_REPORT.md`**
```markdown
# RR Methods Scaling Analysis

## Sample Size Scaling
- Performance vs table size (n=20 to n=10000)
- Memory usage patterns
- Asymptotic behavior

## Sparsity Impact  
- Zero cell scenarios
- Extreme ratio cases
- Convergence characteristics
```

### 5. Automated Benchmark Integration

#### **Enhanced `profiling/performance_profiler.py`**
```python
class ComprehensiveProfiler:
    def __init__(self):
        self.or_methods = OR_METHODS
        self.rr_methods = RR_METHODS  # New
        
    def run_comprehensive_benchmark(self):
        """
        Run both OR and RR method benchmarks.
        Generate comparative analysis.
        """
        
    def generate_performance_summary(self):
        """
        Create unified performance report covering:
        - Method-specific timing
        - OR vs RR algorithmic differences  
        - Recent fix impact analysis
        - Scaling recommendations
        """
```

## Implementation Timeline

### **Phase 1: Infrastructure (1-2 days)**
1. ✅ Extend `performance_profiler.py` with RR methods
2. ✅ Add RR-specific test scenarios 
3. ✅ Create baseline benchmark for current RR performance

### **Phase 2: RR-Specific Analysis (2-3 days)**  
1. ✅ Implement `rr_performance_benchmark.py`
2. ✅ Focus on score method fixes validation
3. ✅ Zero-cell handling performance analysis
4. ✅ Parameter ordering impact assessment

### **Phase 3: Comparative Analysis (1-2 days)**
1. ✅ OR vs RR performance comparison
2. ✅ Cross-method scaling analysis
3. ✅ Memory usage profiling

### **Phase 4: Documentation & Reporting (1 day)**
1. ✅ Generate comprehensive performance reports
2. ✅ Update profiling README with RR methods
3. ✅ Create performance recommendations

## Success Metrics

### **Performance Targets:**
- **Score Methods**: Finite bounds returned in <100ms for standard cases
- **Zero-Cell Detection**: <1ms overhead for delegation logic  
- **Parameter Fix**: No performance regression from API changes
- **Scaling**: Linear or better scaling with table size

### **Validation Targets:**
- **Accuracy**: Results match R `ratesci` where available
- **Robustness**: Handle all edge cases without infinite loops
- **Consistency**: Performance predictable across scenarios

## Expected Deliverables

1. **Extended Profiling Infrastructure** - RR methods integrated into existing framework
2. **Performance Benchmarks** - Comprehensive timing and scaling analysis  
3. **Validation Reports** - Accuracy + performance combined analysis
4. **Optimization Recommendations** - Data-driven suggestions for further improvements

## Integration with Strategic Goals

This profiling extension directly supports the strategic objectives identified in `.devplans/social_media_research.md`:

- **Performance Features**: Validate that Python RR methods outperform R equivalents
- **User Trust**: Demonstrate accuracy + speed through comprehensive benchmarking  
- **Production Readiness**: Ensure robust performance across real-world scenarios
- **Documentation**: Create performance evidence for user adoption

## Next Steps

1. **Immediate**: Extend `performance_profiler.py` with RR methods
2. **Week 1**: Complete RR-specific benchmarking infrastructure  
3. **Week 2**: Generate comprehensive performance reports
4. **Week 3**: Integrate findings into package documentation and optimization roadmap