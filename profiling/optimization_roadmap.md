# ExactCIs Optimization Roadmap

## Performance Bottleneck Summary

Based on comprehensive profiling, the following critical issues have been identified:

### 🔴 Critical Issues (>100x performance impact)
1. **Blaker Method PMF Redundancy**: Recalculating same PMF values in each root-finding iteration
2. **Large Table Scaling**: Exponential time growth with table size for exact methods

### 🟡 Moderate Issues (2-10x performance impact)  
3. **Root-Finding Inefficiency**: Conservative bracketing with excessive safety checks
4. **Support Range Recalculation**: Redundant support range calculations
5. **Memory Allocation Overhead**: Frequent allocation of temporary arrays

### 🟢 Minor Issues (10-50% performance impact)
6. **Import Overhead**: Loading unused dependencies
7. **Logging Overhead**: Excessive debug logging in production
8. **Cache Miss Patterns**: Suboptimal cache usage in batch scenarios

## Detailed Optimization Plan

### Phase 1: Critical Performance Fixes

#### 1.1 Implement Advanced PMF Caching for Blaker Method

**Current Problem**:
```python
# In blaker.py - each root-finding iteration recalculates PMFs
def blaker_p_value(a, n1, n2, m1, theta, s):
    accept_probs_all_k = blaker_acceptability(n1, n2, m1, theta, s.x)  # EXPENSIVE!
    # ... rest of function
```

**Optimization Implementation**:

Create `profiling/optimizations/blaker_cache_optimization.py`:
```python
"""
Advanced caching optimization for Blaker method.
"""
import numpy as np
from functools import lru_cache
from typing import Tuple, Dict, Any
import hashlib

class BlakerPMFCache:
    """High-performance cache for Blaker PMF calculations."""
    
    def __init__(self, max_size: int = 1024):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
        
    def _generate_cache_key(self, n1: int, n2: int, m1: int, 
                          theta: float, support_hash: str) -> str:
        """Generate stable cache key for PMF parameters."""
        theta_rounded = round(theta, 12)  # Avoid floating point precision issues
        return f"{n1}_{n2}_{m1}_{theta_rounded}_{support_hash}"
    
    def get_pmf_values(self, n1: int, n2: int, m1: int, 
                      theta: float, support_x: np.ndarray) -> np.ndarray:
        """Get cached PMF values or calculate if not cached."""
        support_hash = hashlib.md5(support_x.tobytes()).hexdigest()[:8]
        cache_key = self._generate_cache_key(n1, n2, m1, theta, support_hash)
        
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        
        # Calculate PMF values
        from exactcis.core import nchg_pdf
        pmf_values = nchg_pdf(support_x, n1, n2, m1, theta)
        
        # Store in cache with LRU eviction if needed
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = pmf_values
        self.access_count[cache_key] = 1
        
        return pmf_values
    
    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self.access_count:
            return
            
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.access_count.clear()

# Global cache instance
_blaker_cache = BlakerPMFCache()

def optimized_blaker_acceptability(n1: int, n2: int, m1: int, 
                                 theta: float, support_x: np.ndarray) -> np.ndarray:
    """Optimized version of blaker_acceptability with caching."""
    return _blaker_cache.get_pmf_values(n1, n2, m1, theta, support_x)

def optimized_blaker_p_value(a: int, n1: int, n2: int, m1: int, 
                           theta: float, s) -> float:
    """Optimized Blaker p-value calculation with caching."""
    accept_probs_all_k = optimized_blaker_acceptability(n1, n2, m1, theta, s.x)
    
    idx_a = s.offset + a
    if not (0 <= idx_a < len(accept_probs_all_k)):
        return 1.0
    
    current_accept_prob_at_a = accept_probs_all_k[idx_a]
    epsilon = 1e-7
    
    # Vectorized comparison - much faster than loop
    mask = accept_probs_all_k <= current_accept_prob_at_a * (1 + epsilon)
    p_val = np.sum(accept_probs_all_k[mask])
    
    return p_val
```

**Implementation Steps**:
1. Create the optimization module (1 day)
2. Integrate with existing Blaker method (0.5 days)  
3. Add comprehensive tests (0.5 days)
4. Benchmark performance improvement (0.5 days)

**Expected Impact**: 70-80% reduction in Blaker method execution time

#### 1.2 Vectorized PMF Calculations

**Current Problem**:
```python
# Scalar calculations in loops
for k in support_range:
    pmf_value = log_nchg_pmf(k, n1, n2, m1, theta)  # One at a time
```

**Optimization Implementation**:

Create `profiling/optimizations/vectorized_pmf.py`:
```python
"""
Vectorized PMF calculations for improved performance.
"""
import numpy as np
from numba import jit, vectorize
from typing import Union

@vectorize(['float64(int64, int64, int64, int64, float64)'], nopython=True, cache=True)
def vectorized_log_nchg_pmf(k, n1, n2, m1, theta):
    """Vectorized version of log_nchg_pmf for NumPy arrays."""
    from exactcis.core import log_nchg_pmf
    return log_nchg_pmf(k, n1, n2, m1, theta)

def batch_pmf_calculation(support_array: np.ndarray, n1: int, n2: int, 
                         m1: int, theta: float) -> np.ndarray:
    """Calculate PMF for entire support array at once."""
    if len(support_array) < 100:
        # For small arrays, use regular calculation to avoid overhead
        from exactcis.core import log_nchg_pmf
        return np.array([log_nchg_pmf(k, n1, n2, m1, theta) for k in support_array])
    
    # For large arrays, use vectorized calculation
    return vectorized_log_nchg_pmf(support_array, n1, n2, m1, theta)

@jit(nopython=True, cache=True)
def fast_logsumexp(log_values):
    """JIT-compiled logsumexp for improved performance."""
    if len(log_values) == 0:
        return float('-inf')
    
    max_val = np.max(log_values)
    if not np.isfinite(max_val):
        return max_val
    
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))
```

**Expected Impact**: 30-50% improvement in all exact methods

#### 1.3 Smart Root-Finding Initialization

**Current Problem**:
- Conservative initial brackets require many expansions
- No method-specific bracket strategies  
- Excessive safety margins

**Optimization Implementation**:

Create `profiling/optimizations/smart_root_finding.py`:
```python
"""
Smart initialization strategies for root-finding algorithms.
"""
import numpy as np
from typing import Tuple, Callable

def get_smart_initial_bounds(a: int, b: int, c: int, d: int, 
                           alpha: float = 0.05, method: str = 'blaker') -> Tuple[float, float]:
    """Get smart initial bounds based on fast approximations."""
    
    # Use Wald-Haldane as fast approximation
    from exactcis.methods.wald import ci_wald_haldane
    
    try:
        wald_lower, wald_upper = ci_wald_haldane(a, b, c, d, alpha)
        
        # Method-specific expansion factors based on empirical analysis
        expansion_factors = {
            'blaker': (0.7, 1.4),      # Blaker tends to be wider than Wald
            'conditional': (0.8, 1.25), # Fisher's is more conservative  
            'midp': (0.85, 1.15),      # Mid-P is close to Wald
            'unconditional': (0.6, 1.6) # Barnard's can be quite wide
        }
        
        lower_factor, upper_factor = expansion_factors.get(method, (0.5, 2.0))
        
        smart_lower = max(1e-10, wald_lower * lower_factor)
        smart_upper = min(1e10, wald_upper * upper_factor)
        
        return smart_lower, smart_upper
        
    except Exception:
        # Fallback to conservative bounds if Wald fails
        or_point = (a * d) / (b * c) if b * c > 0 else 1.0
        return max(1e-10, or_point / 100), min(1e10, or_point * 100)

def adaptive_root_finding(func: Callable, target: float, 
                         initial_bounds: Tuple[float, float],
                         max_iterations: int = 100,
                         tolerance: float = 1e-8) -> float:
    """Adaptive root-finding with smart bracket management."""
    
    lo, hi = initial_bounds
    
    # Quick convergence check
    lo_val = func(lo) - target
    hi_val = func(hi) - target
    
    if abs(lo_val) < tolerance:
        return lo
    if abs(hi_val) < tolerance:
        return hi
    
    # Use bisection with adaptive step sizing
    for i in range(max_iterations):
        mid = np.exp((np.log(lo) + np.log(hi)) / 2)  # Geometric mean for log-scale
        mid_val = func(mid) - target
        
        if abs(mid_val) < tolerance:
            return mid
        
        # Adaptive tolerance - relax as we get closer
        adaptive_tol = tolerance * (1 + i / max_iterations)
        if abs(mid_val) < adaptive_tol:
            return mid
        
        # Update bounds
        if (lo_val > 0) == (mid_val > 0):
            lo, lo_val = mid, mid_val
        else:
            hi, hi_val = mid, mid_val
    
    # Return best estimate if no exact solution found
    return np.exp((np.log(lo) + np.log(hi)) / 2)
```

**Expected Impact**: 40-60% reduction in root-finding iterations

### Phase 2: Algorithm Improvements

#### 2.1 Adaptive Precision Control

Create `profiling/optimizations/adaptive_precision.py`:
```python
"""
Adaptive precision control based on problem characteristics.
"""
import numpy as np

def get_adaptive_tolerance(total_n: int, method: str, stage: str = 'initial') -> float:
    """Get adaptive tolerance based on problem size and method."""
    
    # Base tolerances by method
    base_tolerances = {
        'blaker': 1e-6,
        'conditional': 1e-7,
        'midp': 1e-7,
        'unconditional': 1e-5,  # Can be more relaxed
        'wald_haldane': 1e-10   # Very fast anyway
    }
    
    base_tol = base_tolerances.get(method, 1e-7)
    
    # Scale by problem size
    if total_n > 1000:
        size_factor = 10  # Relax for very large problems
    elif total_n > 500:
        size_factor = 5
    elif total_n > 100:
        size_factor = 2
    else:
        size_factor = 1
    
    # Scale by computation stage
    stage_factors = {
        'initial': 10,    # Rough bounds first
        'refining': 3,    # Better bounds
        'final': 1        # Precise bounds
    }
    
    stage_factor = stage_factors.get(stage, 1)
    
    return base_tol * size_factor * stage_factor

def multi_stage_root_finding(func, target, bounds, method='blaker', total_n=100):
    """Multi-stage root finding with adaptive precision."""
    
    # Stage 1: Quick rough bounds
    rough_tol = get_adaptive_tolerance(total_n, method, 'initial')
    rough_result = adaptive_root_finding(func, target, bounds, 
                                       max_iterations=20, tolerance=rough_tol)
    
    # Stage 2: Refine bounds around rough result
    refine_factor = 0.1
    refined_bounds = (rough_result * (1 - refine_factor),
                     rough_result * (1 + refine_factor))
    
    refined_tol = get_adaptive_tolerance(total_n, method, 'refining')
    refined_result = adaptive_root_finding(func, target, refined_bounds,
                                         max_iterations=30, tolerance=refined_tol)
    
    # Stage 3: Final precision if needed
    if total_n < 200:  # Only do final stage for smaller problems
        final_factor = 0.01
        final_bounds = (refined_result * (1 - final_factor),
                       refined_result * (1 + final_factor))
        
        final_tol = get_adaptive_tolerance(total_n, method, 'final')
        return adaptive_root_finding(func, target, final_bounds,
                                   max_iterations=20, tolerance=final_tol)
    
    return refined_result
```

#### 2.2 Memory-Efficient Support Calculation

Create `profiling/optimizations/support_optimization.py`:
```python
"""
Memory-efficient support range calculations with caching.
"""
import numpy as np
from functools import lru_cache
from typing import NamedTuple

class OptimizedSupportData(NamedTuple):
    x: np.ndarray
    min_val: int
    max_val: int
    offset: int
    size: int

@lru_cache(maxsize=512)
def cached_support_calculation(n1: int, n2: int, m1: int) -> OptimizedSupportData:
    """Cached support calculation to avoid redundant computation."""
    
    # Calculate support range
    min_k = max(0, n1 - (n1 + n2 - m1))
    max_k = min(n1, m1)
    
    if min_k > max_k:
        # Empty support
        return OptimizedSupportData(np.array([]), min_k, max_k, 0, 0)
    
    # Create support array
    support_array = np.arange(min_k, max_k + 1, dtype=np.int32)
    offset = -min_k
    size = len(support_array)
    
    return OptimizedSupportData(support_array, min_k, max_k, offset, size)

def get_support_size_estimate(n1: int, n2: int, m1: int) -> int:
    """Quick estimate of support size without full calculation."""
    min_k = max(0, n1 - (n1 + n2 - m1))
    max_k = min(n1, m1)
    return max(0, max_k - min_k + 1)
```

### Phase 3: Advanced Optimizations

#### 3.1 Parallel Processing Framework

Create `profiling/optimizations/parallel_pmf.py`:
```python
"""
Parallel processing framework for large PMF calculations.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import os

def should_use_parallel(support_size: int, complexity_factor: float = 1.0) -> bool:
    """Determine if parallel processing would be beneficial."""
    
    # Empirical threshold - parallel overhead is significant
    threshold = 1000 * complexity_factor  
    
    # Consider system resources
    max_workers = os.cpu_count() or 1
    if max_workers < 2:
        return False
    
    return support_size > threshold

def parallel_pmf_calculation(support_array: np.ndarray, n1: int, n2: int, 
                           m1: int, theta: float, n_workers: int = None) -> np.ndarray:
    """Calculate PMF values in parallel for large support arrays."""
    
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 1)  # Conservative default
    
    # Split support array into chunks
    chunk_size = max(1, len(support_array) // n_workers)
    chunks = [support_array[i:i + chunk_size] 
              for i in range(0, len(support_array), chunk_size)]
    
    def calculate_chunk(chunk):
        from exactcis.core import log_nchg_pmf
        return np.array([log_nchg_pmf(k, n1, n2, m1, theta) for k in chunk])
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(calculate_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    
    # Combine results maintaining order
    return np.concatenate(results)

def smart_pmf_calculation(support_array: np.ndarray, n1: int, n2: int,
                         m1: int, theta: float) -> np.ndarray:
    """Smart PMF calculation that chooses optimal strategy."""
    
    support_size = len(support_array)
    
    if support_size < 100:
        # Small arrays - use simple calculation
        from exactcis.core import log_nchg_pmf
        return np.array([log_nchg_pmf(k, n1, n2, m1, theta) for k in support_array])
    
    elif support_size < 1000:
        # Medium arrays - use vectorized calculation  
        from .vectorized_pmf import batch_pmf_calculation
        return batch_pmf_calculation(support_array, n1, n2, m1, theta)
    
    else:
        # Large arrays - consider parallel processing
        if should_use_parallel(support_size):
            return parallel_pmf_calculation(support_array, n1, n2, m1, theta)
        else:
            from .vectorized_pmf import batch_pmf_calculation
            return batch_pmf_calculation(support_array, n1, n2, m1, theta)
```

## Implementation Timeline

### Week 1: Critical Fixes
- **Day 1-2**: Implement Blaker PMF caching
- **Day 3-4**: Add vectorized PMF calculations  
- **Day 5**: Smart root-finding initialization
- **Weekend**: Testing and benchmarking

### Week 2: Algorithm Improvements  
- **Day 1-2**: Adaptive precision control
- **Day 3-4**: Memory-efficient support calculations
- **Day 5**: Integration and testing
- **Weekend**: Performance validation

### Week 3: Advanced Features
- **Day 1-3**: Parallel processing framework
- **Day 4-5**: Final optimization and polishing
- **Weekend**: Comprehensive testing

## Success Metrics

### Quantitative Targets
- **Blaker method**: <30ms for very large tables (current: 270ms)
- **Overall**: 50% improvement in average execution time
- **Memory**: <20% increase in memory usage
- **Accuracy**: No degradation in numerical precision

### Validation Tests
1. **Performance regression suite**: Automated benchmarking
2. **Correctness validation**: Compare against reference implementations  
3. **Memory profiling**: Ensure no memory leaks
4. **Stress testing**: Large batch processing scenarios

## Risk Mitigation

### High-Risk Items
1. **Numerical precision**: Extensive validation required
2. **Cache memory usage**: Implement proper bounds and eviction
3. **Parallel processing overhead**: Conservative thresholds

### Mitigation Strategies
1. **Feature flags**: Enable/disable optimizations individually
2. **Fallback mechanisms**: Graceful degradation to original algorithms
3. **Comprehensive testing**: Unit, integration, and property-based tests
4. **Benchmarking**: Before/after performance comparison

This roadmap provides a systematic approach to addressing the identified performance bottlenecks while maintaining the reliability and accuracy of the ExactCIs package.