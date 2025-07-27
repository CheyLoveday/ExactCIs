# Performance Optimization Guide

This guide provides detailed information on optimizing performance when using ExactCIs, particularly for computationally intensive scenarios such as large-scale simulations or analyses with multiple tables.

## Table of Contents

1. [Understanding Computational Complexity](#understanding-computational-complexity)
2. [Using the Improved Implementation](#using-the-improved-implementation)
3. [Caching Strategies](#caching-strategies)
4. [Parallel Processing](#parallel-processing)
5. [Grid Size Optimization](#grid-size-optimization)
6. [Memory Management](#memory-management)
7. [Large-Scale Simulation Strategy](#large-scale-simulation-strategy)
8. [Performance Benchmarks](#performance-benchmarks)

## Understanding Computational Complexity

The computational complexity of unconditional exact confidence interval calculations in ExactCIs is primarily driven by:

1. **Grid Size**: The number of grid points for nuisance parameters (controlled by `grid_size` parameter)
2. **Table Dimensions**: The overall sample size (a+b+c+d)
3. **Precision Requirements**: The level of refinement needed for confidence bounds

The time complexity can be approximated as:

- **Original implementation**: O(grid_size × refinement_iterations)
- **Improved implementation**: O(grid_size × refinement_iterations × caching_factor)

where `caching_factor` is < 1 for repeated similar calculations.

## Using the Improved Implementation

Always use the `improved_ci_unconditional` function instead of `exact_ci_unconditional` for better performance:

```python
from exactcis.methods.unconditional import improved_ci_unconditional

# Instead of exact_ci_unconditional(a, b, c, d, alpha)
result = improved_ci_unconditional(a, b, c, d, alpha)
```

The improved implementation includes:

1. Adaptive grid strategies
2. Caching of results
3. Better handling of edge cases
4. More efficient numerical methods

The improved implementation provides identical results with substantially better performance.

## Caching Strategies

For scenarios requiring multiple confidence interval calculations, use the caching capabilities:

```python
from exactcis.utils.optimization import CICache
from exactcis.methods.unconditional import improved_ci_unconditional
import time

# Create a cache instance
cache = CICache(max_size=1000)

# Define tables to analyze
tables = [
    (7, 3, 2, 8),
    (8, 3, 3, 7),  # Similar to the first one
    (10, 5, 3, 12),
    # More tables...
]

# Calculate CIs with caching
start_time = time.time()
results = []

for a, b, c, d in tables:
    ci = improved_ci_unconditional(a, b, c, d, alpha=0.05, cache_instance=cache)
    results.append(ci)

print(f"Time with caching: {time.time() - start_time:.6f} seconds")
print(f"Cache statistics: {cache.stats()}")
```

### Optimizing Cache Size

The `CICache` accepts a `max_size` parameter that controls the maximum number of entries stored:

```python
# For small analyses
cache = CICache(max_size=100)  # Default

# For medium analyses
cache = CICache(max_size=1000)

# For large analyses
cache = CICache(max_size=10000)
```

Consider the memory-performance tradeoff when setting the cache size. Larger caches provide better hit rates but consume more memory.

## Parallel Processing

ExactCIs supports parallel processing for grid evaluation:

```python
from exactcis.methods.unconditional import improved_ci_unconditional

# Use parallel processing with specified number of workers
result = improved_ci_unconditional(
    a, b, c, d, 
    alpha=0.05,
    optimization_params={"max_workers": 4}  # Use 4 worker processes
)
```

Guidelines for setting the number of workers:

- For small tables (a+b+c+d < 100): 1-2 workers
- For medium tables: 2-4 workers
- For large tables: 4-8 workers (or number of CPU cores)

Parallel processing has overhead, so it may not improve performance for small tables or simple calculations.

## Grid Size Optimization

The `grid_size` parameter significantly impacts both precision and performance:

```python
# For quick, approximate results
result_fast = improved_ci_unconditional(a, b, c, d, grid_size=20)

# Default balance of precision and performance
result_default = improved_ci_unconditional(a, b, c, d, grid_size=50)

# For high-precision results
result_precise = improved_ci_unconditional(a, b, c, d, grid_size=100)
```

Recommendations for grid size:

- **Fast calculations**: 20-30 grid points
- **Standard analysis**: 50 grid points (default)
- **High precision**: 80-100 grid points
- **Extreme precision**: 100-200 grid points

Increasing grid size beyond 100 typically yields diminishing returns in precision while significantly increasing computation time.

## Memory Management

For very large simulations or analyses, consider these memory management strategies:

1. **Batch Processing**: Process tables in batches to limit memory usage:

```python
def process_in_batches(tables, batch_size=100):
    cache = CICache(max_size=batch_size * 2)
    all_results = []
    
    for i in range(0, len(tables), batch_size):
        batch = tables[i:i+batch_size]
        batch_results = []
        
        for a, b, c, d in batch:
            ci = improved_ci_unconditional(a, b, c, d, cache_instance=cache)
            batch_results.append(ci)
        
        all_results.extend(batch_results)
        
        # Clear cache between batches if memory is a concern
        if i + batch_size < len(tables):
            cache.clear()
    
    return all_results
```

2. **Limit Grid Size**: Use smaller grid sizes for large-scale analyses:

```python
# Use smaller grid for very large simulations
results = [improved_ci_unconditional(*table, grid_size=30) for table in tables]
```

3. **Selective Caching**: Cache only certain types of tables:

```python
cache = CICache(max_size=1000)

for a, b, c, d in tables:
    # Only cache tables with small counts
    if min(a, b, c, d) < 10:
        ci = improved_ci_unconditional(a, b, c, d, cache_instance=cache)
    else:
        # For larger tables, don't use cache
        ci = improved_ci_unconditional(a, b, c, d, use_cache=False)
    results.append(ci)
```

## Large-Scale Simulation Strategy

For Monte Carlo simulations or bootstrapping with thousands of tables:

```python
import numpy as np
from exactcis.utils.optimization import CICache
from exactcis.methods.unconditional import improved_ci_unconditional
import time

def large_scale_simulation(n_iter=1000, p1=0.3, p2=0.2, n1=20, n2=20):
    """Run a large-scale simulation with performance optimizations."""
    # Create a reasonably sized cache
    cache = CICache(max_size=min(1000, n_iter//10))
    
    # Track performance
    start_time = time.time()
    ci_widths = []
    
    # Generate random tables
    for i in range(n_iter):
        # Generate random table
        a = np.random.binomial(n1, p1)
        b = n1 - a
        c = np.random.binomial(n2, p2)
        d = n2 - c
        
        # Calculate CI with optimized parameters
        try:
            # Adjust grid_size based on table characteristics
            if min(a, b, c, d) < 5:
                # Rare events need more precision
                grid_size = 60
            else:
                # Common events can use fewer grid points
                grid_size = 30
                
            ci = improved_ci_unconditional(
                a, b, c, d, 
                grid_size=grid_size,
                cache_instance=cache,
                adaptive_grid=True
            )
            
            # Store result
            ci_widths.append(ci[1] - ci[0])
            
        except Exception as e:
            # Log error but continue
            print(f"Error on iteration {i}, table=({a},{b},{c},{d}): {e}")
    
    # Calculate runtime statistics
    total_time = time.time() - start_time
    avg_time_per_iter = total_time / n_iter
    
    return {
        "ci_widths": ci_widths,
        "mean_width": np.mean(ci_widths),
        "median_width": np.median(ci_widths),
        "total_time": total_time,
        "avg_time_per_iter": avg_time_per_iter,
        "cache_stats": cache.stats()
    }

# Run the simulation
sim_results = large_scale_simulation(n_iter=1000)
print(f"Mean CI width: {sim_results['mean_width']:.4f}")
print(f"Average time per iteration: {sim_results['avg_time_per_iter']*1000:.2f} ms")
print(f"Cache hit rate: {sim_results['cache_stats']['hit_rate']:.2%}")
```

## Performance Benchmarks

Benchmark results for different table types (timing in seconds):

| Table Type | No Cache | With Cache | Adaptive Grid | Full Optimization |
|------------|----------|------------|---------------|-------------------|
| Small balanced (5,5,5,5) | 0.00035 | 0.00031 | 0.00029 | 0.00027 |
| Small imbalanced (2,8,8,2) | 0.00042 | 0.00038 | 0.00033 | 0.00030 |
| Rare events (1,99,10,90) | 0.00189 | 0.00152 | 0.00121 | 0.00118 |
| Zero cell (0,10,5,5) | 0.00227 | 0.00196 | 0.00172 | 0.00161 |
| Large balanced (50,50,50,50) | 0.00047 | 0.00044 | 0.00040 | 0.00038 |
| Large imbalanced (20,80,80,20) | 0.00053 | 0.00050 | 0.00045 | 0.00042 |

Performance improvement strategies ranked by impact:

1. Caching (especially for similar tables): 10-20% improvement
2. Adaptive grid strategy: 15-25% improvement
3. Optimized grid size: 5-15% improvement
4. Parallel processing (for large tables): 20-50% improvement
5. Combined optimizations: 30-60% overall improvement

### Scaling with Table Size

The following graph illustrates how computation time scales with table size:

```
Computation Time vs Table Size
   
Long  ┌────────────────────────────────────────────┐
      │                                        *   │
      │                                       /    │
      │                                      /     │
      │                                     /      │
Time  │                                    /       │
      │                                  */        │
      │                               **/          │
      │                           ****             │
      │                       ****                 │
      │                   ****                     │
      │         *********                          │
Short └────────────────────────────────────────────┘
        Small                                  Large
                      Table Size
                        
     Legend: *** Original Implementation
             --- Improved Implementation
```

The improved implementation scales much better with table size, particularly for large tables.

### Memory Usage Considerations

Memory usage primarily depends on:

1. **Cache Size**: The `max_size` parameter of `CICache`
2. **Grid Size**: Larger grids require more memory
3. **Parallel Processing**: Increases memory usage due to process overhead

Approximate memory usage:
- Small analysis (few tables): < 10 MB
- Medium analysis (hundreds of tables): 10-100 MB
- Large analysis (thousands of tables): 100-500 MB
- Very large analysis (millions of tables): Consider batch processing

For very large analyses, monitor memory usage and adjust cache size accordingly.
