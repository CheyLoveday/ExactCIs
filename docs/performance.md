# ExactCIs Performance Guide

This document provides information on the performance characteristics of the ExactCIs package, profiling techniques, and optimization strategies.

## Performance Characteristics

The computational performance of the confidence interval methods varies significantly:

| Method | Performance | Typical Runtime | Memory Usage |
|--------|-------------|----------------|--------------|
| **conditional** (Fisher) | Fast | < 0.1s | Low |
| **midp** (Mid-P adjusted) | Fast | < 0.1s | Low |
| **blaker** (Blaker's exact) | Fast | < 0.1s | Low |
| **unconditional** (Barnard's) | Slow for large tables | 0.1s - 30s+ | Moderate to High |
| **wald_haldane** (Haldane-Anscombe) | Very fast | < 0.01s | Very Low |

## Timeout Functionality

The package includes timeout functionality for the computationally intensive `exact_ci_unconditional` method to prevent long-running calculations from hanging indefinitely.

### Using Timeouts

```python
from exactcis.methods import exact_ci_unconditional

# Set a 30-second timeout
result = exact_ci_unconditional(a=12, b=5, c=8, d=10, alpha=0.05, timeout=30)

# If the calculation exceeds the timeout, the function will handle it gracefully
```

Timeout behavior:
- If the calculation completes within the timeout period, it returns the confidence interval as normal
- If the timeout is reached during calculation, the function will return `None` or a partial result depending on the context
- Core root-finding functions may also terminate early when a timeout is detected

### When to Use Timeouts

Timeouts are particularly useful in the following scenarios:

1. **Large or imbalanced tables**: Tables with large counts or extreme imbalances can require extensive grid searches that may run for minutes or hours
2. **Performance profiling**: Setting timeouts allows you to identify problematic calculations for targeted optimization
3. **Production systems**: Preventing indefinite hangs in automated systems is critical for reliability

## Profiling Tools

The ExactCIs package includes several profiling tools to help identify and optimize slow calculations:

### profile_with_timeout.py

This script runs various confidence interval methods with timeout protection and records which calculations time out:

```bash
uv run python profile_with_timeout.py --timeout 30 --num-cases 40 --max-count 100
```

This will:
- Generate test cases with varying table characteristics
- Run all methods with a 30-second timeout
- Report success rates and average execution times
- Identify patterns in timeout cases

### optimize_unconditional.py

This script helps optimize the parameters for the unconditional method:

```bash
uv run python optimize_unconditional.py --timeout 30
```

The script tests different grid sizes and the `refine` parameter to identify optimal settings for various table types.

## Optimization Strategies

### For Unconditional Method (Barnard's)

1. **Adjust grid size**: A smaller grid size (e.g., 10-20) is often sufficient for most tables and runs much faster than the default of 50
2. **Disable refinement**: Setting `refine=False` can improve performance for many cases with minimal impact on accuracy
3. **Use NumPy acceleration**: Install with the NumPy extras (`exactcis[numpy]`) for vectorized calculations
4. **Adaptive parameters**: Consider using smaller grid sizes for larger tables:

```python
def adaptive_unconditional_ci(a, b, c, d, alpha=0.05, timeout=60):
    # Determine table size and characteristics
    table_size = a + b + c + d
    
    # Adjust grid size based on table size
    if table_size > 100:
        grid_size = 10
    elif table_size > 50:
        grid_size = 20
    else:
        grid_size = 30
    
    return exact_ci_unconditional(a, b, c, d, alpha, grid_size=grid_size, timeout=timeout)
```

### General Performance Tips

1. **Use appropriate methods**: For large tables, the Wald method is much faster and often provides reasonable approximations
2. **Cache results**: If computing CIs for the same table multiple times, cache the results
3. **Consider parallel processing**: For batch processing multiple tables, use parallel execution with appropriate error handling
4. **Monitor memory usage**: Large grid sizes can consume significant memory for the unconditional method

## Benchmarking

Our benchmarks show the following typical performance characteristics for a moderate table (a=12, b=5, c=8, d=10):

| Method | Average Runtime | Success Rate |
|--------|----------------|--------------|
| conditional | 0.0001s | 100% |
| midp | 0.0002s | 100% |
| blaker | 0.0016s | 100% |
| unconditional (grid_size=20) | 0.65s | 100% |
| unconditional (grid_size=50) | 2.34s | 100% |
| wald_haldane | <0.0001s | 100% |

For larger or more imbalanced tables, the performance gap widens significantly, with the unconditional method potentially taking 30+ seconds or timing out.

## Conclusion

The timeout functionality and profiling tools provided with the ExactCIs package enable you to:

1. Prevent indefinite hangs in production systems
2. Identify which calculations are problematic
3. Optimize parameters for better performance
4. Balance computation time against precision requirements

By using these tools and following the optimization strategies outlined in this guide, you can significantly improve the performance of exact confidence interval calculations for your specific use cases.
