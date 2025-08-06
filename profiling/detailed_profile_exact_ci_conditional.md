# Detailed Performance Analysis: exact_ci_conditional

Generated: 2025-08-05 21:50:58

## Summary

- Total execution time: 0.0832s
- Average time per case: 0.0208s
- Successful cases: 4

## Per-Case Analysis

### Case: very_large

**Execution Time**: 0.0292s
**Result**: (0.7239973188967688, 1.3449540773270283)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| fisher_upper_bound | 1 | 0.0038s | 0.003768s | 0.003768s |
| fisher_lower_bound | 1 | 0.0037s | 0.003747s | 0.003747s |
| validate_bounds | 1 | 0.0000s | 0.000003s | 0.000003s |
| validate_counts | 1 | 0.0000s | 0.000003s | 0.000003s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| get_sf | shared_cache.py | 23 | 0.0001s | 0.000003s |
| get_cdf | shared_cache.py | 20 | 0.0001s | 0.000003s |
| exact_ci_conditional | conditional.py | 1 | 0.0000s | 0.000030s |
| wrapper | shared_cache.py | 23 | 0.0000s | 0.000001s |
| wrapper | shared_cache.py | 20 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 23 | 0.0000s | 0.000000s |
| p_value_func | conditional.py | 20 | 0.0000s | 0.000001s |
| fisher_lower_bound | conditional.py | 1 | 0.0000s | 0.000007s |
| _find_better_bracket | conditional.py | 2 | 0.0000s | 0.000003s |
| fisher_upper_bound | conditional.py | 1 | 0.0000s | 0.000004s |
| get_shared_cache | shared_cache.py | 43 | 0.0000s | 0.000000s |
| validate_bounds | conditional.py | 1 | 0.0000s | 0.000003s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |

### Case: large_balanced

**Execution Time**: 0.0225s
**Result**: (0.5980060266253221, 1.5713690565196878)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| fisher_upper_bound | 1 | 0.0039s | 0.003890s | 0.003890s |
| fisher_lower_bound | 1 | 0.0035s | 0.003520s | 0.003520s |
| validate_bounds | 1 | 0.0000s | 0.000003s | 0.000003s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| get_sf | shared_cache.py | 22 | 0.0001s | 0.000003s |
| get_cdf | shared_cache.py | 21 | 0.0001s | 0.000003s |
| exact_ci_conditional | conditional.py | 1 | 0.0000s | 0.000029s |
| wrapper | shared_cache.py | 21 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 21 | 0.0000s | 0.000001s |
| wrapper | shared_cache.py | 22 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 22 | 0.0000s | 0.000000s |
| _find_better_bracket | conditional.py | 2 | 0.0000s | 0.000003s |
| fisher_lower_bound | conditional.py | 1 | 0.0000s | 0.000006s |
| fisher_upper_bound | conditional.py | 1 | 0.0000s | 0.000005s |
| get_shared_cache | shared_cache.py | 43 | 0.0000s | 0.000000s |
| validate_bounds | conditional.py | 1 | 0.0000s | 0.000003s |
| validate_counts | core.py | 1 | 0.0000s | 0.000001s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |

### Case: large_imbalanced

**Execution Time**: 0.0190s
**Result**: (0.26891383455892315, 1.0)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| fisher_upper_bound | 1 | 0.0043s | 0.004272s | 0.004272s |
| fisher_lower_bound | 1 | 0.0042s | 0.004230s | 0.004230s |
| validate_bounds | 1 | 0.0000s | 0.000003s | 0.000003s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| get_cdf | shared_cache.py | 22 | 0.0001s | 0.000003s |
| get_sf | shared_cache.py | 20 | 0.0001s | 0.000003s |
| exact_ci_conditional | conditional.py | 1 | 0.0000s | 0.000029s |
| p_value_func | conditional.py | 22 | 0.0000s | 0.000001s |
| wrapper | shared_cache.py | 22 | 0.0000s | 0.000001s |
| wrapper | shared_cache.py | 20 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 20 | 0.0000s | 0.000001s |
| fisher_lower_bound | conditional.py | 1 | 0.0000s | 0.000007s |
| _find_better_bracket | conditional.py | 2 | 0.0000s | 0.000004s |
| fisher_upper_bound | conditional.py | 1 | 0.0000s | 0.000006s |
| get_shared_cache | shared_cache.py | 42 | 0.0000s | 0.000000s |
| validate_bounds | conditional.py | 1 | 0.0000s | 0.000004s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |

### Case: medium_balanced

**Execution Time**: 0.0126s
**Result**: (0.29420789105730777, 2.488835148652099)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| fisher_upper_bound | 1 | 0.0027s | 0.002710s | 0.002710s |
| fisher_lower_bound | 1 | 0.0026s | 0.002583s | 0.002583s |
| validate_bounds | 1 | 0.0000s | 0.000003s | 0.000003s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| get_cdf | shared_cache.py | 16 | 0.0000s | 0.000003s |
| get_sf | shared_cache.py | 16 | 0.0000s | 0.000003s |
| exact_ci_conditional | conditional.py | 1 | 0.0000s | 0.000026s |
| wrapper | shared_cache.py | 16 | 0.0000s | 0.000001s |
| wrapper | shared_cache.py | 16 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 16 | 0.0000s | 0.000001s |
| p_value_func | conditional.py | 16 | 0.0000s | 0.000000s |
| fisher_lower_bound | conditional.py | 1 | 0.0000s | 0.000006s |
| _find_better_bracket | conditional.py | 2 | 0.0000s | 0.000002s |
| fisher_upper_bound | conditional.py | 1 | 0.0000s | 0.000004s |
| get_shared_cache | shared_cache.py | 32 | 0.0000s | 0.000000s |
| validate_bounds | conditional.py | 1 | 0.0000s | 0.000003s |
| validate_counts | core.py | 1 | 0.0000s | 0.000001s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |

