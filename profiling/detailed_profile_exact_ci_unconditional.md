# Detailed Performance Analysis: exact_ci_unconditional

Generated: 2025-08-05 21:52:05

## Summary

- Total execution time: 67.4103s
- Average time per case: 16.8526s
- Successful cases: 4

## Per-Case Analysis

### Case: very_large

**Execution Time**: 34.4518s
**Result**: (0.001, 1000.0)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| validate_counts | 1 | 0.0000s | 0.000003s | 0.000003s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| exact_ci_unconditional | unconditional.py | 1 | 0.0000s | 0.000010s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| get_exact | optimization.py | 1 | 0.0000s | 0.000001s |
| validate_alpha | validators.py | 1 | 0.0000s | 0.000000s |
| get_global_cache | optimization.py | 1 | 0.0000s | 0.000000s |

### Case: large_balanced

**Execution Time**: 11.5688s
**Result**: (1.0000000000000002e-07, 1000000)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| validate_counts | 1 | 0.0000s | 0.000003s | 0.000003s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| exact_ci_unconditional | unconditional.py | 1 | 0.0000s | 0.000017s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |
| get_exact | optimization.py | 1 | 0.0000s | 0.000001s |
| validate_alpha | validators.py | 1 | 0.0000s | 0.000000s |
| get_global_cache | optimization.py | 1 | 0.0000s | 0.000000s |

### Case: large_imbalanced

**Execution Time**: 20.8346s
**Result**: (1.0000000000000002e-07, 1000000)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| validate_counts | 1 | 0.0000s | 0.000015s | 0.000015s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| <genexpr> | core.py | 5 | 0.0000s | 0.000003s |
| exact_ci_unconditional | unconditional.py | 1 | 0.0000s | 0.000009s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| get_exact | optimization.py | 1 | 0.0000s | 0.000001s |
| validate_alpha | validators.py | 1 | 0.0000s | 0.000000s |
| get_global_cache | optimization.py | 1 | 0.0000s | 0.000000s |

### Case: medium_balanced

**Execution Time**: 0.5551s
**Result**: (1.0000000000000002e-07, 1000000)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| validate_counts | 1 | 0.0000s | 0.000005s | 0.000005s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| exact_ci_unconditional | unconditional.py | 1 | 0.0000s | 0.000008s |
| <genexpr> | core.py | 5 | 0.0000s | 0.000000s |
| validate_counts | core.py | 1 | 0.0000s | 0.000002s |
| get_exact | optimization.py | 1 | 0.0000s | 0.000001s |
| validate_alpha | validators.py | 1 | 0.0000s | 0.000000s |
| get_global_cache | optimization.py | 1 | 0.0000s | 0.000000s |

