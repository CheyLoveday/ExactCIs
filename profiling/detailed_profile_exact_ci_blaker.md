# Detailed Performance Analysis: exact_ci_blaker

Generated: 2025-08-05 21:50:58

## Summary

- Total execution time: 2.2651s
- Average time per case: 0.5663s
- Successful cases: 4

## Per-Case Analysis

### Case: very_large

**Execution Time**: 2.1117s
**Result**: (0.7284162500123039, 1.364051481075957)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| blaker_p_value | 77 | 0.0427s | 0.000555s | 0.000878s |
| blaker_acceptability | 77 | 0.0419s | 0.000544s | 0.000868s |
| find_smallest_theta | 2 | 0.0402s | 0.020089s | 0.020117s |
| validate_counts | 1 | 0.0000s | 0.000003s | 0.000003s |
| support | 1 | 0.0000s | 0.000001s | 0.000001s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| blaker_acceptability | blaker.py | 77 | 0.0007s | 0.000009s |
| get_pmf | shared_cache.py | 77 | 0.0003s | 0.000004s |
| blaker_p_value | blaker.py | 77 | 0.0002s | 0.000003s |
| sum_probs_less_than_threshold | jit_functions.py | 77 | 0.0001s | 0.000002s |
| bisection_method_log_space | root_finding.py | 2 | 0.0001s | 0.000044s |
| find_smallest_theta | core.py | 2 | 0.0001s | 0.000042s |
| exact_ci_blaker | blaker.py | 1 | 0.0000s | 0.000048s |
| <lambda> | core.py | 69 | 0.0000s | 0.000000s |
| <lambda> | blaker.py | 38 | 0.0000s | 0.000001s |
| _find_better_bracket | blaker.py | 2 | 0.0000s | 0.000010s |
| <lambda> | blaker.py | 39 | 0.0000s | 0.000000s |
| find_root_log_functional | root_finding.py | 2 | 0.0000s | 0.000007s |
| find_root_log_impl_functional | root_finding.py | 2 | 0.0000s | 0.000004s |
| find_root_log | core.py | 2 | 0.0000s | 0.000004s |
| get_shared_cache | shared_cache.py | 77 | 0.0000s | 0.000000s |

### Case: large_balanced

**Execution Time**: 0.0535s
**Result**: (0.6107380458302547, 1.6292938963115875)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| blaker_p_value | 77 | 0.0290s | 0.000376s | 0.001112s |
| blaker_acceptability | 77 | 0.0284s | 0.000369s | 0.001092s |
| find_smallest_theta | 2 | 0.0273s | 0.013671s | 0.015575s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |
| support | 1 | 0.0000s | 0.000001s | 0.000001s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| blaker_acceptability | blaker.py | 77 | 0.0004s | 0.000006s |
| get_pmf | shared_cache.py | 77 | 0.0003s | 0.000004s |
| blaker_p_value | blaker.py | 77 | 0.0002s | 0.000002s |
| sum_probs_less_than_threshold | jit_functions.py | 77 | 0.0001s | 0.000001s |
| bisection_method_log_space | root_finding.py | 2 | 0.0001s | 0.000037s |
| find_smallest_theta | core.py | 2 | 0.0001s | 0.000036s |
| exact_ci_blaker | blaker.py | 1 | 0.0000s | 0.000044s |
| <lambda> | core.py | 69 | 0.0000s | 0.000000s |
| _find_better_bracket | blaker.py | 2 | 0.0000s | 0.000010s |
| <lambda> | blaker.py | 39 | 0.0000s | 0.000000s |
| <lambda> | blaker.py | 38 | 0.0000s | 0.000000s |
| find_root_log_functional | root_finding.py | 2 | 0.0000s | 0.000006s |
| find_root_log | core.py | 2 | 0.0000s | 0.000004s |
| find_root_log_impl_functional | root_finding.py | 2 | 0.0000s | 0.000004s |
| get_shared_cache | shared_cache.py | 77 | 0.0000s | 0.000000s |

### Case: large_imbalanced

**Execution Time**: 0.0675s
**Result**: (1.944222209522358e-05, 0.7922007800122927)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| blaker_p_value | 104 | 0.0371s | 0.000356s | 0.000713s |
| blaker_acceptability | 104 | 0.0364s | 0.000350s | 0.000702s |
| find_smallest_theta | 2 | 0.0359s | 0.017942s | 0.022981s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |
| support | 1 | 0.0000s | 0.000000s | 0.000000s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| blaker_acceptability | blaker.py | 104 | 0.0006s | 0.000005s |
| get_pmf | shared_cache.py | 104 | 0.0004s | 0.000004s |
| blaker_p_value | blaker.py | 104 | 0.0002s | 0.000002s |
| sum_probs_less_than_threshold | jit_functions.py | 104 | 0.0001s | 0.000001s |
| find_smallest_theta | core.py | 2 | 0.0001s | 0.000039s |
| bisection_method_log_space | root_finding.py | 2 | 0.0001s | 0.000038s |
| exact_ci_blaker | blaker.py | 1 | 0.0000s | 0.000043s |
| <lambda> | blaker.py | 65 | 0.0000s | 0.000000s |
| <lambda> | core.py | 69 | 0.0000s | 0.000000s |
| find_plateau_edge | core.py | 1 | 0.0000s | 0.000023s |
| _find_better_bracket | blaker.py | 2 | 0.0000s | 0.000011s |
| <lambda> | blaker.py | 39 | 0.0000s | 0.000000s |
| find_root_log_functional | root_finding.py | 2 | 0.0000s | 0.000005s |
| get_shared_cache | shared_cache.py | 104 | 0.0000s | 0.000000s |
| find_root_log_impl_functional | root_finding.py | 2 | 0.0000s | 0.000004s |

### Case: medium_balanced

**Execution Time**: 0.0324s
**Result**: (0.33197879326473256, 3.0226844678018256)

**Function Call Analysis**:

| Function | Calls | Total Time | Avg Time | Max Time |
|----------|-------|------------|----------|----------|
| blaker_p_value | 77 | 0.0180s | 0.000234s | 0.000514s |
| blaker_acceptability | 77 | 0.0177s | 0.000230s | 0.000504s |
| find_smallest_theta | 2 | 0.0172s | 0.008603s | 0.009085s |
| validate_counts | 1 | 0.0000s | 0.000002s | 0.000002s |
| support | 1 | 0.0000s | 0.000001s | 0.000001s |

**Top Time-Consuming Functions (cProfile)**:

| Function | File | Calls | Total Time | Time/Call |
|----------|------|-------|------------|----------|
| get_pmf | shared_cache.py | 77 | 0.0003s | 0.000003s |
| blaker_acceptability | blaker.py | 77 | 0.0002s | 0.000003s |
| blaker_p_value | blaker.py | 77 | 0.0001s | 0.000002s |
| sum_probs_less_than_threshold | jit_functions.py | 77 | 0.0001s | 0.000001s |
| find_smallest_theta | core.py | 2 | 0.0001s | 0.000041s |
| bisection_method_log_space | root_finding.py | 2 | 0.0001s | 0.000035s |
| exact_ci_blaker | blaker.py | 1 | 0.0000s | 0.000041s |
| <lambda> | core.py | 69 | 0.0000s | 0.000000s |
| <lambda> | blaker.py | 38 | 0.0000s | 0.000000s |
| _find_better_bracket | blaker.py | 2 | 0.0000s | 0.000009s |
| <lambda> | blaker.py | 39 | 0.0000s | 0.000000s |
| find_root_log_functional | root_finding.py | 2 | 0.0000s | 0.000005s |
| find_root_log | core.py | 2 | 0.0000s | 0.000004s |
| find_root_log_impl_functional | root_finding.py | 2 | 0.0000s | 0.000004s |
| get_shared_cache | shared_cache.py | 77 | 0.0000s | 0.000000s |

