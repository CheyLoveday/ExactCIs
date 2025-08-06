# Detailed Performance Analysis: exact_ci_unconditional

Generated: 2025-08-05 21:50:23

## Summary

- Total execution time: 65.7371s
- Average time per case: 16.4343s
- Successful cases: 0

## Per-Case Analysis

### Case: very_large

**ERROR**: 'Stats' object has no attribute 'get_stats'

```
Traceback (most recent call last):
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 204, in analyze_method_performance
    case_results['profile_stats'] = self._extract_profile_data(stats)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 227, in _extract_profile_data
    stats_dict = stats.get_stats()
                 ^^^^^^^^^^^^^^^
AttributeError: 'Stats' object has no attribute 'get_stats'. Did you mean: 'sort_stats'?
```

### Case: large_balanced

**ERROR**: 'Stats' object has no attribute 'get_stats'

```
Traceback (most recent call last):
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 204, in analyze_method_performance
    case_results['profile_stats'] = self._extract_profile_data(stats)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 227, in _extract_profile_data
    stats_dict = stats.get_stats()
                 ^^^^^^^^^^^^^^^
AttributeError: 'Stats' object has no attribute 'get_stats'. Did you mean: 'sort_stats'?
```

### Case: large_imbalanced

**ERROR**: 'Stats' object has no attribute 'get_stats'

```
Traceback (most recent call last):
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 204, in analyze_method_performance
    case_results['profile_stats'] = self._extract_profile_data(stats)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 227, in _extract_profile_data
    stats_dict = stats.get_stats()
                 ^^^^^^^^^^^^^^^
AttributeError: 'Stats' object has no attribute 'get_stats'. Did you mean: 'sort_stats'?
```

### Case: medium_balanced

**ERROR**: 'Stats' object has no attribute 'get_stats'

```
Traceback (most recent call last):
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 204, in analyze_method_performance
    case_results['profile_stats'] = self._extract_profile_data(stats)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chey/Coding_Projects/Archive/ExactCIs/profiling/line_profiler.py", line 227, in _extract_profile_data
    stats_dict = stats.get_stats()
                 ^^^^^^^^^^^^^^^
AttributeError: 'Stats' object has no attribute 'get_stats'. Did you mean: 'sort_stats'?
```

