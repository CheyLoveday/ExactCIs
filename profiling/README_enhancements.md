# ExactCIs Profiling Enhancements

This document provides instructions for using the enhanced profiling tools for the ExactCIs package.

## Overview

The `profiling_enhancements.py` script enhances the existing profiling tools with additional command-line options and functionality to fully support the profiling plan. It provides a unified interface for benchmarking, profiling, and line-by-line profiling of ExactCIs methods.

## Installation

No additional installation is required. The script uses the existing profiling tools in the repository.

## Usage

### Benchmarking

Run comprehensive benchmarks across all methods:

```bash
python profiling/profiling_enhancements.py benchmark
```

Compare current results with a baseline:

```bash
python profiling/profiling_enhancements.py benchmark --compare-baseline
```

Save current results as a baseline for future comparisons:

```bash
python profiling/profiling_enhancements.py benchmark --save-baseline
```

### Performance Profiling

Profile all methods with all scenarios:

```bash
python profiling/profiling_enhancements.py profile
```

Profile a specific method:

```bash
python profiling/profiling_enhancements.py profile --method blaker
```

Profile a specific scenario:

```bash
python profiling/profiling_enhancements.py profile --scenario large
```

Profile a specific method with a specific scenario:

```bash
python profiling/profiling_enhancements.py profile --method blaker --scenario large
```

### Line-by-Line Profiling

Profile a specific method with line-by-line profiling:

```bash
python profiling/profiling_enhancements.py line-profile --method blaker
```

Profile a specific function within a method:

```bash
python profiling/profiling_enhancements.py line-profile --method blaker --function exact_ci_blaker
```

Profile with a specific test case:

```bash
python profiling/profiling_enhancements.py line-profile --method blaker --test-case "250,250,250,250"
```

## Standard Test Scenarios

The script includes the following standard test scenarios:

1. **Small Tables** (N < 100)
   - Balanced: (10, 10, 10, 10)
   - Imbalanced: (2, 18, 3, 17)
   - Zero cells: (0, 20, 5, 15)

2. **Medium Tables** (100 ≤ N < 500)
   - Balanced: (50, 50, 50, 50)
   - Imbalanced: (10, 90, 30, 70)
   - Rare events: (2, 98, 5, 95)

3. **Large Tables** (N ≥ 500)
   - Balanced: (250, 250, 250, 250)
   - Imbalanced: (50, 450, 100, 400)
   - Very rare events: (5, 495, 10, 490)

4. **Edge Cases**
   - Perfect separation: (20, 0, 0, 20)
   - Zero marginals: (0, 0, 10, 10)
   - Large imbalance: (1, 999, 2, 998)

## Output Files

The script generates the following output files:

### Benchmarking

- `profiling/benchmark_results.json`: Raw benchmark results
- `profiling/performance_report.txt`: Human-readable benchmark report
- `profiling/baseline_performance.json`: Baseline performance data (if saved)

### Performance Profiling

- `profiling/enhanced_timing_results.json`: Raw timing results
- `profiling/enhanced_profiling_report.md`: Human-readable profiling report

### Line-by-Line Profiling

- `profiling/detailed_profile_*.md`: Detailed line-by-line profiling reports

## Profiling Workflow

For a comprehensive profiling of the ExactCIs package, follow this workflow:

1. **Initial Benchmarking**
   ```bash
   python profiling/profiling_enhancements.py benchmark --save-baseline
   ```

2. **Identify Bottlenecks**
   ```bash
   python profiling/profiling_enhancements.py profile
   ```

3. **Detailed Line Profiling**
   ```bash
   python profiling/profiling_enhancements.py line-profile --method <slowest_method>
   ```

4. **Implement Optimizations**
   - Focus on the most significant bottlenecks first
   - Implement optimizations based on profiling results

5. **Validate Improvements**
   ```bash
   python profiling/profiling_enhancements.py benchmark --compare-baseline
   ```

## Extending the Script

To add new methods or scenarios:

1. Add new methods to the `METHODS` dictionary in the script
2. Add new scenarios to the `SCENARIOS` dictionary in the script

## Troubleshooting

If you encounter any issues:

1. Make sure you're running the script from the repository root
2. Check that all required dependencies are installed
3. Verify that the ExactCIs package is properly installed or in the Python path

## Further Information

For more detailed information on profiling strategies and guidelines, refer to the `.junie/guidelines.md` file in the repository.

For a comprehensive plan for profiling all methods in the ExactCIs codebase, refer to the `.junie/profiling_plan.md` file.