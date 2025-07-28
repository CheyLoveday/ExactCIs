# ExactCIs Profiling Strategy

This directory contains a comprehensive profiling framework for analyzing the performance, memory usage, and scalability characteristics of the ExactCIs package.

## Overview

The profiling strategy implements a multi-level approach to identify and address performance bottlenecks:

1. **Function-Level Profiling**: Identifies time-consuming functions using cProfile
2. **Line-Level Profiling**: Pinpoints exact bottlenecks within functions using line_profiler  
3. **Memory Analysis**: Understanding memory usage patterns and efficiency
4. **Scalability Analysis**: Computational complexity and scaling behavior
5. **Unified Reporting**: Consolidated analysis with optimization recommendations

## Quick Start

### Basic Usage

```bash
# Quick analysis (5-10 minutes) - recommended first step
cd analysis/analysis_scripts/profiling
uv run python master_profiler.py --quick

# Full comprehensive analysis (30+ minutes)
uv run python master_profiler.py --full

# Targeted analysis for specific methods
uv run python master_profiler.py --targeted --methods blaker unconditional --focus performance memory
```

### Prerequisites

```bash
# Required packages (install if not available)
pip install line_profiler memory-profiler psutil matplotlib pandas numpy
```

## Profiling Scripts

### 1. Master Profiler (`master_profiler.py`)

**Purpose**: Coordinates all profiling analyses and generates unified reports.

**Usage**:
```bash
# Quick analysis (fast methods only, basic profiling)
uv run python master_profiler.py --quick

# Full analysis (all methods, all profiling types)  
uv run python master_profiler.py --full

# Targeted analysis
uv run python master_profiler.py --targeted --methods unconditional blaker --focus scalability
```

**Output**:
- `master_profiling_[type]_[timestamp].json` - Complete results
- `profiling_report_[timestamp].html` - Human-readable HTML report
- Individual profiling output files

### 2. Comprehensive Profiler (`comprehensive_profiler.py`)

**Purpose**: Function-level and line-level performance profiling.

**Key Features**:
- cProfile analysis across different table sizes
- Line-by-line profiling for hotspot identification  
- Test case categorization (small, medium, large, extreme, edge cases)
- Method comparison and bottleneck identification

**Usage**:
```bash
# Profile specific methods
uv run python comprehensive_profiler.py --methods blaker unconditional

# Include line profiling (requires line_profiler)
uv run python comprehensive_profiler.py --methods blaker --include-line-profiling

# Skip scalability analysis for faster results
uv run python comprehensive_profiler.py --no-scalability
```

**Key Outputs**:
- Function-level timing profiles for each method and table category
- Line-by-line profiling for computational hotspots
- Performance ranking and optimization recommendations

### 3. Memory Profiler (`memory_profiler.py`)

**Purpose**: Analyze memory usage patterns and efficiency.

**Key Features**:
- Memory scaling analysis with table size
- Grid method memory profiling (unconditional method)
- Method comparison for memory efficiency
- Peak memory usage and allocation patterns

**Usage**:
```bash
# Full memory analysis
uv run python memory_profiler.py --max-size 200

# Skip grid analysis for faster results  
uv run python memory_profiler.py --no-grid-analysis

# Memory comparison only
uv run python memory_profiler.py --no-grid-analysis --max-size 100
```

**Key Outputs**:
- Memory scaling curves for each method
- Grid size vs memory usage analysis
- Memory efficiency rankings and recommendations

### 4. Scalability Analyzer (`scalability_analyzer.py`)

**Purpose**: Computational complexity and scaling behavior analysis.

**Key Features**:
- Computational complexity curve fitting (linear, quadratic, exponential)
- Parameter sensitivity analysis (grid sizes, timeouts)
- Method scalability comparison
- Break-even point identification

**Usage**:
```bash
# Full scalability analysis
uv run python scalability_analyzer.py --max-size 300

# Skip parameter analysis for faster results
uv run python scalability_analyzer.py --no-parameters

# Specific methods only
uv run python scalability_analyzer.py --methods unconditional blaker --max-size 200
```

**Key Outputs**:
- Computational complexity classifications (O(n), O(n²), etc.)
- Scalability limits for each method
- Parameter optimization recommendations

## Understanding the Results

### Performance Metrics

**Execution Time**: Average time per calculation across different table sizes
- Fast: < 0.1 seconds
- Moderate: 0.1 - 1.0 seconds  
- Slow: 1.0 - 10.0 seconds
- Very Slow: > 10.0 seconds

**Memory Efficiency**: Memory usage per second of computation (MB/s)
- Lower values indicate better efficiency

**Scalability Limits**: Maximum table size before method becomes impractical
- Small tables: < 50 total count
- Medium tables: 50 - 200 total count
- Large tables: > 200 total count

### Method-Specific Insights

**Conditional (Fisher's Exact)**:
- Generally fast and reliable
- Linear scaling with table size
- Low memory usage

**Mid-P**:
- Similar performance to conditional
- Slightly higher computational cost
- Good scalability

**Blaker's Method**:
- Moderate computational cost
- P-value calculations can be bottleneck
- Scales well for most practical table sizes

**Unconditional (Barnard's)**:
- Most computationally intensive
- Grid search over nuisance parameters
- Scaling highly dependent on grid_size parameter
- Benefits significantly from parameter optimization

**Wald-Haldane**:
- Fastest method (asymptotic)
- Constant time regardless of table size
- Minimal memory usage

## Optimization Recommendations

### Immediate Actions (Quick Wins)
1. **Increase cache sizes**: Modify LRU cache parameters in core functions
2. **Reduce grid_size**: Use adaptive grid sizing for unconditional method
3. **Parameter tuning**: Optimize timeout and grid parameters
4. **Progress monitoring**: Add progress bars for long computations

### Short-term Optimizations
1. **Vectorization**: Use NumPy operations for array computations
2. **Algorithm improvements**: Optimize root-finding and grid search
3. **Memory efficiency**: Implement streaming for large tables
4. **Caching strategies**: Cache intermediate probability calculations

### Long-term Improvements  
1. **Parallel processing**: Implement grid computation parallelization
2. **Advanced algorithms**: Research state-of-the-art numerical methods
3. **C extensions**: Consider Cython/C for critical computational kernels
4. **ML parameter optimization**: Use ML models for optimal parameter selection

## Interpreting Profiling Output

### cProfile Output
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   100    0.500    0.005    1.000    0.010 exactcis/methods/blaker.py:42(blaker_p_value)
```
- `ncalls`: Number of function calls
- `tottime`: Total time excluding sub-calls  
- `cumtime`: Total time including sub-calls
- `percall`: Time per call

### Line Profiler Output
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
    42         1          500   500.0     50.0      result = expensive_calculation()
```
- `Hits`: Number of times line was executed
- `Time`: Total time spent on this line
- `% Time`: Percentage of total function time

### Memory Profiler Output
```
Line #    Mem usage    Increment  Occurrences   Line Contents
    42     25.2 MiB      0.5 MiB            1   array = np.zeros(large_size)
```
- `Mem usage`: Total memory usage at this point
- `Increment`: Memory increase from this line

## Troubleshooting

### Common Issues

**Line profiler not available**:
```bash
pip install line_profiler
```

**Memory profiler not available**:
```bash
pip install memory-profiler psutil
```

**Timeouts with unconditional method**:
- Reduce `--max-size` parameter
- Use `--no-parameters` to skip parameter sensitivity analysis
- Increase timeout values in script configuration

**Out of memory errors**:
- Reduce table sizes being tested
- Use `--no-grid-analysis` to skip memory-intensive grid analysis
- Close other applications to free system memory

### Performance Tips

1. **Start with quick analysis** to get overview before full analysis
2. **Use targeted analysis** to focus on specific problematic methods
3. **Run analysis on dedicated system** to avoid interference from other processes
4. **Monitor system resources** during analysis to prevent system overload

## Contributing

When adding new profiling capabilities:

1. Follow the existing module structure and naming conventions
2. Add comprehensive docstrings and type hints
3. Include error handling and timeout protection
4. Generate JSON output compatible with the master profiler
5. Update this README with new script documentation

## File Organization

```
profiling/
├── README.md                     # This documentation
├── master_profiler.py            # Main coordinator script
├── comprehensive_profiler.py     # Function/line-level profiling  
├── memory_profiler.py           # Memory usage analysis
├── scalability_analyzer.py      # Computational complexity analysis
├── profile_slow_functions.py    # Legacy profiler (existing)
├── profile_with_timeout.py      # Legacy timeout profiler (existing)
└── profiling_results/           # Output directory (auto-created)
    ├── *.json                   # Raw profiling data
    ├── *.html                   # Human-readable reports
    ├── *.prof                   # cProfile output files
    └── *.txt                    # Line profiler output files
```

## Example Workflow

1. **Initial Assessment**:
   ```bash
   uv run python master_profiler.py --quick
   ```

2. **Identify Problem Methods** (from quick analysis results)

3. **Detailed Analysis** of problematic methods:
   ```bash
   uv run python master_profiler.py --targeted --methods unconditional --focus performance scalability
   ```

4. **Line-level Investigation**:
   ```bash
   uv run python comprehensive_profiler.py --methods unconditional
   ```

5. **Memory Analysis** for large table scenarios:
   ```bash
   uv run python memory_profiler.py --max-size 150
   ```

6. **Implement Optimizations** based on recommendations

7. **Validation**:
   ```bash
   uv run python master_profiler.py --quick  # Compare with initial results
   ```

This profiling framework provides comprehensive insights into ExactCIs performance characteristics and actionable recommendations for optimization.