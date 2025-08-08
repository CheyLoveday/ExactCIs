# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactCIs is a focused Python package for computing confidence intervals for **odds ratios and relative risks** from 2×2 contingency tables. The package provides comprehensive, peer-reviewed methods for the two most important epidemiological effect measures.

**Production-Ready Methods:**

**Odds Ratio Methods (5 implemented):**
- Fisher's conditional (exact) - Small samples, guaranteed coverage
- Mid-P adjusted (exact) - Adaptive grid search with CI inversion
- Blaker's exact - Recently fixed (Aug 2025), optimal coverage
- Barnard's unconditional exact - Profile likelihood with inflation control
- Wald-Haldane (asymptotic) - Fast approximation for large samples

**Relative Risk Methods (7 implemented - FULLY TESTED Aug 2025):**
- Wald standard and Katz variants - Cross-validated with SciPy ≥1.11
- Wald correlated (for matched data) - Enhanced zero-cell handling
- Score/Tang (Miettinen-Nurminen) - Fixed bracket expansion algorithm
- Score with continuity correction - Fixed parameter ordering and root finding
- Score with strong continuity correction - Alternative correction variant
- U-statistic (nonparametric) - Duan et al. method with proper variance

**Current Focus (Aug 2025):**
- ✅ **All RR methods fixed and production-ready** (57 tests: 50 passed, 7 skipped)
- 🔄 **Performance profiling extension** to RR methods (see `profiling/RR_PROFILING_PLAN.md`)
- 📋 **Next Priority**: Cross-validation with R packages and SciPy benchmarking

## Development Commands

### Testing
- `uv run pytest` - Run fast tests (default, skips slow tests)
- `uv run pytest --run-slow` - Run all tests including slow ones
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest --cov=src/exactcis tests/` - Run tests with coverage
- `pytest tests/test_methods/test_blaker.py -v` - Run specific test file
- `pytest tests/test_methods/test_blaker.py::test_specific_function -v` - Run specific test
- `pytest tests/test_methods/test_relative_risk.py -v` - Test relative risk methods

### Code Quality
- `pre-commit run --all-files` - Run linting with pre-commit
- `make lint` - Alternative lint command
- `ruff check .` - Run ruff linter
- `black .` - Format code with black
- `mypy src/exactcis` - Type checking

### Development Setup
- `uv pip install -e ".[dev]"` - Install in development mode with dev dependencies
- `make dev-install` - Alternative development install
- `make clean` - Clean build artifacts

### Performance Profiling
- `python profiling/performance_profiler.py` - Benchmark all methods
- `python profiling/rr_performance_extension.py` - Benchmark RR methods specifically
- `python profiling/line_profiler.py` - Detailed line-by-line profiling

### Documentation
- `make docs` - Generate Sphinx documentation
- Documentation lives in `docs/source/` with RST files

### Building & Distribution
- `make dist` - Build source and wheel packages (uses `python -m build`)
- `twine check dist/*` - Check package before upload
- `make release` - Full release process (check + upload)

## Architecture Overview

The package follows a modular architecture with clear separation of concerns:

### Core Components
- `src/exactcis/core.py` - Core statistical functions, PMF calculations, root-finding algorithms
- `src/exactcis/__init__.py` - Main public API with `compute_all_cis()` orchestrator function

### Method Implementations (`src/exactcis/methods/`)
- `conditional.py` - Fisher's exact conditional method (noncentral hypergeometric CDF inversion)
- `midp.py` - Mid-P adjusted method (half-weight for observed table)
- `blaker.py` - Blaker's exact method (acceptability function inversion)
- `unconditional.py` - Barnard's unconditional exact method (grid/NumPy optimization over nuisance parameter)
- `wald.py` - Wald-Haldane approximation (normal approximation with Haldane correction)
- `clopper_pearson.py` - Clopper-Pearson method for binomial proportions
- `relative_risk.py` - **NEW (Aug 2025)** - Relative risk confidence intervals (Wald, Katz, Score/Tang, U-statistic methods)

### Utilities (`src/exactcis/utils/`)
- `stats.py` - Statistical utility functions, corrections
- `validators.py` - Input validation functions
- `optimization.py` - Optimization algorithms
- `parallel.py` - Parallel processing utilities
- `root_finding.py` - Root-finding algorithms
- `pmf_functions.py` - Probability mass function calculations
- `shared_cache.py` - Caching utilities

## Key Implementation Details

### Numerical Stability
- All PMF calculations done in log-space to prevent overflow/underflow
- Uses `logsumexp` pattern for stable probability normalization
- Extensive use of `@lru_cache` for expensive computations

### Root Finding Strategy
- Primary: `find_root_log()` for bisection in log-space
- Fallback: `find_plateau_edge()` for detecting flat p-value regions (enhanced Aug 2025)
- Combined approach in `find_smallest_theta()` for robust CI bound detection (relaxed tolerance Aug 2025)
- **Recent improvements**: Statistical tolerances (2% vs 1%) and enhanced boundary detection prevent degenerate solutions

### Performance Features
- Optional NumPy vectorization for unconditional method grid search
- Timeout protection via `timeout_checker` callbacks
- Caching at multiple levels (binomial coefficients, PMF weights, etc.)
- Progress reporting callbacks for long-running calculations

### Recent Algorithm Improvements (Aug 2025)

**Relative Risk Methods - Critical Fixes:**
1. **Score Method Root Finding**: Enhanced bracket expansion in `find_score_ci_bound()` 
   - Fixed infinite upper bounds issue: now returns finite CIs like `(1.18, 2.21)` instead of `(1.18, inf)`
   - Systematic grid search with plateau detection for robust convergence
2. **Parameter Order Standardization**: Fixed `ci_score_cc_rr` signature to match other methods
3. **Zero-Cell Detection**: Enhanced `ci_wald_correlated_rr` with pre-continuity-correction logic
4. **Test Expectation Alignment**: Updated tests to handle legitimate continuity correction effects

**Performance Optimizations:**
- **Enhanced Root Finding**: Relaxed statistical tolerances (2% vs 1%) prevent degenerate solutions
- **Adaptive Grid Search**: Intelligent theta grid generation centered around sample estimates  
- **Batch Processing**: Parallel processing support for computationally intensive methods
- **Shared Caching**: Multi-level caching for repeated calculations

## Testing Strategy

### Test Organization
- `tests/test_methods/` - Method-specific tests
- `tests/test_utils/` - Utility function tests
- Tests use pytest markers: `@pytest.mark.slow`, `@pytest.mark.unconditional`

### Test Data Patterns
- Small contingency tables: `(12, 5, 8, 10)` commonly used
- Edge cases: zero cells, extreme ratios
- Comparative tests against R's exact2x2 package
- Performance benchmarks in `profiling/` directory
- **Comprehensive validation**: `comprehensive_outlier_test.py` validates all methods across 10 diverse scenarios

## Common Development Patterns

### Adding New Methods
1. Create new file in `src/exactcis/methods/`
2. Implement function following signature: `exact_ci_<method>(a, b, c, d, alpha=0.05, **kwargs)`
3. Add to `methods/__init__.py` exports
4. Add to `compute_all_cis()` in main `__init__.py`
5. Create corresponding test file in `tests/test_methods/`

### Debugging Numerical Issues
- Check log-space calculations in `core.py`
- Verify support calculations with `support()` function
- Use `logger` for intermediate values (already configured)
- Test with edge cases like zero cells

### Performance Optimization
- Profile with `line_profiler` (already configured in dev deps)
- Use `profiling/` directory for benchmark scripts
- Consider caching for expensive repeated calculations
- Implement timeout checks for long-running operations

## Current Performance & Profiling Focus

### Performance Profiling Strategy
- **Primary Framework**: `profiling/performance_profiler.py` with comprehensive method benchmarking
- **RR Extension**: `profiling/RR_PROFILING_PLAN.md` outlines integration of RR methods into profiling infrastructure
- **Recent Validation**: All 57 RR tests passing (50 passed, 7 skipped) - methods are production-ready

### Key Profiling Priorities
1. **RR Method Performance**: Benchmark the 6 newly fixed RR methods against R equivalents
2. **Score Algorithm Efficiency**: Measure impact of bracket expansion fixes on convergence speed
3. **Zero-Cell Handling**: Profile delegation logic overhead in enhanced methods
4. **Cross-Method Comparison**: OR vs RR performance characteristics and scaling behavior

## Next Development Priorities

### 1. Performance Profiling Integration (Current Focus)
- **Extend profiling to RR methods**: Add all 6 RR methods to `profiling/performance_profiler.py`
- **Benchmark recent fixes**: Validate that score method fixes maintain performance
- **Cross-validation**: Compare RR results with R `ratesci` and SciPy ≥1.11 implementations
- **Scaling analysis**: Profile performance across table sizes and sparsity patterns

### 2. Enhanced Public API (Planned)
```python
# Unified interface for both OR and RR
from exactcis import or_ci, rr_ci

# Simple usage with smart defaults
result = rr_ci(a, b, c, d)  # Uses best method for data characteristics
result = or_ci(a, b, c, d)  # Auto-selects appropriate OR method

# Explicit method selection
result = rr_ci(a, b, c, d, method="score_cc")  # Score with continuity correction
result = or_ci(a, b, c, d, method="blaker")    # Blaker exact method
```

### 3. Cross-Package Validation
- **SciPy Integration**: Cross-validate with `scipy.stats.contingency.relative_risk()`
- **R Package Comparison**: Automated comparison with R `exact2x2`, `ratesci`, `epitools`
- **Literature Benchmarking**: Reproduce key examples from Agresti, Tang, Duan papers

## Strategic Development Roadmap

### Phase 1: Performance & Validation (Current - Aug 2025)
**Status: 🔄 In Progress**
- **RR Profiling Integration**: Extend performance benchmarking to all 6 RR methods
- **Cross-Validation**: Compare results with R `ratesci`, `exact2x2`, and SciPy ≥1.11
- **Literature Benchmarking**: Validate against published examples (Agresti, Tang, Duan papers)

### Phase 2: Enhanced Public API (Next Release) 
**Priority: High**
- **Unified Interface**: Implement `or_ci()` and `rr_ci()` functions with smart defaults
- **Method Auto-Selection**: Logic to choose optimal method based on sample size and data characteristics  
- **Comprehensive Documentation**: Method selection guidance and cross-reference examples

### Phase 3: Method Expansion (Future)
**Priority: Medium - based on user demand**

**Missing OR Methods:**
- Logit/Score method (Miettinen-Nurminen 1985) 
- Cornfield asymptotic approximation
- Mantel-Haenszel pooled OR

**Additional RR Methods:**  
- Exact unconditional RR (for very small samples)
- Bonett-Price hybrid Fieller interval
- Bootstrap BCa intervals
- Mantel-Haenszel stratified RR

**Selection Criteria**: Methods must appear in major software (R/SAS/Stata) and fill genuine coverage gaps

### Target Coverage Matrix
| Sample Size | Current Coverage | Planned Enhancement |
|-------------|------------------|---------------------|
| **Large (n≥100)** | ✅ Wald variants | + Cross-validation benchmarks |
| **Moderate (30≤n<100)** | ✅ Score/Tang methods | + Bonett-Price alternatives |
| **Small (n<30)** | ✅ Exact conditional/Mid-P | + Exact unconditional options |
| **Zero cells** | ✅ Continuity corrections | + Bootstrap fallbacks |
| **Matched pairs** | ✅ U-statistic, correlated Wald | + McNemar exact OR |
| **Stratified** | ⏳ Planned | + Mantel-Haenszel pooling |

## File Navigation Helpers

### Core Implementation Files
- **Method implementations**: `src/exactcis/methods/<method>.py`
  - `relative_risk.py:1-200` - All RR confidence interval methods
  - `conditional.py`, `midp.py`, `blaker.py`, `unconditional.py`, `wald.py` - OR methods
- **Core algorithms**: `src/exactcis/core.py` (line numbers for key functions):
  - PMF calculations: ~100-400
  - Root finding: ~400-700 (recently enhanced)
  - Support calculations: ~280-320
- **Utilities**: `src/exactcis/utils/`
  - `ci_search.py` - Adaptive grid search with inflation control
  - `root_finding.py` - Enhanced root-finding algorithms (bracket expansion fixes)
  - `stats.py` - Statistical utility functions and corrections

### API and Orchestration  
- **Main API**: `src/exactcis/__init__.py`
  - `compute_all_cis()` at line ~56 - OR methods orchestrator
  - `compute_all_rr_cis()` at line ~83 - RR methods orchestrator
- **CLI interface**: `src/exactcis/cli.py`

### Testing
- **Method-specific tests**: `tests/test_methods/test_<method>.py`
- **Integration tests**: `tests/test_integration.py`, `tests/test_comprehensive_integration.py`
- **RR-specific validation**: `tests/test_methods/test_relative_risk.py`

### Performance & Profiling
- **Main profiler**: `profiling/performance_profiler.py`
- **RR profiling**: `profiling/rr_performance_extension.py`
- **Profiling plan**: `profiling/RR_PROFILING_PLAN.md`