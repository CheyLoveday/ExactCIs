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
- Wald-Haldane (asymptotic) - Refactored with centralized infrastructure (Aug 2025)

**Relative Risk Methods (7 implemented):**
- Wald standard and Katz variants - Refactored with centralized infrastructure (Aug 2025)
- Wald correlated (for matched data) - Enhanced zero-cell handling
- Score/Tang (Miettinen-Nurminen) - Fixed bracket expansion algorithm
- Score with continuity correction - Fixed parameter ordering and root finding
- Score with strong continuity correction - Alternative correction variant
- U-statistic (nonparametric) - Duan et al. method with proper variance

**Phase 0 Refactoring Complete (Aug 2025):**
- ✅ **Centralized validation and corrections** - Single source of truth in `validation.py` and `continuity.py`
- ✅ **Golden parity testing** - Comprehensive fixtures prevent regressions
- ✅ **Core solver infrastructure** - Robust bracketing and root-finding in `solvers.py` and `inversion.py`
- ✅ **Mathematical operations centralized** - `mathops.py` and `estimates.py` consolidate computations
- ✅ **Wald methods refactored** - Both OR and RR use new infrastructure with dataclass safety

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
- `src/exactcis/core.py` - Core statistical functions, PMF calculations, legacy algorithms
- `src/exactcis/__init__.py` - Main public API with `compute_all_cis()` and `compute_all_rr_cis()` orchestrators

### Refactored Infrastructure (`src/exactcis/utils/`)
- `validation.py` - Centralized input validation (Phase 0 ✅)
- `continuity.py` - Unified continuity correction policies (Phase 0 ✅) 
- `solvers.py` - Robust root-finding and bracketing algorithms (Phase 1 ✅)
- `inversion.py` - Standardized CI inversion framework (Phase 1 ✅)
- `mathops.py` - Safe mathematical operations (Phase 2 ✅)
- `estimates.py` - Centralized point estimates and standard errors (Phase 2 ✅)
- `optimization.py` - Optimization algorithms for unconditional methods
- `parallel.py` - Parallel processing utilities
- `shared_cache.py` - Caching utilities

### Method Implementations (`src/exactcis/methods/`)
- `conditional.py` - Fisher's exact conditional method (noncentral hypergeometric CDF inversion)
- `midp.py` - Mid-P adjusted method (half-weight for observed table)
- `blaker.py` - Blaker's exact method (acceptability function inversion)
- `unconditional.py` - Barnard's unconditional exact method (grid/NumPy optimization)
- `wald.py` - **Refactored** Wald-Haldane approximation using centralized infrastructure
- `relative_risk.py` - **Refactored** RR methods using centralized solvers and estimates

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

### Refactoring Achievements (Aug 2025)

**Phase 0 - Validation & Corrections:**
- **Centralized validation**: All methods use single `validate_2x2_table()` from `validation.py`
- **Unified corrections**: Smart continuity correction policies in `continuity.py`
- **Golden parity**: Comprehensive fixtures prevent numerical regressions

**Phase 1 - Core Solvers:**
- **Robust bracketing**: Enhanced bracket expansion with plateau detection in `solvers.py`
- **Standardized inversion**: CI bounds finding unified in `inversion.py` 
- **Score method fixes**: Eliminated infinite upper bounds, improved convergence

**Phase 2 - Mathematical Infrastructure:**
- **Safe operations**: Math utilities centralized in `mathops.py` (zero-safe ratios, log operations)
- **Point estimates**: All OR/RR estimates and standard errors unified in `estimates.py`
- **Dataclass safety**: Structured `EstimationResult` and `CorrectionResult` types
- **Wald refactoring**: Both OR and RR Wald methods use centralized infrastructure

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

### Golden Parity Testing System

**Purpose**: Ensure refactors maintain numerical correctness while accommodating precision changes from algorithmic improvements.

#### **Tiered Warning System** 🚦
- **🟢 LOW (< 0.1%)**: Minor precision difference (likely acceptable refactoring artifact)
- **🟡 MEDIUM (< 1%)**: Moderate precision difference (may indicate algorithm change)  
- **🟠 HIGH (< 10%)**: Significant precision difference (requires investigation)
- **🔴 CRITICAL (≥ 10%)**: Major numerical difference (likely bug or substantial algorithm change)

#### **Configuration Modes**
- **Standard mode**: Fail on differences beyond tight tolerances (1e-9 relative, 1e-12 absolute)
- **Refactoring mode**: Warn for differences < 10%, fail only on CRITICAL (≥ 10%)
- **Strict mode**: Exact bitwise equality (no tolerances)

#### **Usage Examples**
```bash
# Standard mode - tight tolerances for production
uv run pytest tests/test_golden_parity.py

# Refactoring mode - allows acceptable precision changes
EXACTCIS_REFACTORING_MODE=1 uv run pytest tests/test_golden_parity.py

# Investigate specific differences with detailed output
EXACTCIS_REFACTORING_MODE=1 uv run pytest tests/test_golden_parity.py -s -v

# Strict mode - exact match required  
EXACTCIS_STRICT_PARITY=1 uv run pytest tests/test_golden_parity.py

# Custom tolerance
EXACTCIS_REL_TOL=1e-8 uv run pytest tests/test_golden_parity.py
```

#### **When to Use Each Mode**
- **Standard**: Validating final implementations and releases
- **Refactoring**: Active development with infrastructure changes
- **Strict**: Critical validation of core algorithms
- **Custom tolerance**: Fine-tuning precision requirements for specific scenarios

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

## Next Development Priorities

### 1. Cross-Validation & Benchmarking (Current Focus)
- **R Package Validation**: Compare results with `exact2x2`, `ratesci`, `epitools` 
- **SciPy Integration**: Cross-validate with `scipy.stats.contingency.relative_risk()`
- **Literature Benchmarking**: Reproduce examples from Agresti, Tang, Duan papers
- **Performance Profiling**: Extend benchmarking to all refactored methods

### 2. Enhanced Public API (Next Release)
```python
# Unified interface for both OR and RR
from exactcis import or_ci, rr_ci

# Simple usage with smart defaults
result = rr_ci(a, b, c, d)  # Uses best method for data characteristics
result = or_ci(a, b, c, d)  # Auto-selects appropriate OR method

# Explicit method selection
result = rr_ci(a, b, c, d, method="score_cc")
result = or_ci(a, b, c, d, method="blaker")
```

### 3. Remaining Method Refactoring (Phase 3)
- **Exact methods**: Refactor conditional, mid-P, Blaker, unconditional to use centralized infrastructure
- **Registry system**: Method metadata and auto-selection logic
- **Enhanced diagnostics**: Standardized convergence and performance metrics

## Strategic Development Status

### ✅ Completed Phases (Aug 2025)
- **Phase 0**: Validation and corrections centralization
- **Phase 1**: Core solver infrastructure with RR score refactoring  
- **Phase 2**: Mathematical operations and Wald method refactoring

### 🔄 Current Work
- **Cross-validation**: R package and literature benchmarking
- **Performance analysis**: Impact assessment of refactoring
- **Documentation updates**: Reflect new architecture

### 📋 Planned Phases
- **Phase 3**: Exact methods refactoring with method registry
- **Phase 4**: Enhanced public API with smart method selection
- **Phase 5**: Method expansion based on validated user demand

## File Navigation Helpers

### Core Implementation Files
- **Method implementations**: `src/exactcis/methods/`
  - `relative_risk.py` - All RR methods (refactored to use centralized infrastructure)
  - `wald.py` - OR Wald method (refactored to use centralized infrastructure)
  - `conditional.py`, `midp.py`, `blaker.py`, `unconditional.py` - OR exact methods
- **Refactored infrastructure**: `src/exactcis/utils/`
  - `validation.py` - Centralized input validation
  - `continuity.py` - Unified continuity correction policies
  - `solvers.py` - Robust root-finding and bracketing
  - `inversion.py` - Standardized CI inversion framework
  - `mathops.py` - Safe mathematical operations
  - `estimates.py` - Centralized point estimates and standard errors
- **Legacy core**: `src/exactcis/core.py` - PMF calculations and legacy algorithms

### API and Testing
- **Main API**: `src/exactcis/__init__.py` - `compute_all_cis()` and `compute_all_rr_cis()`
- **Golden parity**: `tests/test_golden_parity.py` - Prevents regressions during refactoring
- **Utils tests**: `tests/test_utils/` - Unit tests for refactored infrastructure