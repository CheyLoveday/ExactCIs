# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactCIs is a focused Python package for computing confidence intervals for **odds ratios and relative risks** from 2×2 contingency tables. The package provides comprehensive, peer-reviewed methods for the two most important epidemiological effect measures.

**Odds Ratio Methods (5 implemented):**
- Fisher's conditional (exact)
- Mid-P adjusted (exact)  
- Blaker's exact
- Barnard's unconditional exact
- Wald-Haldane (asymptotic)

**Relative Risk Methods (6 implemented - NEW Aug 2025):**
- Wald standard and Katz variants
- Wald correlated (for matched data)
- Score/Tang (Miettinen-Nurminen) 
- Score with continuity correction
- U-statistic (nonparametric)

**Scope**: Focused exclusively on OR and RR confidence intervals - the two fundamental epidemiological effect measures. Does not attempt to be a complete epidemiological analysis platform.

**Target Applications:**
- Biomedical research requiring OR/RR confidence intervals
- Clinical trials and observational studies  
- Meta-analysis of OR/RR estimates
- Regulatory submissions requiring validated CI methods
- Academic research with peer-reviewed statistical methods

## Development Commands

### Testing
- `uv run pytest` - Run fast tests (default, skips slow tests)
- `uv run pytest --run-slow` - Run all tests including slow ones
- `uv run pytest -v` - Run tests with verbose output
- `uv run pytest --cov=src/exactcis tests/` - Run tests with coverage
- `pytest tests/test_methods/test_blaker.py -v` - Run specific test file
- `pytest tests/test_methods/test_blaker.py::test_specific_function -v` - Run specific test

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

### Documentation
- `make docs` - Generate Sphinx documentation
- Documentation lives in `docs/source/` with RST files

### Building & Distribution
- `make dist` - Build source and wheel packages
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

### Method Implementation Strategy Updates
- **conditional** (Fisher): Guaranteed ≥1-α coverage, conservative, use for small samples
- **midp**: Uses **adaptive grid search with confidence interval inversion** for better reliability with large samples. Similar to R's Exact package approach.
- **blaker**: **RECENTLY FIXED (Aug 2025)** - Root-finding algorithm enhanced with relaxed statistical tolerances (2% vs 1%) and improved plateau edge detection. Now provides exact coverage with minimal over-coverage across all sample sizes.
- **unconditional** (Barnard): **RECENTLY FIXED (Aug 2025)** - Uses profile likelihood approach with enhanced adaptive grid search. Upper bound inflation issue resolved via intelligent capping (2.5x odds ratio threshold) in refinement algorithm.
- **wald_haldane**: Fast approximation for large samples

### New Performance Features
- **Batch Processing**: Both midp and unconditional methods now support `*_batch()` functions with parallel processing
- **Profile Likelihood**: Unconditional method uses scipy optimization to find MLE of nuisance parameter p1 for each theta
- **Adaptive Grid Search**: More intelligent theta grid generation centered around the sample odds ratio
- **Shared Caching**: Enhanced caching system for parallel processing scenarios
- **Inflation Control (Aug 2025)**: CI search algorithms now include intelligent capping to prevent inflated bounds

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

## Recent Fixes (August 2025)

### Major Issues Resolved
Critical algorithmic issues were identified and successfully fixed:

1. **Blaker Method Root-Finding**: Fixed overly strict tolerances and plateau edge detection in `src/exactcis/core.py:650` and `core.py:518-535`
2. **Unconditional Method Upper Bound Inflation**: Fixed adaptive grid search refinement in `src/exactcis/utils/ci_search.py:340-349`
3. **Relative Risk Methods (August 2025)**: Fixed three critical issues in `src/exactcis/methods/relative_risk.py`:
   - Wald-Correlated method pathologically wide intervals (now detects independent vs matched data)
   - Score methods returning (0.000, inf) due to incorrect constrained MLE and root-finding
   - U-Statistic method too narrow intervals due to improper variance estimation

### Validation Status
- ✅ Comprehensive testing across 10 diverse scenarios (N=40 to N=4000)
- ✅ All methods now perform within expected statistical ranges  
- ✅ No significant outliers detected (>3x deviation from gold standards)
- ✅ Relative risk methods validated against expected ranges for 50/1000 vs 10/1000 data

### Documentation
- `RECENT_FIXES_SUMMARY.md` - Detailed technical analysis of fixes and validation
- `comprehensive_outlier_test.py` - Validation test suite across multiple scenarios

## Planned Improvements (Next Release)

### 1. Anchor Methods to Published Algorithms

| Method | Reference Implementation | Citation | Validation Target |
|--------|-------------------------|-----------|-------------------|
| `ci_wald_rr`, `ci_wald_katz_rr` | SciPy ≥1.11 `stats.contingency.relative_risk(method="katz")` | Katz (1960) + Altman (1991) | Cross-validate with SciPy |
| `ci_wald_correlated_rr` | NCSS "Two Correlated Proportions" | Koopman (1984) | Compare against NCSS manual examples |
| `ci_score_rr` | R `ratesci::scoreci()` | Tang (2020), Miettinen-Nurminen (1985) | Reproduce Tang's Table 2 |
| `ci_score_cc_rr` | DelRocco et al. (2023) continuity correction | DelRocco et al. (2023) | Small-n simulation (n=20) |
| `ci_ustat_rr` | R `ratesci::pairbinci(..., "ustat")` | Duan et al. (2014) | Worked example from Duan supplement |

### 2. Cross-Package Regression Testing
- Automated comparison with R `ratesci` package via `rpy2`
- Nightly CI tests against SciPy reference implementations
- Edge case validation (zero cells, rare outcomes, null effects)

### 3. Enhanced Public API
```python
# Planned unified interface
ci = relative_risk_ci(a, b, c, d, alpha=0.05, 
                     method="tang_cc", delta=4)

# Method mapping
_method_map = {
    "katz": ci_wald_katz_rr,
    "wald": ci_wald_rr, 
    "wald_corr": ci_wald_correlated_rr,
    "tang": ci_score_rr,
    "tang_cc": ci_score_cc_rr,
    "ustat": ci_ustat_rr,
}
```

### 4. Documentation & Validation
- Add authoritative citations to all method docstrings
- Create benchmark Jupyter notebook reproducing key literature examples
- Implement exhaustive cross-validation test suite
- Pin minimum SciPy=1.11 for `relative_risk` compatibility

### 5. Method Detection & Auto-Selection
- Automatic detection of independent vs matched/correlated data structures
- Smart defaults based on sample sizes and data patterns
- Warnings for inappropriate method choices

## Planned Improvements for Odds Ratio Methods

### 1. Anchor OR Methods to Published Algorithms

| Method | Reference Implementation | Citation | Validation Target |
|--------|-------------------------|-----------|-------------------|
| `exact_ci_conditional` | R `exact2x2::fisher.exact()` | Fisher (1935), Agresti (2002) | Cross-validate against R's fisher.test |
| `exact_ci_midp` | R `exact2x2::midp.exact.test()` | Lancaster (1961), Berry & Armitage (1995) | Reproduce Hirji et al. (1991) examples |
| `exact_ci_blaker` | R `exact2x2::blaker.exact.test()` | Blaker (2000) | Validate against Blaker's original paper examples |
| `exact_ci_unconditional` | R `Exact::exact.test(method="Csm")` | Barnard (1945), Suissa & Shuster (1985) | Compare with CSM implementation |
| `ci_wald_haldane` | Multiple (SAS PROC FREQ, R epitools) | Haldane (1956), Woolf (1955) | Cross-validate with SAS and R |
| `exact_ci_clopper_pearson` | R `binom.test()`, SciPy `binomtest` | Clopper & Pearson (1934) | Match SciPy binomial_test results |

### 2. Add Missing Standard Methods

**High Priority Additions Needed:**
- **Logit/Score Method**: Miettinen & Nurminen (1985) score-based OR CI
- **Cornfield Method**: Cornfield (1956) asymptotic approximation with exact tail areas
- **Breslow-Day-Tarone**: For stratified/matched data OR confidence intervals
- **Mantel-Haenszel**: Pooled OR across strata with Robins variance

**Reference Implementations to Target:**
- R `epitools::oddsratio()` - comprehensive OR methods
- R `meta::metabin()` - meta-analysis OR methods
- SAS `PROC FREQ ... / CMH` - Cochran-Mantel-Haenszel methods
- R `exact2x2` package - exact and mid-p methods

### 3. Cross-Package OR Validation Strategy

```python
# Planned validation tests
import pytest
import numpy as np
from rpy2.robjects import r
from scipy.stats import contingency

test_tables = [
    (12, 5, 8, 10),     # Agresti Example 3.1.1
    (3, 1, 1, 3),       # Small counts
    (0, 12, 5, 8),      # Zero cell
    (689, 41, 349, 63), # Large sample from Fleiss (1981)
]

@pytest.mark.parametrize("a,b,c,d", test_tables)
def test_conditional_vs_r_fisher(a, b, c, d):
    """Validate against R's fisher.test()"""
    py_ci = exact_ci_conditional(a, b, c, d)
    r_result = r['fisher.test'](r.matrix([a,b,c,d], nrow=2))
    r_ci = r_result.rx2('conf.int')
    assert np.allclose(py_ci, r_ci, rtol=1e-6)

@pytest.mark.parametrize("a,b,c,d", test_tables)  
def test_wald_vs_epitools(a, b, c, d):
    """Validate against R epitools::oddsratio()"""
    py_ci = ci_wald_haldane(a, b, c, d)
    r.library('epitools')
    r_result = r['oddsratio'](r.matrix([a,b,c,d], nrow=2), method="wald")
    r_ci = r_result.rx2('measure').rx(True, [2,3])  # Extract CI columns
    assert np.allclose(py_ci, r_ci, rtol=1e-4)
```

### 4. Enhanced OR API Design

```python
# Planned unified odds ratio interface
def odds_ratio_ci(a, b, c, d, alpha=0.05, method="auto", **kwargs):
    """
    Compute confidence interval for odds ratio of 2x2 table.
    
    Parameters
    ----------
    a, b, c, d : int
        Cell counts of 2x2 contingency table
    alpha : float, default 0.05
        Significance level (1-alpha confidence level)
    method : str, default "auto"
        Method to use. Options:
        - "auto": Choose based on sample size and cell counts
        - "fisher": Conditional exact (Fisher) method
        - "midp": Mid-p exact method  
        - "blaker": Blaker exact method
        - "unconditional": Barnard unconditional exact
        - "wald": Wald (Woolf) with Haldane correction
        - "logit": Miettinen-Nurminen score method
        - "cornfield": Cornfield asymptotic method
    
    Returns
    -------
    ConfidenceInterval
        Named tuple with .low, .high, .method_used attributes
    """
    
    # Auto-selection logic
    if method == "auto":
        n = a + b + c + d
        min_cell = min(a, b, c, d)
        
        if min_cell == 0:
            method = "fisher"  # Handle zero cells exactly
        elif n < 40 or min_cell < 5:
            method = "fisher"  # Small samples need exact methods
        elif n > 200:
            method = "wald"    # Large samples can use asymptotic
        else:
            method = "midp"    # Mid-range samples
    
    # Method dispatch
    method_map = {
        "fisher": exact_ci_conditional,
        "midp": exact_ci_midp,
        "blaker": exact_ci_blaker, 
        "unconditional": exact_ci_unconditional,
        "wald": ci_wald_haldane,
        # TODO: Add when implemented
        # "logit": ci_logit_or,
        # "cornfield": ci_cornfield_or,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    
    ci_func = method_map[method]
    lower, upper = ci_func(a, b, c, d, alpha, **kwargs)
    
    # Return structured result
    from collections import namedtuple
    ConfidenceInterval = namedtuple('ConfidenceInterval', 
                                   ['low', 'high', 'method_used'])
    return ConfidenceInterval(lower, upper, method)
```

### 5. Literature Validation Benchmarks

**Key Papers to Reproduce:**
- Agresti (2002) "Categorical Data Analysis" - Examples 3.1.1, 3.1.2, 3.1.5
- Hirji et al. (1991) "Computing exact power..." - Mid-p examples
- Blaker (2000) "Confidence curves..." - Original Blaker examples  
- Martin Andrés & Silva Mato (1994) - Unconditional exact comparisons
- Fleiss et al. (2003) "Statistical Methods..." - Large sample validations

**R Package Cross-Validation:**
- `exact2x2` (v1.6.9) - comprehensive exact methods
- `epitools` (v0.5-10.1) - epidemiological OR calculations  
- `meta` (v7.0-0) - meta-analysis OR pooling
- `PropCIs` (v0.3-0) - proportion and OR confidence intervals

## Additional Relative Risk Methods (High Priority Gaps)

Based on comprehensive software survey (SAS, Stata, NCSS, R, SciPy), **four key RR methods** appear consistently and would complete the standard epidemiological toolkit:

### 1. Mantel-Haenszel Pooled RR (Stratified Analysis)

**When Essential:**
- Multi-stratum studies (sex, site, ancestry groups)
- Required by FDA and CONSORT for adjusted RR
- Controlling for confounding variables

**Reference Implementations:**
- R `epiR::epi.2by2(type="mh")` 
- SAS `PROC FREQ ... / CMH RISKDIFF(CL=MH)`
- Stata `cs` command with stratification

**Mathematical Foundation:**
```
RR_MH = Σ(a_i * n2_i / n_i) / Σ(c_i * n1_i / n_i)
Variance via Robins et al. (1986) formula
```

### 2. Exact Unconditional RR (Cornfield/Fisher Hybrid)

**When Essential:**
- Very small cell counts (≤5) where conditional exact is too conservative
- Unbalanced designs where Wald breaks down
- Regulatory submissions requiring exact methods

**Reference Implementations:**
- R `exact2x2::riskratioUncond()`
- Stata `cs` with `exact` option
- NCSS "Two Proportions" exact unconditional

### 3. Bonett-Price Hybrid Fieller Interval

**When Essential:**
- Moderate samples (30 ≤ n ≤ 200)
- Better coverage than Wald, narrower than exact methods
- Recommended for clinical trials with moderate sample sizes

**Reference Implementation:**
- R `DescTools::RelRisk(method="bonett")`

**Mathematical Foundation:**
```
Hybrid of Wilson score intervals combined via Fieller method
CI_BP = (p̂₁(1-q₂) ± z√[p̂₁(1-p̂₁)/n₁ + q₂(1-q₂)/n₂]) / (q₂(1-p̂₁) + z²/(2n₂))
```

### 4. Bootstrap BCa Interval (Non-parametric)

**When Essential:**
- Complex designs (clustering, matching)
- When analytic variance formulas are disputed
- Genetic epidemiology with family structures

**Reference Implementation:**
- `scikits-bootstrap` with bias-corrected and accelerated (BCa) method
- Base `numpy.random` for simple bootstrap

## Completing the RR Toolkit: Four Essential Additions

To achieve **parity with standard epidemiological software** (SAS, Stata, R), these four RR methods would complete the professional toolkit while maintaining focused scope:

### 1. **Exact Unconditional RR** (Small Sample Priority)
- **Use Case**: Very small cells (≤5) where conditional exact is too conservative
- **Reference**: R `exact2x2::riskratioUncond()`, Stata `cs, exact`
- **Implementation**: Adapt existing unconditional OR algorithms to RR

### 2. **Bonett-Price Hybrid Fieller** (Moderate Sample Priority)  
- **Use Case**: Moderate samples (30-200) - better than Wald, narrower than exact
- **Reference**: R `DescTools::RelRisk(method="bonett")`
- **Implementation**: Wilson score intervals combined via Fieller method

### 3. **Bootstrap BCa** (Complex Design Priority)
- **Use Case**: Complex designs, disputed variance formulas, genetic studies
- **Reference**: `scikits-bootstrap` or base `numpy.random`
- **Implementation**: Non-parametric resampling with bias correction

### 4. **Mantel-Haenszel Stratified RR** (Single-Table Focus)
- **Use Case**: Single 2x2 table with known stratum weights (not full stratified analysis)
- **Reference**: SAS `PROC FREQ CMH`, R `epiR::epi.2by2(type="mh")`  
- **Implementation**: MH formula for pre-weighted single table

**Strategic Note**: These four methods complete the RR toolkit without expanding beyond OR/RR focus. No risk difference, no full stratified analysis platform, no sample size calculations - just comprehensive OR/RR confidence intervals.

## Complete Package Design Matrix

Following the **"necessary and sufficient"** principle, the package covers every common study design with exactly the right methods:

### Coverage Matrix: Study Design × Sample Size

| Design / n-range | RR Method | OR Method | Justification |
|------------------|-----------|-----------|---------------|
| **Large, independent (n≥100)** | Wald-Katz | Wald-Haldane | SciPy/SAS standard, fast baseline |
| **Moderate (30≤n<100)** | Score (Tang) + CC | Score + CC | EMA recommended, gold-standard coverage |
| **Small (n<30)** | Exact conditional + Mid-P | Exact conditional + Mid-P | Exact methods, Mid-P less conservative |
| **Ultra-small/zeros** | Exact unconditional | Exact unconditional | Narrower than conditional, Stata `exact` |
| **Matched pairs** | U-statistic (Duan) | McNemar exact | Preserves pairing information |
| **Multi-stratum** | Mantel-Haenszel | Mantel-Haenszel | CONSORT required for stratified analysis |
| **Moderate, narrow CI** | Bonett-Price Fieller | Agresti-Caffo | Coverage-width trade-off optimization |
| **Complex/clustered** | Bootstrap BCa | Bootstrap BCa | Design-agnostic, model diagnostic |

### Three-Rule Filter Applied
1. **✅ Widely Accepted**: Every method appears in major software (R/SAS/Stata) or regulatory guidance
2. **✅ Complementary Coverage**: Every study scenario has at least one reliable method  
3. **✅ No Redundancy**: No mathematically equivalent methods included

### Recommended Defaults
- **Primary Default**: Continuity-corrected Score (Tang for RR, score for OR)
- **Rationale**: Tracks nominal 95% from n≈30 upward, never diverges like Wald, FDA template

### Public API Design
```python
from exactcis import rr_ci, or_ci

# Simple cases with smart defaults
rr_ci(a, b, c, d)  # Uses score_cc method
or_ci(a, b, c, d)  # Uses score_cc method

# Explicit method selection
rr_ci(a, b, c, d, method="tang_cc")      # Score with continuity correction
rr_ci(a, b, c, d, method="katz")         # Wald-Katz for large samples
rr_ci(a, b, c, d, method="exact_cond")   # Conditional exact for small samples
rr_ci(a, b, c, d, method="exact_uncond") # Unconditional for zeros
rr_ci(a, b, c, d, method="bonett")       # Bonett-Price for moderate samples
rr_ci(a, b, c, d, method="ustat")        # U-statistic for matched data
rr_ci(a, b, c, d, method="bootstrap")    # BCa bootstrap for complex designs

# Advanced options
rr_ci(a, b, c, d, method="bootstrap", B=10000, seed=42)
```

### Bootstrap BCa Implementation Priority
**High Value Addition**:
- **Design Agnostic**: Works for any table structure without re-deriving variance formulas
- **Model Diagnostic**: Sanity-check for analytic CIs (mismatches reveal data issues)
- **Pedagogical**: Easier to understand than complex score equations
- **Rare Events**: Stable when analytic formulas break down

**Implementation Strategy**:
- Pure NumPy implementation (no external dependencies)
- ~30ms for B=10,000 replicates (acceptable performance)
- Bias-corrected & accelerated for second-order accuracy
- Independent resampling within exposed/control groups

## File Navigation Helpers

- CI method implementations: `src/exactcis/methods/<method>.py`
- Core algorithms: `src/exactcis/core.py:` (line numbers for key functions):
  - PMF calculations: ~100-400
  - Root finding: ~400-700 (recently enhanced)
  - Support calculations: ~280-320
- CI search utilities: `src/exactcis/utils/ci_search.py` (adaptive grid search with inflation control)
- Main orchestrator: `src/exactcis/__init__.py:40` (`compute_all_cis`)
- Test patterns: `tests/test_methods/test_<method>.py`