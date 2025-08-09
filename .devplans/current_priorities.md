# Current Development Priorities - ExactCIs

**Status: Active Development Plan (Aug 2025)**

## Recently Completed Work

### Phase 0-2 Refactoring ✅
- **Phase 0**: Centralized validation and corrections (`validation.py`, `continuity.py`)
- **Phase 1**: Core solver infrastructure (`solvers.py`, `inversion.py`) with RR score refactoring
- **Phase 2**: Mathematical operations (`mathops.py`, `estimates.py`) with Wald method refactoring

### Key Achievements
- **Golden parity testing**: Comprehensive fixtures prevent numerical regressions
- **Infrastructure consolidation**: 60%+ reduction in code duplication
- **Dataclass safety**: Structured `EstimationResult` and `CorrectionResult` types
- **Robust algorithms**: Enhanced bracketing, plateau detection, infinite bound handling

## Current Focus Areas

### 1. Cross-Validation & Benchmarking (Priority: High)
**Objective**: Validate refactored methods against external standards

**Tasks**:
- [ ] **R Package Comparison**: Cross-validate results with `exact2x2`, `ratesci`, `epitools`
  - Automated comparison scripts for systematic validation
  - Document any legitimate differences and their statistical justification
- [ ] **SciPy Integration**: Compare with `scipy.stats.contingency.relative_risk()` (≥1.11)
- [ ] **Literature Benchmarking**: Reproduce key examples from Agresti, Tang, Duan papers
- [ ] **Performance Assessment**: Measure impact of refactoring on computation times

### 2. Documentation & Architecture Updates (Priority: Medium)
**Objective**: Reflect new modular architecture in documentation

**Tasks**:
- [ ] Update method documentation to reference centralized infrastructure
- [ ] Create architecture diagrams showing refactored data flow
- [ ] Document migration path for any breaking changes
- [ ] Update CLI help text and examples

### 3. Quality Assurance (Priority: High)
**Objective**: Ensure production readiness after major refactoring

**Tasks**:
- [ ] **Comprehensive testing**: Run full test suite with all combinations
- [ ] **Edge case validation**: Verify zero-cell, extreme ratio handling
- [ ] **Performance regression testing**: Compare pre/post refactoring benchmarks
- [ ] **Memory usage analysis**: Profile memory consumption patterns

## Planned Future Phases

### Phase 3: Exact Methods Refactoring (Next Quarter)
- Refactor conditional, mid-P, Blaker, unconditional methods
- Implement method registry system for metadata and auto-selection
- Standardize diagnostics across all methods

### Phase 4: Enhanced Public API (Following Release)
- Implement unified `or_ci()` and `rr_ci()` functions
- Add smart method selection based on data characteristics
- Comprehensive API documentation and examples

### Phase 5: Method Expansion (Based on Demand)
- Additional OR methods (logit/score, Cornfield)  
- Additional RR methods (exact unconditional, Bootstrap BCa)
- Stratified analysis (Mantel-Haenszel pooling)

## Success Metrics

### Quality Gates
- [ ] **Golden parity**: All fixtures pass with established tolerances
- [ ] **R parity**: Results match R implementations within statistical tolerances  
- [ ] **Performance**: No >10% regression in computation times
- [ ] **Coverage**: Test coverage maintained at >95%

### Deliverables
- [ ] **Validation report**: Comparison with R packages and literature
- [ ] **Performance analysis**: Before/after benchmarking results
- [ ] **Updated documentation**: Architecture guides and API references
- [ ] **Release notes**: Summary of improvements and any breaking changes

## Development Guidelines

### Code Quality
- Maintain golden parity testing for all refactoring work
- Use centralized infrastructure (`utils/`) for new method development
- Follow dataclass patterns for structured data flow
- Include comprehensive unit tests for all new utilities

### Testing Strategy
- Run `pytest --run-slow` for comprehensive validation
- Use `EXACTCIS_STRICT_PARITY=1` for exact numerical matching during refactoring
- Profile performance impact of any algorithmic changes
- Cross-validate statistical correctness with external implementations

### Architecture Principles
- **Single source of truth**: Centralize common operations in `utils/`
- **Structured data flow**: Use dataclasses for complex data structures
- **Separation of concerns**: Keep method-specific logic in `methods/`, shared logic in `utils/`
- **Backward compatibility**: Preserve existing public API behavior