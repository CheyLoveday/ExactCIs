# ExactCIs Development Plans

This directory contains strategic development plans and roadmaps for the ExactCIs project.

## 📋 **Current Status (August 2025)**

**Phase 0-2 Complete**: Core infrastructure refactoring finished
- ✅ Centralized validation and corrections  
- ✅ Robust solver infrastructure
- ✅ Mathematical operations consolidation

## 🗂️ **Active Plans**

### **Strategic Overview**
- [`current_priorities.md`](./current_priorities.md) - Active development priorities and immediate focus areas
- [`refactoring_summary.md`](./refactoring_summary.md) - Complete summary of Phase 0-2 achievements

### **Future Roadmaps** 
- [`phase3_mathematical_core.md`](./phase3_mathematical_core.md) - **Phase 3**: Consolidate all mathematical operations into unified module
- [`phase4_method_registry.md`](./phase4_method_registry.md) - **Phase 4**: Intelligent method selection and enhanced public API

### **Maintenance**
- [`legacy_cleanup_roadmap.md`](./legacy_cleanup_roadmap.md) - Systematic plan for removing outdated code marked with TODO tags

### **Feature Development**
- [`feature_cc/`](./feature_cc/) - Continuity correction enhancement plans
- [`feature_rr/`](./feature_rr/) - Relative risk method expansion plans

## 🎯 **Next Actions**

1. **Cross-validation & Benchmarking** (Current Priority)
   - Compare results with R packages (`exact2x2`, `ratesci`, `epitools`)
   - Validate numerical accuracy and performance
   - Document any differences and their justification

2. **Legacy Code Cleanup** (Ongoing)
   - Remove TODO-marked functions systematically
   - Consolidate duplicate functionality
   - Update mixed legacy/modern import patterns

3. **Phase 3 Planning** (Upcoming)
   - Consolidate mathematical operations
   - Unify caching strategies
   - Prepare for method registry implementation

## 📊 **High-Value Refactoring Opportunities**

### **Mathematical Operations Consolidation** 🧮
**Value**: Unified caching, performance optimization, single source of truth
- Consolidate `support`, `log_binom_coeff`, `logsumexp` from `core.py`
- Merge safe operations from `utils/mathops.py`
- Create `utils/math_core.py` with centralized caching strategy

### **Method Registry and Smart Selection** 🎯  
**Value**: Intelligent defaults, better UX, extensible architecture
- Method metadata system with characteristics and constraints
- Automatic method selection based on data analysis
- Enhanced public API: `or_ci(a, b, c, d, method="auto")`
- Educational features with method recommendations

## 🧹 **Cleanup Status**

All legacy code has been marked with TODO tags for systematic removal:
- ✅ Legacy wrappers in `core.py` 
- ✅ Duplicate functions between modules
- ✅ Mixed legacy/modern import patterns  
- ✅ Deprecated utility functions

See [`legacy_cleanup_roadmap.md`](./legacy_cleanup_roadmap.md) for detailed removal plan.

## 🏗️ **Development Principles**

### **Functional & Modular Architecture**
- Pure functions with clear inputs/outputs
- Dataclasses for structured data (no business logic classes)
- Centralized utilities with single responsibility
- Golden parity testing prevents regressions

### **Quality Gates**
- Maintain >95% test coverage
- Golden parity tests pass with established tolerances  
- Performance benchmarking prevents regressions
- Cross-validation with R packages ensures correctness

---

*Last updated: August 2025 - Post Phase 2 completion*