# Phase 4: Method Registry and Smart Selection System

**Status: Planned for Post-Phase 3 Implementation**

## Overview

Implement intelligent method registry with automatic selection capabilities and enhanced public API, transforming ExactCIs from a collection of functions into a unified, user-friendly statistical toolkit.

## Current State Analysis

### Existing API Limitations
- **Manual method selection**: Users must know specific method names
- **Scattered orchestration**: `compute_all_cis()` and `compute_all_rr_cis()` are hardcoded
- **No guidance**: No recommendations for optimal method selection
- **Inconsistent APIs**: Different calling patterns across methods

### User Experience Problems
- **Decision paralysis**: 12+ methods available with no guidance
- **Suboptimal choices**: Users default to first method they find
- **Trial and error**: No systematic way to compare method suitability
- **Domain knowledge required**: Must understand statistical nuances to choose correctly

## Proposed Solution

### Core Components

#### 1. Method Metadata System

```python
# utils/method_registry.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

@dataclass
class MethodMetadata:
    """Comprehensive method characteristics and constraints."""
    name: str
    category: str  # "odds_ratio" | "relative_risk"
    method_type: str  # "exact" | "asymptotic" | "score"
    handles_zeros: bool
    min_sample_size: Optional[int]
    computational_cost: int  # 1-5 scale (1=fastest, 5=slowest)
    recommended_for: List[str]  # ["small_samples", "zero_cells", "speed", "accuracy"]
    description: str
    reference: str  # Scientific paper or textbook reference
    
# Complete registry definitions
OR_METHODS: Dict[str, MethodMetadata] = {
    "conditional": MethodMetadata(
        name="conditional",
        category="odds_ratio",
        method_type="exact", 
        handles_zeros=True,
        min_sample_size=None,
        computational_cost=3,
        recommended_for=["small_samples", "exact_coverage", "zero_cells"],
        description="Fisher's exact conditional CI",
        reference="Fisher (1935), Agresti (2002)"
    ),
    "wald_haldane": MethodMetadata(
        name="wald_haldane", 
        category="odds_ratio",
        method_type="asymptotic",
        handles_zeros=True,
        min_sample_size=30,
        computational_cost=1,
        recommended_for=["speed", "large_samples"],
        description="Wald CI with Haldane correction",
        reference="Haldane (1956), Agresti (2002)"
    ),
    "blaker": MethodMetadata(
        name="blaker",
        category="odds_ratio",
        method_type="exact",
        handles_zeros=True,
        min_sample_size=None,
        computational_cost=4,
        recommended_for=["small_samples", "optimal_coverage"],
        description="Blaker's exact unconditional CI", 
        reference="Blaker (2000)"
    ),
    # ... complete definitions for all OR methods
}

RR_METHODS: Dict[str, MethodMetadata] = {
    "score": MethodMetadata(
        name="score",
        category="relative_risk",
        method_type="score",
        handles_zeros=True,
        min_sample_size=10,
        computational_cost=2, 
        recommended_for=["accuracy", "moderate_samples"],
        description="Miettinen-Nurminen score CI",
        reference="Tang et al. (2020)"
    ),
    "score_cc": MethodMetadata(
        name="score_cc",
        category="relative_risk", 
        method_type="score",
        handles_zeros=True,
        min_sample_size=10,
        computational_cost=3,
        recommended_for=["accuracy", "small_samples", "conservative"],
        description="Continuity-corrected score CI",
        reference="Tang et al. (2020)"
    ),
    # ... complete definitions for all RR methods
}
```

#### 2. Data Analysis and Scoring Functions

```python
def analyze_data_characteristics(a: int, b: int, c: int, d: int) -> Dict[str, any]:
    """Analyze data to determine optimal method characteristics."""
    n = a + b + c + d
    has_zeros = any(x == 0 for x in (a, b, c, d))
    min_cell = min(a, b, c, d)
    max_cell = max(a, b, c, d)
    
    # Calculate effect size for extreme ratio detection
    if b > 0 and c > 0:
        or_estimate = (a * d) / (b * c)
        is_extreme_ratio = or_estimate > 10 or or_estimate < 0.1
    else:
        is_extreme_ratio = True
        
    return {
        "total_n": n,
        "has_zeros": has_zeros,
        "zero_count": sum(x == 0 for x in (a, b, c, d)),
        "min_cell_count": min_cell,
        "max_cell_count": max_cell,
        "is_small_sample": n < 30 or min_cell < 5,
        "is_very_small_sample": n < 20 or min_cell < 3,
        "has_extreme_ratio": is_extreme_ratio,
        "is_sparse": (a + c) < 10 or (b + d) < 10,  # Small margins
        "balance_ratio": min_cell / max_cell if max_cell > 0 else 0
    }

def score_method_suitability(method_meta: MethodMetadata,
                           data_chars: Dict[str, any], 
                           priority: str = "accuracy") -> float:
    """Score how suitable a method is for given data (0-1 scale)."""
    score = 1.0
    
    # Hard constraints (eliminate method if violated)
    if data_chars["has_zeros"] and not method_meta.handles_zeros:
        return 0.0
        
    if (method_meta.min_sample_size and 
        data_chars["total_n"] < method_meta.min_sample_size):
        score *= 0.3  # Strong penalty but not elimination
    
    # Soft preferences based on data characteristics
    if data_chars["is_very_small_sample"]:
        if "small_samples" in method_meta.recommended_for:
            score *= 1.3
        if method_meta.method_type == "exact":
            score *= 1.2
    
    if data_chars["has_extreme_ratio"]:
        if "exact_coverage" in method_meta.recommended_for:
            score *= 1.2
        if method_meta.method_type == "asymptotic":
            score *= 0.7  # Asymptotic methods struggle with extreme ratios
    
    if data_chars["is_sparse"]:
        if "zero_cells" in method_meta.recommended_for:
            score *= 1.2
    
    # Priority-based adjustments
    if priority == "speed":
        # Prefer low computational cost
        score *= (6 - method_meta.computational_cost) / 5
    elif priority == "accuracy":
        # Prefer exact methods and those recommended for accuracy
        if "accuracy" in method_meta.recommended_for:
            score *= 1.3
        if method_meta.method_type == "exact":
            score *= 1.1
    elif priority == "conservative":
        # Prefer methods known for conservative coverage
        if "conservative" in method_meta.recommended_for:
            score *= 1.2
        if method_meta.method_type == "exact":
            score *= 1.1
    
    return min(score, 1.0)

def recommend_best_method(a: int, b: int, c: int, d: int,
                         category: str,
                         priority: str = "accuracy") -> str:
    """Recommend single best method for given data."""
    data_chars = analyze_data_characteristics(a, b, c, d)
    registry = OR_METHODS if category == "odds_ratio" else RR_METHODS
    
    best_method = None
    best_score = -1.0
    
    for method_name, method_meta in registry.items():
        score = score_method_suitability(method_meta, data_chars, priority)
        if score > best_score:
            best_score = score
            best_method = method_name
    
    return best_method

def get_suitable_methods(a: int, b: int, c: int, d: int,
                        category: str, 
                        min_score: float = 0.5) -> List[Tuple[str, float]]:
    """Get all suitable methods ranked by suitability score."""
    data_chars = analyze_data_characteristics(a, b, c, d)
    registry = OR_METHODS if category == "odds_ratio" else RR_METHODS
    
    scored_methods = []
    for method_name, method_meta in registry.items():
        score = score_method_suitability(method_meta, data_chars)
        if score >= min_score:
            scored_methods.append((method_name, score))
    
    # Sort by score descending
    scored_methods.sort(key=lambda x: x[1], reverse=True)
    return scored_methods
```

#### 3. Enhanced Public API

```python
# In __init__.py - Clean, functional dispatch
def or_ci(a: int, b: int, c: int, d: int,
          method: str = "auto",
          alpha: float = 0.05, 
          priority: str = "accuracy",
          **kwargs) -> Tuple[float, float]:
    """
    Unified odds ratio confidence interval with smart method selection.
    
    Parameters
    ----------
    a, b, c, d : int
        Cell counts of the 2x2 contingency table
    method : str, default "auto"
        Method to use. If "auto", selects best method based on data
    alpha : float, default 0.05
        Significance level (e.g., 0.05 for 95% CI)
    priority : str, default "accuracy"
        Selection priority: "accuracy", "speed", or "conservative"
    **kwargs
        Additional method-specific parameters
        
    Returns
    -------
    lower, upper : float
        Confidence interval bounds
        
    Examples
    --------
    >>> # Automatic method selection
    >>> or_ci(12, 5, 8, 10)  # Uses best method for this data
    (0.89, 8.91)
    
    >>> # Explicit method selection  
    >>> or_ci(12, 5, 8, 10, method="conditional")
    (0.85, 9.12)
    
    >>> # Speed-optimized selection
    >>> or_ci(120, 50, 80, 100, priority="speed")
    (1.23, 2.89)
    """
    from .utils.method_registry import recommend_best_method
    from .utils.validation import validate_table_and_alpha
    
    # Validate inputs
    a, b, c, d = validate_table_and_alpha(a, b, c, d, alpha)
    
    # Method selection
    if method == "auto":
        method = recommend_best_method(a, b, c, d, "odds_ratio", priority)
    
    # Functional dispatch table
    method_functions = {
        "conditional": lambda: ci_conditional(a, b, c, d, alpha, **kwargs),
        "wald_haldane": lambda: ci_wald_haldane(a, b, c, d, alpha, **kwargs),
        "blaker": lambda: ci_blaker(a, b, c, d, alpha, **kwargs),
        "unconditional": lambda: ci_unconditional(a, b, c, d, alpha, **kwargs),
        "midp": lambda: ci_midp(a, b, c, d, alpha, **kwargs),
    }
    
    if method not in method_functions:
        available = ", ".join(method_functions.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    
    return method_functions[method]()

def rr_ci(a: int, b: int, c: int, d: int,
          method: str = "auto",
          alpha: float = 0.05,
          priority: str = "accuracy", 
          **kwargs) -> Tuple[float, float]:
    """
    Unified relative risk confidence interval with smart method selection.
    
    [Similar docstring structure as or_ci]
    """
    # Similar implementation for relative risk methods

def list_methods(category: str = "all", 
                 data_context: Optional[Tuple[int, int, int, int]] = None) -> None:
    """
    Display available methods with recommendations.
    
    Parameters
    ----------  
    category : str, default "all"
        Method category: "all", "odds_ratio", or "relative_risk"
    data_context : tuple, optional
        If provided, shows suitability scores for this specific data
        
    Examples
    --------
    >>> list_methods("odds_ratio")
    Available Odds Ratio Methods:
    =============================
    conditional     - Fisher's exact conditional CI [exact]
    wald_haldane    - Wald CI with Haldane correction [asymptotic] 
    blaker          - Blaker's exact unconditional CI [exact]
    ...
    
    >>> list_methods("odds_ratio", data_context=(2, 8, 1, 9))
    Recommended Methods for Your Data (n=20, zeros=0):
    =================================================
    1. conditional   (score: 0.95) - Fisher's exact conditional CI
    2. blaker       (score: 0.92) - Blaker's exact unconditional CI  
    3. wald_haldane (score: 0.45) - Wald CI with Haldane correction
    ...
    """
```

## Implementation Plan

### Phase 4.1: Registry Infrastructure 🏗️
- [ ] Create method metadata definitions for all existing methods
- [ ] Implement data analysis and scoring functions  
- [ ] Add comprehensive unit tests for selection algorithms
- [ ] Create method recommendation validation suite

### Phase 4.2: Enhanced API Development 🚀
- [ ] Implement `or_ci()` and `rr_ci()` unified functions
- [ ] Create functional dispatch system with clean error handling
- [ ] Add `list_methods()` help function with formatting
- [ ] Implement method comparison utilities

### Phase 4.3: Testing and Validation 🧪
- [ ] **Recommendation validation**: Test selection logic against known-good choices
- [ ] **Edge case coverage**: Ensure robust handling of all data scenarios
- [ ] **API consistency**: Validate uniform behavior across all methods
- [ ] **Performance testing**: Ensure dispatch overhead is minimal

### Phase 4.4: Documentation and Examples 📚
- [ ] **Complete API documentation**: Comprehensive docstrings and examples
- [ ] **Method selection guide**: When to use each method
- [ ] **Tutorial notebooks**: Real-world usage examples  
- [ ] **Migration guide**: Moving from old to new API

## Benefits

### User Experience Improvements
- **Beginner-friendly**: Automatic method selection removes decision paralysis
- **Expert-friendly**: Full control retained with explicit method selection
- **Educational**: Built-in recommendations teach best practices
- **Consistent**: Unified API reduces cognitive load

### Technical Advantages  
- **Extensible**: Easy to add new methods without breaking changes
- **Maintainable**: Method metadata centralizes characteristics
- **Testable**: Clear separation of selection logic and implementation
- **Future-proof**: Registry system enables advanced features

## Success Metrics

### Quality Gates
- [ ] **Selection accuracy**: >90% of recommendations match expert choices
- [ ] **API consistency**: All methods accessible through unified interface
- [ ] **Performance**: <1ms overhead for method selection
- [ ] **Coverage**: 100% test coverage for registry and selection logic

### User Experience Goals
- [ ] **Reduced support questions**: Fewer "which method?" inquiries
- [ ] **Increased adoption**: More users trying advanced methods
- [ ] **Positive feedback**: User surveys show improved satisfaction
- [ ] **Documentation clarity**: Self-service capability for common questions

This phase will transform ExactCIs from a collection of statistical functions into an intelligent, user-friendly toolkit that guides users to optimal choices while maintaining full flexibility for advanced users.