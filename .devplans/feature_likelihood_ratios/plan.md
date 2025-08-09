# Feature Plan: Likelihood Ratios for Effect Size Evidence

**Status: Planned - Advanced Feature for Phase 5+**

## Overview

Implement likelihood ratio methods for quantifying evidence about **clinically meaningful effect sizes**, moving beyond traditional null hypothesis significance testing to provide interpretable measures of evidence strength.

## Concept & Motivation

### **The Problem with Traditional Testing**
Current epidemiological practice focuses on statistical significance rather than clinical importance:

```python
# Traditional approach
or_ci(12, 88, 8, 92)  # CI: (1.02, 3.89), p=0.04
# Result: "Statistically significant" - but is OR=1.02 clinically meaningful?

# What we really want to know
"How strong is the evidence for a clinically important effect (OR ≥ 2.0)
 compared to no meaningful effect (OR ≈ 1.0)?"
```

### **Likelihood Ratios as Solution**
Likelihood ratios provide **quantified evidence** for competing hypotheses:
- **LR > 1**: Data favor the effect hypothesis  
- **LR < 1**: Data favor the null hypothesis
- **LR = 1**: Data provide equal evidence for both

**Interpretation Scale**:
- LR ≥ 10: Strong evidence for H₁
- LR ≥ 3: Moderate evidence for H₁  
- 0.33 ≤ LR ≤ 3: Weak evidence either way
- LR ≤ 0.33: Moderate evidence for H₀
- LR ≤ 0.1: Strong evidence for H₀

## Technical Foundation

### **Likelihood Function for 2×2 Tables**
For contingency table with cells (a, b, c, d), the likelihood is:
```
L(θ) = P(data | θ) ∝ θᵃ × (1 + θ)^(-(a+b)) × ... [hypergeometric or binomial]
```

Where θ is the odds ratio or relative risk parameter.

### **Effect Size Likelihood Ratio**
```
LR = L(data | H₁) / L(data | H₀)

Where:
H₁: "Effect is clinically meaningful" (e.g., OR ≥ 2.0)  
H₀: "Effect is negligible" (e.g., 0.9 ≤ OR ≤ 1.1)
```

### **Computational Approach**
1. **Point hypotheses**: Direct likelihood evaluation
2. **Composite hypotheses**: Numerical integration over parameter regions
3. **Maximum likelihood**: Use supremum over hypothesis region

## Proposed API Design

### **Basic Interface**
```python
def lr_effect_evidence(a: int, b: int, c: int, d: int,
                       effect_type: str = "OR",           # "OR" or "RR"
                       meaningful_threshold: float = 2.0,  # Clinically important effect
                       null_region: Tuple[float, float] = (0.9, 1.1),  # Negligible effect
                       direction: str = "greater") -> LikelihoodRatioResult:
    """
    Compute likelihood ratio for meaningful vs negligible effect.
    
    Quantifies evidence strength for clinically important effect sizes
    rather than just statistical significance.
    
    Parameters
    ----------
    a, b, c, d : int
        Cell counts of 2x2 contingency table
    effect_type : {"OR", "RR"}
        Type of effect measure
    meaningful_threshold : float
        Threshold for clinically meaningful effect
    null_region : tuple of float
        Range of effect sizes considered negligible
    direction : {"greater", "less", "two_sided"}
        Direction of meaningful effect
        
    Returns
    -------
    LikelihoodRatioResult
        Contains LR value, interpretation, and diagnostic info
        
    Examples
    --------
    >>> # Strong evidence for meaningful protective effect?
    >>> lr_effect_evidence(5, 95, 20, 80, 
    ...                    effect_type="RR", 
    ...                    meaningful_threshold=0.5,
    ...                    direction="less")
    LikelihoodRatioResult(
        lr=8.3,
        interpretation="Moderate evidence for meaningful protective effect",
        h1_description="RR ≤ 0.5 (meaningful protection)",
        h0_description="0.9 ≤ RR ≤ 1.1 (negligible effect)",
        strength_category="moderate_h1"
    )
    
    >>> # Evidence for harmful effect vs no effect?
    >>> lr_effect_evidence(25, 75, 10, 90,
    ...                    meaningful_threshold=2.0) 
    LikelihoodRatioResult(
        lr=12.4,
        interpretation="Strong evidence for meaningful harmful effect",
        h1_description="OR ≥ 2.0 (meaningful harm)",
        h0_description="0.9 ≤ OR ≤ 1.1 (negligible effect)",
        strength_category="strong_h1"
    )
    """
```

### **Advanced Interface**
```python
def lr_custom_hypotheses(a: int, b: int, c: int, d: int,
                        effect_type: str,
                        h1_region: Union[Callable, Tuple[float, float]],
                        h0_region: Union[Callable, Tuple[float, float]]) -> LikelihoodRatioResult:
    """
    Compute likelihood ratio for custom hypothesis regions.
    
    Examples
    --------
    >>> # Complex hypothesis: moderate harm vs protective
    >>> lr_custom_hypotheses(15, 85, 8, 92, "OR",
    ...                      h1_region=(1.5, 4.0),  # Moderate to strong harm
    ...                      h0_region=(0.3, 0.8))  # Protective effect
    """

def lr_threshold_scan(a: int, b: int, c: int, d: int,
                     effect_type: str,
                     thresholds: List[float]) -> Dict[float, LikelihoodRatioResult]:
    """
    Scan across multiple meaningful effect thresholds.
    
    Useful for finding the threshold where evidence becomes compelling.
    """
```

### **Data Models**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LikelihoodRatioResult:
    """Result from likelihood ratio analysis."""
    lr: float                          # Likelihood ratio value
    log_lr: float                      # Natural log of LR (for stability)
    interpretation: str                # Plain English interpretation
    strength_category: str             # "strong_h1", "moderate_h1", "weak", "moderate_h0", "strong_h0"
    h1_description: str               # Description of effect hypothesis
    h0_description: str               # Description of null hypothesis
    h1_likelihood: float              # Likelihood under H1
    h0_likelihood: float              # Likelihood under H0
    point_estimate: float             # MLE of effect size
    diagnostics: Dict[str, float]     # Computational diagnostics
```

## Implementation Plan

### **Phase 5.1: Foundation & Simple Cases** 🏗️
- [ ] **Mathematical infrastructure**:
  - Implement likelihood functions for OR and RR
  - Create numerical integration utilities for composite hypotheses
  - Add likelihood maximization over parameter regions

- [ ] **Basic point vs point hypotheses**:
  - `lr_point_hypotheses(a, b, c, d, h1_value, h0_value)`
  - Simple cases like H₁: OR = 2.0 vs H₀: OR = 1.0

- [ ] **Core data structures**:
  - `LikelihoodRatioResult` dataclass
  - Interpretation mapping (LR → strength category)

### **Phase 5.2: Threshold-Based Interface** 🎯
- [ ] **Primary user interface**:
  - `lr_effect_evidence()` function with intuitive parameters
  - Smart defaults for "meaningful" vs "negligible" effect regions
  - Clear interpretation strings

- [ ] **Validation & testing**:
  - Unit tests against known analytical results
  - Simulation studies to verify numerical integration
  - Edge case handling (zero cells, extreme ratios)

### **Phase 5.3: Advanced Features** 🚀
- [ ] **Custom hypothesis regions**:
  - Support for arbitrary parameter regions
  - Both tuple ranges and callable specifications
  - Composite hypothesis integration

- [ ] **Scanning utilities**:
  - `lr_threshold_scan()` for exploring evidence strength
  - Visualization helpers for LR profiles
  - Decision threshold recommendations

### **Phase 5.4: Integration & Polish** ✨
- [ ] **API integration**:
  - Add to main module imports
  - Consistent error handling with existing functions
  - Documentation and examples

- [ ] **Performance optimization**:
  - Caching for repeated likelihood evaluations  
  - Adaptive numerical integration
  - Parallel computation for threshold scanning

## Usage Examples

### **Clinical Trial Evaluation**
```python
# Phase II trial: Is this treatment worth pursuing?
# Traditional: p < 0.05 → "statistically significant"
# Better: How much evidence for clinically meaningful benefit?

a, b, c, d = 15, 35, 8, 42  # Treatment vs control outcomes

result = lr_effect_evidence(
    a, b, c, d, 
    effect_type="OR",
    meaningful_threshold=2.0,  # Clinically meaningful benefit
    null_region=(0.8, 1.2)     # Negligible effect
)

print(f"Evidence strength: {result.interpretation}")
print(f"LR = {result.lr:.1f}")
# Output: "Moderate evidence for meaningful benefit (LR = 4.2)"
```

### **Public Health Decision Making**
```python
# Environmental exposure study
# Question: Strong enough evidence for regulatory action?

exposure_cases = (45, 155, 23, 177)

# Scan multiple regulatory thresholds
thresholds = [1.5, 2.0, 2.5, 3.0]
evidence_scan = lr_threshold_scan(*exposure_cases, "RR", thresholds)

for threshold, result in evidence_scan.items():
    print(f"RR ≥ {threshold}: {result.strength_category} (LR = {result.lr:.1f})")
    
# Help decide: At what effect size do we have compelling evidence?
```

### **Meta-Analysis Prior Evidence**
```python
# Individual study from meta-analysis
# How much does this study contribute to evidence for meaningful effect?

study_data = (8, 42, 12, 38)

# Evidence for moderate protective effect
protective_evidence = lr_effect_evidence(
    *study_data,
    effect_type="RR", 
    meaningful_threshold=0.7,
    direction="less"
)

# Quantify study's contribution to overall evidence base
print(f"Study evidence weight: LR = {protective_evidence.lr:.2f}")
```

## Technical Challenges & Solutions

### **Numerical Integration Complexity**
**Challenge**: Composite hypotheses require integration over parameter regions
```python
# H1: 2.0 ≤ OR ≤ 5.0 requires ∫[2,5] L(θ) dθ
```

**Solution**: 
- Adaptive quadrature with error bounds
- Importance sampling for extreme parameter values
- Cached likelihood evaluations for performance

### **Likelihood Function Stability**
**Challenge**: Hypergeometric likelihoods can have numerical issues

**Solution**:
- All computations in log-space
- Use existing `log_nchg_pmf` infrastructure  
- Robust handling of boundary cases

### **User Interface Complexity**
**Challenge**: Balance flexibility with usability

**Solution**:
- Simple defaults for common use cases
- Advanced interface for power users
- Rich help system with epidemiological context

## Benefits & Impact

### **Methodological Advancement**
- **Beyond p-values**: Focus on effect size evidence
- **Clinical relevance**: Meaningful vs statistical significance  
- **Quantified evidence**: Interpretable likelihood ratios

### **Competitive Advantage**
- **Unique in Python**: No existing package provides this functionality
- **Research appeal**: Attracts serious epidemiologists and biostatisticians
- **Publication potential**: Novel methodology implementation

### **Educational Value**
- **Better statistics**: Teaches effect size thinking
- **Evidence-based medicine**: Supports clinical decision making
- **Research quality**: Promotes more thoughtful hypothesis testing

## Success Metrics

### **Quality Gates**
- [ ] **Numerical accuracy**: Match analytical results where available
- [ ] **Computational stability**: Robust handling of edge cases
- [ ] **Performance**: Reasonable computation time for interactive use
- [ ] **API usability**: Clear, intuitive interface with good defaults

### **Adoption Indicators**
- [ ] **User feedback**: Positive response from epidemiological community
- [ ] **Academic interest**: Citations in methodology papers
- [ ] **Real-world usage**: Applied in published research
- [ ] **Teaching adoption**: Used in biostatistics courses

## Risk Assessment

### **🟡 Medium Risk: Complexity**
- **Risk**: Feature too complex for typical users
- **Mitigation**: Excellent documentation, simple defaults, educational examples
- **Monitoring**: User feedback and support question frequency

### **🟢 Low Risk: Numerical Issues**  
- **Risk**: Integration failures or numerical instability
- **Mitigation**: Leverage existing robust mathematical infrastructure
- **Monitoring**: Comprehensive test suite with edge cases

### **🟡 Medium Risk: Interpretation Confusion**
- **Risk**: Users misinterpret likelihood ratios
- **Mitigation**: Clear interpretation strings, educational documentation
- **Monitoring**: Community feedback on documentation clarity

This feature would establish ExactCIs as the premier package for sophisticated epidemiological inference, moving the field beyond simple significance testing toward evidence-based effect size evaluation.