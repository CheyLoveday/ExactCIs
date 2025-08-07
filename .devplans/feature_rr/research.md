# Comprehensive Summary: Confidence Intervals for Relative Risk in Genetic Epidemiological Studies

This summary consolidates the mathematical foundations, implementation methods, and practical guidance for computing confidence intervals for relative risk in biomedical and genetic epidemiological research.

## Core Methods Overview

### **Score-Based Methods** (Recommended)
- **Asymptotic Score Interval**: Most widely recommended method that inverts the score test statistic
- **Modified Score Interval with Continuity Correction**: Enhanced version for small samples with correction parameter δ
- **Theoretical Equivalence**: Proven mathematically equivalent to Nam-Blackwelder's constrained MLE approach

### **Nonparametric Methods**
- **U-Statistic-Based Interval**: Robust rank-based method with no distributional assumptions
- **Conservative t-Distribution**: Uses n-1 degrees of freedom for improved small-sample coverage

### **Classical Methods**
- **Wald Intervals**: Simple normal-theory approach on log scale
- **Independent Proportions**: Katz method for case-control designs  
- **Correlated Proportions**: Modified variance accounting for covariance structure

## Mathematical Implementation Framework

### Score Interval Foundation
For a 2×2 contingency table, the score test statistic under H₀: θ_RR = θ₀ is:

$$S(\theta_0) = \frac{(x_{11} + x_{12}) - (x_{11} + x_{21})\theta_0}{\sqrt{n(1+\theta_0)\tilde{p}_{21} + (x_{11} + x_{12} + x_{21})(\theta_0 - 1)}}$$

**Key Implementation Steps:**
1. **Constrained MLE**: Compute p̃₂₁ using quadratic formula solution
2. **Quartic Equations**: Confidence bounds require solving quartic equations via Ferrari's method
3. **Root Selection**: Choose biologically meaningful positive roots

### Continuity-Corrected Score Interval
Enhanced score statistic with correction strength δ:

$$S_{\delta}(\theta_0) = \frac{|(x_{11} + x_{12}) - (x_{11} + x_{21})\theta_0| - \frac{1}{\delta n}(x_{11} + x_{21})}{\text{denominator}}$$

**Correction Levels:**
- **δ = 2**: High correction (ASCC-H)
- **δ = 4**: Medium correction (ASCC-M) - **recommended**
- **δ = 8**: Low correction (ASCC-L)

### U-Statistic Nonparametric Method
**Variance of log RR** using rank-based estimation:
$$\widehat{V}_{LRR} = \frac{1}{n^2(n-1)\widehat{p}_1^2\widehat{p}_2^2} \times \text{[complex variance expression]}$$

**Conservative CI**: Uses t-distribution with n-1 degrees of freedom

### Wald Method Implementations

#### Independent Proportions (Katz Method)
$$\operatorname{Var}(\log\widehat{\theta}_{RR}) = \frac{1-p_1}{x_{11}} + \frac{1-p_2}{x_{21}}$$

#### Correlated Proportions
$$\operatorname{Var}(\log\widehat{\theta}_{RR}) = \frac{\widehat{\operatorname{Var}}(\hat{p}_1)}{\hat{p}_1^2} + \frac{\widehat{\operatorname{Var}}(\hat{p}_2)}{\hat{p}_2^2} - 2\frac{\widehat{\operatorname{Cov}}(\hat{p}_1,\hat{p}_2)}{\hat{p}_1\hat{p}_2}$$

**Critical Covariance Term**: $\widehat{\operatorname{Cov}}(\hat{p}_1,\hat{p}_2) = \frac{x_{11}x_{22}-x_{12}x_{21}}{n(n-1)}$

## Implementation Pseudocode Framework

### Complete Function Library
```
// Score-based methods
function asymptotic_score_interval(x11, x12, x21, x22, alpha)
function modified_score_interval(x11, x12, x21, x22, delta, alpha)
function ferrari_quartic_solver(coefficients)

// Nonparametric method  
function u_statistic_interval(x11, x12, x21, x22, alpha)

// Wald methods
function wald_rr_independent(x11, x12, x21, x22, alpha)
function wald_rr_correlated(x11, x12, x21, x22, alpha)

// Utility functions
function constrained_mle_p21(x11, x12, x21, x22, theta0)
function normal_quantile(p)
function t_quantile(df, p)
```

## Method Selection Guidelines

### Sample Size Recommendations
| Sample Size | Recommended Method | Rationale |
|-------------|-------------------|-----------|
| **n < 50** | U-statistic nonparametric | Robust coverage without distributional assumptions |
| **50 ≤ n < 200** | ASCC-M (δ = 4) | Optimal balance of accuracy and coverage |
| **n ≥ 200** | Original score method | Asymptotic properties adequate |

### Study Design Considerations
- **Case-Control Studies**: Use independent proportion methods
- **Family-Based Studies**: Always use correlated methods (twin studies, sibling pairs)
- **Matched Designs**: Correlated proportion framework required
- **Population Stratification**: Apply Mantel-Haenszel pooling across genetic ancestry strata

### Performance Characteristics
- **Score-based intervals**: Superior coverage, especially with increasing correlation
- **Continuity-corrected score**: Best overall performance for moderate samples
- **Wald intervals**: Simple implementation but poor small-sample performance
- **U-statistic method**: Most robust but potentially wider intervals

## Practical Implementation Notes

### Critical Implementation Details
1. **Cell Count Handling**: Check for zero cells; add 0.5 continuity correction if needed
2. **Numerical Stability**: Use log-scale computations to avoid overflow
3. **Root Selection**: Ensure biologically meaningful positive values
4. **Correlation Structure**: Always account for family/matching structure in genetic studies

### Common Applications in Genetic Epidemiology
- **SNP Association Studies**: Compare disease risk between genotype groups
- **Family Segregation Analysis**: Assess genetic risk within families
- **Twin Studies**: Control for shared genetic/environmental factors  
- **Linkage Analysis**: Account for correlation due to linkage disequilibrium

### Software Integration
- **R Packages**: `epiR`, `pairedPrev`, custom implementations
- **SAS Procedures**: `PROC FREQ` with `RELRISK` and `AGREE` options
- **Python Libraries**: `scipy.stats`, `statsmodels` with custom extensions

## Key Theoretical Results

### Equivalence Theorem
**Proven mathematical equivalence** between:
- Tang's asymptotic score interval using S(θ)
- Nam-Blackwelder's constrained MLE interval using Fieller-type statistic T(θ)

This **unifies two major methodological approaches** and provides practitioners flexibility in method choice without loss of statistical rigor.

### Coverage Properties
- **Score-based methods**: Maintain nominal coverage across correlation structures
- **Continuity corrections**: Improve small-sample coverage properties
- **Wald methods**: Coverage deteriorates with small samples or extreme proportions

## Conclusion

The **ASCC-M method** (medium continuity correction, δ = 4) represents the **optimal choice** for most genetic epidemiological studies with moderate sample sizes, providing robust coverage and computational efficiency. For small samples or when distributional assumptions are questionable, the **U-statistic nonparametric method** offers the most reliable coverage properties. The **comprehensive implementation framework** provided enables researchers to select and implement the most appropriate method based on study design, sample size, and correlation structure considerations.

Sources
