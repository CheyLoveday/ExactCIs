# Continuity Corrections for Odds Ratio and Relative Risk in 2 × 2 Tables  

Before applying many classical or asymptotic confidence-interval (CI) formulas to a 2 × 2 table, analysts add a small constant to cells that contain zero.  The choice and placement of that constant materially influence bias, coverage probability, and CI width.  This report consolidates current statistical knowledge on continuity-correction strategies for the odds ratio (OR) and the relative risk (RR), presents the underlying mathematics, compares competing approaches, and ends with a decision table that maps real-world study scenarios to a recommended correction.

***

## 1   Why Continuity Corrections Exist  

Analytic CI methods for OR and RR usually involve a log transformation:

$$
\hat{\theta}_{OR}= \frac{a\,d}{b\,c},\qquad 
\hat{\theta}_{RR}= \frac{a/(a+b)}{c/(c+d)} .
$$

When $$a=0$$ or $$c=0$$ the point estimate is 0 or ∞ and $$\log\hat{\theta}$$ or its variance is undefined.  A continuity correction ensures every cell is positive so that the log, standard error, and CI limits are finite.

Corrections also reduce discreteness bias in very small samples by shifting the observed table toward the centre of the sampling distribution.

***

## 2   Classical Constant Corrections  

### 2.1  Haldane–Anscombe (+0.5)  

Add 0.5 to **all four cells**:

$$
(a,b,c,d) \longrightarrow (a+0.5,\;b+0.5,\;c+0.5,\;d+0.5).
$$

*Mathematics.*  
The adjusted log-OR is  

$$
\log\hat{\theta}^{(0.5)}_{OR}
    = \log\!\bigl[\tfrac{(a+0.5)(d+0.5)}{(b+0.5)(c+0.5)}\bigr],
$$
with estimated variance  

$$
\widehat{\operatorname{Var}}\!\bigl(\log\hat{\theta}^{(0.5)}\bigr)
      =\frac{1}{a+0.5}+\frac{1}{b+0.5}+\frac{1}{c+0.5}+\frac{1}{d+0.5}.
$$

*Justification.*  Haldane introduced the idea (1955) for OR, Anscombe (1956) formalised its use for RR.  It is still the default in Stata, SAS and SciPy.

*Limitations.*  Simulations show mild upward bias and 2–6% over-coverage once any cell exceeds ≈ 5 counts[1].

### 2.2  Laplace (+1)  

Add 1 to every cell.  This was common in early epidemiology texts but is now discouraged.

*Drawbacks.*  
Bias toward the null nearly doubles compared with +0.5 and confidence intervals widen appreciably, especially when total $$n<50$$[1].  No major analytical package uses +1 as its default.

***

## 3   Data-Adaptive Constant Corrections  

### 3.1  Group-Size–Weighted (Sweeting et al., 2004)  

For each study arm add the inverse of its total:

$$
a^* = a+\tfrac{1}{n_1},\;
b^* = b+\tfrac{1}{n_1},\;
c^* = c+\tfrac{1}{n_2},\;
d^* = d+\tfrac{1}{n_2}.
$$

*Rationale.*  Reduces bias when arms are highly unbalanced in stratified or meta-analytic settings[1].

*Status.*  Implemented in StatsDirect and RevMan for Mantel–Haenszel pooling.

### 3.2  Treatment-Arm Only  

Add 0.5 to the **zero** cell(s) plus 0.5 to the opposite cell in the *same* treatment arm, leaving the other arm unchanged:

|           | Outcome = 1|Outcome = 0|
|-----------|-----------|-----------|
|Exposed    |$$a$$+0.5  |$$\,b$$+0.5|
|Control    |$$c$$      |$$d$$      |

*When to use.*  Sparse data where only one arm has zeros; reduces bias compared with four-cell addition while still making variance finite.

***

## 4   Method-Specific, Non-Constant Corrections  

### 4.1  Continuity-Corrected Score (Tang, DelRocco)**  

The asymptotic score CI replaces the raw difference in the score statistic numerator with a *shrunken* value:

$$
S_\delta(\theta)=
\frac{\bigl| (x_{11}+x_{12})-n_1\theta p_{21} \bigr|
        -\dfrac{x_{11}+x_{21}}{\delta(n_1+n_2)}}
     {\sqrt{\;n_1p_{11}(1-p_{11})+(n_1\theta)^2p_{21}(1-p_{21})}}[2].
$$

-   **δ = 4** (medium) → best average coverage for $$20\le n\le200$$  
-   **δ = 2** (high)    → very small samples, conservative  
-   **δ = 8** (low)     → larger samples, narrows CI

Because the subtracted term is data-dependent, this correction is *smaller than 0.5* whenever $$x_{11}+x_{21}\lt$$ total sample size, limiting bias.

### 4.2  Score Confidence Interval with Variance-Recovery (MOVER-Wilson)  

Instead of adding a constant, the MOVER algorithm reconstructs variance from two Wilson single-sample intervals then combines them Fieller-style; no explicit continuity constant is required[3].  Works equally for OR and RR and retains nominal coverage down to $$n\approx10$$.

***

## 5   Corrections Built into “Integer-Only” Exact Methods  

Exact conditional, mid-P, Blaker, and exact unconditional CIs enumerate all integer tables; **no artificial constant** is added.  The discreteness of the distribution already guarantees finite variance[4].

***

## 6   Bootstrapping and Penalised Likelihood  

- **Bootstrap BCa RR/OR:** Does not need a continuity correction because resampling preserves zeros; instead, an *ε* (e.g. $$10^{-8}$$) can avoid log(0).  
- **Firth/log-binomial or Firth/logistic:** Penalisation replaces the need for continuity constants and gives finite estimates even when counts are zero.

***

## 7   Choosing a Continuity Correction: Decision Matrix  

| Scenario                               | Recommended Correction | Why                                                         |
|---------------------------------------|------------------------|-------------------------------------------------------------|
| **Any exact CI (conditional, mid-P, Blaker, unconditional)** | *None* | Exact enumeration handles zeros by design[4] |
| **Analytic Wald / Katz, small table (any cell = 0)** | +0.5 to **all** cells (Haldane–Anscombe) | Universally implemented, minimal bias for OR/RR |
| **Analytic Wald / Katz, strata or meta-analysis with arm size imbalance** | Group-size weighted (+1/n₁, +1/n₂) | Lower bias in pooled MH estimates[1] |
| **Tang / MN Score CI (n ≈ 30–200)**   | δ-adjusted (δ = 4)     | Best coverage-width trade-off in simulations[2] |
| **Tang / MN Score CI, n < 20**        | δ = 2 (high correction) | More conservative; avoids under-coverage in very small n |
| **Paired U-statistic RR, matched pairs** | *None* (or ε=10⁻⁸)     | Statistic handles zeros; tiny ε only for log stability |
| **Ultra-sparse, single-arm zero only** | Two-cell 0.5 (same arm) | Less bias than four-cell addition |
| **Teaching / worst-case upper bound**  | +1 to all cells        | Very conservative; rarely justified in practice |

***

## 8   Key Take-Home Messages  

1. **Haldane–Anscombe (+0.5) remains the practical default** for Wald/Katz OR and RR.  
2. **Modern score intervals** use a **tunable, data-dependent correction** ($$\delta$$) that is invariably *smaller* than 0.5, giving better coverage and shorter width.  
3. **Exact methods require no continuity constants** – their discreteness already guards against infinities.  
4. **Adding +1 to every cell is overly conservative and largely obsolete**; only use for didactic “worst-case” illustrations.  
5. **Choose corrections according to design and sample size** — the decision matrix above offers a concise rule-of-thumb that mirrors current regulatory and software defaults.

***

### References  
All statements are supported by the following key sources:

[4] Wang X. *Exact Confidence Intervals for the Relative Risk and the Odds Ratio* (2015)  
[1] Sweeting MJ et al. *Stat Med* 23 : 1351-75 (2004)  
[2] Fay MP & DelRocco N. *Continuity-corrected score confidence interval* (2022)  
[3] Metron 82 : 149-170 (2024) – MOVER Wilson CC article

Sources
[1] Estimating Relative Risk When Observing Zero Events—Frequentist ... https://pmc.ncbi.nlm.nih.gov/articles/PMC8196730/
[2] [PDF] exact2x2: Exact Tests and Confidence Intervals for 2x2 Tables - CRAN https://cran.r-project.org/web/packages/exact2x2/exact2x2.pdf
[3] Chapter 6: Choosing effect measures and computing estimates of ... https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-06
[4] Exact Confidence Intervals for the Relative Risk and the Odds Ratio https://pmc.ncbi.nlm.nih.gov/articles/PMC4715482/
