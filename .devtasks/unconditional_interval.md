Here is a **detailed breakdown of the mathematics and computational strategies** used for the unconditional exact confidence interval (CI) in the R `Exact` package, including key formulas and pseudocode reflecting the precise approach.

## 1. **Mathematics: Unconditional Exact CI in the Two-Sample Binomial Case**

Suppose you have two independent binomial samples:  
- Group 1: $$ n_1 $$ trials, $$ x_1 $$ successes, unknown probability $$ p_1 $$  
- Group 2: $$ n_2 $$ trials, $$ x_2 $$ successes, unknown probability $$ p_2 $$  
You're interested in a confidence interval for the difference in proportions, $$ \Delta = p_1 - p_2 $$.

### **Formulation**

- **Hypothesis to Invert:**  
  $$
  H_0: p_1 - p_2 = \Delta_0
  $$
  for candidate values $$ \Delta_0 $$ throughout some interval, e.g. $$[-1, 1]$$.

- **Test Statistic:**  
  Generally, a Z-type statistic is used:
  $$
  Z = \frac{(\hat{p}_1 - \hat{p}_2) - \Delta_0}{\sqrt{\frac{\tilde{p}_1(1-\tilde{p}_1)}{n_1} + \frac{\tilde{p}_2(1-\tilde{p}_2)}{n_2}}}
  $$
  where
  - $$ \tilde{p}_2 $$ is the nuisance parameter (varied),
  - $$ \tilde{p}_1 = \tilde{p}_2 + \Delta_0 $$,
  - $$ \hat{p}_1 = x_1 / n_1 $$, $$ \hat{p}_2 = x_2 / n_2 $$.

  Other orderings such as likelihood or CSM statistic may also be used.

- **Unconditional p-value:**  
  For each candidate value $$ \Delta_0 $$, and for each possible value of the nuisance parameter $$ p_2 $$:
  - Calculate probability of all tables that are as or more extreme (per the chosen ordering) as your observed table.
  - The **unconditional p-value** for $$ \Delta_0 $$ is the **supremum (maximum) over all $$ p_2 $$** in $$$$.

### **Confidence Interval Construction**

$$
CI = \{\, \Delta_0 : \sup_{p_2 \in } P(\text{observed or more extreme table} \mid p_1 = p_2 + \Delta_0, \, p_2) \geq \alpha \, \}
$$
where $$ \alpha $$ is your significance level (e.g., 0.05).

## 2. **Computational Strategy (as in R’s `Exact` Package)**

### **Step-by-step Algorithm**

1. **Grid over candidate differences ($$ \Delta_0 $$)**:  
   Create a fine grid of possible difference values over which to invert the test.

2. **For each candidate $$ \Delta_0 $$:**
   - **Grid over the nuisance parameter ($$ p_2 $$)**:  
     Use a grid (default: 100 points, adjustable) to discretize $$ p_2 $$ over .
   - For each $$ p_2 $$:
     - **Enumerate all possible (k1, k2) tables**:  
       For $$ k_1 $$ in $$ 0,...,n_1 $$; $$ k_2 $$ in $$ 0,...,n_2 $$.
     - **Compute test statistic** (order definition) for each table.
     - **Identify more extreme tables** relative to the observed statistic (by chosen ordering: e.g., Z-statistic, CSM, Boschloo, likelihood).
     - **Sum probabilities** (using binomial likelihood) over these more extreme tables.
   - **Find supremum (maximum) of p-value over all $$ p_2 $$**.
3. **Include $$ \Delta_0 $$ in CI if maximized p-value $$\geq \alpha$$.**
4. **CI boundaries:**  
   The smallest and largest $$ \Delta_0 $$ values included define the confidence interval.

### **Algorithmic Optimization Used in R**

- **Convexity assumption**: For many orderings, the rejection region is convex, enabling speedup.
- **Stored ordering and matrices**: For “CSM” (Barnard’s ordering), speeds up repeatedly checking which tables are more extreme.
- **Refined maximization**: If the maximum p-value over the grid is close to $$\alpha$$, an optimizer (e.g., R’s `optimize()`) hones in on the precise maximum.
- **Parallelization**: Optionally used for large sample sizes.

## 3. **Pseudocode**

Here’s high-level pseudocode reflecting R’s implementation approach:

```r
for each candidate_diff in diff_grid:  # e.g., -1 to 1 in 0.001 steps
  max_pvalue = 0
  for each p2 in seq(0, 1, length=npNumbers):
    p1 = candidate_diff + p2
    if p1 < 0 or p1 > 1:
      next
    # For all possible (k1 in 0..n1, k2 in 0..n2)
    for k1 in 0:n1:
      for k2 in 0:n2:
        # Compute test statistic for (k1, k2)
        stat = calc_statistic(k1, n1, k2, n2, p1, p2, method="CSM")  # or "z-pooled" etc
    # Identify which tables are as or more extreme as observed table
    # Sum binomial probabilities for those tables to get p-value
    observed_pvalue = sum_probs_of_extreme_tables(...)
    if observed_pvalue > max_pvalue:
      max_pvalue = observed_pvalue
  if max_pvalue >= alpha:
    include candidate_diff in CI
# CI = set of all candidate_diff values included
```

## 4. **Summary Table**

| Step                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Discretize $$ \Delta_0 $$| Fine grid of possible differences to test                                   |
| Discretize $$ p_2 $$     | Grid (npNumbers) of values from , efficiently handled                  |
| Enumerate tables         | All (k1, k2) tables for given (n1, n2)                                     |
| Test statistics/orderings| CSM, z-pooled, Boschloo, likelihood                                         |
| Sum probabilities        | Over tables as extreme or more extreme for each candidate value             |
| Maximize over nuisance   | The p-value is the largest found across all $$ p_2 $$                       |
| Invert test              | All $$ \Delta_0 $$ where p-value ≥ alpha are included in confidence interval|
| Refinements/optimizations| Convexity, stored matrices, and precise optimization near maxima             |

**In summary:**  
R’s implementation systematically explores possible effect sizes and nuisance parameters, computing the probability (under binomial models) of observing results as extreme as the data across all possible 2x2 tables, maximizes p-value over the nuisance parameter for each candidate effect size, and collects the interval of all those effect sizes not rejected by the data. Optimizations such as convexity and stored orderings are used for efficiency.

Sources
