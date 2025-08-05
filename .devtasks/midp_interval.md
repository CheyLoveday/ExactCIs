Here are the **mathematics and R implementation strategies for the mid-p method** for exact inference, particularly as applicable to 2x2 tables and the approaches available in R’s `Exact` package.

## 1. **Mathematical Definition of Mid-p**

- The **mid-p method** is a modification of the classical exact p-value that aims to reduce conservatism (the tendency of exact methods to produce intervals that are too wide).
- If $$ H_0 $$ is your null hypothesis and the distribution of your test statistic $$ T $$ is discrete, then for an observed value $$ t_{\text{obs}} $$:
  $$
  \text{mid-}p = P(T < t_{\text{obs}} | H_0) + 0.5 \cdot P(T = t_{\text{obs}} | H_0)
  $$
- For a two-sided alternative, the mid-p value sums half the probability of tables equally as or more extreme as observed, and the full probability of tables strictly more extreme.

## 2. **Applying Mid-p to Binomial Proportions and 2x2 Tables**

- In the context of **exact unconditional tests** for two independent binomial proportions:
  - Determine all possible tables and their p-values under the null (for each value of the nuisance parameter).
  - For the observed table, the classical exact p-value sums the probabilities of all tables as or more extreme than observed.
  - The mid-p adjusts this sum by only adding half the probability of the observed table (i.e., the so-called "mid" adjustment).

**Mathematically:**
- Let $$ \Omega $$ be the set of all tables (k1, k2) with test statistic at least as extreme as the observed.
- Let $$ P_{\text{null}}(k_1, k_2) $$ be the null probability of the table.
- The **mid-p** is:
  $$
  \text{mid-}p = \sum_{(k_1, k_2) \in \Omega^*} P_{\text{null}}(k_1, k_2) + 0.5 \cdot P_{\text{null}}(\text{Observed table})
  $$
  where $$ \Omega^* $$ is the set of strictly more extreme tables.

## 3. **R’s Implementation Strategy for Mid-p in the `Exact` Package**

### **a. Table Enumeration and Test Statistic**

- For given data and method (e.g., `"z-pooled"`, `"boschloo"`, `"csm"`), enumerate all possible 2x2 tables for the given marginal totals.
- For each table, compute the chosen test statistic under the null hypothesis and nuisance parameter.

### **b. Calculation of the Mid-p Value**

- For each value of the nuisance parameter (e.g., the common success probability under the null), calculate the probabilities of all possible tables.
- For the observed table:
  - **Classical exact p-value:** sum probabilities of all tables with statistic as or more extreme.
  - **Mid-p**: sum:
    - **Half the probability** of the observed table.
    - **Full probability** of all strictly more extreme tables.

- For confidence interval construction and hypothesis inversion, this mid-p value is maximized over the grid of nuisance parameter values, just as in the classical method.

### **c. Confidence Interval Construction**

- Invert the test: The confidence interval consists of those values of the parameter (e.g., difference in proportions) that would **not be rejected** according to the mid-p test (i.e., mid-p value $$\geq \alpha$$ for the chosen significance level).

## 4. **Pseudocode Outline for Mid-p Method**

```r
for each candidate_diff in diff_grid:  # e.g., -1 to 1 in 0.001 steps
  max_mid_p = 0
  for each p2 in seq(0, 1, length=npNumbers):
    p1 = candidate_diff + p2
    if p1 < 0 or p1 > 1:
      next
    for all (k1 in 0...n1, k2 in 0...n2):
      stat = calc_statistic(k1, n1, k2, n2, p1, p2, method)
    get tables strictly more extreme than observed: S
    mid_p = sum(P(k1, k2) for (k1, k2) in S) + 0.5 * P(observed table)
    if mid_p > max_mid_p:
      max_mid_p = mid_p
  if max_mid_p >= alpha:
    include candidate_diff in CI
```

- The only difference with the classical approach is in the calculation of the **mid-p**: instead of summing the full probability of the observed table, only half is added.

## 5. **Example of R Usage**

- In R, specifying `midp=TRUE` (in the right context or package, e.g., some exact test functions in packages like `exact2x2` or setting option in `Exact`) computes the mid-p interval.
- The implementation in `Exact` package for unconditional tests supports this option for the main test types: `"z-pooled"`, `"z-unpooled"`, `"boschloo"`, `"santner and snell"`, `"csm"`.

## 6. **Summary Table**

| Step                        | Classical      | Mid-p Modification                               |
|-----------------------------|---------------|--------------------------------------------------|
| Table inclusion criterion   | As or more extreme | Strictly more extreme + half observed       |
| Sum probability             | Full          | 0.5 × observed + all strictly more extreme       |
| CI construction             | Invert test   | Invert test using mid-p instead of exact p       |

**In summary:**  
The **mid-p method** produces a less conservative p-value and confidence interval by only assigning half probability to the observed table, while otherwise following the same enumeration and maximization strategy as other exact tests. In R’s `Exact` package, all test strategies (ordering methods) may implement the mid-p option by summing over strictly more extreme tables and then adding half the observed probability when calculating p-values on each grid value of the nuisance parameter. This mid-p value is maximized (for each candidate parameter value) over the nuisance grid, and those parameter values not rejected are included in the confidence interval. The pseudocode highlights the only real difference: how the observed-probability table is handled in the sum.

Sources
