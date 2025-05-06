# References & Citations

This document provides a comprehensive list of academic references related to the methods implemented in ExactCIs, as well as instructions on how to cite the package.

## Citing ExactCIs

If you use ExactCIs in your research, please cite it as follows:

```
@software{exactcis,
  author       = {ExactCIs Contributors},
  title        = {ExactCIs: A Python Package for Exact Confidence Intervals},
  year         = {2023},
  publisher    = {GitHub},
  url          = {https://github.com/username/exactcis}
}
```

## Academic References

### Theoretical Foundation

1. Barnard, G. A. (1945). A New Test for 2×2 Tables. *Nature*, 156(3954), 177.
   
   *The original paper introducing the unconditional exact test for 2×2 tables.*

2. Barnard, G. A. (1947). Significance Tests for 2×2 Tables. *Biometrika*, 34(1/2), 123-138.
   
   *Detailed exposition of Barnard's test, including computational considerations.*

3. Boschloo, R. D. (1970). Raised Conditional Level of Significance for the 2×2-table when Testing the Equality of Two Probabilities. *Statistica Neerlandica*, 24(1), 1-35.
   
   *Description of Boschloo's test, a modification of Fisher's exact test that typically has greater power.*

4. Suissa, S., & Shuster, J. J. (1985). Exact Unconditional Sample Sizes for the 2×2 Binomial Trial. *Journal of the Royal Statistical Society. Series A (General)*, 148(4), 317-327.
   
   *Discussion of exact unconditional methods for sample size determination.*

5. Agresti, A., & Min, Y. (2001). On Small-Sample Confidence Intervals for Parameters in Discrete Distributions. *Biometrics*, 57(3), 963-971.
   
   *Comprehensive discussion of methods for constructing confidence intervals for discrete data.*

### Computational Methods

6. Mehta, C. R., & Patel, N. R. (1983). A Network Algorithm for Performing Fisher's Exact Test in r×c Contingency Tables. *Journal of the American Statistical Association*, 78(382), 427-434.
   
   *Efficient algorithm for computing Fisher's exact test p-values.*

7. Silva-Maia, J. & Jara, A. & Sánchez, L. (2020). Optimal p-values for exact unconditional tests. *Austrian Journal of Statistics*, 49, 1-19.
   
   *Efficient computational approaches for exact unconditional tests.*

8. Fagerland, M. W., Lydersen, S., & Laake, P. (2013). The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. *BMC Medical Research Methodology*, 13(1), 91.
   
   *Discussion of mid-p and exact methods for paired binary data.*

### Method Comparisons

9. Lydersen, S., Fagerland, M. W., & Laake, P. (2009). Recommended tests for association in 2×2 tables. *Statistics in Medicine*, 28(7), 1159-1175.
   
   *Comprehensive comparison of methods for 2×2 tables, including recommendations.*

10. Crans, G. G., & Shuster, J. J. (2008). How conservative is Fisher's exact test? A quantitative evaluation of the two-sample comparative binomial trial. *Statistics in Medicine*, 27(18), 3598-3611.
    
    *Evaluates Fisher's exact test compared to unconditional approaches.*

11. Agresti, A. (2001). Exact inference for categorical data: recent advances and continuing controversies. *Statistics in Medicine*, 20(17-18), 2709-2722.
    
    *Discussion of exact methods for categorical data, including 2×2 tables.*

12. Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions*. John Wiley & Sons.
    
    *Comprehensive reference on statistical methods for categorical data.*

### Rare Events

13. Grizzle, J. E. (1967). Continuity Correction in the χ² Test for 2×2 Tables. *The American Statistician*, 21(4), 28-32.
    
    *Discussion of continuity corrections for chi-square tests.*

14. Hirji, K. F. (2006). *Exact Analysis of Discrete Data*. Chapman and Hall/CRC.
    
    *Comprehensive text on exact methods for discrete data, including rare events.*

15. Bradburn, M. J., Deeks, J. J., Berlin, J. A., & Russell Localio, A. (2007). Much ado about nothing: a comparison of the performance of meta‐analytical methods with rare events. *Statistics in Medicine*, 26(1), 53-77.
    
    *Comparison of methods for meta-analysis with rare events.*

16. Greenland, S., & Robins, J. M. (1985). Estimation of a common effect parameter from sparse follow-up data. *Biometrics*, 41(1), 55-68.
    
    *Discussion of methods for estimating effect parameters with sparse data.*

### Implementation Considerations

17. Agresti, A., & Coull, B. A. (1998). Approximate is better than "exact" for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119-126.
    
    *Discussion of the advantages of approximate methods in certain contexts.*

18. Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101-133.
    
    *Comprehensive discussion of methods for interval estimation of binomial proportions.*

19. Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion: comparison of seven methods. *Statistics in Medicine*, 17(8), 857-872.
    
    *Comparison of methods for confidence intervals for proportions.*

20. Reiczigel, J., Földi, J., & Ózsvári, L. (2010). Exact confidence limits for prevalence of a disease with an imperfect diagnostic test. *Epidemiology & Infection*, 138(11), 1674-1678.
    
    *Discussion of confidence limits with imperfect diagnostic tests.*

## Related Software

1. R's `exact2x2` Package:
   Fay, M. P. (2010). Two-sided exact tests and matching confidence intervals for discrete data. *R Journal*, 2(1), 53-58.

2. StatXact:
   Mehta, C. R., & Patel, N. R. (1995). *StatXact 3 for Windows: Statistical Software for Exact Nonparametric Inference*. Cytel Software Corporation.

3. SAS PROC FREQ:
   SAS Institute Inc. (2014). SAS/STAT® 13.2 User's Guide: The FREQ Procedure. SAS Institute Inc.

4. NumPy and SciPy:
   Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

## Additional Reading

For further reading on exact methods for categorical data, we recommend:

- Agresti, A. (2013). *Categorical Data Analysis*. John Wiley & Sons.
- Altman, D. G. (1991). *Practical Statistics for Medical Research*. Chapman and Hall/CRC.
- Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology*. Lippincott Williams & Wilkins.
- Breslow, N. E., & Day, N. E. (1980). *Statistical Methods in Cancer Research: Volume I - The Analysis of Case-Control Studies*. International Agency for Research on Cancer.

## Online Resources

- [Exact Confidence Intervals for Proportions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1120692/)
- [PASS Sample Size Software](https://www.ncss.com/software/pass/)
- [OpenEpi](https://www.openepi.com) - Open Source Epidemiologic Statistics for Public Health
