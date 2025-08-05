# Confidence Intervals for 2x2 Tables

This table compares confidence interval methods across different implementations (ExactCIs, SciPy, and R) for various 2x2 tables.

| Case | a | b | c | d | OR | Conditional CI | MidP CI | Blaker CI | Unconditional CI | Wald-Haldane CI |
|---|---|---|---|---|---|---|---|---|---|---|
| README example | 12 | 5 | 8 | 10 | 3.0 | (0.614, 8.868) | (0.732, 12.604) | (0.72, 12.711) | (0.069, 3.0) | (0.727, 10.837) |
| 2x README example | 24 | 10 | 16 | 20 | 3.0 | (1.008, 6.946) | (1.11, 7.753) | (1.017, 8.257) | (0.121, 3.0) | (1.098, 7.657) |
| 0.5x README example | 6 | 2 | 4 | 5 | 3.75 | (0.337, 14.267) | (0.45, 35.708) | (0.424, 35.999) | (0.029, 3.75) | (0.467, 21.647) |
| Balanced OR=9 | 15 | 5 | 5 | 15 | 9.0 | (1.794, 26.254) | (2.222, 38.275) | (1.819, 46.201) | (0.022, 9.0) | (2.006, 31.443) |
| Balanced OR=1 | 10 | 10 | 10 | 10 | 1.0 | (0.244, 2.753) | (0.297, 3.37) | (0.278, 3.601) | (0.281, 3.556) | (0.298, 3.353) |
| Balanced OR=0.11 | 5 | 15 | 15 | 5 | 0.111 | (0.021, 1.0) | (0.026, 0.45) | (0.022, 0.55) | (0.111, 44.984) | (0.032, 0.498) |
| Minimal counts | 1 | 1 | 1 | 1 | 1.0 | (0.006, 4.557) | (0.014, 71.494) | (0.013, 76.249) | (0.013, 79.06) | (0.041, 24.565) |
| Zero in one cell | 0 | 5 | 8 | 10 | 0.0 | (0.0, 1.867) | (0.0, 1.275) | (0.0, 1.432) | (0.0, 0.0) | (0.005, 2.333) |
| Large imbalance | 50 | 5 | 2 | 20 | 100.0 | (15.439, 323.446) | (19.116, 659.319) | (99.0, 665.411) | (0.002, 100.0) | (15.492, 365.903) |
| Strong diagonal | 20 | 5 | 5 | 20 | 16.0 | (3.399, 45.216) | (4.15, 62.226) | (15.84, 16.0) | (0.017, 16.0) | (3.669, 52.598) |
| Strong anti-diagonal | 5 | 20 | 20 | 5 | 0.062 | (0.012, 1.0) | (0.016, 0.241) | (0.013, 0.062) | (0.062, 59.636) | (0.019, 0.273) |
| Row imbalance | 25 | 25 | 5 | 5 | 1.0 | (0.202, 2.989) | (0.241, 4.15) | (0.255, 3.927) | (0.212, 4.715) | (0.272, 3.682) |
| Column imbalance | 5 | 5 | 25 | 25 | 1.0 | (0.202, 2.989) | (0.241, 4.15) | (0.255, 3.927) | (0.212, 4.715) | (0.272, 3.682) |

## Full Comparison Data

For a complete comparison including SciPy and R implementations, see the CSV file `method_comparison.csv` in the project root directory.


## Interpretation

- **Conditional CI**: Fisher's exact test, conditions on marginal totals
- **MidP CI**: Mid-P adjusted Fisher's exact test, less conservative than Fisher's
- **Blaker CI**: Blaker's exact test, typically narrower than Fisher's
- **Unconditional CI**: Barnard's unconditional exact test, doesn't condition on margins
- **Wald-Haldane CI**: Normal approximation with Haldane correction

The confidence intervals show how different methods can produce different results for the same data. In general, unconditional methods are more conservative (wider intervals) than conditional methods, and exact methods are more reliable for small sample sizes than approximate methods.
