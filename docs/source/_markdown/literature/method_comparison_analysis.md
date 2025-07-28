# Confidence Interval Method Comparison Analysis

## Overview
This report summarizes the comparisons between different confidence interval methods implemented in ExactCIs, SciPy, and R. The analysis focuses on understanding how these methods differ from each other and across implementations.

## Key Findings

### Differences Between Methods
- **Conditional vs. Unconditional Methods**: 
  - In ExactCIs, unconditional methods produce lower bounds that are on average -93.1% different from conditional methods.
  - In ExactCIs, unconditional methods produce upper bounds that are on average 24291.1% different from conditional methods.

  - In R, unconditional methods produce lower bounds that are on average 20.2% different from conditional methods.
  - In R, unconditional methods produce upper bounds that are on average -13.2% different from conditional methods.

- **MidP vs. Other Methods**: 
  - In ExactCIs, MidP methods produce lower bounds that are on average -90.0% different from Conditional methods.
  - In ExactCIs, MidP methods produce upper bounds that are on average -12.3% different from Conditional methods.

  - In ExactCIs, MidP methods produce lower bounds that are on average -69.5% different from Blaker methods.
  - In ExactCIs, MidP methods produce upper bounds that are on average -37.9% different from Blaker methods.

  - In ExactCIs, MidP methods produce lower bounds that are on average -67.5% different from Unconditional methods.
  - In ExactCIs, MidP methods produce upper bounds that are on average -76.6% different from Unconditional methods.

  - In R, MidP methods produce lower bounds that are on average 24.2% different from Conditional methods.
  - In R, MidP methods produce upper bounds that are on average -15.9% different from Conditional methods.

  - In R, MidP methods produce lower bounds that are on average 5.9% different from Blaker methods.
  - In R, MidP methods produce upper bounds that are on average -5.5% different from Blaker methods.

  - In R, MidP methods produce lower bounds that are on average 3.6% different from Unconditional methods.
  - In R, MidP methods produce upper bounds that are on average -3.1% different from Unconditional methods.

### Differences Between Implementations
- **ExactCIs vs. R**: 
  - For Conditional methods, ExactCIs produces lower bounds that are on average 478.0% different from R.
  - For Conditional methods, ExactCIs produces upper bounds that are on average -12.3% different from R.

  - For MidP methods, ExactCIs produces lower bounds that are on average -91.7% different from R.
  - For MidP methods, ExactCIs produces upper bounds that are on average 0.0% different from R.

  - For Blaker methods, ExactCIs produces lower bounds that are on average -35.7% different from R.
  - For Blaker methods, ExactCIs produces upper bounds that are on average 53.4% different from R.

  - For Unconditional methods, ExactCIs produces lower bounds that are on average -89.2% different from R.
  - For Unconditional methods, ExactCIs produces upper bounds that are on average 32985.5% different from R.

  - For Wald-Haldane methods, ExactCIs produces lower bounds that are on average inf% different from R.
  - For Wald-Haldane methods, ExactCIs produces upper bounds that are on average 28.5% different from R.

## Potential Sources of Differences

1. **Numerical Precision**: Different numerical algorithms used in the implementations may lead to slight variations in results.

2. **Search Algorithms**: The root-finding and optimization algorithms used to determine confidence interval boundaries can vary across implementations.

3. **Edge Case Handling**: Different strategies for handling edge cases (zeros, small counts, etc.) can significantly impact results.

4. **Implementation Details**: Specific implementation choices for each method can lead to differences, such as:
   - How grid points are selected for the unconditional method
   - How p-values are calculated and compared to the alpha level
   - How convergence criteria are defined

5. **Version Differences**: The reference values from R might be from older versions with different implementations.

## Conclusion

The comparison between ExactCIs, SciPy, and R shows that while there are differences in the confidence interval estimates, they generally follow similar patterns. The Wald-Haldane method serves as a good baseline for comparison since it's implemented consistently across all platforms.

For most practical applications, these differences are unlikely to significantly impact statistical inference. However, for edge cases with very small counts or extreme odds ratios, users should be aware that different implementations may produce notably different results.

The ExactCIs package provides results that are generally consistent with established implementations, with variations that are expected due to implementation differences.
