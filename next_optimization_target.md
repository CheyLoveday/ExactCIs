# Next Optimization Target After Blaker Method

Based on the profiling results and code analysis, the **Conditional (Fisher's) Method** is the next slowest method after Blaker's, with an average execution time of 5.2ms.

## Key Bottlenecks

The specific bottlenecks in this method are the `fisher_lower_bound` and `fisher_upper_bound` functions, which both exhibit:
- Complex bracket expansion logic with nested loops
- Multiple fallback strategies
- 4 levels of nesting
- 7 parameters each
- Approximately 112-117 lines of code

## Performance Issues

According to the profiling data, the Conditional Method has "moderate issues" with performance:
- Average execution time: 5.2ms
- For very large tables: 19ms
- Root cause: "Conservative bracketing algorithms"

## Recommended Optimization Approaches

The optimization roadmap suggests several approaches that could be applied:
1. Smart root-finding initialization
2. Adaptive precision control
3. Memory-efficient support calculations

## Conclusion

The Conditional (Fisher's) Method, specifically the `fisher_lower_bound` and `fisher_upper_bound` functions, should be the next priority for optimization after Blaker's method.