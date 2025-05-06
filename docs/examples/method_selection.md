# Method Selection Guide

This guide provides a structured approach to selecting the most appropriate confidence interval method for your specific scenario.

## Decision Flowchart

Use the following decision flowchart to guide your method selection:

```
Start
  │
  ├─ Is sample size very large (all cells > 10)?
  │   │
  │   ├─ Yes → Is computational speed critical?
  │   │         │
  │   │         ├─ Yes → Use Normal Approximation
  │   │         │
  │   │         └─ No  → Do you need exact methods for protocol adherence?
  │   │                   │
  │   │                   ├─ Yes → Use Barnard's Unconditional (ExactCIs)
  │   │                   │
  │   │                   └─ No  → Use Normal Approximation
  │   │
  │   └─ No  → Are any cells less than 5?
  │             │
  │             ├─ Yes → Are margins fixed by design?
  │             │         │
  │             │         ├─ Yes → Use Fisher's Exact Test
  │             │         │
  │             │         └─ No  → Use Barnard's Unconditional (ExactCIs)
  │             │
  │             └─ No  → Are you dealing with rare events (rate < 1%)?
  │                       │
  │                       ├─ Yes → Use Barnard's Unconditional (ExactCIs)
  │                       │
  │                       └─ No  → Are margins fixed by design?
  │                                 │
  │                                 ├─ Yes → Use Fisher's Exact Test
  │                                 │
  │                                 └─ No  → Use Barnard's Unconditional (ExactCIs)
```

## Interactive Method Selector

The following Python function can help you select the appropriate method based on your dataset characteristics:

```python
def recommend_ci_method(a, b, c, d, fixed_margins=False, need_exact=False, speed_critical=False):
    """
    Recommends the most appropriate confidence interval method.
    
    Parameters:
    - a, b, c, d: Counts in the 2×2 table
    - fixed_margins: Whether margins are fixed by design
    - need_exact: Whether exact methods are required by protocol
    - speed_critical: Whether computational speed is critical
    
    Returns:
    - Recommended method(s) and reasoning
    """
    min_count = min(a, b, c, d)
    total_count = a + b + c + d
    
    # Calculate event rates
    rate1 = a / (a + b) if (a + b) > 0 else 0
    rate2 = c / (c + d) if (c + d) > 0 else 0
    
    # Check for zero cells
    has_zero = min_count == 0
    
    # Check for rare events (less than 1% in either group)
    rare_events = rate1 < 0.01 or rate2 < 0.01
    
    # Check if all cells are large
    all_large = min_count >= 10
    
    # Generate recommendation
    methods = []
    reasons = []
    
    if all_large and speed_critical and not need_exact:
        methods.append("Normal Approximation")
        reasons.append("All cells are large (≥10) and computation speed is prioritized")
    elif has_zero:
        methods.append("Barnard's Unconditional (ExactCIs)")
        reasons.append("Table contains zero cell(s), which requires careful handling")
        if fixed_margins:
            methods.append("Fisher's Exact Test")
            reasons.append("Margins are fixed by design (secondary recommendation)")
    elif rare_events:
        methods.append("Barnard's Unconditional (ExactCIs)")
        reasons.append("Rare events present (<1%), requiring exact unconditional methods")
    elif min_count < 5:
        if fixed_margins:
            methods.append("Fisher's Exact Test")
            reasons.append("Small cell counts (<5) with fixed margins")
        else:
            methods.append("Barnard's Unconditional (ExactCIs)")
            reasons.append("Small cell counts (<5) without fixed margins")
    elif need_exact:
        methods.append("Barnard's Unconditional (ExactCIs)")
        reasons.append("Exact methods required by protocol")
    elif all_large:
        methods.append("Normal Approximation")
        reasons.append("All cells are large (≥10), making asymptotic methods appropriate")
    else:
        if fixed_margins:
            methods.append("Fisher's Exact Test")
            reasons.append("Moderate sample size with fixed margins")
        else:
            methods.append("Barnard's Unconditional (ExactCIs)")
            reasons.append("Moderate sample size without fixed margins")
    
    return {
        "primary_method": methods[0],
        "alternative_methods": methods[1:] if len(methods) > 1 else None,
        "reasons": reasons
    }

# Example usage
tables = [
    {"name": "Standard table", "counts": (7, 3, 2, 8)},
    {"name": "Large table", "counts": (50, 50, 50, 50)},
    {"name": "Rare events", "counts": (1, 999, 10, 990)},
    {"name": "Zero cell", "counts": (0, 100, 10, 90)}
]

for table in tables:
    a, b, c, d = table["counts"]
    print(f"\n{table['name']} ({a}, {b}, {c}, {d}):")
    
    rec1 = recommend_ci_method(a, b, c, d, fixed_margins=False)
    print(f"- Without fixed margins: {rec1['primary_method']}")
    print(f"  Reason: {rec1['reasons'][0]}")
    
    rec2 = recommend_ci_method(a, b, c, d, fixed_margins=True)
    print(f"- With fixed margins: {rec2['primary_method']}")
    print(f"  Reason: {rec2['reasons'][0]}")
```

## Method Selection Table

| Scenario | ExactCIs (Barnard's) | Fisher's Exact | Normal Approximation |
|----------|----------------------|----------------|----------------------|
| Small sample size (any cell < 5) | ✓✓✓ | ✓✓ | ✗ |
| Zero cells present | ✓✓✓ | ✓✓ | ✗ |
| Rare events (<1%) | ✓✓✓ | ✓✓ | ✗ |
| Moderate sample size | ✓✓ | ✓✓ | ✓ |
| Large sample size (all cells > 10) | ✓ | ✓ | ✓✓✓ |
| Fixed margins by design | ✓ | ✓✓✓ | ✓ |
| Computational speed critical | ✗ | ✓ | ✓✓✓ |
| Conservative inference needed | ✓✓✓ | ✓✓ | ✗ |

Legend: ✓✓✓ Highly recommended, ✓✓ Recommended, ✓ Acceptable, ✗ Not recommended

## Detailed Method Characteristics

### ExactCIs (Barnard's Unconditional)

**Best for**:
- Small sample sizes
- Rare events
- Zero cells
- When conservative inference is needed
- When margins are not fixed by design

**Implementation details**:
- `improved_ci_unconditional()` - Recommended for most cases
- `exact_ci_unconditional()` - Original implementation

**Computational characteristics**:
- More computationally intensive
- Excellent statistical properties
- Most conservative (wider intervals)

### Fisher's Exact Test

**Best for**:
- Fixed margins by design
- Protocol-mandated scenarios requiring Fisher's test
- Moderate sample sizes with small cells

**Implementation details**:
- Available in SciPy: `scipy.stats.fisher_exact()`
- Available in R: `exact2x2` package

**Computational characteristics**:
- Faster than Barnard's unconditional
- Still exact, but less conservative
- Well-established in literature

### Normal Approximation

**Best for**:
- Large sample sizes (all cells > 10)
- When computational speed is critical
- Preliminary analyses
- When exact methods are not required

**Implementation details**:
- Simple to implement
- Available in most statistical packages

**Computational characteristics**:
- Very fast computation
- Less precise for small samples
- Can produce implausible intervals

## Example Comparison of Methods

```python
from exactcis.methods.unconditional import improved_ci_unconditional
import numpy as np
import scipy.stats as stats

def normal_approx_ci(a, b, c, d, alpha=0.05):
    # Normal approximation with Haldane's correction
    a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    or_est = (a*d)/(b*c)
    log_or = np.log(or_est)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = stats.norm.ppf(1 - alpha/2)
    return np.exp(log_or - z*se), np.exp(log_or + z*se)

def ci_width(ci):
    return ci[1] - ci[0]

# Test scenarios with different characteristics
scenarios = [
    {"name": "Small balanced", "counts": (5, 5, 5, 5)},
    {"name": "Small imbalanced", "counts": (2, 8, 8, 2)},
    {"name": "Rare events", "counts": (1, 999, 10, 990)},
    {"name": "Zero cell", "counts": (0, 10, 5, 5)},
    {"name": "Large balanced", "counts": (50, 50, 50, 50)},
    {"name": "Large imbalanced", "counts": (20, 80, 80, 20)}
]

print("Comparison of Confidence Interval Methods\n")
print("Scenario | Point Est | ExactCIs CI | Normal CI | Width Ratio")
print("---------|-----------|-------------|-----------|------------")

for scenario in scenarios:
    a, b, c, d = scenario["counts"]
    
    # Point estimate of odds ratio
    if b*c == 0:
        or_est = float('inf') if a*d > 0 else float('nan')
    else:
        or_est = (a*d)/(b*c)
    
    # Calculate CIs
    try:
        exact_ci = improved_ci_unconditional(a, b, c, d)
        exact_width = ci_width(exact_ci)
    except Exception as e:
        exact_ci = (float('nan'), float('nan'))
        exact_width = float('nan')
    
    try:
        normal_ci = normal_approx_ci(a, b, c, d)
        normal_width = ci_width(normal_ci)
    except Exception as e:
        normal_ci = (float('nan'), float('nan'))
        normal_width = float('nan')
    
    # Calculate width ratio if both methods succeeded
    if not np.isnan(exact_width) and not np.isnan(normal_width):
        width_ratio = exact_width / normal_width
    else:
        width_ratio = float('nan')
    
    print(f"{scenario['name']:<10} | {or_est:>8.2f} | ({exact_ci[0]:>.2f}, {exact_ci[1]:<.2f}) | ({normal_ci[0]:>.2f}, {normal_ci[1]:<.2f}) | {width_ratio:>5.2f}x")
```

## Computational Complexity Considerations

When selecting a method, consider the computational resources required:

| Method | Time Complexity | Memory Usage | Relative Speed |
|--------|----------------|--------------|---------------|
| ExactCIs (improved_ci_unconditional) | O(n × grid_size) | O(grid_size) | 1x (baseline) |
| ExactCIs (exact_ci_unconditional) | O(n × grid_size) | O(grid_size) | 0.8x |
| Fisher's Exact Test | O(n) | O(1) | ~5x faster |
| Normal Approximation | O(1) | O(1) | ~50x faster |

Where n is related to table dimensions and grid_size is the precision parameter (default=50).

For ExactCIs, using caching can significantly improve performance when calculating multiple confidence intervals:

```python
from exactcis.utils.optimization import CICache
from exactcis.methods.unconditional import improved_ci_unconditional

# Create cache instance
cache = CICache(max_size=1000)

# Use the same cache instance for multiple calculations
ci1 = improved_ci_unconditional(7, 3, 2, 8, cache_instance=cache)
ci2 = improved_ci_unconditional(8, 2, 3, 7, cache_instance=cache)  # Similar table, will use cache
```

## Conclusion

When selecting a confidence interval method:

1. For small sample sizes, rare events, or zero cells, use **Barnard's Unconditional Method** (ExactCIs).

2. When margins are fixed by design and exactness is required, use **Fisher's Exact Test**.

3. For large sample sizes where computational speed is important, use **Normal Approximation**.

4. When in doubt, the **ExactCIs implementation** provides the most conservative and statistically valid approach, especially for small to moderate sample sizes.

5. For multiple calculations, use **caching** to improve performance.

Always consider the specific requirements of your analysis, including the need for exactness, computational resources, and the characteristics of your data.
