# ExactCIs Quick Reference Guide

This guide provides a visual overview of confidence interval methods and when to use ExactCIs.

## Visual Method Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ CONFIDENCE INTERVAL WIDTH COMPARISON - 95% CI                   │
│                                                                 │
│  Wide ┌───────────────────────────────────────────┐            │
│       │                                           │            │
│       │     Barnard's Unconditional (ExactCIs)    │            │
│       │     ******************************        │            │
│       │    *                              *       │            │
│  CI   │   *                                *      │            │
│ Width │  *                                  *     │            │
│       │ *                                    *    │            │
│       │*           Fisher's Exact             ****│            │
│       │--------------------------------------     │            │
│       │                                           │            │
│       │................Normal Approximation.......│            │
│ Narrow└───────────────────────────────────────────┘            │
│          Small                              Large              │
│                         Sample Size                            │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Tree

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ METHOD SELECTION DECISION TREE                                 │
│                                                                │
│             ┌──────────┐                                       │
│             │  START   │                                       │
│             └────┬─────┘                                       │
│                  │                                             │
│                  ▼                                             │
│         ┌─────────────────┐     Yes                           │
│         │ Any cell count  ├─────────────┐                     │
│         │     < 5?        │             │                     │
│         └────────┬────────┘             │                     │
│                  │ No                   │                     │
│                  ▼                      ▼                     │
│         ┌─────────────────┐    ┌────────────────────┐        │
│         │  Rare events    │ Yes│   Use Barnard's    │        │
│         │  (rate < 1%)?   ├────►  Unconditional     │        │
│         └────────┬────────┘    │   (ExactCIs)       │        │
│                  │ No          └────────────────────┘        │
│                  ▼                                           │
│         ┌─────────────────┐                                  │
│         │   All cells     │     Yes                          │
│         │    > 10?        ├─────────────┐                    │
│         └────────┬────────┘             │                    │
│                  │ No                   │                    │
│                  │                      ▼                    │
│                  │             ┌────────────────────┐       │
│                  │             │ Computational speed│  Yes   │
│                  │             │     critical?      ├────┐   │
│                  │             └─────────┬──────────┘    │   │
│                  │                       │ No            │   │
│                  │                       │               │   │
│                  │                       ▼               ▼   │
│                  │            ┌────────────────┐ ┌────────────┐
│                  └───────────►│ Use Barnard's  │ │ Use Normal │
│                               │ Unconditional  │ │ Approximation│
│                               │  (ExactCIs)    │ │            │
│                               └────────────────┘ └────────────┘
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Method Comparison Table

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ METHOD CHARACTERISTICS COMPARISON                              │
│                                                                │
│ ┌───────────────────┬───────────┬───────────┬─────────────┐   │
│ │                   │ Barnard's │ Fisher's  │   Normal    │   │
│ │  Characteristic   │ (ExactCIs)│  Exact    │Approximation│   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Statistical       │    ●●●    │    ●●     │      ●      │   │
│ │ Validity          │           │           │             │   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Small Sample      │    ●●●    │    ●●     │      ✗      │   │
│ │ Performance       │           │           │             │   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Rare Event        │    ●●●    │    ●●     │      ✗      │   │
│ │ Handling          │           │           │             │   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Computational     │     ●     │    ●●     │     ●●●     │   │
│ │ Speed             │           │           │             │   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Ease of           │    ●●     │    ●●●    │     ●●●     │   │
│ │ Implementation    │           │           │             │   │
│ ├───────────────────┼───────────┼───────────┼─────────────┤   │
│ │ Large Sample      │    ●●     │    ●●     │     ●●●     │   │
│ │ Performance       │           │           │             │   │
│ └───────────────────┴───────────┴───────────┴─────────────┘   │
│                                                                │
│  Legend: ●●● Excellent   ●● Good   ● Fair   ✗ Poor            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Runtime Performance

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ RUNTIME PERFORMANCE COMPARISON                                 │
│                                                                │
│  Slow  ┌───────────────────────────────────────────┐          │
│        │*                                           │          │
│        │*                                           │          │
│        │*                                           │          │
│        │ *                                          │          │
│ Runtime│ *                                          │          │
│        │  *                                         │          │
│        │   *                                        │          │
│        │    *                                       │          │
│        │     **                                     │          │
│        │       ****                                 │          │
│        │           *********....................    │          │
│  Fast  └───────────────────────────────────────────┘          │
│          Small                                Large           │
│                         Table Size                            │
│                                                               │
│     Legend: *** ExactCIs (Barnard's Unconditional)            │
│             ... Normal Approximation                          │
│                                                               │
└────────────────────────────────────────────────────────────────┘
```

## ExactCIs Algorithm Flow

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ EXACTCIS ALGORITHM OVERVIEW                                    │
│                                                                │
│     ┌──────────────────┐                                       │
│     │  Initialize with │                                       │
│     │  adaptive grid   │                                       │
│     └────────┬─────────┘                                       │
│              │                                                 │
│              ▼                                                 │
│ ┌────────────────────────────┐                                │
│ │  For each θ (odds ratio)   │                                │
│ │  calculate max p-value     │                                │
│ │  over nuisance parameters  │                                │
│ └────────────┬───────────────┘                                │
│              │                                                 │
│              ▼                                                 │
│     ┌──────────────────┐                                       │
│     │ Find θ values    │                                       │
│     │ where p = α/2    │                                       │
│     └────────┬─────────┘                                       │
│              │                                                 │
│              ▼                                                 │
│     ┌──────────────────┐                                       │
│     │  Refine bounds   │                                       │
│     │  with binary     │                                       │
│     │  search          │                                       │
│     └────────┬─────────┘                                       │
│              │                                                 │
│              ▼                                                 │
│     ┌──────────────────┐                                       │
│     │ Return confidence│                                       │
│     │ interval bounds  │                                       │
│     └──────────────────┘                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Example Tables

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ EXAMPLE CONFIDENCE INTERVALS                                   │
│                                                                │
│  Standard Table (7,3,2,8) - OR = 9.33                          │
│  ┌────────────┬─────────────────────┬────────────────────────┐ │
│  │ Method     │ 95% CI              │ Width                  │ │
│  ├────────────┼─────────────────────┼────────────────────────┤ │
│  │ ExactCIs   │ (1.047, 104.72)     │ 103.67                 │ │
│  │ Fisher's   │ (0.882, 127.06)     │ 126.18                 │ │
│  │ Normal     │ (1.193, 72.99)      │ 71.80                  │ │
│  └────────────┴─────────────────────┴────────────────────────┘ │
│                                                                │
│  Rare Events Table (1,1000,10,1000) - OR = 0.10                │
│  ┌────────────┬─────────────────────┬────────────────────────┐ │
│  │ Method     │ 95% CI              │ Width                  │ │
│  ├────────────┼─────────────────────┼────────────────────────┤ │
│  │ ExactCIs   │ (0.011, 0.980)      │ 0.969                  │ │
│  │ Fisher's   │ (0.013, 0.826)      │ 0.813                  │ │
│  │ Normal     │ (0.013, 0.783)      │ 0.770                  │ │
│  └────────────┴─────────────────────┴────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## When to Use ExactCIs

Use ExactCIs (Barnard's Unconditional Method) when:

1. **Small sample sizes** - Particularly when any cell count is < 5
2. **Rare events** - Event rates < 1% or tables with zero cells
3. **Conservative inference** - When you need the most statistically valid approach
4. **Unconditional design** - When margins are not fixed by design
5. **Regulatory/Safety** - For applications requiring the most rigorous approach

## Quick Start Example

```python
from exactcis.methods.unconditional import exact_ci_unconditional

# Example 2×2 table
#      Success   Failure
# Grp1    7         3
# Grp2    2         8

# Calculate 95% confidence interval
lower, upper = exact_ci_unconditional(7, 3, 2, 8, alpha=0.05)
print(f"95% CI for odds ratio: ({lower:.4f}, {upper:.4f})")
# Output: 95% CI for odds ratio: (1.0472, 104.7200)
```
