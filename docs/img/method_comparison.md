# Method Comparison Diagrams

This file contains ASCII diagrams for visually comparing confidence interval methods. You can convert these to proper images using tools like mermaid.js, PlantUML, or other diagramming software.

## Confidence Interval Width Comparison

```
                  Width of 95% Confidence Intervals
   
Wide  ┌────────────────────────────────────────────┐
      │                                             │
      │                                             │
      │       ╭─────────────╮                       │
      │      /               \                      │
      │     /                 \                     │
CI    │    /                   \                    │
Width │   /                     \                   │
      │  /                       \                  │
      │ /                         \________________ │
      │/                                            │
      │                                             │
Narrow└────────────────────────────────────────────┘
        Small                                  Large
                      Sample Size
                        
     Legend: ━━━ Barnard's Unconditional (ExactCIs)
             ╌╌╌ Fisher's Exact Test
             ··· Normal Approximation
```

## Decision Tree for Method Selection

```
┌─────────────┐
│    Start    │
└──────┬──────┘
       │
       ▼
┌──────────────────┐     Yes     ┌───────────────────────┐
│ Are any cells <5? ├────────────►Is computational speed │
└──────┬───────────┘             │      critical?        │
       │ No                      └──────────┬────────────┘
       ▼                                    │
┌──────────────────┐                        │     ┌───────────────────┐
│Is total sample   │     Yes                │ Yes │ Use Fisher's      │
│   size >1000?    ├────────────────────────┼────►│ Exact Test        │
└──────┬───────────┘                        │     └───────────────────┘
       │ No                                 │
       ▼                                    │     ┌───────────────────┐
┌──────────────────┐                        └────►│ Use Barnard's     │
│ Are margins fixed│     Yes                      │ Unconditional     │
│   by design?     ├─────────────────────────────►│ (ExactCIs)        │
└──────┬───────────┘                              └───────────────────┘
       │ No
       ▼
┌───────────────────┐
│ Use Barnard's     │
│ Unconditional     │
│ (ExactCIs)        │
└───────────────────┘
```

## Confidence Level Coverage Comparison

```
                Coverage Probability vs Sample Size
   
100% ┌────────────────────────────────────────────┐
     │   *****                                     │
     │  *     *                                    │
     │ *       *                                   │
95%  │*         *******___________________________ │
     │                                             │
     │                    .....................    │
     │                 ...                     ... │
90%  │            .....                            │
     │        ....                                 │
     │    ....                                     │
     │....                                         │
85%  └────────────────────────────────────────────┘
       Small                                   Large
                       Sample Size
                        
     Legend: *** Barnard's Unconditional (ExactCIs)
             ___ Fisher's Exact Test
             ... Normal Approximation
```

## Implementation Comparison

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│            Implementation Characteristics                  │
│                                                           │
├───────────────┬───────────────┬───────────────┬───────────┤
│               │  ExactCIs     │  SciPy        │  R        │
│ Characteristic│ (Barnard's)   │ (Fisher's)    │(exact2x2) │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Conditional?  │      No       │     Yes       │    Yes    │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Search        │  Adaptive     │  Direct       │  Root     │
│ Strategy      │  Grid         │  Calculation  │  Finding  │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Statistical   │  Most         │  Less         │  Less     │
│ Conservation  │  Conservative │  Conservative │Conservative│
├───────────────┼───────────────┼───────────────┼───────────┤
│ Computational │     Slow      │    Medium     │    Fast   │
│ Speed         │               │               │           │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Memory Usage  │    Medium     │     Low       │    Low    │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Zero Cell     │ Special       │ Standard      │ Special   │
│ Handling      │ Handling      │ Correction    │ Methods   │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Rare Event    │ Excellent     │ Good          │ Good      │
│ Performance   │               │               │           │
├───────────────┼───────────────┼───────────────┼───────────┤
│ Availability  │ Python        │ Python        │ R         │
│               │               │               │           │
└───────────────┴───────────────┴───────────────┴───────────┘
```

## Algorithmic Flow Diagram

```
┌─────────────────────────────┐
│ ExactCIs Algorithm Flow     │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────┐
│1. Initialize adaptive grid    │
│   for nuisance parameters     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│2. For each theta value:       │
│   ┌────────────────────┐     │
│   │Calculate p-values   │     │
│   │over nuisance grid   │     │
│   └────────┬───────────┘     │
│            │                 │
│            ▼                 │
│   ┌────────────────────┐     │
│   │Find maximum p-value│     │
│   └────────────────────┘     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│3. Find theta values where     │
│   max p-value = alpha/2      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│4. Refine bounds using         │
│   binary search               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│5. Return confidence interval  │
│   (lower_bound, upper_bound)  │
└──────────────────────────────┘
```

## Implementation Detail Comparison

```
┌──────────────────────────────────────────────────────────────┐
│          Detailed Algorithm Implementation Comparison         │
└──────────────────────────────────────────────────────────────┘

            ┌─────────────────┐     ┌─────────────────┐
            │    ExactCIs     │     │     SciPy       │
            └────────┬────────┘     └────────┬────────┘
                     │                       │
                     ▼                       ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│Maximizes over all nuisance   │ │Conditions on fixed margins   │
│parameters (p₁, p₂)           │ │(hypergeometric distribution) │
└──────────────┬───────────────┘ └─────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│Uses adaptive grid for search  │ │Direct calculation of p-value │
│with refinement                │ │(no grid needed)             │
└──────────────┬───────────────┘ └─────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│Advanced caching for similar   │ │Optimized C implementation   │
│problems                       │ │for speed                    │
└──────────────┬───────────────┘ └─────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│Special handling for extreme   │ │Standard approach for all    │
│tables and edge cases          │ │tables                       │
└──────────────────────────────┘ └─────────────────────────────┘
```
