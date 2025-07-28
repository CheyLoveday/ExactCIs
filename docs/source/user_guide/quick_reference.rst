ExactCIs Quick Reference Guide
===========================

This guide provides a visual overview of confidence interval methods and when to use ExactCIs.

Visual Method Comparison
-----------------------

.. code-block:: text

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

Decision Tree
-----------

.. code-block:: text

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

Method Comparison Table
---------------------

.. code-block:: text

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

Common Use Cases
--------------

1. **Small Sample Sizes (n < 50)**
   
   When working with small samples, especially with cell counts less than 5, 
   use ExactCIs for reliable confidence intervals.

   .. code-block:: python

      from exactcis.methods import exact_ci_barnard
      
      # Small sample example
      a, b, c, d = 3, 2, 1, 5  # Small cell counts
      lower, upper = exact_ci_barnard(a, b, c, d)
      print(f"95% CI: ({lower:.4f}, {upper:.4f})")

2. **Rare Events**
   
   For rare events (e.g., rare disease studies), ExactCIs provides more 
   accurate intervals than asymptotic methods.

   .. code-block:: python

      # Rare event example (0.5% incidence)
      a, b, c, d = 5, 995, 1, 999
      lower, upper = exact_ci_barnard(a, b, c, d)
      print(f"95% CI: ({lower:.4f}, {upper:.4f})")

3. **Zero Cells**
   
   When one or more cells contain zero, ExactCIs handles this gracefully.

   .. code-block:: python

      # Zero cell example
      a, b, c, d = 10, 90, 0, 100
      lower, upper = exact_ci_barnard(a, b, c, d)
      print(f"95% CI: ({lower:.4f}, {upper:.4f})")

4. **Imbalanced Tables**
   
   For tables with large imbalances between cells, ExactCIs provides 
   more reliable intervals.

   .. code-block:: python

      # Imbalanced table
      a, b, c, d = 50, 5, 10, 100
      lower, upper = exact_ci_barnard(a, b, c, d)
      print(f"95% CI: ({lower:.4f}, {upper:.4f})")

Method Selection Guide
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Recommended Method
   * - Small sample (n < 50)
     - ``exact_ci_barnard`` (Unconditional)
   * - Zero cells present
     - ``exact_ci_barnard`` (Unconditional)
   * - Rare events (< 1%)
     - ``exact_ci_barnard`` (Unconditional)
   * - Balanced, large sample
     - ``exact_ci_midp`` (Mid-P) for speed, or ``exact_ci_barnard`` for accuracy
   * - Need fastest computation
     - ``exact_ci_midp`` (Mid-P)
   * - Maximum statistical validity
     - ``exact_ci_barnard`` (Unconditional)

Performance Considerations
------------------------

- **Computation Time**: Unconditional methods are more computationally intensive than mid-P or conditional methods.
- **Memory Usage**: For very large tables (all cells > 1000), consider using mid-P methods to reduce memory usage.
- **Parallelization**: For batch processing, use the parallel implementation:

  .. code-block:: python

     from exactcis.methods import exact_ci_barnard_parallel
     
     # Process multiple tables in parallel
     tables = [
         (10, 20, 5, 25),
         (15, 15, 10, 20),
         (8, 12, 4, 16)
     ]
     
     results = exact_ci_barnard_parallel(tables)
     for i, (lower, upper) in enumerate(results):
         print(f"Table {i+1}: 95% CI ({lower:.4f}, {upper:.4f})")