Method Selection Guide
====================

This guide provides a structured approach to selecting the most appropriate confidence interval method for your specific scenario.

Decision Flowchart
----------------

Use the following decision flowchart to guide your method selection:

.. code-block:: text

   Start
     │
     ├─ Are you estimating a single proportion rather than odds ratio?
     │   │
     │   ├─ Yes → Use Clopper-Pearson Method
     │   │
     │   └─ No  → Is sample size very large (all cells > 10)?
     │             │
     │             ├─ Yes → Is computational speed critical?
     │             │         │
     │             │         ├─ Yes → Use Normal Approximation
     │             │         │
     │             │         └─ No  → Do you need exact methods for protocol adherence?
     │             │                   │
     │             │                   ├─ Yes → Use Barnard's Unconditional (ExactCIs)
     │             │                   │
     │             │                   └─ No  → Use Normal Approximation
     │             │
     │             └─ No  → Are any cells less than 5?
     │                       │
     │                       ├─ Yes → Are margins fixed by design?
     │                       │         │
     │                       │         ├─ Yes → Use Fisher's Exact Test
     │                       │         │
     │                       │         └─ No  → Use Barnard's Unconditional (ExactCIs)
     │                       │
     │                       └─ No  → Are you dealing with rare events (rate < 1%)?
     │                                 │
     │                                 ├─ Yes → Use Barnard's Unconditional (ExactCIs)
     │                                 │
     │                                 └─ No  → Are margins fixed by design?
     │                                           │
     │                                           ├─ Yes → Use Fisher's Exact Test
     │                                           │
     │                                           └─ No  → Use Barnard's Unconditional (ExactCIs)

Interactive Method Selector
------------------------

The following Python function can help you select the appropriate method based on your dataset characteristics:

.. code-block:: python

   def recommend_ci_method(a, b, c, d, fixed_margins=False, need_exact=False, speed_critical=False, estimate_proportion=False, group=1):
       """
       Recommends the most appropriate confidence interval method.
       
       Parameters:
       - a, b, c, d: Counts in the 2×2 table
       - fixed_margins: Whether margins are fixed by design
       - need_exact: Whether exact methods are required by protocol
       - speed_critical: Whether computational speed is critical
       - estimate_proportion: Whether to estimate a proportion rather than odds ratio
       - group: Which group to estimate proportion for (1 or 2), only used if estimate_proportion=True
       
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
       
       # If estimating a proportion rather than odds ratio
       if estimate_proportion:
           methods.append("Clopper-Pearson Method")
           if group == 1:
               reasons.append(f"Estimating proportion for group 1 (p1 = {rate1:.4f})")
           else:
               reasons.append(f"Estimating proportion for group 2 (p2 = {rate2:.4f})")
           return {
               "recommended_methods": methods,
               "reasons": reasons,
               "table_properties": {
                   "min_count": min_count,
                   "total_count": total_count,
                   "has_zero": has_zero,
                   "rare_events": rare_events,
                   "all_large": all_large,
                   "group1_rate": rate1,
                   "group2_rate": rate2
               }
           }
       
       # For odds ratio estimation
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
           "recommended_methods": methods,
           "reasons": reasons,
           "table_properties": {
               "min_count": min_count,
               "total_count": total_count,
               "has_zero": has_zero,
               "rare_events": rare_events,
               "all_large": all_large,
               "group1_rate": rate1,
               "group2_rate": rate2
           }
       }

Example Usage
-----------

Here are examples of using the method selector with different scenarios:

.. code-block:: python

   # Example 1: Small sample with zero cell
   table1 = (5, 10, 0, 15)
   result1 = recommend_ci_method(*table1)
   print(f"Table: {table1}")
   print(f"Recommended method: {result1['recommended_methods'][0]}")
   print(f"Reason: {result1['reasons'][0]}")
   print()

   # Example 2: Large sample with speed requirements
   table2 = (50, 50, 40, 60)
   result2 = recommend_ci_method(*table2, speed_critical=True)
   print(f"Table: {table2}")
   print(f"Recommended method: {result2['recommended_methods'][0]}")
   print(f"Reason: {result2['reasons'][0]}")
   print()

   # Example 3: Rare events
   table3 = (2, 998, 1, 999)
   result3 = recommend_ci_method(*table3)
   print(f"Table: {table3}")
   print(f"Recommended method: {result3['recommended_methods'][0]}")
   print(f"Reason: {result3['reasons'][0]}")

Method Comparison
--------------

The following table compares the key characteristics of different confidence interval methods:

.. list-table::
   :header-rows: 1
   :widths: 18 18 16 16 16 16

   * - Characteristic
     - Barnard's Unconditional
     - Fisher's Exact
     - Mid-P
     - Normal Approximation
     - Clopper-Pearson
   * - Statistical Validity
     - Excellent
     - Very Good
     - Good
     - Fair
     - Excellent
   * - Small Sample Performance
     - Excellent
     - Good
     - Good
     - Poor
     - Excellent
   * - Rare Event Handling
     - Excellent
     - Good
     - Good
     - Poor
     - Excellent
   * - Computational Speed
     - Slow
     - Moderate
     - Fast
     - Very Fast
     - Fast
   * - Handles Zero Cells
     - Yes
     - Yes
     - Yes
     - No (requires correction)
     - Yes
   * - Recommended Sample Size
     - Any
     - Any
     - n > 20
     - n > 50
     - Any
   * - Primary Use Case
     - Odds Ratio
     - Odds Ratio
     - Odds Ratio
     - Odds Ratio
     - Single Proportion

Implementation in ExactCIs
-----------------------

Here's how to implement each method using ExactCIs:

Barnard's Unconditional Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from exactcis.methods import exact_ci_barnard
   
   # Example table
   a, b, c, d = 7, 3, 2, 8
   
   # Calculate 95% confidence interval
   lower, upper = exact_ci_barnard(a, b, c, d)
   print(f"Barnard's Unconditional: 95% CI ({lower:.4f}, {upper:.4f})")

Mid-P Method
^^^^^^^^^

.. code-block:: python

   from exactcis.methods import exact_ci_midp
   
   # Example table
   a, b, c, d = 7, 3, 2, 8
   
   # Calculate 95% confidence interval
   lower, upper = exact_ci_midp(a, b, c, d)
   print(f"Mid-P: 95% CI ({lower:.4f}, {upper:.4f})")

Fisher's Exact Method
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from exactcis.methods import exact_ci_fisher
   
   # Example table
   a, b, c, d = 7, 3, 2, 8
   
   # Calculate 95% confidence interval
   lower, upper = exact_ci_fisher(a, b, c, d)
   print(f"Fisher's Exact: 95% CI ({lower:.4f}, {upper:.4f})")

Normal Approximation
^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import scipy.stats as stats
   
   def normal_approx_ci(a, b, c, d, alpha=0.05):
       # Add 0.5 to each cell (Haldane's correction)
       a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
       
       # Calculate odds ratio and log odds ratio
       or_est = (a*d)/(b*c)
       log_or = np.log(or_est)
       
       # Standard error of log odds ratio
       se = np.sqrt(1/a + 1/b + 1/c + 1/d)
       
       # Critical value
       z = stats.norm.ppf(1 - alpha/2)
       
       # Confidence interval for log odds ratio
       log_lower = log_or - z*se
       log_upper = log_or + z*se
       
       # Convert back to odds ratio scale
       return np.exp(log_lower), np.exp(log_upper)
   
   # Example table
   a, b, c, d = 7, 3, 2, 8
   
   # Calculate 95% confidence interval
   lower, upper = normal_approx_ci(a, b, c, d)
   print(f"Normal Approximation: 95% CI ({lower:.4f}, {upper:.4f})")

Clopper-Pearson Method
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from exactcis.methods import exact_ci_clopper_pearson
   
   # Example table
   a, b, c, d = 7, 3, 2, 8
   
   # Calculate 95% confidence interval for proportion in group 1
   lower1, upper1 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=1)
   print(f"Clopper-Pearson (Group 1): 95% CI ({lower1:.4f}, {upper1:.4f})")
   
   # Calculate 95% confidence interval for proportion in group 2
   lower2, upper2 = exact_ci_clopper_pearson(a, b, c, d, alpha=0.05, group=2)
   print(f"Clopper-Pearson (Group 2): 95% CI ({lower2:.4f}, {upper2:.4f})")
   
   # Interpretation
   p1 = a / (a + b)  # Proportion in group 1
   p2 = c / (c + d)  # Proportion in group 2
   print(f"Group 1 proportion: {p1:.4f}, CI: ({lower1:.4f}, {upper1:.4f})")
   print(f"Group 2 proportion: {p2:.4f}, CI: ({lower2:.4f}, {upper2:.4f})")

Special Considerations
-------------------

Zero Cells
^^^^^^^^^

When one or more cells contain zero, special handling is required:

.. code-block:: python

   from exactcis.methods import exact_ci_barnard
   
   # Table with a zero cell
   a, b, c, d = 10, 5, 0, 15
   
   # Calculate 95% confidence interval
   lower, upper = exact_ci_barnard(a, b, c, d)
   print(f"Zero cell handling: 95% CI ({lower:.4f}, {upper:.4f})")

Fixed Margins
^^^^^^^^^^^

When margins are fixed by design (e.g., in matched case-control studies), consider using Fisher's exact method:

.. code-block:: python

   from exactcis.methods import exact_ci_fisher
   
   # Example for matched case-control study
   a, b, c, d = 15, 5, 5, 15  # Matched pairs
   
   # Calculate 95% confidence interval
   lower, upper = exact_ci_fisher(a, b, c, d)
   print(f"Fixed margins: 95% CI ({lower:.4f}, {upper:.4f})")

Computational Efficiency
^^^^^^^^^^^^^^^^^^^^^

For large datasets or batch processing, consider using caching or parallel processing:

.. code-block:: python

   from exactcis.methods import exact_ci_barnard_parallel
   
   # Define multiple tables
   tables = [
       (7, 3, 2, 8),
       (10, 5, 3, 12),
       (15, 5, 7, 13)
   ]
   
   # Process in parallel
   results = exact_ci_barnard_parallel(tables)
   
   # Print results
   for i, (lower, upper) in enumerate(results):
       print(f"Table {i+1}: 95% CI ({lower:.4f}, {upper:.4f})")