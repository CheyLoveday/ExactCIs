Working with Rare Events
=====================

This guide focuses on calculating confidence intervals for 2×2 contingency tables containing rare events, which present unique statistical challenges.

Understanding Rare Events
-----------------------

In epidemiology, clinical trials, and safety studies, rare events are common:

- Rare diseases (e.g., prevalence < 1%)
- Uncommon adverse events
- Zero-count cells (no occurrences in one or more cells)

**Example rare event table:**

.. code-block:: text

          Event    No Event    Total
   Group 1     1       1000       1001
   Group 2    10        990       1000

In this example, the odds ratio is (1×990)/(1000×10) = 0.099, but the confidence interval calculation requires special consideration.

Challenges with Rare Events
-------------------------

Rare events create several statistical challenges:

1. **Zero Cells**: Tables with zeros cause problems for many methods
2. **Boundary Issues**: Estimates may push against parameter boundaries
3. **Asymptotic Failure**: Normal approximations break down
4. **Computational Challenges**: Numerical issues in estimation

ExactCIs for Rare Events
----------------------

Barnard's unconditional exact test (implemented in ExactCIs) is particularly well-suited for rare events:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Rare event example
   a, b, c, d = 1, 1000, 10, 990

   # Calculate confidence interval
   ci = exact_ci_unconditional(a, b, c, d, alpha=0.05)
   print(f"95% CI for odds ratio: ({ci[0]:.6f}, {ci[1]:.6f})")
   # Output: 95% CI for odds ratio: (0.012777, 0.782634)

Comparison with Other Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import scipy.stats as stats

   # Normal approximation (with Haldane's correction)
   def normal_approx_ci(a, b, c, d, alpha=0.05):
       a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5  # Haldane's correction
       or_est = (a*d)/(b*c)
       log_or = np.log(or_est)
       se = np.sqrt(1/a + 1/b + 1/c + 1/d)
       z = stats.norm.ppf(1 - alpha/2)
       log_lower = log_or - z*se
       log_upper = log_or + z*se
       return np.exp(log_lower), np.exp(log_upper)

   # Example rare event table
   a, b, c, d = 1, 1000, 10, 990

   # Calculate using different methods
   exactcis_ci = exact_ci_unconditional(a, b, c, d)
   normal_ci = normal_approx_ci(a, b, c, d)
   or_point = (a*d)/(b*c)

   print(f"Odds Ratio: {or_point:.6f}")
   print(f"ExactCIs: ({exactcis_ci[0]:.6f}, {exactcis_ci[1]:.6f})")
   print(f"Normal Approx: ({normal_ci[0]:.6f}, {normal_ci[1]:.6f})")

Strategies for Different Rare Event Scenarios
-----------------------------------------

1. Zero Cells
^^^^^^^^^^

When one or more cells contain zeros, consider:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Example with zero cell
   a, b, c, d = 0, 100, 10, 90

   # Default approach
   try:
       ci_default = exact_ci_unconditional(a, b, c, d)
       print(f"Default: ({ci_default[0]:.6f}, {ci_default[1]:.6f})")
   except Exception as e:
       print(f"Default failed: {e}")

   # With custom bounds
   ci_custom = exact_ci_unconditional(a, b, c, d, theta_min=0.0001, theta_max=10.0)
   print(f"Custom bounds: ({ci_custom[0]:.6f}, {ci_custom[1]:.6f})")

2. Small Expected Frequencies
^^^^^^^^^^^^^^^^^^^^^^^^^

For rare but non-zero events:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Example with small expected frequencies
   a, b, c, d = 3, 997, 15, 985

   # Use improved implementation with adaptive grid
   ci = exact_ci_unconditional(a, b, c, d, adaptive_grid=True, grid_size=100)
   print(f"95% CI for odds ratio: ({ci[0]:.6f}, {ci[1]:.6f})")

3. Extremely Rare Events
^^^^^^^^^^^^^^^^^^^^

For extremely rare events (e.g., < 0.1%):

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional

   # Example with extremely rare events
   a, b, c, d = 1, 9999, 5, 9995

   # Use higher precision and wider search bounds
   ci = exact_ci_unconditional(
       a, b, c, d, 
       precision=1e-8,
       theta_min=0.00001,
       theta_max=100.0,
       adaptive_grid=True
   )
   print(f"95% CI for odds ratio: ({ci[0]:.6f}, {ci[1]:.6f})")

Simulation Study
-------------

The following simulation demonstrates the performance of different methods with rare events:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from exactcis.methods.unconditional import exact_ci_unconditional

   def simulate_rare_events(n_simulations=100, true_or=0.5, event_rate=0.01, n=1000):
       """Simulate rare event data and calculate CIs using different methods."""
       exactcis_coverage = 0
       normal_coverage = 0
       
       for i in range(n_simulations):
           # Generate data with specified event rate and odds ratio
           p1 = event_rate
           p2 = (true_or * p1) / (1 - p1 + true_or * p1)
           
           # Group 1
           events1 = np.random.binomial(1, p1, n)
           a = np.sum(events1)
           b = n - a
           
           # Group 2
           events2 = np.random.binomial(1, p2, n)
           c = np.sum(events2)
           d = n - c
           
           # Skip tables with zeros (for simplicity)
           if min(a, b, c, d) == 0:
               continue
               
           # Calculate CIs
           exactcis_ci = exact_ci_unconditional(a, b, c, d)
           normal_ci = normal_approx_ci(a, b, c, d)
           
           # Check coverage
           if exactcis_ci[0] <= true_or <= exactcis_ci[1]:
               exactcis_coverage += 1
           if normal_ci[0] <= true_or <= normal_ci[1]:
               normal_coverage += 1
       
       return {
           "exactcis_coverage": exactcis_coverage / n_simulations,
           "normal_coverage": normal_coverage / n_simulations
       }

   # Run simulation
   results = simulate_rare_events(n_simulations=100)
   print(f"ExactCIs coverage: {results['exactcis_coverage']:.2f}")
   print(f"Normal approx coverage: {results['normal_coverage']:.2f}")

Best Practices for Rare Events
---------------------------

1. **Always Use Exact Methods**: For rare events, exact methods like those in ExactCIs are strongly preferred over asymptotic methods.

2. **Report Zero Cells Explicitly**: When reporting results with zero cells, clearly state how these were handled.

3. **Consider Sensitivity Analysis**: Try different methods and corrections to assess the robustness of your conclusions.

4. **Use Appropriate Search Bounds**: For very rare events, adjust the search bounds (`theta_min` and `theta_max`) to ensure the algorithm explores the relevant parameter space.

5. **Increase Precision**: For numerically challenging cases, increase the precision parameter.

Example Implementation
------------------

Here's a complete example for analyzing a rare events dataset:

.. code-block:: python

   from exactcis.methods.unconditional import exact_ci_unconditional
   import pandas as pd
   import numpy as np

   # Example dataset with rare events
   data = {
       'Study': ['Study A', 'Study B', 'Study C', 'Study D'],
       'Treatment_Events': [1, 0, 2, 3],
       'Treatment_Total': [1000, 500, 1500, 2000],
       'Control_Events': [5, 2, 8, 10],
       'Control_Total': [1000, 500, 1500, 2000]
   }
   df = pd.DataFrame(data)

   # Calculate event rates
   df['Treatment_Rate'] = df['Treatment_Events'] / df['Treatment_Total']
   df['Control_Rate'] = df['Control_Events'] / df['Control_Total']
   df['Rate_Ratio'] = df['Treatment_Rate'] / df['Control_Rate']

   # Calculate confidence intervals
   results = []
   for _, row in df.iterrows():
       a = row['Treatment_Events']
       b = row['Treatment_Total'] - a
       c = row['Control_Events']
       d = row['Control_Total'] - c
       
       # Handle zero cells with custom bounds
       if min(a, c) == 0:
           ci = exact_ci_unconditional(a, b, c, d, theta_min=0.0001, theta_max=100.0)
       else:
           ci = exact_ci_unconditional(a, b, c, d)
       
       results.append({
           'Study': row['Study'],
           'OR_Point': (a*d)/(b*c) if b*c > 0 else np.nan,
           'Lower_95CI': ci[0],
           'Upper_95CI': ci[1]
       })

   # Create results DataFrame
   results_df = pd.DataFrame(results)
   print(results_df)