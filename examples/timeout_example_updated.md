# Using Timeout Functionality in ExactCIs

This notebook demonstrates how to use the timeout functionality in ExactCIs to prevent calculations from running too long, especially when using computationally intensive methods like Barnard's unconditional test.

```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exactcis import compute_all_cis
from exactcis.methods import (
    exact_ci_unconditional,
    exact_ci_conditional,
    exact_ci_midp,
    exact_ci_blaker,
    ci_wald_haldane
)
from exactcis.utils import create_timeout_checker
```

## 1. Why Timeouts Are Necessary

Computing exact confidence intervals, especially using unconditional methods, can be computationally intensive for certain tables. Without a timeout mechanism, calculations might:

- Run for an unexpectedly long time
- Cause a Jupyter notebook or application to appear unresponsive
- Consume excessive computational resources

Let's demonstrate this with an example of a table that requires significant computation time.

```python
# Example computationally intensive table
# Large imbalanced table that will take significant time with unconditional method
a, b, c, d = 120, 5, 7, 200

print("Computationally intensive table:\n")
print(f"     | Cases | Controls")
print(f"-----|-------|----------")
print(f"Exp. |  {a:3d}  |    {b:3d}")
print(f"Unex.|  {c:3d}  |    {d:3d}\n")

# First, let's try the fast methods to establish baseline comparison
print("Results from fast methods:")
fast_methods = [
    ("conditional", exact_ci_conditional),
    ("midp", exact_ci_midp),
    ("wald_haldane", ci_wald_haldane),
]

for name, method in fast_methods:
    start_time = time.time()
    result = method(a, b, c, d)
    elapsed = time.time() - start_time
    lower, upper = result
    print(f"{name:12s} ({lower:.3f}, {upper:.3f}) - computed in {elapsed:.4f}s")
```

## 2. Using a Timeout with the Unconditional Method

Now let's use the timeout parameter to limit the computation time for the unconditional method.

```python
# Try with a 2-second timeout
timeout_seconds = 2

try:
    print(f"\nAttempting unconditional method with {timeout_seconds}-second timeout...")
    start_time = time.time()
    result = exact_ci_unconditional(a, b, c, d, grid_size=200, timeout=timeout_seconds)
    elapsed = time.time() - start_time
    lower, upper = result
    print(f"Success! Result: ({lower:.3f}, {upper:.3f}) - computed in {elapsed:.4f}s")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"Computation timed out after {elapsed:.4f}s with error: {str(e)}")
```

## 3. Finding the Right Timeout Value

The optimal timeout value depends on your specific needs. Let's try different timeout values to find one that allows the computation to complete.

```python
def try_with_timeout(a, b, c, d, timeout, grid_size=100):
    try:
        print(f"Attempting with {timeout}-second timeout...")
        start_time = time.time()
        result = exact_ci_unconditional(a, b, c, d, grid_size=grid_size, timeout=timeout)
        elapsed = time.time() - start_time
        lower, upper = result
        return True, (lower, upper), elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, str(e), elapsed

# Try with increasing timeouts
timeouts = [1, 5, 10, 15]
smaller_table = (50, 5, 7, 60)  # A slightly smaller table that should compute faster

results = []
for timeout in timeouts:
    success, result, elapsed = try_with_timeout(*smaller_table, timeout=timeout)
    if success:
        lower, upper = result
        print(f"Success with {timeout}s timeout: ({lower:.3f}, {upper:.3f}) in {elapsed:.4f}s")
    else:
        print(f"Failed with {timeout}s timeout after {elapsed:.4f}s: {result}")
    results.append((timeout, success, elapsed))
```

```python
# Visualize the relationship between timeout and success
timeouts = [r[0] for r in results]
elapsed_times = [r[2] for r in results]
successes = [r[1] for r in results]

plt.figure(figsize=(10, 6))
plt.bar(timeouts, elapsed_times, color=['red' if not s else 'green' for s in successes])
plt.axhline(y=max(elapsed_times), color='black', linestyle='--', alpha=0.5)
plt.xlabel('Timeout (seconds)')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Computation Time vs. Timeout Setting')
plt.xticks(timeouts)
plt.grid(axis='y', alpha=0.3)

# Add labels for success/failure
for i, (timeout, success, elapsed) in enumerate(results):
    label = "Success" if success else "Timeout"
    plt.text(timeout, elapsed + 0.1, label, ha='center')

plt.tight_layout()
plt.show()
```

## 4. Practical Timeout Strategies

Here are some practical strategies for using timeouts in your analyses:

```python
def compute_with_fallback(a, b, c, d, alpha=0.05, preferred_method="unconditional", timeout=10):
    """
    Attempt to compute CI with preferred method, falling back to alternatives if timeout occurs.
    
    Parameters:
    -----------
    a, b, c, d : int
        Cell counts
    alpha : float
        Significance level
    preferred_method : str
        Preferred method ("unconditional", "conditional", "midp", "blaker", "wald_haldane")
    timeout : float
        Timeout in seconds for unconditional method
        
    Returns:
    --------
    tuple : (lower, upper, method_used, elapsed_time)
    """
    method_functions = {
        "unconditional": lambda: exact_ci_unconditional(a, b, c, d, alpha=alpha, timeout=timeout),
        "conditional": lambda: exact_ci_conditional(a, b, c, d, alpha=alpha),
        "midp": lambda: exact_ci_midp(a, b, c, d, alpha=alpha),
        "blaker": lambda: exact_ci_blaker(a, b, c, d, alpha=alpha),
        "wald_haldane": lambda: ci_wald_haldane(a, b, c, d, alpha=alpha)
    }
    
    # Try preferred method first
    print(f"Attempting '{preferred_method}' method...")
    start_time = time.time()
    try:
        result = method_functions[preferred_method]()
        elapsed = time.time() - start_time
        return (*result, preferred_method, elapsed)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Failed: {str(e)} after {elapsed:.4f}s")
        
        # Fall back to other methods in order of preference
        fallback_order = ["midp", "conditional", "blaker", "wald_haldane"]
        
        for method in fallback_order:
            if method != preferred_method:
                print(f"Falling back to '{method}' method...")
                try:
                    fallback_start = time.time()
                    result = method_functions[method]()
                    fallback_elapsed = time.time() - fallback_start
                    return (*result, f"{method} (fallback)", fallback_elapsed)
                except Exception as fallback_e:
                    print(f"Fallback failed: {str(fallback_e)}")
        
        # If all methods fail, raise the original error
        raise Exception(f"All methods failed. Original error: {str(e)}")

# Try the fallback approach on our example table
try:
    lower, upper, method, elapsed = compute_with_fallback(a, b, c, d, timeout=3)
    print(f"\nFinal result using {method}: ({lower:.3f}, {upper:.3f}) in {elapsed:.4f}s")
except Exception as e:
    print(f"All methods failed: {str(e)}")
```

## 5. Recommended Timeout Settings

Based on experience with various table sizes and structures, here are some recommended timeout settings:

| Table Characteristics | Recommended Timeout | Notes |
|----------------------|---------------------|-------|
| Small tables (all cells < 10) | 5 seconds | Usually completes quickly |
| Medium tables (10-50 per cell) | 10-30 seconds | Balance between precision and speed |
| Large tables (> 50 per cell) | 60 seconds or use alternative | Consider midp or conditional methods |
| Highly imbalanced tables | 30-60 seconds | Or use midp as a good alternative |
| Production environments | 10-30 seconds | Set reasonable limits for user-facing applications |
| Batch processing | 60+ seconds | Can be more generous when running offline |

Remember that computation times can vary significantly based on hardware, so adjust these recommendations to your specific environment.

## 6. Preemptive Detection of Computationally Intensive Tables

You can also implement heuristics to predict which tables might take a long time and adjust your approach accordingly:

```python
def predict_computation_intensity(a, b, c, d):
    """
    Predict the computational intensity of a 2x2 table and recommend an approach.
    
    Returns:
    --------
    tuple: (intensity_score, recommended_method, recommended_timeout)
    """
    # Calculate some metrics
    total = a + b + c + d
    max_count = max(a, b, c, d)
    min_count = min(a, b, c, d)
    imbalance_ratio = max_count / (min_count + 1)  # +1 to avoid division by zero
    
    # Simple heuristic for computational intensity
    intensity_score = total * (imbalance_ratio ** 0.5) / 100
    
    # Determine recommended method and timeout
    if intensity_score < 1:
        return intensity_score, "unconditional", 5
    elif intensity_score < 5:
        return intensity_score, "unconditional", 15
    elif intensity_score < 10:
        return intensity_score, "midp", 10
    else:
        return intensity_score, "conditional", 5

# Test the prediction on various tables
test_tables = [
    (5, 3, 2, 4),       # Small balanced table
    (50, 40, 30, 60),   # Medium balanced table
    (100, 5, 7, 200),   # Imbalanced table
    (2, 0, 1, 3),       # Small table with zero
    (300, 250, 280, 320) # Large balanced table
]

print("Table                 | Intensity | Recommended Method | Timeout")
print("-" * 65)
for table in test_tables:
    a, b, c, d = table
    intensity, method, timeout = predict_computation_intensity(a, b, c, d)
    print(f"({a:3d}, {b:3d}, {c:3d}, {d:3d}) | {intensity:9.2f} | {method:18s} | {timeout:4d}s")
```

## 7. Using Method-Specific Timeouts

The `compute_all_cis` function does not directly accept a timeout parameter, but you can use method-specific timeouts when you need more control:

```python
# Use method-specific timeouts with compute_all_cis
moderate_table = (25, 8, 12, 30)
print(f"Table: {moderate_table}\n")

# First, compute results using individual methods with timeouts
start_time = time.time()
results = {}

try:
    # Use different timeouts for different methods
    results["conditional"] = exact_ci_conditional(*moderate_table)
    results["midp"] = exact_ci_midp(*moderate_table)
    results["blaker"] = exact_ci_blaker(*moderate_table)
    results["unconditional"] = exact_ci_unconditional(*moderate_table, timeout=5)
    results["wald_haldane"] = ci_wald_haldane(*moderate_table)
    
    elapsed = time.time() - start_time
    
    print(f"Results computed in {elapsed:.4f}s:")
    print("Method        Lower   Upper   Width")
    print("-" * 40)
    for method, (lower, upper) in results.items():
        width = upper - lower
        print(f"{method:12s} {lower:.3f}   {upper:.3f}   {width:.3f}")
except Exception as e:
    print(f"Error: {str(e)}")

# Alternative: Use compute_all_cis but be aware it doesn't support timeout
print("\nUsing compute_all_cis (no timeout support):")
try:
    start_time = time.time()
    all_results = compute_all_cis(*moderate_table)
    elapsed = time.time() - start_time
    
    print(f"Results computed in {elapsed:.4f}s")
except Exception as e:
    print(f"Error: {str(e)}")
```

## 8. Creating a Custom Timeout Checker

You can also create your own timeout checker for more advanced use cases:

```python
# Create a custom timeout checker with progress tracking
def create_custom_timeout_checker(timeout, callback=None):
    """
    Create a timeout checker function with optional progress callback.
    
    Parameters:
    -----------
    timeout : float
        Timeout in seconds
    callback : callable, optional
        Function to call periodically during execution
        
    Returns:
    --------
    function
        A function that returns True if timeout has been exceeded
    """
    start_time = time.time()
    last_callback = start_time
    
    def check_timeout():
        nonlocal last_callback
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Call the callback function every second if provided
        if callback and current_time - last_callback >= 1.0:
            callback(elapsed, timeout)
            last_callback = current_time
            
        return elapsed > timeout
    
    return check_timeout

# Example progress callback
def progress_callback(elapsed, timeout):
    percent_complete = min(100, elapsed / timeout * 100)
    print(f"Progress: {elapsed:.1f}s / {timeout:.1f}s ({percent_complete:.1f}%)")

# Create a custom timeout checker with progress reporting
print("Custom timeout checker with progress reporting:")
checker = create_custom_timeout_checker(3.0, callback=progress_callback)

# Simulate a long computation
start = time.time()
while not checker():
    # Simulate work
    time.sleep(0.5)

print(f"\nFinished after {time.time() - start:.1f}s (either completed or timed out)")
```

## 9. Summary and Best Practices

### Key Takeaways

1. **Always use timeouts** for the unconditional method, especially in production environments
2. **Provide fallback methods** when timeout occurs
3. **Adjust timeout values** based on table size and characteristics
4. **Monitor computation times** to optimize performance

### Best Practices

1. **Start with fast methods** for exploratory analysis
2. **Use the unconditional method** with appropriate timeouts for final results when needed
3. **Pre-screen tables** to identify potentially slow computations
4. **Cache results** for repeated calculations on the same table
5. **Document the methods used** in your analysis reports

By following these guidelines, you can effectively balance computational accuracy with practical time constraints in your statistical analyses.