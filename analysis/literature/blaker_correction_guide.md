# Guide to Correcting src/exactcis/methods/blaker.py

This guide outlines the steps and logic to refactor and correct the Blaker's confidence interval implementation in `src/exactcis/methods/blaker.py`.

## 1. Imports

Ensure the following are imported from `exactcis.core`:

```python
from exactcis.core import (
    validate_counts, 
    support, 
    Support,  # Type hint for support object
    find_smallest_theta, 
    logsumexp, 
    nchg_pdf, # For calculating non-central hypergeometric probabilities
    estimate_point_or # For point estimate logging
)
```

## 2. `blaker_acceptability` Function

This helper function calculates the acceptability `P(X=k|theta)` for all `k` in the support for a given `theta`.

**Signature:**
```python
import numpy as np # Ensure numpy is imported as np

def blaker_acceptability(n1: int, n2: int, m1: int, theta: float, support_x: np.ndarray) -> np.ndarray:
```

**Logic:**
```python
    # Calculates P(X=k | theta) for each k in support_x.
    # nchg_pdf (from exactcis.core) should return an array of probabilities.
    return nchg_pdf(support_x, n1, n2, m1, theta)
```

## 3. `blaker_p_value` Function

This function calculates Blaker's p-value for a given `theta` and observed count `a`.

**Signature:**
```python
from exactcis.core import Support # Ensure Support type is available
import numpy as np

def blaker_p_value(a: int, n1: int, n2: int, m1: int, theta: float, s: Support) -> float:
```

**Logic:**
```python
    # s is the Support object from core.support(n1, n2, m1)
    # s.x is the array of all possible k values (counts in cell a)
    # s.offset is used to map 'a' to an index in s.x if s.x doesn't start from 0

    # 1. Calculate acceptability P(X=k|theta) for all k in the support s.x
    accept_probs_all_k = blaker_acceptability(n1, n2, m1, theta, s.x)
    
    # 2. Get the acceptability probability for the observed 'a'
    # The index for 'a' in the support_x array is (a - s.min_val) or (s.offset + a)
    # Assuming s.offset correctly maps 'a' to its position in s.x relative to s.x[0]
    idx_a = s.offset + a 
    # It's crucial that idx_a is a valid index for accept_probs_all_k.
    # Add basic bounds check for robustness, though kmin/kmax checks on 'a' should prevent issues.
    if not (0 <= idx_a < len(accept_probs_all_k)):
        # This case should ideally not happen if 'a' is within kmin/kmax and support is correct.
        # Handle appropriately, e.g., log error, return 1.0 as p-value might imply impossibility.
        # For now, assume idx_a is valid based on upstream logic.
        # If P(a|theta) is truly 0 and unrepresented, this logic needs care.
        logger.warning(f"Blaker p-value: Calculated index for 'a' ({idx_a}) is out of bounds for acceptability array (len {len(accept_probs_all_k)}). a={a}, s.offset={s.offset}")
        # Fallback or error here if needed.

    current_accept_prob_at_a = accept_probs_all_k[idx_a]
    
    # 3. Sum probabilities for k where P(k|theta) <= P(a|theta) * (1 + epsilon)
    # Epsilon is a small tolerance factor for floating point comparisons.
    epsilon = 1e-7 
    p_val_terms = accept_probs_all_k[accept_probs_all_k <= current_accept_prob_at_a * (1 + epsilon)]
    p_val = np.sum(p_val_terms)
    
    return p_val
```

## 4. Modifications to `exact_ci_blaker` Function

This is the main function that calculates the confidence interval.

**Inside `exact_ci_blaker`:**

*   **Point Estimate (for logging):**
    ```python
    or_point_est = estimate_point_or(a,b,c,d, correction_type='haldane')
    logger.info(f"Blaker exact_ci_blaker: Point OR estimate (Haldane corrected) for ({a},{b},{c},{d}) is {or_point_est:.4f}")
    ```

*   **Remove `p_obs`**: If there's any calculation of `p_obs = nchg_pdf(a, ...)` or similar, it should be removed as it's not passed to the new `blaker_p_value`.

*   **Update Lambda Functions for `find_smallest_theta`**: The lambda functions passed to `find_smallest_theta` need to use the new `blaker_p_value` signature.
    ```python
    # For lower bound search:
    blaker_p_value_lower = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
    raw_theta_low = find_smallest_theta(
        blaker_p_value_lower, 
        alpha / 2.0,  # Use alpha/2 for two-sided CIs
        lo=1e-9, hi=1e7, # Or other appropriate search range
        increasing=False # p-value decreases as theta increases for lower bound
    )

    # For upper bound search:
    blaker_p_value_upper = lambda theta_val: blaker_p_value(a, n1, n2, m1, theta_val, s)
    raw_theta_high = find_smallest_theta(
        blaker_p_value_upper, 
        alpha / 2.0, # Use alpha/2
        lo=1e-9, hi=1e7, # Or other appropriate search range
        increasing=True  # p-value increases as theta increases for upper bound
    )
    ```

*   **Bounds Check (Crucial):** After `theta_low` and `theta_high` are determined (typically after the `try...except` block that converts `raw_theta_low/high` to `float`), add a check for crossed or invalid bounds:
    ```python
    # Inside the try block, after theta_low and theta_high are finalized floats
    if not both_finite: # 'both_finite' from existing logging
        logger.warning(f"Blaker CI calculation resulted in non-finite bounds for ({a},{b},{c},{d}): low={theta_low}, high={theta_high}. Defaulting to (0, inf).")
        return 0.0, float('inf')

    # New check for crossed bounds
    if theta_low > theta_high:
        logger.warning(f"Blaker CI: Lower bound {theta_low:.4f} > Upper bound {theta_high:.4f} for ({a},{b},{c},{d}). This can happen. Consider returning (0, inf) or investigating further. For now, returning (0, inf).")
        # Depending on statistical requirements, could also be (min(theta_low, theta_high), max(theta_low, theta_high)), 
        # or specific handling if one bound is 0 or inf.
        # A common safe return for problematic intervals is (0, inf) or (NaN, NaN).
        return 0.0, float('inf')

    return theta_low, theta_high
    ```

## 5. Cleanup (Recommended)

*   **Remove Obsolete `blaker_p_value`**: There might be an older definition of `blaker_p_value` near the top of `blaker.py` with a signature like `def blaker_p_value(a: int, b: int, c: int, d: int, theta: float) -> float:`. If this function is no longer used by the `exact_ci_blaker` logic (which now uses the version taking `s: Support`), it should be removed to avoid confusion and potential errors.

## Final Structure Reminder

Ensure the `blaker_acceptability` and the corrected `blaker_p_value` (taking `s: Support`) are defined in the file *before* they are called by `exact_ci_blaker` or its lambda functions.

This guide should help you in restructuring and correcting the `blaker.py` module. Let me know if any part needs clarification.
