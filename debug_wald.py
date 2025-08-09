#!/usr/bin/env python3

import sys
import os
import math

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exactcis.methods.wald import ci_wald_haldane
from exactcis.utils.estimates import compute_log_or_with_se
from exactcis.utils.stats import normal_quantile

# Test case from golden fixtures that's failing
a, b, c, d, alpha = 1, 1, 1, 1, 0.05

print("=== Debug Wald Method Refactoring ===")
print(f"Table: ({a}, {b}, {c}, {d}), alpha={alpha}")

# Original manual calculation (what golden fixtures expect)
a2, b2, c2, d2 = a+0.5, b+0.5, c+0.5, d+0.5
or_hat_manual = (a2 * d2) / (b2 * c2)
se_manual = math.sqrt(1/a2 + 1/b2 + 1/c2 + 1/d2)
z = normal_quantile(1 - alpha/2)
lo_manual = math.exp(math.log(or_hat_manual) - z*se_manual)
hi_manual = math.exp(math.log(or_hat_manual) + z*se_manual)

print(f"\nOriginal manual calculation:")
print(f"  Corrected counts: ({a2}, {b2}, {c2}, {d2})")
print(f"  OR estimate: {or_hat_manual}")
print(f"  SE: {se_manual}")
print(f"  Z critical: {z}")
print(f"  CI: ({lo_manual}, {hi_manual})")

# Centralized calculation
from exactcis.utils.estimates import se_log_or_wald
from exactcis.utils.mathops import wald_variance_or

log_or, se_log_or = compute_log_or_with_se(a, b, c, d, method="wald_haldane")

# Debug the variance calculation
print(f"\nDebugging variance calculation:")
print(f"  wald_variance_or(1,1,1,1, haldane_corrected=True): {wald_variance_or(1,1,1,1, haldane_corrected=True)}")
print(f"  wald_variance_or(1,1,1,1, haldane_corrected=False): {wald_variance_or(1,1,1,1, haldane_corrected=False)}")
print(f"  se_log_or_wald(1,1,1,1, haldane=True): {se_log_or_wald(1,1,1,1, haldane=True)}")
print(f"  se_log_or_wald(1,1,1,1, haldane=False): {se_log_or_wald(1,1,1,1, haldane=False)}")

lo_centralized = math.exp(log_or - z * se_log_or)
hi_centralized = math.exp(log_or + z * se_log_or)

print(f"\nCentralized calculation:")
print(f"  Log OR: {log_or}")
print(f"  SE log OR: {se_log_or}")
print(f"  CI: ({lo_centralized}, {hi_centralized})")

# Current refactored method
current_result = ci_wald_haldane(a, b, c, d, alpha)
print(f"\nCurrent refactored method: {current_result}")

# Compare
print(f"\n=== Comparison ===")
print(f"Manual log(OR): {math.log(or_hat_manual)}")
print(f"Centralized log(OR): {log_or}")
print(f"Difference in log(OR): {abs(math.log(or_hat_manual) - log_or)}")

print(f"\nManual SE: {se_manual}")
print(f"Centralized SE: {se_log_or}")
print(f"Difference in SE: {abs(se_manual - se_log_or)}")

# Expected result from golden fixtures
expected = (0.040708779638011366, 2.459893048496844)
print(f"\nExpected from golden: {expected}")
print(f"Actual from refactored: {current_result}")
print(f"Lower bound diff: {abs(expected[0] - current_result[0])}")
print(f"Upper bound diff: {abs(expected[1] - current_result[1])}")