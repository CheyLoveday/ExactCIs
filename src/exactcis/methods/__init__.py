"""
Confidence interval methods for odds ratios.

This package provides various methods for calculating confidence intervals
for the odds ratio of a 2x2 contingency table.
"""

from exactcis.methods.conditional import exact_ci_conditional
from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.wald import ci_wald_haldane
from exactcis.methods.fixed_ci import improved_ci_unconditional

__all__ = [
    "exact_ci_conditional",
    "exact_ci_midp",
    "exact_ci_blaker",
    "exact_ci_unconditional",
    "ci_wald_haldane",
    "improved_ci_unconditional",
]