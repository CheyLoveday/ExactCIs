"""
Confidence interval methods for odds ratios and proportions.

This package provides various methods for calculating confidence intervals
for the odds ratio of a 2x2 contingency table and for binomial proportions.
"""

from exactcis.methods.conditional import exact_ci_conditional
from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.wald import ci_wald_haldane
from exactcis.methods.clopper_pearson import exact_ci_clopper_pearson
from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr,
)

__all__ = [
    "exact_ci_conditional",
    "exact_ci_midp",
    "exact_ci_blaker",
    "exact_ci_unconditional",
    "ci_wald_haldane",
    "exact_ci_clopper_pearson",
    "ci_wald_rr",
    "ci_wald_katz_rr",
    "ci_wald_correlated_rr",
    "ci_score_rr",
    "ci_score_cc_rr",
    "ci_ustat_rr",
]