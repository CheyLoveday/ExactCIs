"""
ExactCIs: Exact Confidence Intervals for Odds Ratios and Relative Risks

This package provides methods to compute confidence intervals for the odds ratio
of a 2×2 contingency table. It includes various methods such as conditional (Fisher),
mid-P adjusted, Blaker's exact, Barnard's unconditional exact, and Haldane-Anscombe Wald.
It also provides methods for relative risk confidence intervals.
"""

from typing import Dict, Tuple

from exactcis.core import validate_counts, calculate_odds_ratio, calculate_relative_risk
from exactcis.utils.stats import (
    add_haldane_correction, 
    calculate_odds_ratio_with_correction,
    calculate_standard_error
)
from exactcis.methods.conditional import exact_ci_conditional
from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.wald import ci_wald_haldane
from exactcis.methods.relative_risk import (
    ci_wald_rr,
    ci_wald_katz_rr,
    ci_wald_correlated_rr,
    ci_score_rr,
    ci_score_cc_rr,
    ci_ustat_rr,
)

__version__ = "0.2.0"

__all__ = [
    "compute_all_cis",
    "compute_all_rr_cis",
    "exact_ci_conditional",
    "exact_ci_midp",
    "exact_ci_blaker",
    "exact_ci_unconditional",
    "ci_wald_haldane",
    "ci_wald_rr",
    "ci_wald_katz_rr",
    "ci_wald_correlated_rr",
    "ci_score_rr",
    "ci_score_cc_rr",
    "ci_ustat_rr",
    "calculate_odds_ratio",
    "calculate_relative_risk",
    "add_haldane_correction",
    "calculate_odds_ratio_with_correction",
    "calculate_standard_error",
]


def compute_all_cis(a: int, b: int, c: int, d: int,
                    alpha: float = 0.05, grid_size: int = 200
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for the odds ratio using all available methods.

    Args:
        a: Count in cell (1,1)
        b: Count in cell (1,2)
        c: Count in cell (2,1)
        d: Count in cell (2,2)
        alpha: Significance level (default: 0.05)
        grid_size: Number of grid points for optimization in Barnard's method (default: 200)

    Returns:
        Dictionary mapping method names to confidence intervals (lower_bound, upper_bound)
    """
    validate_counts(a, b, c, d)
    return {
        "conditional": exact_ci_conditional(a, b, c, d, alpha),
        "midp": exact_ci_midp(a, b, c, d, alpha),
        "blaker": exact_ci_blaker(a, b, c, d, alpha),
        "unconditional": exact_ci_unconditional(a, b, c, d, alpha=alpha, grid_size=grid_size),
        "wald_haldane": ci_wald_haldane(a, b, c, d, alpha),
    }


def compute_all_rr_cis(a: int, b: int, c: int, d: int, 
                     alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for the relative risk using all available methods.

    Args:
        a: Count in cell (1,1) - exposed with outcome
        b: Count in cell (1,2) - exposed without outcome
        c: Count in cell (2,1) - unexposed with outcome
        d: Count in cell (2,2) - unexposed without outcome
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary mapping method names to confidence intervals (lower_bound, upper_bound)
    """
    validate_counts(a, b, c, d)
    return {
        "wald": ci_wald_rr(a, b, c, d, alpha),
        "wald_katz": ci_wald_katz_rr(a, b, c, d, alpha),
        "wald_correlated": ci_wald_correlated_rr(a, b, c, d, alpha),
        "score": ci_score_rr(a, b, c, d, alpha),
        "score_cc": ci_score_cc_rr(a, b, c, d, 4.0, alpha),  # Medium correction
        "score_cc_strong": ci_score_cc_rr(a, b, c, d, 2.0, alpha),  # Strong correction
        "ustat": ci_ustat_rr(a, b, c, d, alpha),
    }
