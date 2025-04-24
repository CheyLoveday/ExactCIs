"""
ExactCIs: Exact Confidence Intervals for Odds Ratios

This package provides methods to compute confidence intervals for the odds ratio
of a 2×2 contingency table. It includes various methods such as conditional (Fisher),
mid-P adjusted, Blaker's exact, Barnard's unconditional exact, and Haldane-Anscombe Wald.
"""

from typing import Dict, Tuple

from exactcis.core import validate_counts
from exactcis.methods.conditional import exact_ci_conditional
from exactcis.methods.midp import exact_ci_midp
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.methods.unconditional import exact_ci_unconditional
from exactcis.methods.wald import ci_wald_haldane

__version__ = "0.1.0"

__all__ = [
    "compute_all_cis",
    "exact_ci_conditional",
    "exact_ci_midp",
    "exact_ci_blaker",
    "exact_ci_unconditional",
    "ci_wald_haldane",
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
        "unconditional": exact_ci_unconditional(a, b, c, d, alpha, grid_size),
        "wald_haldane": ci_wald_haldane(a, b, c, d, alpha),
    }
