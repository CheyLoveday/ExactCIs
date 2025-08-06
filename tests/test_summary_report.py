#!/usr/bin/env python3
"""pytest test that runs all CI methods for a few 2×2 contingency tables
and prints a human-readable summary of the odds ratio and confidence
intervals returned by :func:`exactcis.compute_all_cis`.

The purpose is *diagnostic* – to show at a glance that every method runs
without errors on representative examples.  There are **no strict
numerical assertions** because the reference values depend on numerous
parameters (grid size, optimisation tolerance, …).  We simply verify
that each method returns a tuple of two numeric floats and is ordered
(lower ≤ upper).
"""
from typing import List, Tuple

import pytest

from exactcis import compute_all_cis
from exactcis.core import calculate_odds_ratio

# A handful of representative 2×2 tables.
TABLES: List[Tuple[int, int, int, int]] = [
    (50, 950, 10, 990),   # moderate counts, same denominator
    (12, 5, 8, 10),       # small counts where mid-P had special-casing
    (0, 20, 5, 15),       # zero cell requiring Haldane correction in some methods
]


def _fmt_float(x: float) -> str:
    """Pretty-print a float; handle infinity."""
    return f"{x:.4f}" if x != float("inf") else "inf"


@pytest.mark.parametrize("a,b,c,d", TABLES)
def test_summary_report(capsys, a: int, b: int, c: int, d: int) -> None:
    """Run every CI method on one table and emit a summary to stdout."""
    or_est = calculate_odds_ratio(a, b, c, d)
    print("\n" + "=" * 80)
    print(f"Table ({a},{b},{c},{d}) – Observed OR = {_fmt_float(or_est)}")
    print("=" * 80)

    results = compute_all_cis(a, b, c, d, alpha=0.05)

    # Basic validation & pretty report
    hdr = f"{'Method':<15}  {'Lower':<10}  {'Upper':<10}  {'Contains OR':<12}"
    print(hdr)
    print("-" * len(hdr))

    for method, (lower, upper) in results.items():
        # Sanity checks so that pytest fails if result looks wrong
        assert isinstance(lower, float) and isinstance(upper, float)
        assert lower <= upper or upper == float("inf")
        contains_or = lower <= or_est <= upper
        print(f"{method:<15}  {_fmt_float(lower):<10}  {_fmt_float(upper):<10}  {contains_or}")

    # Ensure the captured output is displayed when pytest -s is *not* used
    # (pytest captures during test, so we explicitly flush here).
    capsys.readouterr()
