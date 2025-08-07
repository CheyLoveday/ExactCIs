Relative Risk Methods
=====================

This module provides confidence interval methods for the relative risk (risk ratio) in 2x2 tables.
It complements the odds-ratio focused methods with options suitable for risk-based measures.

.. automodule:: exactcis.methods.relative_risk
   :members: ci_wald_rr, ci_wald_katz_rr, ci_wald_correlated_rr, ci_score_rr, ci_score_cc_rr, ci_ustat_rr
   :undoc-members:
   :show-inheritance:

Overview
--------

The relative risk (RR) compares the probability of an outcome in an exposed group to that in
an unexposed group: RR = [a/(a+b)] / [c/(c+d)]. This module implements several approaches
for constructing confidence intervals for RR, handling zero cells and small-sample situations.

Key Functions
-------------

- ``ci_wald_rr``
  - Standard Wald CI on the log scale with continuity correction for zero cells
- ``ci_wald_katz_rr``
  - Katz-adjusted Wald CI using a variance expression suitable for independent proportions
- ``ci_wald_correlated_rr``
  - Wald CI with correlation adjustment for matched or correlated designs; falls back to standard
    Wald for large independent samples
- ``ci_score_rr``
  - Score-based CI via inversion of a corrected score test (Tang et al.)
- ``ci_score_cc_rr``
  - Continuity-corrected score-based CI with adjustable correction strength ``delta``
- ``ci_ustat_rr``
  - U-statistic-based CI using variance estimation per Duan et al.; uses a t critical value with
    small-sample adjustment

Usage Examples
--------------

Basic Wald CI:

.. code-block:: python

   from exactcis.methods.relative_risk import ci_wald_rr

   # 2x2 table: [[a, b], [c, d]] = [[10, 20], [5, 25]]
   lower, upper = ci_wald_rr(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI (Wald RR): ({lower:.4f}, {upper:.4f})")

Katz-adjusted Wald CI:

.. code-block:: python

   from exactcis.methods.relative_risk import ci_wald_katz_rr

   lower, upper = ci_wald_katz_rr(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI (Katz RR): ({lower:.4f}, {upper:.4f})")

Score-based CI (corrected score inversion):

.. code-block:: python

   from exactcis.methods.relative_risk import ci_score_rr

   lower, upper = ci_score_rr(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI (Score RR): ({lower:.4f}, {upper:.4f})")

Continuity-corrected score CI (tunable correction):

.. code-block:: python

   from exactcis.methods.relative_risk import ci_score_cc_rr

   lower, upper = ci_score_cc_rr(10, 20, 5, 25, delta=4.0, alpha=0.05)
   print(f"95% CI (Score CC RR): ({lower:.4f}, {upper:.4f})")

U-statistic-based CI:

.. code-block:: python

   from exactcis.methods.relative_risk import ci_ustat_rr

   lower, upper = ci_ustat_rr(10, 20, 5, 25, alpha=0.05)
   print(f"95% CI (U-stat RR): ({lower:.4f}, {upper:.4f})")

Limitations
----------

- Zero cells: Functions internally apply a small continuity correction where appropriate.
  Intervals may be one-sided in extreme cases (e.g., a == 0 or c == 0 for score-based methods).
- Design considerations: ``ci_wald_correlated_rr`` is intended for matched/correlated designs;
  it reverts to a standard Wald CI for large independent samples to avoid overly wide intervals.
- Small samples: Score-based and U-statistic approaches generally offer better coverage than
  unadjusted Wald intervals when samples are small or risks are near 0 or 1.
