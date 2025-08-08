# Golden Parity Test Plan

Purpose
- Lock in current method behavior so refactors don’t change results unintentionally.
- Provide deterministic, reviewable fixtures for CI to catch drift.

Principles
- Exact parity for existing outputs; no tolerances for exact methods and previously agreed semantics (including legitimate ∞ bounds).
- Update fixtures only with an explicit, reviewed decision (and method_version bump where relevant).

What is covered
- Public aggregators: `compute_all_cis` (OR) and `compute_all_rr_cis` (RR)
- Core methods: OR (conditional, midp [non-Haldane], midp_haldane [if invoked directly], blaker, unconditional, wald_haldane), RR (wald, wald_katz, wald_correlated, score, score_cc, ustat)
- Alphas: 0.10, 0.05, 0.01 (configurable)

Fixture schema (JSON)
- File locations
  - `tests/fixtures/golden_or.json`
  - `tests/fixtures/golden_rr.json`
- Structure (array of records):
```
[
  {
    "table": {"a": 12, "b": 5, "c": 8, "d": 10},
    "alpha": 0.05,
    "results": {
      "or": {
        "conditional": {"lower": 0.73, "upper": 3.12},
        "midp": {"lower": "0.0", "upper": "inf"},
        "blaker": {"lower": 0.68, "upper": 3.05},
        "unconditional": {"lower": 0.71, "upper": 3.22},
        "wald_haldane": {"lower": 0.65, "upper": 3.41}
      },
      "rr": {
        "wald": {"lower": 0.80, "upper": 2.50},
        "score": {"lower": 0.78, "upper": "inf"}
      }
    }
  },
  ...
]
```
- Notes
  - Infinity is serialized as the string "inf" (and "-inf" if needed) to remain JSON-compatible.
  - All numeric values are floats; rounding in fixtures should match current test expectations.

Table grid (curated)
- Edge/zero cells (ensure corrections and ∞ handling)
  - (0, b, c, d), (a, 0, c, d), (a, b, 0, d), (a, b, c, 0) with small positive companions
  - Examples: (0,1,5,7), (1,0,5,7), (5,7,0,1), (5,7,1,0)
- Small n exhaustive-ish (n_total ≤ 12)
  - Enumerate a subset ensuring coverage of all margins and balances; sample to ~200 tables
- Imbalanced margins
  - Examples: (1,20,3,60), (2,50,4,100), (10,2,50,5)
- Symmetric/moderate
  - Examples: (5,5,5,5), (10,10,10,10), (6,9,7,8)
- Known pathological/previous regressions
  - OR mid-P case: (12,5,8,10) where midp (non-Haldane) is the reference semantics
  - RR score cases that previously produced inflated ∞ upper bounds

Generation procedure
1) Freeze current code on a tagged branch (e.g., `pre-refactor-parity`).
2) Run generator scripts:
   - `scripts/generate_golden_or.py` -> `tests/fixtures/golden_or.json`
   - `scripts/generate_golden_rr.py` -> `tests/fixtures/golden_rr.json`
3) Inspect diff; ensure legitimate ∞ and zeros are preserved.
4) Commit fixtures with the tag in the commit message.

Test harness
- `tests/test_parity_or.py` and `tests/test_parity_rr.py`:
  - Load fixtures, loop through records, call compute_all_* for each `table`+`alpha`.
  - Assert exact string match for "inf"; numeric equality for floats (or exact stringified match to avoid float repr drift).
  - Ensure required methods are present; skip optional ones if not applicable.

Updating fixtures (only when behavior is intentionally changed)
- Document motivation (e.g., statistical correction), show validation against R or theory.
- Bump a `method_version` in the code path that changed (used by caches and artifact tags).
- Regenerate fixtures and update docs/CHANGELOG.

R parity (separate from golden parity)
- Maintain CSVs generated from R for validation of specific methods (e.g., OR unconditional via `uncondExact2x2`, RR score via ratesci/epiR).
- Store in `tests/fixtures/r_parity/*.csv` with metadata: method, alpha, R package versions.
- Tests compare within tight tolerances (or exact where applicable), accepting ∞ where R does.

Automation & CI
- Add a CI job that runs parity tests on every PR touching `src/exactcis/**`.
- Optional env flag `EXACTCIS_STRICT_PARITY=1` to enforce zero diffs on refactor PRs.

Ownership
- Any changes to fixtures require review by maintainers familiar with the statistical methods.
