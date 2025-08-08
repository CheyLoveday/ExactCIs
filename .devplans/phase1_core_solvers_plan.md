# Phase 1 Plan — Core Solvers and Inversion, RR Score Refactor

Objective
- Introduce shared solver and inversion utilities used by CI methods.
- Refactor RR score and RR score_cc to delegate to the shared utilities with strict golden parity.

Non-goals (Phase 1)
- No public API changes.
- No behavioral/numerical changes; parity with golden fixtures must be exact.
- No GPU or caching integration yet (comes later).

Scope
1) New modules
- `src/exactcis/core/solvers.py`
  - Bracketing and root-finding on log-theta domain with robust safeguards.
- `src/exactcis/core/inversion.py`
  - Standardized CI inversion helpers, including infinite-bound detection.

2) Refactors
- `src/exactcis/methods/relative_risk.py`
  - Score-based methods (score, score_cc) refactored to use `core.inversion` and `core.solvers`.
  - Math/logic unchanged; only plumbing delegated.

3) Tests
- Unit tests for `core.solvers` and `core.inversion` on synthetic functions.
- Golden parity tests unchanged but must remain green.

Design details
- Domain for θ (RR, OR where applicable): `(1e-12, 1e12)`; configurable via arguments.
- Evaluate functions on the log scale for numerical stability when expanding brackets.
- Root-finding algorithm: safe bisection (guaranteed convergence with sign change) with hooks to swap a faster method later.
- Bracketing strategy
  - Seed around θ̂ (MLE) on log scale using multiplicative expansions: θ_lo = θ̂ / f, θ_hi = θ̂ * f.
  - Expand until sign-change or hitting domain limits; cap expansions to prevent runaway.
- Infinite bound detection
  - If no sign-change in the direction of search within domain and monotonicity holds, report a legitimate infinite bound.
  - Document rules clearly; never coerce to a finite value.
- Diagnostics (optional/verbose)
  - iterations, bracket_expansions, f_calls, monotone_flag, plateau_flag, domain_hits, final_residual.

API sketch
- solvers.py
  - `bracket_log_space(f, theta0, domain=(1e-12, 1e12), factor=2.0, max_expand=60) -> (lo, hi, meta)`
  - `find_sign_change_interval(f, seed, domain=(1e-12, 1e12), factor=2.0, max_expand=60) -> (lo, hi, meta)`
  - `is_monotone_on_log_grid(f, grid) -> {"monotone": bool, "direction": -1|0|1}`
  - `bisection_safe(f, lo, hi, tol=1e-10, max_iter=200) -> (root, meta)`
- inversion.py
  - `invert_bound(f_minus_target, is_lower, theta_hat, domain, tol, seed_factor, diagnostics=False) -> (bound, diag)`
  - `invert_two_sided_ci(f_minus_target, theta_hat, domain, tol, diagnostics=False) -> (lower, upper, diag)`

RR score integration
- Define `f_minus_target(theta) = score_statistic(theta) - z_crit^2` (or exact variant used currently).
- Use `invert_bound(..., is_lower=True/False)` to compute one-sided bounds.
- Two-sided CI constructed by calling for both sides.
- Preserve all current corrections and options (e.g., continuity correction in score_cc) before the f evaluation.

Testing strategy
- tests/core/test_solvers.py
  - Monotone increasing/decreasing functions on log domain; verify bracket expansion and sign-change discovery.
  - Bisection convergence and diagnostic counts.
  - Plateau case synthetic function; ensure detection and safe behavior.
- tests/core/test_inversion.py
  - Synthetic f_minus_target with known analytic roots; test finite and infinite bounds via domain.
- Golden parity
  - Run full `tests/test_golden_parity.py`. Must pass with EXACTCIS_STRICT_PARITY=1.

Acceptance criteria
- All unit tests for new modules pass locally and in CI.
- Golden fixtures parity 100% unchanged.
- relative_risk.py score and score_cc delegate to core utilities; method outputs identical.
- New modules have clear docstrings and typing; basic diagnostics available.

Risks & mitigations
- Hidden behavioral changes: keep refactor minimal; write adapter functions if needed to match current edge-case handling.
- Numeric drift: use bisection (deterministic) for now; later swap in faster method only with explicit review and version bump.
- Performance regressions: acceptable in Phase 1; measure and optimize in Phase 2+ with warm-starts and caching.

Work breakdown (PR-sized steps)
1) Add `core/solvers.py` + tests (no integration).
2) Add `core/inversion.py` + tests (no integration).
3) Refactor RR score in `relative_risk.py` to use inversion/solvers; run golden parity.
4) Refactor RR score_cc similarly; run golden parity.

Configuration
- Respect `EXACTCIS_STRICT_PARITY=1` in CI; block merges if parity drifts.
- Numerical defaults from `src/exactcis/constants.py` (EPS/TOL/SMALL_POS/MAX_ITER).

Follow-ups (Phase 2)
- Introduce warm-start θ grids and bracket caching.
- Move Wald/Katz SEs and risk/odds helpers into `core/estimates.py`.
- Begin OR method integrations incrementally.
