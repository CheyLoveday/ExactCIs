# ExactCIs Refactoring Roadmap

Purpose
- Reduce duplication, increase correctness and maintainability, and make future method work faster/safer.
- Preserve public API and outputs while internal structure is improved.

Guiding principles
- Small, incremental PRs; strict parity with current outputs; add tests before refactors.
- Extract shared logic into core/util and thin method files.
- Document invariants and avoid hidden behavior changes.

Scope (what we refactor)
- Validation and corrections
  - Single source of truth for validate_counts and Haldane/continuity corrections.
- Numerical plumbing
  - Root finding, bracketing, plateau detection, test inversion, math-safe helpers.
- Data and results
  - Table2x2 model and CiResult result type (while preserving existing dict API externally).
- Method registration
  - Central method registry to standardize names, defaults, and metadata.

Proposed module layout
- exactcis/models
  - table.py: Table2x2 dataclass (a, b, c, d, derived totals)
  - result.py: CiResult dataclass {lower, upper, estimate, method, alpha, notes, diagnostics}
- exactcis/exceptions.py: InvalidCountsError, ConvergenceError, NumericalStabilityWarning
- exactcis/constants.py: EPS, TOL, MAX_ITER, DEFAULT_ALPHA, SMALL_POS, DEFAULT_CORRECTION
- exactcis/utils
  - validation.py: validate_counts, normalize_inputs
  - formatting.py: format_ci, safe_inf, sanitize_output
  - logging.py: helpers to attach diagnostics (iterations, bracket widths, flags)
- exactcis/core
  - corrections.py: Haldane and continuity correction policies
  - estimates.py: compute_or, compute_rr, risks/odds, Wald/Katz SEs
  - mathops.py: logsafe, exp_safe, clip_probs, ratio_safe
  - solvers.py: robust root/bracket utils (log-scale brackets, plateau-edge, safeguarded brentq)
  - inversion.py: standard test inversion, ∞-bound detection, sign conventions
  - optim.py: wrappers for global/local optimization (for unconditional/Barnard), with bounds
- exactcis/registry.py: method metadata (name, family, callable, defaults, options schema)
- exactcis/methods: keep algorithmic specifics; call into shared core where possible

Phased rollout
- Phase 0: Golden outputs + minimal centralization (1–2 days)
  - Generate golden fixtures for compute_all_cis and compute_all_rr_cis across a curated grid (including zeros and imbalanced tables). Store JSON fixtures.
  - Centralize validate_counts and Haldane/continuity corrections. Replace local duplicates in one file at a time (e.g., relative_risk.py first).
- Phase 1: Core solvers and inversion (3–5 days)
  - Implement solvers.py and inversion.py with unit tests (plateau, monotone checks, ∞ detection).
  - Refactor RR score/score_cc to use inversion pipeline. Strict parity with fixtures.
- Phase 2: Estimates and mathops (3–4 days)
  - Move risks/odds and SEs (Wald/Katz) into estimates.py and use in methods (Wald OR/RR).
  - Introduce mathops.py for safe log/ratios and probability clipping.
- Phase 3: Unconditional (Barnard) via core.optim (3–4 days)
  - Wrap optimizer calls with robust bounds and diagnostics. Maintain output parity.
- Phase 4: Registry + compute_all refactor (2–3 days)
  - Register methods, expose "midp" and "midp_haldane" distinctly, keep old API behavior.
  - compute_all_* loops over registry to assemble results.

Safety and guardrails
- Golden fixtures parity tests (no tolerances for exact methods; strict match for ∞ and zero).
- Unit tests for each new core utility; integration tests for compute_all_*.
- CI flag EXACTCIS_STRICT_PARITY=1 to enforce identical outputs during refactor phases.
- One-file-at-a-time PRs for early phases; revert on any parity drift.

Acceptance criteria
- All pre-refactor tests pass unchanged; fixtures match exactly.
- Method files shrink and delegate to core/util; no duplicated validation/corrections.
- New utilities covered by tests; coverage does not drop.

Risks and mitigations
- Subtle numeric drift: use golden tests and incremental changes.
- Hidden coupling: refactor per-method family, keep PRs scoped; document invariants.
- Developer confusion: add short dev notes and lints against re-implementing core utilities.

Developer checklist per PR
- [ ] New/updated tests written and passing locally
- [ ] Golden parity verified (no diffs)
- [ ] Only intended files changed; no algorithm edits slipped in
- [ ] Docs: brief note of what moved where and why
- [ ] CI green with STRICT_PARITY enabled

Unconditional ("uc") method notes
- Scope: Barnard’s unconditional exact method for OR is in scope; treat as first-class in refactor and validation.
- Validation: Benchmark and parity-check against R implementations (e.g., uncondExact2x2). Record R package/version in fixtures.
- Invariants and diagnostics:
  - Global optimum must be found with robust bounds; report convergence status, iterations, and objective at optimum.
  - Accept legitimate infinite bounds where mathematically correct.
  - Include method_version so caches and offline artifacts invalidate cleanly on algorithm changes.
- Refactor hooks:
  - Use core.optim for search with safe bounds and retries.
  - Register in registry.py with options and defaults; surface diagnostics in CiResult.

Mid-P semantics and invariants
- By policy, compute_all_cis maps midp to mid-P without Haldane correction (haldane=False) to match existing tests/expectations.
- If Haldane-corrected mid-P is desired via aggregators, expose a distinct key midp_haldane.
- Document this explicitly in method docs and ensure the registry advertises both variants clearly.
- Parity tests must reflect these semantics; treat any drift as a failure unless intentionally changed and reviewed.

Cross-links
- Related docs:
  - .devplans/precompute_and_gpu_plan.md (precomputation/caching/GPU strategy)
  - .devplans/golden_parity_plan.md (fixtures and parity testing to guard refactors)
  - .devplans/planned_expansion.md (broader roadmap and context)
