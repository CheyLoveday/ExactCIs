# Precomputation, Caching, and GPU Acceleration Plan

Goals
- Reduce repeated work and accelerate heavy methods (RR score/score-cc, OR unconditional, exact tails).
- Provide optional batch and offline acceleration without changing results.
- Keep determinism and correctness; public API unchanged.

Principles
- Opt-in accelerations; identical numerical outputs vs CPU-only paths.
- Batch-first design: GPU/parallel paths trigger when work is sizable.
- Versioned caches to avoid stale results.

Components
1) Central caching (core/cache.py)
- Memory LRU for in-session speedups; optional disk cache (sqlite/joblib/npz).
- Cache key = hash(canonical_table, method, alpha, options, method_version).
- Config toggles: EXACTCIS_CACHE, EXACTCIS_DISK_CACHE, size/TTL caps.

2) Canonicalization (core/canonical.py)
- Map 2x2 to canonical form via row/column permutations preserving method semantics.
- Return transform for mapping CI back (e.g., reciprocals); validate per method family.

3) Precomputed primitives (core/primitives.py)
- Memoize: hypergeometric pmf/cdf arrays for fixed margins; log-choose via gammaln; z-critical for common alphas.

4) Solver warm-starts (core/solvers.py)
- Vectorized θ-grid evaluation to find sign-change intervals (for RR score/score-cc).
- Cache last-good brackets per canonical table; reuse across calls/batches.

5) Small-N offline tables (optional)
- Methods: Fisher/conditional, mid-P, Blaker for N ≤ 50 (configurable).
- CLI: `exactcis-cache build --max-n 50 --methods conditional,midp,blaker --alphas 0.05,0.01`.
- Runtime: load table; transform from canonical; fallback to compute when miss.

6) GPU acceleration (optional)
- Backend selector (EXACTCIS_ACCEL=off|auto|cupy|jax); start with CuPy.
- Use GPU for batched score evaluations, hypergeometric tails, nuisance grid search.
- Keep final root-finding/optimization on CPU; only mass evaluations on GPU.

Phased rollout
- P0 (1–2 days): Memory LRU, primitive memoization, z-critical cache; parity tests.
- P1 (2–4 days): Canonicalization + warm-start grids for RR score/score-cc; batch API hooks.
- P2 (3–5 days): Disk cache + small-N table builder and loader; versioned artifacts.
- P3 (2–3 days): GPU backend for θ-grid and tail computations; auto-gating by batch size.
- P4 (2–3 days): Benchmarks, docs, usage examples; CI perf checks.

Benchmarks & targets
- RR score/score-cc: 2–5x wall-clock on ≥10k tables; 5–20x fewer CPU evals.
- OR unconditional warm-starts: 3–10x speedups in candidate search.
- Precompute jobs: 10–50x faster on GPU for lookup generation.

Risks & mitigations
- Cache invalidation: include method_version; clear-on-upgrade policy.
- Transfer overhead: batch-only GPU usage; minimize host-device copies.
- Invariance assumptions: validate canonicalization per method; disable where unsafe.
- Dependency bloat: GPU is optional extra; default CPU path remains first-class.

Developer checklist
- [ ] Add/verify parity tests for methods using cache/accelerations
- [ ] Ensure cache keys include options and method_version
- [ ] Provide toggles/env vars and sensible defaults
- [ ] Document when accelerations trigger and expected speedups
- [ ] Bench results recorded and tracked over time

Cloud usage notes
- Provide a GPU-ready Dockerfile and job script to build small-N tables and populate disk caches.
- Store artifacts with versioned metadata; loader validates compatibility at runtime.
