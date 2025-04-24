Targeted updates by file (no timeline):

⸻

src/exactcis/core.py
	•	validate_counts: restore correct 2×2 checks (non‑neg ints + no empty margins).
	•	Add log‑space helpers:

def logsumexp(log_terms): …
def log_binom_coeff(n,k): …
def log_nchg_pmf(k,n1,n2,m1,theta): …
def log_nchg_cdf(k,…): …
def log_nchg_sf(k,…): …


	•	Root finding on log(theta): implement

def find_root_log(f, lo, hi): …    # bracket + bisection on log(theta)
def find_plateau_edge(f, target, lo, hi): …  # refine for smallest/logical theta


	•	Remove any @lru_cache on functions taking float theta.

⸻

src/exactcis/utils/stats.py
	•	Ensure only generic routines:
	•	normal_quantile(p) (AS approximation)
	•	Optionally log_gamma, but move all logsumexp into core.
	•	No CI logic here.

⸻

src/exactcis/methods/wald.py
	•	Drop SciPy: import normal_quantile from utils/stats.py.
	•	ASCII only: rename any θ → theta.
	•	Guard zero margins via validate_counts.

⸻

src/exactcis/methods/conditional.py
	•	Use log‑space: replace direct PMF/CDF with log_nchg_pmf, log_nchg_cdf, log_nchg_sf.
	•	Operate on log_theta: call find_root_log/find_plateau_edge on log‑scale.
	•	Split α/2 correctly in tails.
	•	Short‑circuit a==k_min→0, a==k_max→∞.

⸻

src/exactcis/methods/midp.py
	•	Log‑space mid‑P: compute

log_P_eq, log_P_lt, log_P_gt
f_midp = logsumexp([log_P_gt, log_P_eq+log(0.5)]) *or* lower tail


	•	Invert at α (not α/2) on log_theta.
	•	Use core’s root‑finder and plateau refinement.

⸻

src/exactcis/methods/blaker.py
	•	Compute acceptability in log‑space:

log_acc_lo[i]=log_nchg_cdf(i,…)
log_acc_hi[i]=log_nchg_sf(i,…)
log_acc[i]=min(...)


	•	Sum log_nchg_pmf(i,…) for log_acc[i]≤log_acc[a] via logsumexp.
	•	Invert at log(alpha) with find_root_log + plateau edge.

⸻

src/exactcis/methods/unconditional.py
	•	Full joint enumeration over (k,ℓ) in [0,n1]×[0,n2]:

logP_kl = log_binom_coeff(n1,k) + k*log(p1) + … + log_binom_coeff(n2,l) + …


	•	Compute logP_obs at (a,c), sum all logP_kl≤logP_obs via logsumexp.
	•	Optimize over p1∈(ε,1−ε) using a 1D solver (Brent/golden‑section) to approximate sup p‑value.
	•	Invert that worst‑case p‑value = α/2 on log_theta with plateau refinement.

⸻

Edge‑case & naming sweep (all files):
	•	Rename θ → theta.
	•	Enforce validate_counts at function start.
	•	Document each method with the exact mathematical reference and edge‑case behavior.