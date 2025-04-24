import code

# Example from the README
a, b, c, d = 12, 5, 8, 10
alpha = 0.05

# Test conditional CI
lower, upper = code.exact_ci_conditional(a, b, c, d, alpha)
print(f"Conditional CI: ({lower:.3f}, {upper:.3f})")

# Test all CIs
results = code.compute_all_cis(a, b, c, d, alpha, grid_size=500)
for method, (lo, hi) in results.items():
    print(f"{method:12s} CI: ({lo:.3f}, {hi:.3f})")