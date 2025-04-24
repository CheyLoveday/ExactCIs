import code

def print_separator():
    print("\n" + "-"*60 + "\n")

def run_test_case(a, b, c, d, alpha=0.05, grid_size=500):
    print(f"Test case: a={a}, b={b}, c={c}, d={d}, alpha={alpha}")
    
    # Test individual CI methods
    methods = [
        ("conditional", lambda: code.exact_ci_conditional(a, b, c, d, alpha)),
        ("midp", lambda: code.exact_ci_midp(a, b, c, d, alpha)),
        ("blaker", lambda: code.exact_ci_blaker(a, b, c, d, alpha)),
        ("barnard", lambda: code.exact_ci_unconditional(a, b, c, d, alpha, grid_size)),
        ("wald_haldane", lambda: code.ci_wald_haldane(a, b, c, d, alpha))
    ]
    
    for name, method in methods:
        try:
            lower, upper = method()
            print(f"{name:12s} CI: ({lower:.3f}, {upper:.3f})")
        except Exception as e:
            print(f"{name:12s} ERROR: {type(e).__name__}: {e}")
    
    # Test the orchestrator
    print("\nUsing orchestrator:")
    try:
        results = code.compute_all_cis(a, b, c, d, alpha, grid_size)
        for method, (lo, hi) in results.items():
            print(f"{method:12s} CI: ({lo:.3f}, {hi:.3f})")
    except Exception as e:
        print(f"ERROR with orchestrator: {type(e).__name__}: {e}")

# Test case from README
print_separator()
print("Example from README:")
run_test_case(12, 5, 8, 10)

# Edge cases
print_separator()
print("Edge case - small counts:")
run_test_case(1, 1, 1, 1)

print_separator()
print("Edge case - zero in one cell:")
run_test_case(0, 5, 8, 10)

print_separator()
print("Edge case - large imbalance:")
run_test_case(50, 5, 2, 20)

# Calculate odds ratio for comparison
def odds_ratio(a, b, c, d):
    return (a * d) / (b * c)

print_separator()
print("Original example: OR =", odds_ratio(12, 5, 8, 10))
