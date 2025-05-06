from exactcis.methods.fixed_ci import improved_ci_unconditional

# Test case
result, metadata = improved_ci_unconditional(10, 5, 6, 12, alpha=0.05)
print(f'CI: {result}')
print(f'Method Used: {metadata["method_used"]}')
print(f'Calculation Time: {metadata["calculation_time"]:.4f} seconds')
print(f'Fallback Used: {metadata["fallback_used"]}')
print(f'Initial Bounds Method: {metadata["initial_bounds_method"]}')
print(f'Warnings: {metadata["warnings"]}')
