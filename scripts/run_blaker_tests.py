import logging
from exactcis.methods.blaker import exact_ci_blaker
from exactcis.core import estimate_point_or, validate_counts

# Configure basic logging to see output from the CI functions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

test_tables = [
    {"a": 3, "b": 1, "c": 1, "d": 3, "desc": "Std symmetric (OR=9)"},
    {"a": 1, "b": 9, "c": 9, "d": 1, "desc": "Std symmetric (OR approx 0.0123)"},
    {"a": 5, "b": 0, "c": 2, "d": 8, "desc": "Zero cell b=0"},
    {"a": 0, "b": 5, "c": 8, "d": 2, "desc": "Zero cell a=0"},
    {"a": 10, "b": 7, "c": 9990, "d": 9993, "desc": "Large N, OR approx 1.429"},
    {"a": 2, "b": 8, "c": 8, "d": 2, "desc": "Symmetric, OR=0.0625 (R exact2x2 ex.)"},
    {"a": 1, "b": 1, "c": 1, "d": 20, "desc": "Asymmetric, OR=20"},
]

alpha = 0.05

print(f"--- Python Blaker CI Results (alpha={alpha}) ---")

for table_data in test_tables:
    a, b, c, d = table_data['a'], table_data['b'], table_data['c'], table_data['d']
    desc = table_data['desc']
    
    print(f"\n=== Table: {desc} (a={a}, b={b}, c={c}, d={d}) ===")
    try:
        # Validate counts first, as exact_ci_blaker would
        validate_counts(a,b,c,d)
        
        point_or_uncorrected = estimate_point_or(a, b, c, d)
        point_or_haldane = estimate_point_or(a, b, c, d, correction_type='haldane')
        print(f"  Point OR (uncorrected): {point_or_uncorrected:.6f}")
        print(f"  Point OR (Haldane)  : {point_or_haldane:.6f}")
        
        lower, upper = exact_ci_blaker(a, b, c, d, alpha=alpha)
        # Ensure numeric output for consistency in comparison
        lower_val = float(lower) if isinstance(lower, (str, float, int)) and lower not in [float('inf'), float('-inf')] else lower
        upper_val = float(upper) if isinstance(upper, (str, float, int)) and upper not in [float('inf'), float('-inf')] else upper

        print(f"  Python Blaker CI    : Lower={lower_val:.6f}, Upper={upper_val:.6f}")
        
    except ValueError as e:
        print(f"  Error for table ({a},{b},{c},{d}): {e}")
    except Exception as e:
        print(f"  Unexpected error for table ({a},{b},{c},{d}): {e.__class__.__name__} - {e}")
