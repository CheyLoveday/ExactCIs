import numpy as np
from scipy.stats.contingency import odds_ratio
from exactcis.methods import exact_ci_conditional
import json

def get_our_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    try:
        lower, upper = exact_ci_conditional(a, b, c, d, alpha=alpha)
        return round(lower, 6), round(upper, 6) if upper != float('inf') else float('inf')
    except Exception as e:
        return str(e), str(e)

def get_scipy_ci(a: int, b: int, c: int, d: int, alpha: float = 0.05):
    table_data = np.array([[a, b], [c, d]])
    try:
        res_or = odds_ratio(table_data)
        conf_int = res_or.confidence_interval(confidence_level=1-alpha)
        lower, upper = conf_int.low, conf_int.high
        return round(lower, 6), round(upper, 6) if upper != float('inf') else float('inf')
    except Exception as e:
        return str(e), str(e)

def main():
    tables_to_test = [
        {"name": "Table 1 (5,2,9995,9998)", "data": (5, 2, 9995, 9998)},
        {"name": "Table 2 (10,7,9990,9993)", "data": (10, 7, 9990, 9993)},
        {"name": "Table 3 (3,0,9997,10000)", "data": (3, 0, 9997, 10000)}
    ]

    results = {}

    for table_info in tables_to_test:
        a, b, c, d = table_info["data"]
        results[table_info["name"]] = {
            "our_method": get_our_ci(a, b, c, d),
            "scipy": get_scipy_ci(a, b, c, d)
        }

    print("Python Results (Our Method vs SciPy):")
    for name, res_data in results.items():
        print(f"\n{name}:")
        print(f"  Our Implementation: Lower={res_data['our_method'][0]}, Upper={res_data['our_method'][1]}")
        print(f"  SciPy:              Lower={res_data['scipy'][0]}, Upper={res_data['scipy'][1]}")

if __name__ == "__main__":
    main()
