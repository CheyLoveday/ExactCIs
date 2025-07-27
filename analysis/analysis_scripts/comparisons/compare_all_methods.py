import numpy as np
from scipy.stats.contingency import odds_ratio
from scipy import stats # For stats.norm.ppf in Wald manual calculation if needed

from exactcis.methods import (
    exact_ci_conditional,
    ci_wald_haldane,
    exact_ci_blaker,
    exact_ci_midp,
    exact_ci_unconditional 
)
import json

# Helper to format CI results consistently
def format_ci(func, a, b, c, d, alpha=0.05, method_name="", **kwargs):
    try:
        lower, upper = func(a, b, c, d, alpha=alpha, **kwargs)
        return round(lower, 6), round(upper, 6) if upper != float('inf') else float('inf')
    except Exception as e:
        # print(f"Error in {method_name} for table ({a},{b},{c},{d}): {e}") # Optional debug print
        return str(e), str(e)

# --- SciPy Comparison Functions ---
def get_scipy_fisher_ci(a, b, c, d, alpha=0.05):
    table = np.array([[a,b],[c,d]])
    try:
        # Specify kind='conditional' for Fisher's exact test based OR
        res_or = odds_ratio(table, kind='conditional') 
        conf_int = res_or.confidence_interval(confidence_level=1-alpha)
        return round(conf_int.low, 6), round(conf_int.high, 6) if conf_int.high != float('inf') else float('inf')
    except Exception as e:
        return str(e), str(e)

def get_scipy_wald_logit_ci(a, b, c, d, alpha=0.05):
    table = np.array([[a,b],[c,d]])
    try:
        # Specify kind='sample' for sample OR, used for Wald CI
        res_or = odds_ratio(table, kind='sample') 
        conf_int = res_or.confidence_interval(confidence_level=1-alpha) 
        return round(conf_int.low, 6), round(conf_int.high, 6) if conf_int.high != float('inf') else float('inf')
    except Exception as e:
        return str(e), str(e)


def main():
    tables_to_test = [
        {"name": "Table 1 (5,2,9995,9998)", "data": (5, 2, 9995, 9998)},
        {"name": "Table 2 (10,7,9990,9993)", "data": (10, 7, 9990, 9993)},
        {"name": "Table 3 (3,0,9997,10000)", "data": (3, 0, 9997, 10000)}
    ]

    all_results = {}

    for table_info in tables_to_test:
        a, b, c, d = table_info["data"]
        table_name = table_info["name"]
        all_results[table_name] = {
            "Our_Conditional_Exact": format_ci(exact_ci_conditional, a, b, c, d, method_name="Our_Conditional_Exact"),
            "SciPy_Fisher_Exact": get_scipy_fisher_ci(a,b,c,d),
            "Our_Wald_Logit_Haldane": format_ci(ci_wald_haldane, a, b, c, d, method_name="Our_Wald_Logit_Haldane"),
            "SciPy_Wald_Logit": get_scipy_wald_logit_ci(a,b,c,d),
            "Our_Blaker_Exact": format_ci(exact_ci_blaker, a, b, c, d, method_name="Our_Blaker_Exact"), 
            "Our_MidP_Conditional": format_ci(exact_ci_midp, a, b, c, d, method_name="Our_MidP_Conditional"), 
            "Our_Unconditional_Exact": format_ci(exact_ci_unconditional, a, b, c, d, method_name="Our_Unconditional_Exact"),
        }
        # Add more methods as needed

    print("--- Comparison of CI Methods --- (alpha=0.05)")
    for table_name, methods_data in all_results.items():
        print(f"\n=== {table_name} ===")
        print(f"  Input: a={tables_to_test[next(i for i, t in enumerate(tables_to_test) if t['name'] == table_name)]['data'][0]}, "
              f"b={tables_to_test[next(i for i, t in enumerate(tables_to_test) if t['name'] == table_name)]['data'][1]}, "
              f"c={tables_to_test[next(i for i, t in enumerate(tables_to_test) if t['name'] == table_name)]['data'][2]}, "
              f"d={tables_to_test[next(i for i, t in enumerate(tables_to_test) if t['name'] == table_name)]['data'][3]}")
        
        for method_name, ci_val in methods_data.items():
            print(f"  {method_name:<28}: Lower={ci_val[0]}, Upper={ci_val[1]}")

if __name__ == "__main__":
    main()
