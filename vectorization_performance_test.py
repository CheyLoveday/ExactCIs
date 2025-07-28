import time
import numpy as np
from exactcis.core import _pmf_weights_impl, log_nchg_pmf, support
from exactcis.methods.midp import exact_ci_midp

def test_pmf_weights_impl_performance():
    """Test the performance of the vectorized _pmf_weights_impl function."""
    print("\nTesting _pmf_weights_impl performance:")
    
    # Test cases with increasing sample sizes
    test_cases = [
        # n1, n2, m, theta
        (20, 20, 10, 1.0),  # Small case
        (50, 50, 25, 1.0),  # Medium case
        (100, 100, 50, 1.0),  # Large case
        (200, 200, 100, 1.0),  # Very large case
    ]
    
    for n1, n2, m, theta in test_cases:
        print(f"\nTest case: n1={n1}, n2={n2}, m={m}, theta={theta}")
        
        # Get support for this case
        supp = support(n1, n2, m)
        
        # Measure execution time
        start_time = time.time()
        result = _pmf_weights_impl(n1, n2, m, theta)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Support size: {len(supp.x)}")

def test_log_nchg_pmf_performance():
    """Test the performance of the vectorized log_nchg_pmf function."""
    print("\nTesting log_nchg_pmf performance:")
    
    # Test cases with increasing sample sizes
    test_cases = [
        # n1, n2, m1, theta
        (20, 20, 10, 1.0),  # Small case
        (50, 50, 25, 1.0),  # Medium case
        (100, 100, 50, 1.0),  # Large case
        (200, 200, 100, 1.0),  # Very large case
    ]
    
    for n1, n2, m1, theta in test_cases:
        print(f"\nTest case: n1={n1}, n2={n2}, m1={m1}, theta={theta}")
        
        # Get support for this case
        supp = support(n1, n2, m1)
        
        # Measure execution time for a single call
        k = supp.x[len(supp.x) // 2]  # Middle value in support
        start_time = time.time()
        result = log_nchg_pmf(k, n1, n2, m1, theta)
        end_time = time.time()
        
        single_execution_time = end_time - start_time
        print(f"Single call execution time: {single_execution_time:.6f} seconds")
        
        # Measure execution time for multiple calls (to test vectorization)
        start_time = time.time()
        results = np.vectorize(log_nchg_pmf)(supp.x, n1, n2, m1, theta)
        end_time = time.time()
        
        vectorized_execution_time = end_time - start_time
        print(f"Vectorized execution time for {len(supp.x)} values: {vectorized_execution_time:.6f} seconds")
        print(f"Average time per value: {vectorized_execution_time / len(supp.x):.6f} seconds")

def test_midp_performance():
    """Test the performance of the vectorized midp_pval_func function through exact_ci_midp."""
    print("\nTesting Mid-P method performance:")
    
    # Test cases with increasing sample sizes
    test_cases = [
        # a, b, c, d
        (5, 5, 5, 5),  # Small case
        (10, 10, 10, 10),  # Medium case
        (20, 20, 20, 20),  # Large case
    ]
    
    for a, b, c, d in test_cases:
        print(f"\nTest case: a={a}, b={b}, c={c}, d={d}")
        
        # Measure execution time
        start_time = time.time()
        lower, upper = exact_ci_midp(a, b, c, d)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Result: ({lower:.4f}, {upper:.4f})")

if __name__ == "__main__":
    test_pmf_weights_impl_performance()
    test_log_nchg_pmf_performance()
    test_midp_performance()