import time
import numpy as np
from exactcis.methods.unconditional import _process_grid_point

def test_performance():
    """Test the performance of _process_grid_point with large sample sizes."""
    print("Testing performance of _process_grid_point with large sample sizes...")
    
    # Test cases with increasing sample sizes
    test_cases = [
        # p1, a, c, n1, n2, theta
        (0.5, 10, 5, 20, 20, 1.0),  # Small case
        (0.5, 20, 10, 50, 50, 1.0),  # Medium case
        (0.5, 30, 15, 100, 100, 1.0),  # Large case
        (0.5, 50, 25, 200, 200, 1.0),  # Very large case
    ]
    
    for i, (p1, a, c, n1, n2, theta) in enumerate(test_cases):
        print(f"\nTest case {i+1}: n1={n1}, n2={n2}")
        
        # Measure execution time
        start_time = time.time()
        result = _process_grid_point((p1, a, c, n1, n2, theta))
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Result: {np.exp(result) if result != float('-inf') else 0.0}")

if __name__ == "__main__":
    test_performance()