import logging
import sys

# Configure logging to show INFO and DEBUG messages
# Ensure logs from exactcis.methods.blaker and exactcis.core are captured
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Print logs to stdout so they are captured by run_command
)

# If specific loggers are too verbose or quiet, adjust their levels here
# logging.getLogger('exactcis.core').setLevel(logging.DEBUG) 
# logging.getLogger('exactcis.methods.blaker').setLevel(logging.DEBUG)

from exactcis.methods.blaker import exact_ci_blaker

tables = {
    "Table 1": (5, 2, 9995, 9998),
    "Table 2": (10, 7, 9990, 9993),
    "Table 3": (3, 0, 9997, 10000)
}

alpha = 0.05

if __name__ == "__main__":
    for name, (a, b, c, d) in tables.items():
        print(f"\n--- Testing Blaker CI for {name} ({a},{b},{c},{d}) alpha={alpha} ---")
        try:
            low, high = exact_ci_blaker(a, b, c, d, alpha=alpha)
            print(f"Result for {name}: Lower={low}, Upper={high}")
        except Exception as e:
            print(f"Error calculating Blaker CI for {name}: {e}")
        print(f"--- Finished Blaker CI for {name} ---")
