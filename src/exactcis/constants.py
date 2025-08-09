"""
Constants for ExactCIs package.

This module defines all numerical constants, tolerances, and default values
used throughout the package for consistent behavior.
"""

import math
from typing import Literal

# Type Aliases for controlled vocabularies
CorrectionMethod = Literal["haldane", "continuity", None]
WaldMethod = Literal["wald", "agresti_caffo"]
Distribution = Literal["normal", "t"]

# Default significance level
DEFAULT_ALPHA = 0.05

# Numerical tolerances
EPS = 1e-15  # Machine epsilon for floating point comparisons
TOL = 1e-10  # General tolerance for root finding and convergence
ROOT_TOL = 1e-12  # Stricter tolerance for root finding algorithms
STATISTICAL_TOL = 0.02  # 2% tolerance for statistical convergence (relaxed from 1%)

# Iteration limits
MAX_ITER = 1000  # Maximum iterations for iterative algorithms
MAX_ROOT_ITER = 500  # Maximum iterations for root finding
MAX_GRID_SIZE = 10000  # Maximum grid size for optimization

# Boundary and safety values
SMALL_POS = 1e-10  # Small positive number to avoid log(0)
LARGE_POS = 1e10   # Large positive number for upper bounds
LOG_SMALL = -23.0  # log(SMALL_POS), precomputed for efficiency
LOG_LARGE = 23.0   # log(LARGE_POS), precomputed for efficiency

# Default corrections
DEFAULT_HALDANE_CORRECTION = 0.5
DEFAULT_CONTINUITY_CORRECTION = 0.5

# Grid search parameters
DEFAULT_GRID_SIZE = 200
MIN_GRID_SIZE = 10
ADAPTIVE_GRID_EXPANSION_FACTOR = 2.0
INFLATION_CAP_FACTOR = 2.5  # Cap for inflated bounds

# Plateau detection parameters
PLATEAU_TOL = 1e-8  # Tolerance for detecting flat p-value regions
MIN_PLATEAU_WIDTH = 1e-6  # Minimum width for a valid plateau

# Probability bounds
MIN_PROB = 1e-16  # Minimum probability value
MAX_PROB = 1.0 - 1e-16  # Maximum probability value (slightly less than 1)

# Log-space computation constants
LOG_2PI = math.log(2 * math.pi)  # log(2π) for normal approximations
SQRT_2PI = math.sqrt(2 * math.pi)  # √(2π) for normal approximations

# Method-specific defaults
DEFAULT_BLAKER_BOUNDS = (1e-5, 1e7)  # Default search bounds for Blaker method
DEFAULT_MIDP_GRID_SIZE = 51  # Default grid size for Mid-P method
DEFAULT_UNCONDITIONAL_GRID_SIZE = 200  # Default grid size for unconditional method

# Caching parameters
DEFAULT_CACHE_SIZE = 8192  # Default LRU cache size
SHARED_CACHE_SIZE = 16384  # Size for shared process cache

# Environment variable names for configuration
ENV_STRICT_PARITY = "EXACTCIS_STRICT_PARITY"
ENV_REGEN_GOLDEN = "EXACTCIS_REGEN_GOLDEN"
ENV_DEBUG = "EXACTCIS_DEBUG"
ENV_CACHE_SIZE = "EXACTCIS_CACHE_SIZE"

# Validation constants
MIN_ALPHA = 1e-10  # Minimum allowed alpha
MAX_ALPHA = 0.999  # Maximum allowed alpha
MAX_COUNT = 1e10   # Maximum reasonable count value

# Error messages
ERROR_NEGATIVE_COUNTS = "All counts must be non-negative numbers"
ERROR_ZERO_MARGIN = "Row or column margin cannot be zero"
ERROR_INVALID_ALPHA = "Alpha must be between 0 and 1"
ERROR_CONVERGENCE = "Algorithm failed to converge within maximum iterations"
ERROR_NUMERICAL_STABILITY = "Numerical instability detected"

# Method family identifiers
OR_METHODS = {"conditional", "midp", "blaker", "unconditional", "wald_haldane"}
RR_METHODS = {"wald", "wald_katz", "wald_correlated", "score", "score_cc", "score_cc_strong", "ustat"}
EXACT_METHODS = {"conditional", "midp", "blaker", "unconditional"}
ASYMPTOTIC_METHODS = {"wald_haldane", "wald", "wald_katz", "wald_correlated"}

# Performance profiling
PROFILING_WARMUP_RUNS = 3
PROFILING_TIMING_RUNS = 10
PROFILING_TIMEOUT_SECONDS = 300