"""
Data models for ExactCIs.

This module contains dataclasses for structured data in the ExactCIs package,
providing type safety and clear data contracts.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union


@dataclass
class EstimationResult:
    """
    Result of a statistical estimation with point estimate and standard error.
    
    This dataclass provides a structured way to return estimation results
    from functions that compute point estimates and their uncertainties.
    """
    point_estimate: float
    standard_error: float
    log_estimate: Optional[float] = None
    method: Optional[str] = None
    correction_applied: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the estimation result after initialization."""
        if not isinstance(self.point_estimate, (int, float)):
            raise TypeError("point_estimate must be numeric")
        if not isinstance(self.standard_error, (int, float)):
            raise TypeError("standard_error must be numeric")
        if self.standard_error < 0:
            raise ValueError("standard_error must be non-negative")


@dataclass  
class CorrectionResult:
    """
    Result of applying a continuity correction to a 2x2 table.
    
    This dataclass encapsulates the corrected counts and metadata about
    the correction that was applied.
    """
    a: float
    b: float
    c: float
    d: float
    correction_applied: bool
    correction_amount: float
    original_counts: Optional[tuple] = None
    method: Optional[str] = None
    
    def __post_init__(self):
        """Validate the correction result after initialization."""
        for count in [self.a, self.b, self.c, self.d]:
            if not isinstance(count, (int, float)) or count < 0:
                raise ValueError("All corrected counts must be non-negative numbers")
        if not isinstance(self.correction_applied, bool):
            raise TypeError("correction_applied must be boolean")
        if not isinstance(self.correction_amount, (int, float)) or self.correction_amount < 0:
            raise ValueError("correction_amount must be non-negative")


@dataclass
class ConfidenceInterval:
    """
    Confidence interval result with bounds and metadata.
    
    This dataclass represents a confidence interval with additional
    information about how it was computed.
    """
    lower: float
    upper: float
    alpha: float
    method: str
    point_estimate: Optional[float] = None
    converged: bool = True
    iterations: Optional[int] = None
    diagnostics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the confidence interval after initialization."""
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        # Allow infinite bounds for legitimate statistical cases
        if self.lower > self.upper and not (
            self.lower == float('inf') or self.upper == float('-inf')
        ):
            raise ValueError("lower bound must be less than or equal to upper bound")
    
    @property 
    def width(self) -> float:
        """Calculate the width of the confidence interval."""
        if self.lower == float('-inf') or self.upper == float('inf'):
            return float('inf')
        return self.upper - self.lower
    
    @property
    def confidence_level(self) -> float:
        """Get the confidence level (1 - alpha)."""
        return 1.0 - self.alpha


# Type aliases for common data structures
TableCounts = tuple[float, float, float, float]  # (a, b, c, d)
CIBounds = tuple[float, float]  # (lower, upper)