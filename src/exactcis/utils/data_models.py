"""
Data-only classes for ExactCIs following functional programming principles.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


@dataclass(frozen=True)
class TableData:
    """Immutable 2x2 contingency table data."""
    a: int
    b: int
    c: int
    d: int
    
    @property
    def n1(self) -> int:
        """Row 1 total."""
        return self.a + self.b
    
    @property
    def n2(self) -> int:
        """Row 2 total."""
        return self.c + self.d
    
    @property
    def m1(self) -> int:
        """Column 1 total."""
        return self.a + self.c
    
    @property
    def m2(self) -> int:
        """Column 2 total."""
        return self.b + self.d
    
    @property
    def total(self) -> int:
        """Total count."""
        return self.a + self.b + self.c + self.d


@dataclass(frozen=True)
class UnconditionalConfig:
    """Configuration for unconditional CI calculation."""
    alpha: float = 0.05
    grid_size: int = 15
    theta_min: Optional[float] = None
    theta_max: Optional[float] = None
    custom_range: Optional[Tuple[float, float]] = None
    theta_factor: float = 5.0
    haldane: bool = False
    timeout: Optional[float] = None
    adaptive_grid: bool = True
    use_cache: bool = True
    optimization_strategy: str = "auto"


@dataclass(frozen=True)
class ThetaRange:
    """Theta search range with computed bounds."""
    min_theta: float
    max_theta: float
    or_value: float


@dataclass(frozen=True)
class GridConfig:
    """Grid configuration for p1 values."""
    p1_values: np.ndarray
    grid_size: int
    p1_mle: float


@dataclass(frozen=True)
class BoundResult:
    """Result of confidence interval bound calculation."""
    value: float
    iterations: int
    method: str


@dataclass(frozen=True)
class CIResult:
    """Final confidence interval result."""
    lower: float
    upper: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PMFWeightsConfig:
    """Configuration for PMF weights calculation."""
    n1: float
    n2: float
    m: float
    theta: float
    
    
@dataclass(frozen=True)
class PMFWeightsResult:
    """Result of PMF weights calculation."""
    support: Tuple[int, ...]
    weights: Tuple[float, ...]
    method: str


@dataclass(frozen=True)
class RootFindingConfig:
    """Configuration for root finding operations."""
    lo: float = 1e-8
    hi: float = 1.0
    tol: float = 1e-8
    maxiter: int = 60
    timeout: Optional[float] = None


@dataclass(frozen=True)
class RootFindingResult:
    """Result of root finding operation."""
    value: Optional[float]
    iterations: int
    converged: bool
    method: str