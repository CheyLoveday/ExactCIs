"""
Standard benchmark scenarios for ExactCIs performance testing.

Provides curated sets of 2x2 tables representing different use cases,
difficulty levels, and edge cases for comprehensive method evaluation.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Type alias for 2x2 table
Table = Tuple[int, int, int, int]

@dataclass
class BenchmarkScenario:
    """
    A benchmark scenario with multiple tables and metadata.
    
    Attributes
    ----------
    name : str
        Scenario name
    description : str
        Description of what this scenario tests
    tables : list of tuple
        List of (a, b, c, d) tuples representing 2x2 tables
    expected_difficulty : str
        Expected computational difficulty: 'easy', 'medium', 'hard'
    use_cases : list of str
        Real-world applications this scenario represents
    """
    name: str
    description: str
    tables: List[Table]
    expected_difficulty: str
    use_cases: List[str]

# Standard epidemiological scenarios
EPIDEMIOLOGICAL_CASES = BenchmarkScenario(
    name="epidemiological",
    description="Typical epidemiological study designs with realistic effect sizes",
    tables=[
        (20, 80, 10, 90),    # RR ≈ 2.0, OR ≈ 2.25 - moderate effect
        (90, 910, 10, 990),  # RR ≈ 9.0, OR ≈ 9.9 - strong effect (smoking-lung cancer)
        (45, 455, 15, 485),  # RR ≈ 3.2, OR ≈ 3.4 - moderate-strong effect
        (5, 95, 8, 92),      # RR ≈ 0.6, OR ≈ 0.57 - protective effect
        (35, 165, 70, 330),  # RR ≈ 1.0, OR ≈ 1.0 - null effect
    ],
    expected_difficulty="easy",
    use_cases=["Case-control studies", "Cohort studies", "Meta-analysis"]
)

CLINICAL_TRIAL_CASES = BenchmarkScenario(
    name="clinical_trial", 
    description="Clinical trial scenarios with treatment vs control comparisons",
    tables=[
        (15, 85, 25, 75),    # RR ≈ 0.6 - protective treatment
        (12, 88, 20, 80),    # RR ≈ 0.6 - treatment effect
        (8, 42, 12, 38),     # RR ≈ 0.6 - smaller trial
        (30, 70, 35, 65),    # RR ≈ 0.8 - modest effect
        (22, 78, 22, 78),    # RR = 1.0 - no effect
    ],
    expected_difficulty="easy",
    use_cases=["Randomized controlled trials", "Drug efficacy studies"]
)

SMALL_SAMPLE_CASES = BenchmarkScenario(
    name="small_sample",
    description="Small sample scenarios where exact methods are essential",
    tables=[
        (2, 3, 4, 5),        # N=14 - very small
        (1, 2, 3, 4),        # N=10 - very small  
        (5, 5, 5, 5),        # N=20 - small balanced
        (3, 7, 2, 8),        # N=20 - small imbalanced
        (1, 9, 4, 6),        # N=20 - small with extreme cells
    ],
    expected_difficulty="medium",
    use_cases=["Pilot studies", "Rare disease research", "Subgroup analyses"]
)

ZERO_CELL_CASES = BenchmarkScenario(
    name="zero_cells",
    description="Zero cell scenarios testing continuity corrections and boundary handling",
    tables=[
        (0, 10, 5, 5),       # Zero in exposed outcome
        (5, 5, 0, 10),       # Zero in unexposed outcome  
        (0, 10, 0, 10),      # Zero in both outcomes
        (10, 0, 5, 5),       # Zero in exposed no-outcome
        (5, 5, 10, 0),       # Zero in unexposed no-outcome
    ],
    expected_difficulty="hard",
    use_cases=["Rare outcomes", "Safety studies", "Negative trials"]
)

LARGE_SAMPLE_CASES = BenchmarkScenario(
    name="large_sample",
    description="Large sample scenarios testing computational efficiency and asymptotic behavior",
    tables=[
        (200, 800, 100, 900),   # N=2000 - large balanced
        (150, 350, 200, 300),   # N=1000 - large imbalanced
        (500, 500, 300, 700),   # N=2000 - large balanced, different split
        (75, 925, 50, 950),     # N=2000 - large with rare outcome
        (1000, 1000, 800, 1200), # N=4000 - very large
    ],
    expected_difficulty="medium",  # Should be fast for asymptotic methods
    use_cases=["Population studies", "Large cohorts", "Registry analyses"]
)

EXTREME_CASES = BenchmarkScenario(
    name="extreme",
    description="Extreme scenarios testing numerical stability and edge case handling",
    tables=[
        (1, 1, 1, 1),        # Minimal counts
        (1, 999, 1, 999),    # Extreme imbalance, rare outcome
        (999, 1, 999, 1),    # Extreme imbalance, common outcome
        (50, 0, 25, 25),     # Perfect association in one direction
        (0, 50, 25, 25),     # Perfect separation
    ],
    expected_difficulty="hard",
    use_cases=["Stress testing", "Edge case validation", "Numerical stability testing"]
)

# Score method fix validation cases (specific to recent RR improvements)
SCORE_FIX_CASES = BenchmarkScenario(
    name="score_fixes", 
    description="Test cases that previously caused score methods to return infinite bounds",
    tables=[
        (15, 5, 10, 10),     # Primary fix case - previously returned (1.18, inf)
        (12, 8, 6, 14),      # Secondary case with similar pattern
        (3, 7, 2, 8),        # Small sample that had legitimate infinite bounds
        (20, 30, 15, 35),    # Larger sample for scaling test
        (8, 12, 4, 16),      # Another pattern that was problematic
    ],
    expected_difficulty="medium",
    use_cases=["Algorithm validation", "Regression testing", "Fix verification"]
)

class StandardScenarios:
    """
    Manager for standard benchmark scenarios.
    
    Provides easy access to different scenario types and utilities
    for running benchmarks across multiple scenarios.
    """
    
    def __init__(self):
        self.scenarios = {
            'epidemiological': EPIDEMIOLOGICAL_CASES,
            'clinical_trial': CLINICAL_TRIAL_CASES,
            'small_sample': SMALL_SAMPLE_CASES,
            'zero_cells': ZERO_CELL_CASES,
            'large_sample': LARGE_SAMPLE_CASES,
            'extreme': EXTREME_CASES,
            'score_fixes': SCORE_FIX_CASES,
        }
    
    def get(self, scenario_name: str) -> BenchmarkScenario:
        """Get scenario by name."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                           f"Available: {list(self.scenarios.keys())}")
        return self.scenarios[scenario_name]
    
    def get_tables(self, scenario_name: str) -> List[Table]:
        """Get just the tables for a scenario."""
        return self.get(scenario_name).tables
    
    def list_scenarios(self) -> List[str]:
        """List available scenario names."""
        return list(self.scenarios.keys())
    
    def get_all_tables(self, exclude_extreme: bool = True) -> List[Tuple[Table, str]]:
        """
        Get all tables from all scenarios with scenario labels.
        
        Parameters
        ----------
        exclude_extreme : bool, default True
            Whether to exclude extreme cases that might cause failures
            
        Returns
        -------
        list of (table, scenario_name) tuples
        """
        all_tables = []
        for name, scenario in self.scenarios.items():
            if exclude_extreme and name == 'extreme':
                continue
            for table in scenario.tables:
                all_tables.append((table, name))
        return all_tables
    
    def get_by_difficulty(self, difficulty: str) -> List[BenchmarkScenario]:
        """
        Get scenarios by expected difficulty level.
        
        Parameters
        ---------- 
        difficulty : {'easy', 'medium', 'hard'}
            Difficulty level to filter by
            
        Returns
        -------
        list of BenchmarkScenario
        """
        return [scenario for scenario in self.scenarios.values() 
                if scenario.expected_difficulty == difficulty]
    
    def get_by_use_case(self, use_case: str) -> List[BenchmarkScenario]:
        """
        Get scenarios relevant to a specific use case.
        
        Parameters
        ----------
        use_case : str
            Use case to search for (partial match)
            
        Returns
        -------
        list of BenchmarkScenario
        """
        matching = []
        for scenario in self.scenarios.values():
            if any(use_case.lower() in uc.lower() for uc in scenario.use_cases):
                matching.append(scenario)
        return matching

# Convenience collections for common benchmark patterns
FAST_BENCHMARK_TABLES = [
    # Quick tests that should complete in <1ms each
    (10, 10, 10, 10),    # Balanced, medium
    (20, 80, 10, 90),    # Standard epi
    (5, 5, 5, 5),        # Small balanced
]

COMPREHENSIVE_BENCHMARK_TABLES = [
    # Full test suite covering all major patterns
    *EPIDEMIOLOGICAL_CASES.tables,
    *CLINICAL_TRIAL_CASES.tables,
    *SMALL_SAMPLE_CASES.tables[:3],  # Subset of small samples
    *ZERO_CELL_CASES.tables,
    *LARGE_SAMPLE_CASES.tables[:3],  # Subset of large samples
]

STRESS_TEST_TABLES = [
    # Challenging cases for performance stress testing
    *LARGE_SAMPLE_CASES.tables,
    *EXTREME_CASES.tables,
    *ZERO_CELL_CASES.tables,
]

def get_scenario_summary() -> Dict[str, Any]:
    """
    Get summary statistics about all available scenarios.
    
    Returns
    -------
    dict
        Summary with counts, difficulty distribution, etc.
    """
    scenarios = StandardScenarios()
    
    summary = {
        'total_scenarios': len(scenarios.scenarios),
        'total_tables': sum(len(s.tables) for s in scenarios.scenarios.values()),
        'by_difficulty': {},
        'by_use_case': {},
        'scenario_details': {}
    }
    
    # Count by difficulty
    for difficulty in ['easy', 'medium', 'hard']:
        matching = scenarios.get_by_difficulty(difficulty)
        summary['by_difficulty'][difficulty] = {
            'count': len(matching),
            'scenario_names': [s.name for s in matching]
        }
    
    # Scenario details
    for name, scenario in scenarios.scenarios.items():
        summary['scenario_details'][name] = {
            'table_count': len(scenario.tables),
            'difficulty': scenario.expected_difficulty,
            'use_cases': scenario.use_cases
        }
    
    return summary