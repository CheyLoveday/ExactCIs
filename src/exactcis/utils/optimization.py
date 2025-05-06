"""
Optimization utilities for ExactCIs.

This module provides tools for optimizing confidence interval calculations,
including caching mechanisms and adaptive search strategies.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import time

logger = logging.getLogger(__name__)

class CICache:
    """
    Caching system for confidence interval calculations.
    
    This cache stores both exact matches and similar tables to guide
    future searches based on patterns in previous calculations.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store in the similar cache
        """
        self.exact_cache: Dict[Tuple[int, int, int, int, float], Tuple[Tuple[float, float], Dict[str, Any]]] = {}
        self.similar_cache: List[Tuple[Tuple[int, int, int, int], Tuple[float, float], Dict[str, Any]]] = []
        self.max_size = max_size
        self.cache_hits = 0
        self.similar_hits = 0
        self.total_lookups = 0
    
    def get_exact(self, a: int, b: int, c: int, d: int, alpha: float) -> Optional[Tuple[Tuple[float, float], Dict[str, Any]]]:
        """
        Get exact cache hit.
        
        Args:
            a, b, c, d: Cell counts for 2x2 table
            alpha: Significance level
        
        Returns:
            Tuple of (confidence interval, parameters) if found, None otherwise
        """
        self.total_lookups += 1
        key = (a, b, c, d, alpha)
        result = self.exact_cache.get(key)
        
        if result is not None:
            self.cache_hits += 1
            logger.debug(f"Cache hit for {key}, hit rate: {self.cache_hits}/{self.total_lookups} = {self.cache_hits/self.total_lookups:.2%}")
        
        return result
    
    def get_similar(self, a: int, b: int, c: int, d: int, alpha: float, similarity_threshold: float = 0.2) -> List[Tuple[Tuple[int, int, int, int], Tuple[float, float], Dict[str, Any]]]:
        """
        Find similar tables to guide the search.
        
        Args:
            a, b, c, d: Cell counts for 2x2 table
            alpha: Significance level
            similarity_threshold: Minimum similarity score to include
        
        Returns:
            List of (counts, confidence interval, parameters) tuples for similar tables
        """
        if not self.similar_cache:
            return []
        
        # Calculate metrics for similarity
        target_or = (a * d) / (b * c) if b * c > 0 else float('inf')
        target_size = a + b + c + d
        
        # Find similar entries
        similar_entries = []
        for entry in self.similar_cache:
            counts, ci, params = entry
            entry_a, entry_b, entry_c, entry_d = counts
            
            entry_or = (entry_a * entry_d) / (entry_b * entry_c) if entry_b * entry_c > 0 else float('inf')
            entry_size = entry_a + entry_b + entry_c + entry_d
            
            # Calculate similarity score
            if target_or == 0 or entry_or == 0:
                or_ratio = 0  # Handle zeros to avoid division by zero
            elif target_or == float('inf') or entry_or == float('inf'):
                or_ratio = 0  # Handle infinity cases
            else:
                or_ratio = min(target_or, entry_or) / max(target_or, entry_or)
            
            size_ratio = min(target_size, entry_size) / max(target_size, entry_size)
            
            # More emphasis on odds ratio similarity
            similarity = 0.7 * or_ratio + 0.3 * size_ratio
            
            if similarity >= similarity_threshold:
                similar_entries.append((similarity, entry))
        
        # Sort by similarity
        similar_entries.sort(key=lambda x: x[0], reverse=True)
        
        result = [entry for _, entry in similar_entries[:5]]  # Return top 5 most similar
        
        if result:
            self.similar_hits += 1
            logger.debug(f"Similar pattern found for ({a},{b},{c},{d}), pattern hits: {self.similar_hits}/{self.total_lookups}")
        
        return result
    
    def add(self, a: int, b: int, c: int, d: int, alpha: float, ci: Tuple[float, float], params: Dict[str, Any]) -> None:
        """
        Add a result to both caches.
        
        Args:
            a, b, c, d: Cell counts for 2x2 table
            alpha: Significance level
            ci: Confidence interval (lower, upper)
            params: Parameters used for the calculation
        """
        # Add to exact cache
        key = (a, b, c, d, alpha)
        self.exact_cache[key] = (ci, params)
        
        # Add to similar cache
        self.similar_cache.append(((a, b, c, d), ci, params))
        
        # Prune if needed
        if len(self.similar_cache) > self.max_size:
            self.similar_cache = self.similar_cache[-self.max_size:]
            
    # Alias methods for compatibility with different naming conventions
    def store(self, a: int, b: int, c: int, d: int, alpha: float, ci: Tuple[float, float], params: Dict[str, Any]) -> None:
        """Alias for add() method."""
        return self.add(a, b, c, d, alpha, ci, params)
    
    def lookup(self, a: int, b: int, c: int, d: int, alpha: float) -> Optional[Tuple[float, float]]:
        """
        Alias for get_exact() method with simpler return type.
        
        Returns just the confidence interval without parameters.
        """
        result = self.get_exact(a, b, c, d, alpha)
        if result is not None:
            return result[0]  # Just return the CI tuple
        return None


def derive_search_params(a: int, b: int, c: int, d: int, similar_entries: List[Tuple[Tuple[int, int, int, int], Tuple[float, float], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Extract patterns from similar entries to optimize search.
    
    Args:
        a, b, c, d: Cell counts for 2x2 table
        similar_entries: List of similar table entries from cache
    
    Returns:
        Dictionary of search parameters
    """
    if not similar_entries:
        return {"grid_size": 50}
    
    # Analyze the similar entries to extract patterns
    grid_sizes = []
    theta_ranges = []
    total_iterations = []
    
    # Current table's odds ratio
    current_or = (a * d) / (b * c) if b * c > 0 else float('inf')
    if current_or == float('inf'):
        current_or = 100.0  # For computational purposes
    
    # Process each similar entry
    for entry in similar_entries:
        counts, ci, params = entry
        entry_a, entry_b, entry_c, entry_d = counts
        
        # Extract useful parameters
        grid_sizes.append(params.get("grid_size", 50))
        
        if "total_iterations" in params:
            total_iterations.append(params["total_iterations"])
        
        # Previous odds ratio
        prev_or = (entry_a * entry_d) / (entry_b * entry_c) if entry_b * entry_c > 0 else float('inf')
        if prev_or == float('inf'):
            prev_or = 100.0
        
        # Calculate relative OR scaling factor
        or_scale = current_or / prev_or if prev_or != 0 else 1.0
        
        # Scale the previous CI to estimate for current table
        lower, upper = ci
        if upper == float('inf'):
            upper = lower * 100  # Arbitrary but reasonable upper bound
        
        scaled_lower = lower * or_scale
        scaled_upper = upper * or_scale
        
        theta_ranges.append((scaled_lower, scaled_upper))
    
    # Determine optimal parameters based on patterns
    if grid_sizes:
        optimal_grid_size = int(sum(grid_sizes) / len(grid_sizes))
    else:
        optimal_grid_size = 50
    
    # Estimate appropriate theta range from similar tables
    if theta_ranges:
        lower_bounds = [r[0] for r in theta_ranges]
        upper_bounds = [r[1] for r in theta_ranges]
        
        # Use a conservative range (broader than individual entries)
        estimated_lower = max(0.5 * min(lower_bounds), 1e-6)
        estimated_upper = min(2.0 * max(upper_bounds), 1e6)
        
        predicted_range = (estimated_lower, estimated_upper)
    else:
        predicted_range = None
    
    # Estimate iterations needed
    if total_iterations:
        estimated_iterations = int(sum(total_iterations) / len(total_iterations))
    else:
        estimated_iterations = 100
    
    return {
        "grid_size": optimal_grid_size,
        "predicted_theta_range": predicted_range,
        "estimated_iterations": estimated_iterations
    }


def adaptive_grid_search(f: Callable[[float], float], bounds: Tuple[float, float], 
                         target_value: float = 0.0, initial_points: int = 10, 
                         refinement_rounds: int = 3, 
                         timeout_checker: Optional[Callable[[], bool]] = None) -> List[float]:
    """
    Perform adaptive grid search with progressive refinement.
    
    Args:
        f: Function to evaluate
        bounds: (lower, upper) bounds for search
        target_value: Target value to find crossings for
        initial_points: Number of initial grid points
        refinement_rounds: Number of refinement iterations
        timeout_checker: Optional function that returns True if timeout occurred
    
    Returns:
        List of x values where f(x) = target_value
    """
    # Ensure bounds are valid
    if bounds[0] >= bounds[1]:
        logger.warning(f"Invalid bounds for adaptive grid search: {bounds}")
        return []
    
    # Initialize grid
    grid = np.linspace(bounds[0], bounds[1], initial_points)
    evaluations = []
    
    # Evaluate initial grid
    for x in grid:
        if timeout_checker and timeout_checker():
            logger.warning("Timeout during initial grid evaluation")
            return []
        
        y = f(x)
        if y is None:  # Handle function evaluation failure
            continue
        evaluations.append((x, y))
    
    if len(evaluations) < 2:
        logger.warning("Too few valid evaluations for adaptive grid search")
        return []
    
    # Sort evaluations by x value
    evaluations.sort(key=lambda p: p[0])
    
    # Extract x and y arrays
    grid = np.array([p[0] for p in evaluations])
    values = np.array([p[1] for p in evaluations])
    
    for round_idx in range(refinement_rounds):
        if len(grid) < 2:
            break
            
        # Find regions with steepest derivatives or near target value
        derivatives = np.abs(np.diff(values) / np.diff(grid))
        target_proximity = np.abs(values[:-1] - target_value) + np.abs(values[1:] - target_value)
        
        # Combine metrics - focus on areas with steep derivatives and near target
        interest_metric = derivatives / (target_proximity + 1e-10)
        
        # Determine threshold for adding points (top 30% most interesting regions)
        if len(interest_metric) > 0:
            threshold = np.percentile(interest_metric, 70)
        else:
            threshold = 0
        
        # Add more points in interesting regions
        new_points = []
        for i in range(len(interest_metric)):
            if interest_metric[i] > threshold:
                # Add a point in this region
                new_x = (grid[i] + grid[i+1]) / 2
                new_points.append(new_x)
        
        # Evaluate new points
        new_evaluations = []
        for x in new_points:
            if timeout_checker and timeout_checker():
                logger.warning(f"Timeout during refinement round {round_idx+1}")
                break
                
            y = f(x)
            if y is not None:  # Skip None results
                new_evaluations.append((x, y))
        
        # Merge new points with existing grid
        all_points = list(zip(grid, values)) + new_evaluations
        all_points.sort(key=lambda p: p[0])
        
        if all_points:
            grid = np.array([p[0] for p in all_points])
            values = np.array([p[1] for p in all_points])
    
    # Find where we cross the target value
    crossings = []
    for i in range(len(values)-1):
        diff1 = values[i] - target_value
        diff2 = values[i+1] - target_value
        
        if diff1 * diff2 <= 0:  # Sign change or exact hit
            # If exact hit
            if diff1 == 0:
                crossings.append(grid[i])
            elif diff2 == 0:
                crossings.append(grid[i+1])
            else:
                # Linear interpolation to find crossing point
                t = -diff1 / (diff2 - diff1)
                x = grid[i] + t * (grid[i+1] - grid[i])
                crossings.append(x)
    
    return crossings


def batch_optimize_ci(tables: List[Tuple[int, int, int, int]], 
                     ci_function: Callable[[int, int, int, int, float, Dict[str, Any]], Tuple[Tuple[float, float], Dict[str, Any]]],
                     alpha: float = 0.05) -> Dict[Tuple[int, int, int, int], Tuple[float, float]]:
    """
    Optimize confidence interval calculation for multiple tables.
    
    Args:
        tables: List of (a,b,c,d) tuples representing 2x2 tables
        ci_function: Function to calculate CI for a single table
        alpha: Significance level
    
    Returns:
        Dictionary mapping tables to their confidence intervals
    """
    # Initialize cache
    cache = CICache()
    
    # Estimate complexity (product of counts as a simple heuristic)
    def estimate_complexity(table):
        a, b, c, d = table
        return (a+1) * (b+1) * (c+1) * (d+1)
    
    # Group similar tables based on odds ratio and total counts
    def group_similar_tables(tables_list):
        # Calculate key metrics for each table
        table_metrics = []
        for table in tables_list:
            a, b, c, d = table
            
            # Calculate odds ratio
            or_value = (a * d) / (b * c) if b * c > 0 else float('inf')
            if or_value == float('inf'):
                or_value = 100.0  # For computational purposes
            
            # Calculate total count
            total = a + b + c + d
            
            table_metrics.append((table, or_value, total))
        
        # Sort by odds ratio (primary) and total count (secondary)
        table_metrics.sort(key=lambda x: (x[1], x[2]))
        
        # Group similar tables
        groups = []
        current_group = []
        
        for i, (table, or_val, total) in enumerate(table_metrics):
            if i == 0:
                current_group = [table]
            else:
                prev_or = table_metrics[i-1][1]
                prev_total = table_metrics[i-1][2]
                
                # If similar to previous, add to current group
                or_ratio = min(or_val, prev_or) / max(or_val, prev_or)
                size_ratio = min(total, prev_total) / max(total, prev_total)
                
                if or_ratio > 0.7 and size_ratio > 0.7:
                    current_group.append(table)
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [table]
        
        if current_group:
            groups.append(current_group)
            
        return groups
    
    # Group the tables
    table_groups = group_similar_tables(tables)
    
    # Process each group
    results = {}
    start_time = time.time()
    
    for group_idx, group in enumerate(sorted(table_groups, key=lambda g: estimate_complexity(g[0]))):
        group_representative = group[0]
        logger.info(f"Processing group {group_idx+1}/{len(table_groups)} with {len(group)} tables")
        
        # Process representative first
        rep_a, rep_b, rep_c, rep_d = group_representative
        search_params = {}
        
        # For representatives, use similar entries from cache if available
        similar_entries = cache.get_similar(rep_a, rep_b, rep_c, rep_d, alpha)
        if similar_entries:
            search_params = derive_search_params(rep_a, rep_b, rep_c, rep_d, similar_entries)
        
        # Calculate CI for representative
        rep_ci, rep_details = ci_function(rep_a, rep_b, rep_c, rep_d, alpha, search_params)
        results[group_representative] = rep_ci
        
        # Add to cache
        cache.add(rep_a, rep_b, rep_c, rep_d, alpha, rep_ci, rep_details)
        
        # Process remaining tables in the group
        for table in group[1:]:
            a, b, c, d = table
            
            # Check exact cache first
            cached_result = cache.get_exact(a, b, c, d, alpha)
            if cached_result:
                results[table] = cached_result[0]
                continue
            
            # Use representative's result to guide search
            similar_entries = [((rep_a, rep_b, rep_c, rep_d), rep_ci, rep_details)]
            search_params = derive_search_params(a, b, c, d, similar_entries)
            
            # Calculate CI
            ci, details = ci_function(a, b, c, d, alpha, search_params)
            results[table] = ci
            
            # Add to cache
            cache.add(a, b, c, d, alpha, ci, details)
    
    elapsed = time.time() - start_time
    logger.info(f"Processed {len(tables)} tables in {elapsed:.2f} seconds")
    logger.info(f"Cache hit rate: {cache.cache_hits}/{cache.total_lookups} = {cache.cache_hits/max(1,cache.total_lookups):.2%}")
    
    return results


# Global cache for confidence intervals
_global_ci_cache = CICache(max_size=5000)

def get_global_cache() -> CICache:
    """Get the global CI cache instance."""
    return _global_ci_cache
