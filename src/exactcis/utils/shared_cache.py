"""
Shared inter-process cache implementation for ExactCIs parallel processing.

This module provides process-safe caching mechanisms to eliminate 
redundant computations across worker processes.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from multiprocessing import Manager, Lock
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


class SharedProcessCache:
    """
    Process-safe cache using multiprocessing.Manager for sharing across workers.
    
    This cache enables worker processes to share expensive computation results,
    particularly CDF/SF calculations and PMF values that are repeated frequently
    during root-finding algorithms.
    """
    
    def __init__(self, max_size: int = 8192, manager=None):
        """Initialize shared cache with Manager for inter-process communication."""
        if manager is None:
            self.manager = Manager()
        else:
            self.manager = manager
        self.max_size = max_size
        
        # Separate caches for different data types
        self.cdf_cache = self.manager.dict()
        self.sf_cache = self.manager.dict()
        self.pmf_cache = self.manager.dict()
        self.support_cache = self.manager.dict()
        
        # Statistics tracking
        self.stats = self.manager.dict()
        self.stats['hits'] = 0
        self.stats['misses'] = 0
        self.stats['total_lookups'] = 0
        
        # Thread-safe access
        self.lock = self.manager.Lock()
        
        logger.info(f"Initialized SharedProcessCache with max_size={max_size}")
    
    def _evict_if_needed(self, cache_dict):
        """Simple FIFO eviction when cache is full."""
        if len(cache_dict) >= self.max_size // 4:  # Per-cache limit
            # Remove oldest 25% of entries (FIFO)
            keys_to_remove = list(cache_dict.keys())[:len(cache_dict) // 4]
            for key in keys_to_remove:
                cache_dict.pop(key, None)
    
    def get_cdf(self, a: int, N: int, c1: int, r1: int, psi: float) -> Optional[float]:
        """Get cached CDF value or None if not found."""
        # Round psi for cache stability
        psi_rounded = round(psi, 12)
        key = (a, N, c1, r1, psi_rounded)
        
        with self.lock:
            self.stats['total_lookups'] += 1
            result = self.cdf_cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
                return result
            else:
                self.stats['misses'] += 1
                return None
    
    def set_cdf(self, a: int, N: int, c1: int, r1: int, psi: float, value: float):
        """Store CDF value in cache."""
        psi_rounded = round(psi, 12)
        key = (a, N, c1, r1, psi_rounded)
        
        with self.lock:
            self._evict_if_needed(self.cdf_cache)
            self.cdf_cache[key] = value
    
    def get_sf(self, a: int, N: int, c1: int, r1: int, psi: float) -> Optional[float]:
        """Get cached SF value or None if not found."""
        psi_rounded = round(psi, 12)
        key = (a, N, c1, r1, psi_rounded)
        
        with self.lock:
            self.stats['total_lookups'] += 1
            result = self.sf_cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
                return result
            else:
                self.stats['misses'] += 1
                return None
    
    def set_sf(self, a: int, N: int, c1: int, r1: int, psi: float, value: float):
        """Store SF value in cache."""
        psi_rounded = round(psi, 12)
        key = (a, N, c1, r1, psi_rounded)
        
        with self.lock:
            self._evict_if_needed(self.sf_cache)
            self.sf_cache[key] = value
    
    def get_pmf(self, n1: int, n2: int, m1: int, theta: float, support_tuple: tuple) -> Optional[np.ndarray]:
        """Get cached PMF values or None if not found."""
        theta_rounded = round(theta, 12)
        key = (n1, n2, m1, theta_rounded, support_tuple)
        
        with self.lock:
            self.stats['total_lookups'] += 1
            result = self.pmf_cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
                # Convert back to numpy array
                return np.array(result)
            else:
                self.stats['misses'] += 1
                return None
    
    def set_pmf(self, n1: int, n2: int, m1: int, theta: float, support_tuple: tuple, values: np.ndarray):
        """Store PMF values in cache."""
        theta_rounded = round(theta, 12)
        key = (n1, n2, m1, theta_rounded, support_tuple)
        
        with self.lock:
            self._evict_if_needed(self.pmf_cache)
            # Convert numpy array to list for serialization
            self.pmf_cache[key] = values.tolist()
    
    def get_support(self, n1: int, n2: int, m1: int) -> Optional[Tuple]:
        """Get cached support data or None if not found."""
        key = (n1, n2, m1)
        
        with self.lock:
            self.stats['total_lookups'] += 1
            result = self.support_cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
                return result
            else:
                self.stats['misses'] += 1
                return None
    
    def set_support(self, n1: int, n2: int, m1: int, support_data: Tuple):
        """Store support data in cache."""
        key = (n1, n2, m1)
        
        with self.lock:
            self._evict_if_needed(self.support_cache)
            self.support_cache[key] = support_data
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.stats['total_lookups']
        if total == 0:
            return 0.0
        return 100.0 * self.stats['hits'] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'], 
            'total_lookups': self.stats['total_lookups'],
            'hit_rate_percent': self.get_hit_rate(),
            'cache_sizes': {
                'cdf': len(self.cdf_cache),
                'sf': len(self.sf_cache),
                'pmf': len(self.pmf_cache),
                'support': len(self.support_cache),
            }
        }
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.cdf_cache.clear()
            self.sf_cache.clear()
            self.pmf_cache.clear()
            self.support_cache.clear()
            self.stats['hits'] = 0
            self.stats['misses'] = 0
            self.stats['total_lookups'] = 0
        logger.info("Cleared all shared caches")


# Global shared cache instance
_shared_cache = None
_shared_manager = None


def get_shared_cache() -> SharedProcessCache:
    """Get or create the global shared cache instance."""
    global _shared_cache, _shared_manager
    if _shared_cache is None:
        if _shared_manager is None:
            _shared_manager = Manager()
        _shared_cache = SharedProcessCache()
    return _shared_cache


def init_shared_cache_for_parallel():
    """Initialize shared cache for parallel processing - call before spawning workers."""
    global _shared_cache, _shared_manager
    _shared_manager = Manager()
    _shared_cache = SharedProcessCache(manager=_shared_manager)
    return _shared_cache


def reset_shared_cache():
    """Reset the shared cache (mainly for testing)."""
    global _shared_cache, _shared_manager
    _shared_cache = None
    _shared_manager = None


def cached_cdf_function(func):
    """Decorator to add shared caching to CDF functions."""
    @wraps(func)
    def wrapper(a, N, c1, r1, psi):
        cache = get_shared_cache()
        
        # Try cache first
        cached_result = cache.get_cdf(a, N, c1, r1, psi)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = func(a, N, c1, r1, psi)
        cache.set_cdf(a, N, c1, r1, psi, result)
        return result
    
    return wrapper


def cached_sf_function(func):
    """Decorator to add shared caching to SF functions."""
    @wraps(func)
    def wrapper(a, N, c1, r1, psi):
        cache = get_shared_cache()
        
        # Try cache first
        cached_result = cache.get_sf(a, N, c1, r1, psi)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = func(a, N, c1, r1, psi)
        cache.set_sf(a, N, c1, r1, psi, result)
        return result
    
    return wrapper