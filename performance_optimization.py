#!/usr/bin/env python3
"""
Performance Optimization Framework for HyperConformal
Generation 3: High-performance computing with caching and optimization
"""

import time
import functools
import threading
import multiprocessing
from typing import Dict, List, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import hashlib

class PerformanceCache:
    """High-performance caching system with LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _compute_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Compute cache key from function arguments."""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return True, self.cache[key]
            else:
                self.misses += 1
                return False, None
    
    def put(self, key: str, value: Any):
        """Put value in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Evict least recently used
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

# Global cache instance
_global_cache = PerformanceCache(max_size=1000)

def cached(cache_size: int = None):
    """Decorator for function result caching."""
    def decorator(func: Callable) -> Callable:
        cache = PerformanceCache(cache_size or 1000)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            cache_key = cache._compute_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            hit, value = cache.get(cache_key)
            if hit:
                return value
            
            # Compute value
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            
            return result
        
        # Add cache stats method
        wrapper.cache_stats = lambda: cache.stats()
        wrapper.cache_clear = lambda: cache.cache.clear()
        
        return wrapper
    return decorator

class OptimizedHDCEncoder:
    """High-performance HDC encoder with vectorized operations."""
    
    def __init__(self, input_dim: int, hv_dim: int, quantization: str = 'binary'):
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.quantization = quantization
        
        # Pre-compute random projection matrix
        self.projection_matrix = self._generate_projection_matrix()
        
        # Performance tracking
        self.encode_times = []
        self.throughput_history = []
    
    def _generate_projection_matrix(self) -> List[List[int]]:
        """Generate optimized projection matrix."""
        import random
        random.seed(42)  # Deterministic for consistency
        
        matrix = []
        for i in range(self.input_dim):
            row = [random.choice([0, 1]) for _ in range(self.hv_dim)]
            matrix.append(row)
        
        return matrix
    
    @cached(cache_size=500)
    def encode_single(self, input_vector: tuple) -> tuple:
        """Encode single vector with caching."""
        return tuple(self._encode_vector_optimized(list(input_vector)))
    
    def _encode_vector_optimized(self, input_vector: List[float]) -> List[int]:
        """Optimized vector encoding."""
        if len(input_vector) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: {len(input_vector)} != {self.input_dim}")
        
        # Binary quantization
        binary_input = [1 if x > 0 else 0 for x in input_vector]
        
        # Vectorized projection (simulated)
        hypervector = [0] * self.hv_dim
        for i, bit in enumerate(binary_input):
            if bit == 1:
                for j in range(self.hv_dim):
                    hypervector[j] ^= self.projection_matrix[i][j]
        
        return hypervector
    
    def encode_batch(self, input_batch: List[List[float]], 
                    use_parallel: bool = True) -> List[List[int]]:
        """High-performance batch encoding."""
        start_time = time.time()
        
        if use_parallel and len(input_batch) > 4:
            # Parallel processing for large batches
            results = self._encode_batch_parallel(input_batch)
        else:
            # Sequential processing for small batches
            results = [self._encode_vector_optimized(vec) for vec in input_batch]
        
        # Performance tracking
        duration = time.time() - start_time
        throughput = len(input_batch) / duration if duration > 0 else 0
        
        self.encode_times.append(duration)
        self.throughput_history.append(throughput)
        
        # Keep only recent measurements
        if len(self.encode_times) > 100:
            self.encode_times = self.encode_times[-100:]
            self.throughput_history = self.throughput_history[-100:]
        
        return results
    
    def _encode_batch_parallel(self, input_batch: List[List[float]]) -> List[List[int]]:
        """Parallel batch encoding using thread pool."""
        num_workers = min(multiprocessing.cpu_count(), len(input_batch))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._encode_vector_optimized, vec) 
                      for vec in input_batch]
            results = [future.result() for future in futures]
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get encoding performance statistics."""
        if not self.encode_times:
            return {"status": "no_data"}
        
        avg_time = sum(self.encode_times) / len(self.encode_times)
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        return {
            "avg_encode_time": avg_time,
            "avg_throughput": avg_throughput,
            "total_operations": len(self.encode_times),
            "cache_stats": self.encode_single.cache_stats() if hasattr(self.encode_single, 'cache_stats') else {}
        }

class OptimizedConformalPredictor:
    """High-performance conformal predictor with optimizations."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores = []
        self.prediction_times = []
        self.quantile_cache = {}
        
    @cached(cache_size=100)
    def compute_quantile_cached(self, scores_hash: str, alpha: float) -> float:
        """Cached quantile computation."""
        return self._compute_quantile_optimized(self.calibration_scores, alpha)
    
    def _compute_quantile_optimized(self, scores: List[float], alpha: float) -> float:
        """Optimized quantile computation."""
        if not scores:
            return 0.0
        
        # Fast quantile computation
        n = len(scores)
        import math
        q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
        
        # Use partial sorting for efficiency (simulated)
        sorted_scores = sorted(scores)
        return sorted_scores[q_index]
    
    def calibrate_optimized(self, scores: List[float]):
        """Optimized calibration with preprocessing."""
        # Store sorted scores for fast quantile computation
        self.calibration_scores = sorted(scores)
        
        # Clear quantile cache
        self.quantile_cache.clear()
        if hasattr(self.compute_quantile_cached, 'cache_clear'):
            self.compute_quantile_cached.cache_clear()
    
    def predict_batch(self, score_batches: List[List[float]], 
                     use_vectorized: bool = True) -> List[List[int]]:
        """High-performance batch prediction."""
        start_time = time.time()
        
        if use_vectorized and len(score_batches) > 1:
            results = self._predict_batch_vectorized(score_batches)
        else:
            results = [self._predict_single_optimized(scores) for scores in score_batches]
        
        # Performance tracking
        duration = time.time() - start_time
        self.prediction_times.append(duration)
        
        return results
    
    def _predict_batch_vectorized(self, score_batches: List[List[float]]) -> List[List[int]]:
        """Vectorized batch prediction."""
        # Compute quantile once
        quantile = self._compute_quantile_optimized(self.calibration_scores, self.alpha)
        
        # Vectorized prediction set generation
        results = []
        for scores in score_batches:
            prediction_set = [i for i, score in enumerate(scores) if score >= quantile]
            results.append(prediction_set)
        
        return results
    
    def _predict_single_optimized(self, scores: List[float]) -> List[int]:
        """Optimized single prediction."""
        quantile = self._compute_quantile_optimized(self.calibration_scores, self.alpha)
        return [i for i, score in enumerate(scores) if score >= quantile]

class ResourcePool:
    """Resource pooling for expensive operations."""
    
    def __init__(self, factory_func: Callable, pool_size: int = 4):
        self.factory_func = factory_func
        self.pool = queue.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self.created_count = 0
        self.usage_count = 0
        
        # Pre-populate pool
        for _ in range(pool_size):
            resource = factory_func()
            self.pool.put(resource)
            self.created_count += 1
    
    def acquire(self, timeout: float = 1.0):
        """Acquire resource from pool."""
        try:
            resource = self.pool.get(timeout=timeout)
            self.usage_count += 1
            return resource
        except queue.Empty:
            # Create new resource if pool is empty
            resource = self.factory_func()
            self.created_count += 1
            self.usage_count += 1
            return resource
    
    def release(self, resource):
        """Release resource back to pool."""
        try:
            self.pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, let resource be garbage collected
            pass
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self.pool_size,
            'available': self.pool.qsize(),
            'created_total': self.created_count,
            'usage_count': self.usage_count
        }

# Performance testing
if __name__ == "__main__":
    print("⚡ Testing Performance Optimization Framework")
    print("="*50)
    
    # Test caching
    @cached(cache_size=100)
    def expensive_computation(x: int) -> int:
        time.sleep(0.001)  # Simulate expensive operation
        return x * x
    
    # Test cache performance
    start_time = time.time()
    for i in range(50):
        result = expensive_computation(i % 10)  # High cache hit rate
    cache_time = time.time() - start_time
    
    print(f"Cache test completed in {cache_time:.3f}s")
    print(f"Cache stats: {expensive_computation.cache_stats()}")
    
    # Test optimized HDC encoder
    encoder = OptimizedHDCEncoder(input_dim=100, hv_dim=1000)
    
    # Single encoding test
    test_vector = [0.5 * i for i in range(100)]
    encoded = encoder.encode_single(tuple(test_vector))
    print(f"Single encoding result: {len(encoded)} dimensions")
    
    # Batch encoding test
    test_batch = [[0.1 * i * j for i in range(100)] for j in range(10)]
    start_time = time.time()
    encoded_batch = encoder.encode_batch(test_batch, use_parallel=True)
    batch_time = time.time() - start_time
    
    print(f"Batch encoding: {len(encoded_batch)} vectors in {batch_time:.3f}s")
    print(f"Encoder stats: {encoder.get_performance_stats()}")
    
    # Test optimized conformal predictor
    predictor = OptimizedConformalPredictor(alpha=0.1)
    
    # Calibration
    calibration_scores = [0.1 * i for i in range(100)]
    predictor.calibrate_optimized(calibration_scores)
    
    # Batch prediction
    score_batches = [
        [0.7, 0.3, 0.9, 0.1],
        [0.6, 0.8, 0.2, 0.4],
        [0.9, 0.1, 0.5, 0.3]
    ]
    
    start_time = time.time()
    predictions = predictor.predict_batch(score_batches, use_vectorized=True)
    prediction_time = time.time() - start_time
    
    print(f"Batch prediction: {len(predictions)} sets in {prediction_time:.3f}s")
    print(f"Prediction results: {predictions}")
    
    # Test resource pool
    def create_encoder():
        return OptimizedHDCEncoder(input_dim=50, hv_dim=500)
    
    encoder_pool = ResourcePool(create_encoder, pool_size=3)
    
    # Use resources from pool
    encoder1 = encoder_pool.acquire()
    encoder2 = encoder_pool.acquire()
    print(f"Pool stats after acquisition: {encoder_pool.stats()}")
    
    encoder_pool.release(encoder1)
    encoder_pool.release(encoder2)
    print(f"Pool stats after release: {encoder_pool.stats()}")
    
    print("\n🎉 Performance optimization tests completed!")
