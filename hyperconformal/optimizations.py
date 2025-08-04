"""
Performance optimization utilities for HyperConformal.
Includes caching, batching, and parallel processing optimizations.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import threading
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        with self.lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.metrics[f"{operation}_time"].append(duration)
                del self.start_times[operation]
                return duration
            return 0.0
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a custom metric."""
        with self.lock:
            self.metrics[name].append(value)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        with self.lock:
            for metric_name, values in self.metrics.items():
                if values:
                    stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()


def timed_operation(operation_name: str):
    """Decorator to time operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'perf_monitor'):
                self.perf_monitor.start_timer(operation_name)
                result = func(self, *args, **kwargs)
                self.perf_monitor.end_timer(operation_name)
                return result
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class CachedEncoder:
    """Wrapper for encoders with caching capability."""
    
    def __init__(self, encoder, cache_size: int = 1000):
        self.encoder = encoder
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
    
    def _get_cache_key(self, x: torch.Tensor) -> str:
        """Generate cache key for input tensor."""
        # Use hash of tensor values for caching
        try:
            # Convert to numpy and then to bytes for hashing
            return str(hash(x.detach().cpu().numpy().tobytes()))
        except:
            # Fallback: use tensor shape and some values
            return f"{x.shape}_{x.sum().item():.6f}_{x.mean().item():.6f}"
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with caching."""
        cache_key = self._get_cache_key(x)
        
        with self.lock:
            if cache_key in self._cache:
                self.cache_hits += 1
                # Move to end (most recent)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return self._cache[cache_key].clone()
            
            self.cache_misses += 1
        
        # Compute encoding
        encoded = self.encoder.encode(x)
        
        with self.lock:
            # Add to cache
            if len(self._cache) >= self.cache_size:
                # Remove oldest
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[cache_key] = encoded.clone()
            self._access_order.append(cache_key)
        
        return encoded
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size
            }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        with self.lock:
            self._cache.clear()
            self._access_order.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped encoder."""
        return getattr(self.encoder, name)


class BatchProcessor:
    """Efficient batch processing for predictions."""
    
    def __init__(self, batch_size: int = 64, num_workers: int = None):
        self.batch_size = batch_size
        self.num_workers = num_workers or min(4, mp.cpu_count())
    
    def process_batched(self, 
                       data: torch.Tensor, 
                       process_fn,
                       combine_fn=torch.cat) -> torch.Tensor:
        """Process data in batches."""
        if len(data) <= self.batch_size:
            return process_fn(data)
        
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = process_fn(batch)
            results.append(batch_result)
        
        return combine_fn(results, dim=0)
    
    def process_parallel_batches(self,
                               data: torch.Tensor,
                               process_fn,
                               combine_fn=torch.cat) -> torch.Tensor:
        """Process batches in parallel (CPU only)."""
        if len(data) <= self.batch_size or self.num_workers <= 1:
            return self.process_batched(data, process_fn, combine_fn)
        
        # Split into batches
        batches = [data[i:i + self.batch_size] 
                  for i in range(0, len(data), self.batch_size)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_fn, batches))
        
        return combine_fn(results, dim=0)


class MemoryPool:
    """Memory pool for tensor reuse to reduce allocations."""
    
    def __init__(self):
        self.pools = defaultdict(list)  # shape -> list of tensors
        self.lock = threading.Lock()
        self.allocations = 0
        self.reuses = 0
    
    def get_tensor(self, shape: Tuple[int, ...], dtype=torch.float32, device=None) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one."""
        key = (shape, dtype, device)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()  # Clear contents
                self.reuses += 1
                return tensor
        
        # Allocate new tensor
        self.allocations += 1
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse."""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        
        with self.lock:
            if len(self.pools[key]) < 10:  # Limit pool size
                self.pools[key].append(tensor.detach())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_requests = self.allocations + self.reuses
            reuse_rate = self.reuses / total_requests if total_requests > 0 else 0.0
            
            pool_sizes = {str(key): len(tensors) for key, tensors in self.pools.items()}
            
            return {
                'allocations': self.allocations,
                'reuses': self.reuses,
                'reuse_rate': reuse_rate,
                'pool_sizes': pool_sizes,
                'total_pooled_tensors': sum(len(tensors) for tensors in self.pools.values())
            }
    
    def clear(self) -> None:
        """Clear all pools."""
        with self.lock:
            self.pools.clear()
            self.allocations = 0
            self.reuses = 0


class OptimizedSimilarityComputer:
    """Optimized similarity computation with vectorization and GPU acceleration."""
    
    def __init__(self, use_gpu: bool = True, chunk_size: int = 1000):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.chunk_size = chunk_size
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
    
    def compute_similarities_vectorized(self,
                                      encoded: torch.Tensor,
                                      prototypes: torch.Tensor,
                                      similarity_type: str = 'cosine') -> torch.Tensor:
        """Compute similarities using vectorized operations."""
        # Move to optimal device
        encoded = encoded.to(self.device)
        prototypes = prototypes.to(self.device)
        
        if similarity_type == 'cosine':
            # Normalize for cosine similarity
            encoded_norm = F.normalize(encoded, dim=-1)
            prototypes_norm = F.normalize(prototypes, dim=-1)
            similarities = torch.mm(encoded_norm, prototypes_norm.t())
            
        elif similarity_type == 'dot':
            similarities = torch.mm(encoded, prototypes.t())
            
        elif similarity_type == 'hamming':
            # For binary vectors
            similarities = torch.mm(encoded, prototypes.t()) / encoded.shape[-1]
            
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarities
    
    def compute_similarities_chunked(self,
                                   encoded: torch.Tensor,
                                   prototypes: torch.Tensor,
                                   similarity_type: str = 'cosine') -> torch.Tensor:
        """Compute similarities in chunks to manage memory."""
        if len(encoded) <= self.chunk_size:
            return self.compute_similarities_vectorized(encoded, prototypes, similarity_type)
        
        results = []
        for i in range(0, len(encoded), self.chunk_size):
            chunk = encoded[i:i + self.chunk_size]
            chunk_similarities = self.compute_similarities_vectorized(
                chunk, prototypes, similarity_type
            )
            results.append(chunk_similarities.cpu())  # Move back to CPU to save GPU memory
        
        return torch.cat(results, dim=0)


class AdaptiveCalibration:
    """Adaptive calibration that adjusts based on performance metrics."""
    
    def __init__(self, target_coverage: float = 0.9, adaptation_rate: float = 0.1):
        self.target_coverage = target_coverage
        self.adaptation_rate = adaptation_rate
        self.coverage_history = []
        self.alpha_history = []
        self.current_alpha = 1 - target_coverage
    
    def update_alpha(self, observed_coverage: float) -> float:
        """Update alpha based on observed coverage."""
        self.coverage_history.append(observed_coverage)
        self.alpha_history.append(self.current_alpha)
        
        # Simple adaptive adjustment
        coverage_error = self.target_coverage - observed_coverage
        alpha_adjustment = self.adaptation_rate * coverage_error
        
        # Update alpha with bounds checking
        new_alpha = self.current_alpha + alpha_adjustment
        self.current_alpha = max(0.01, min(0.99, new_alpha))
        
        logger.debug(f"Coverage: {observed_coverage:.3f}, Target: {self.target_coverage:.3f}, "
                    f"New alpha: {self.current_alpha:.3f}")
        
        return self.current_alpha
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        if not self.coverage_history:
            return {}
        
        return {
            'current_alpha': self.current_alpha,
            'mean_coverage': np.mean(self.coverage_history),
            'coverage_std': np.std(self.coverage_history),
            'coverage_trend': np.polyfit(range(len(self.coverage_history)), 
                                       self.coverage_history, 1)[0] if len(self.coverage_history) > 1 else 0,
            'num_updates': len(self.coverage_history)
        }


class ResourceManager:
    """Manage computational resources and auto-scaling."""
    
    def __init__(self):
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.processing_times = []
        self.lock = threading.Lock()
    
    def record_resource_usage(self, cpu_percent: float, memory_mb: float, processing_time: float):
        """Record resource usage metrics."""
        with self.lock:
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_mb)
            self.processing_times.append(processing_time)
            
            # Keep only recent history
            max_history = 100
            if len(self.cpu_usage_history) > max_history:
                self.cpu_usage_history = self.cpu_usage_history[-max_history:]
                self.memory_usage_history = self.memory_usage_history[-max_history:]
                self.processing_times = self.processing_times[-max_history:]
    
    def suggest_batch_size(self, current_batch_size: int) -> int:
        """Suggest optimal batch size based on resource usage."""
        if len(self.processing_times) < 5:
            return current_batch_size
        
        with self.lock:
            recent_cpu = np.mean(self.cpu_usage_history[-10:]) if self.cpu_usage_history else 50
            recent_memory = np.mean(self.memory_usage_history[-10:]) if self.memory_usage_history else 500
            recent_times = np.mean(self.processing_times[-10:]) if self.processing_times else 1.0
        
        # Simple heuristic for batch size adjustment
        if recent_cpu < 50 and recent_memory < 1000:  # Low resource usage
            return min(current_batch_size * 2, 512)
        elif recent_cpu > 90 or recent_memory > 2000:  # High resource usage
            return max(current_batch_size // 2, 8)
        else:
            return current_batch_size
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        with self.lock:
            if not self.cpu_usage_history:
                return {}
            
            return {
                'avg_cpu_usage': np.mean(self.cpu_usage_history),
                'avg_memory_mb': np.mean(self.memory_usage_history),
                'avg_processing_time': np.mean(self.processing_times),
                'cpu_trend': np.polyfit(range(len(self.cpu_usage_history)), 
                                       self.cpu_usage_history, 1)[0] if len(self.cpu_usage_history) > 1 else 0,
                'memory_trend': np.polyfit(range(len(self.memory_usage_history)), 
                                         self.memory_usage_history, 1)[0] if len(self.memory_usage_history) > 1 else 0
            }


# Global instances for shared use
_global_memory_pool = MemoryPool()
_global_perf_monitor = PerformanceMonitor()
_global_resource_manager = ResourceManager()

def get_memory_pool() -> MemoryPool:
    """Get global memory pool instance."""
    return _global_memory_pool

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _global_perf_monitor

def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    return _global_resource_manager