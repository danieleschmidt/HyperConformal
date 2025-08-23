"""
Performance optimization and scaling framework for HyperConformal.
Implements caching, batching, parallel processing, and auto-scaling.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
import pickle
import hashlib
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 128
    enable_batching: bool = True
    batch_size: int = 32
    enable_parallel: bool = True
    num_workers: int = None
    use_gpu_batching: bool = True
    prefetch_factor: int = 2


class LRUCache:
    """Thread-safe LRU cache for tensor operations."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.current_time = 0
        self.lock = threading.RLock()
    
    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Create hash key for tensor."""
        # Use tensor data, shape, and dtype for hash
        if tensor.numel() > 1000:  # For large tensors, sample for efficiency
            sampled = tensor.flatten()[::max(1, tensor.numel() // 100)]
            data_bytes = sampled.cpu().numpy().tobytes()
        else:
            data_bytes = tensor.cpu().numpy().tobytes()
        
        shape_str = str(tuple(tensor.shape))
        dtype_str = str(tensor.dtype)
        
        return hashlib.md5(
            data_bytes + shape_str.encode() + dtype_str.encode()
        ).hexdigest()
    
    def get(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached result."""
        with self.lock:
            hash_key = self._hash_tensor(key)
            if hash_key in self.cache:
                self.current_time += 1
                self.access_times[hash_key] = self.current_time
                return self.cache[hash_key].clone()
            return None
    
    def put(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store result in cache."""
        with self.lock:
            hash_key = self._hash_tensor(key)
            
            # Evict least recently used if at capacity
            if len(self.cache) >= self.max_size and hash_key not in self.cache:
                lru_key = min(self.access_times.keys(), 
                             key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.current_time += 1
            self.cache[hash_key] = value.clone()
            self.access_times[hash_key] = self.current_time
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_time = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size
            }


class BatchProcessor:
    """Intelligent batching for tensor operations."""
    
    def __init__(self, batch_size: int = 32, device: torch.device = None):
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
    
    def process_batched(self, data: torch.Tensor, 
                       processor_func: Callable[[torch.Tensor], torch.Tensor],
                       progress_callback: Optional[Callable] = None) -> torch.Tensor:
        """Process data in batches to manage memory."""
        results = []
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = processor_func(batch)
            results.append(batch_result)
            
            if progress_callback:
                progress_callback(i // self.batch_size + 1, num_batches)
        
        return torch.cat(results, dim=0)
    
    def optimal_batch_size(self, data_size: int, memory_limit_mb: int = 1024) -> int:
        """Estimate optimal batch size based on available memory."""
        # Simple heuristic: assume each sample needs ~1KB
        max_batch_from_memory = (memory_limit_mb * 1024) // data_size
        return min(self.batch_size, max_batch_from_memory, data_size)


class ParallelProcessor:
    """Parallel processing for CPU-intensive operations."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or min(8, mp.cpu_count())
        self._thread_pool = None
        self._process_pool = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Cleanup thread and process pools."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
    
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._thread_pool
    
    @property 
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool executor."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        return self._process_pool
    
    def map_threaded(self, func: Callable, data: List[Any]) -> List[Any]:
        """Map function over data using threads."""
        return list(self.thread_pool.map(func, data))
    
    def map_processes(self, func: Callable, data: List[Any]) -> List[Any]:
        """Map function over data using processes."""
        return list(self.process_pool.map(func, data))


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "prediction_times": [],
            "batch_sizes": [],
            "memory_usage": [],
            "throughput": []
        }
        self.config = PerformanceConfig()
        self.lock = threading.Lock()
    
    def record_prediction(self, duration: float, batch_size: int, 
                         memory_mb: float, throughput: float) -> None:
        """Record performance metrics."""
        with self.lock:
            self.metrics["prediction_times"].append(duration)
            self.metrics["batch_sizes"].append(batch_size)
            self.metrics["memory_usage"].append(memory_mb)
            self.metrics["throughput"].append(throughput)
            
            # Keep only recent metrics
            for key in self.metrics:
                if len(self.metrics[key]) > 100:
                    self.metrics[key] = self.metrics[key][-100:]
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on metrics."""
        with self.lock:
            if not self.metrics["prediction_times"]:
                return {"message": "No metrics available"}
            
            avg_time = np.mean(self.metrics["prediction_times"])
            avg_memory = np.mean(self.metrics["memory_usage"])
            avg_throughput = np.mean(self.metrics["throughput"])
            
            recommendations = {}
            
            # Batch size recommendations
            if avg_time > 1.0:  # If predictions are slow
                current_batch = np.mean(self.metrics["batch_sizes"])
                recommendations["batch_size"] = int(current_batch * 1.5)
                recommendations["reason"] = "Increase batch size for better throughput"
            
            # Memory optimization
            if avg_memory > 512:  # If using too much memory
                recommendations["enable_gradient_checkpointing"] = True
                recommendations["reduce_precision"] = "fp16"
            
            # Parallel processing
            if avg_throughput < 100:  # If throughput is low
                recommendations["enable_parallel"] = True
                recommendations["num_workers"] = min(8, mp.cpu_count())
            
            return recommendations
    
    def auto_tune(self) -> PerformanceConfig:
        """Automatically tune performance configuration."""
        recommendations = self.get_recommendations()
        
        if "batch_size" in recommendations:
            self.config.batch_size = recommendations["batch_size"]
        
        if "enable_parallel" in recommendations:
            self.config.enable_parallel = recommendations["enable_parallel"]
            if "num_workers" in recommendations:
                self.config.num_workers = recommendations["num_workers"]
        
        return self.config


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = LRUCache(self.config.cache_size) if self.config.enable_caching else None
        self.batch_processor = BatchProcessor(self.config.batch_size) if self.config.enable_batching else None
        self.parallel_processor = ParallelProcessor(self.config.num_workers) if self.config.enable_parallel else None
        self.auto_scaler = AutoScaler()
        
        logger.info(f"Performance optimizer initialized: {self.config}")
    
    def __enter__(self):
        if self.parallel_processor:
            self.parallel_processor.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parallel_processor:
            self.parallel_processor.__exit__(exc_type, exc_val, exc_tb)
    
    def cached_encode(self, encoder_func: Callable, data: torch.Tensor) -> torch.Tensor:
        """Cached encoding operation."""
        if not self.cache:
            return encoder_func(data)
        
        # Check cache first
        cached_result = self.cache.get(data)
        if cached_result is not None:
            logger.debug("Cache hit for encoding operation")
            return cached_result
        
        # Compute and cache result
        start_time = time.time()
        result = encoder_func(data)
        computation_time = time.time() - start_time
        
        self.cache.put(data, result)
        logger.debug(f"Encoded and cached result in {computation_time:.3f}s")
        
        return result
    
    def batched_prediction(self, predictor_func: Callable, 
                          data: torch.Tensor) -> torch.Tensor:
        """Batched prediction with performance monitoring."""
        if not self.batch_processor:
            return predictor_func(data)
        
        start_time = time.time()
        
        # Use optimal batch size
        optimal_batch_size = self.batch_processor.optimal_batch_size(
            data.shape[1] * 4,  # Estimate memory per sample
            memory_limit_mb=512
        )
        
        if optimal_batch_size != self.batch_processor.batch_size:
            logger.info(f"Using optimal batch size: {optimal_batch_size}")
            self.batch_processor.batch_size = optimal_batch_size
        
        result = self.batch_processor.process_batched(data, predictor_func)
        
        # Record metrics
        duration = time.time() - start_time
        memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        throughput = len(data) / duration
        
        self.auto_scaler.record_prediction(duration, optimal_batch_size, memory_mb, throughput)
        
        return result
    
    def parallel_similarity(self, similarity_func: Callable, 
                           data: List[torch.Tensor]) -> List[float]:
        """Parallel similarity computations."""
        if not self.parallel_processor or len(data) < 4:
            return [similarity_func(item) for item in data]
        
        return self.parallel_processor.map_threaded(similarity_func, data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "config": self.config.__dict__,
            "auto_scaling_recommendations": self.auto_scaler.get_recommendations()
        }
        
        if self.cache:
            stats["cache"] = self.cache.stats()
        
        return stats
    
    def optimize_for_inference(self) -> None:
        """Optimize configuration for inference workloads."""
        self.config.enable_caching = True
        self.config.cache_size = 256
        self.config.enable_batching = True
        self.config.batch_size = 64
        self.config.enable_parallel = True
        
        logger.info("Optimized configuration for inference")
    
    def optimize_for_training(self) -> None:
        """Optimize configuration for training workloads."""
        self.config.enable_caching = False  # Training data changes
        self.config.enable_batching = True
        self.config.batch_size = 32
        self.config.enable_parallel = True
        
        logger.info("Optimized configuration for training")


# Performance decorators
def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {duration:.3f}s")
        return result
    return wrapper


def memory_efficient(func: Callable) -> Callable:
    """Decorator to make functions more memory efficient."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = func(*args, **kwargs)
        
        # Clear cache again after computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    return wrapper


# Global performance optimizer
_global_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

def set_performance_config(config: PerformanceConfig) -> None:
    """Set global performance configuration."""
    global _global_optimizer
    _global_optimizer = PerformanceOptimizer(config)

def enable_performance_optimization() -> None:
    """Enable all performance optimizations."""
    config = PerformanceConfig(
        enable_caching=True,
        cache_size=128,
        enable_batching=True,
        batch_size=32,
        enable_parallel=True,
        num_workers=min(8, mp.cpu_count())
    )
    set_performance_config(config)