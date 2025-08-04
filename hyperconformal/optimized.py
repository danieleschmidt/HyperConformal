"""
Optimized implementations of HyperConformal classes with performance enhancements.
"""

import time
import logging
import warnings
from typing import List, Optional, Union, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
import psutil
import threading

from .hyperconformal import ConformalHDC, AdaptiveConformalHDC
from .encoders import BaseEncoder
from .optimizations import (
    CachedEncoder, BatchProcessor, MemoryPool, OptimizedSimilarityComputer,
    AdaptiveCalibration, ResourceManager, PerformanceMonitor, 
    timed_operation, get_memory_pool, get_performance_monitor, get_resource_manager
)

logger = logging.getLogger(__name__)


class OptimizedConformalHDC(ConformalHDC):
    """
    Optimized version of ConformalHDC with performance enhancements:
    - Caching for repeated computations
    - Vectorized operations
    - Memory pooling
    - Batch processing
    - GPU acceleration
    - Performance monitoring
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        score_type: str = 'aps',
        calibration_split: float = 0.2,
        device: Optional[torch.device] = None,
        validate_inputs: bool = True,
        random_state: Optional[int] = None,
        # Optimization parameters
        enable_caching: bool = True,
        cache_size: int = 1000,
        batch_size: int = 64,
        use_gpu_acceleration: bool = True,
        enable_memory_pooling: bool = True,
        num_workers: int = None,
        auto_optimize: bool = True
    ):
        """
        Initialize OptimizedConformalHDC.
        
        Args:
            encoder: HDC encoder for feature transformation
            num_classes: Number of target classes
            alpha: Miscoverage level (1-alpha is target coverage)
            score_type: Type of conformal score
            calibration_split: Fraction of training data for calibration
            device: PyTorch device
            validate_inputs: Whether to validate inputs
            random_state: Random seed for reproducibility
            enable_caching: Whether to enable encoder result caching
            cache_size: Size of the encoding cache
            batch_size: Batch size for processing
            use_gpu_acceleration: Whether to use GPU for acceleration
            enable_memory_pooling: Whether to use memory pooling
            num_workers: Number of worker threads
            auto_optimize: Whether to automatically optimize parameters
        """
        super().__init__(encoder, num_classes, alpha, score_type, calibration_split,
                         device, validate_inputs, random_state)
        
        # Optimization settings
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.use_gpu_acceleration = use_gpu_acceleration and torch.cuda.is_available()
        self.enable_memory_pooling = enable_memory_pooling
        self.num_workers = num_workers or min(4, torch.get_num_threads())
        self.auto_optimize = auto_optimize
        
        # Optimization components
        if self.enable_caching:
            self.encoder = CachedEncoder(self.encoder, cache_size)
        
        self.batch_processor = BatchProcessor(batch_size, num_workers)
        self.similarity_computer = OptimizedSimilarityComputer(
            use_gpu=self.use_gpu_acceleration,
            chunk_size=batch_size * 4
        )
        
        # Performance monitoring
        self.perf_monitor = get_performance_monitor()
        self.resource_manager = get_resource_manager()
        self.memory_pool = get_memory_pool() if enable_memory_pooling else None
        
        # Auto-optimization state
        self.optimization_stats = {}
        self.last_optimization_check = time.time()
        self.optimization_interval = 60.0  # Check every minute
        
        logger.info(f"Initialized OptimizedConformalHDC with GPU={self.use_gpu_acceleration}, "
                   f"Caching={self.enable_caching}, Batch={self.batch_size}")
    
    @timed_operation("training")
    def fit(self, 
           X: Union[torch.Tensor, np.ndarray], 
           y: Union[torch.Tensor, np.ndarray]) -> 'OptimizedConformalHDC':
        """Fit with performance optimizations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Call parent fit method
        result = super().fit(X, y)
        
        # Record performance metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        self.resource_manager.record_resource_usage(
            cpu_percent, end_memory, end_time - start_time
        )
        
        # Auto-optimize if enabled
        if self.auto_optimize:
            self._auto_optimize_parameters()
        
        logger.info(f"Optimized training completed: {end_time - start_time:.2f}s, "
                   f"Memory: {end_memory - start_memory:.1f}MB")
        
        return result
    
    @timed_operation("prediction")
    def _predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Optimized probability prediction with vectorized operations."""
        try:
            # Process in batches if large input
            if len(X) > self.batch_size:
                return self.batch_processor.process_batched(
                    X, self._predict_proba_batch
                )
            else:
                return self._predict_proba_batch(X)
        except Exception as e:
            logger.error(f"Optimized prediction failed, falling back: {e}")
            return super()._predict_proba(X)
    
    def _predict_proba_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Predict probabilities for a batch with optimizations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Encode input (uses caching if enabled)
        encoded = self.encoder.encode(X)
        
        # Validate and clean encoded output
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            warnings.warn("NaN or Inf values detected in encoded features", UserWarning)
            encoded = torch.where(
                torch.isnan(encoded) | torch.isinf(encoded),
                torch.randn_like(encoded) * 0.1,
                encoded
            )
        
        # Use optimized similarity computation
        similarities = self.similarity_computer.compute_similarities_vectorized(
            encoded, self.class_prototypes, 
            similarity_type=self._get_similarity_type()
        )
        
        # Convert to probabilities with temperature scaling
        temperature = 1.0
        try:
            probabilities = torch.softmax(similarities / temperature, dim=1)
        except Exception as e:
            logger.warning(f"Softmax failed, using uniform probabilities: {e}")
            probabilities = torch.ones_like(similarities) / self.num_classes
        
        return probabilities
    
    def _get_similarity_type(self) -> str:
        """Determine similarity type based on encoder."""
        if hasattr(self.encoder, 'quantization'):
            if self.encoder.quantization in ['binary', 'ternary']:
                return 'hamming'
            elif self.encoder.quantization == 'complex':
                return 'dot'
        return 'cosine'
    
    @timed_operation("batch_prediction")
    def predict_batch_parallel(self, 
                              X: Union[torch.Tensor, np.ndarray],
                              batch_size: Optional[int] = None,
                              num_workers: Optional[int] = None) -> np.ndarray:
        """Predict labels for large datasets with parallel processing."""
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        if len(X) <= batch_size or num_workers <= 1:
            return self.predict(X)
        
        # Split into batches
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        
        # Process batches in parallel (CPU-based parallelism)
        def predict_batch(batch):
            return self.predict(batch.cpu())
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(predict_batch, batches))
        
        return np.concatenate(results)
    
    def predict_set_optimized(self, 
                            X: Union[torch.Tensor, np.ndarray],
                            use_adaptive_threshold: bool = True) -> List[List[int]]:
        """Generate prediction sets with additional optimizations."""
        # Get probabilities
        probabilities = self.predict_proba(X)
        
        # Use adaptive threshold if enabled and we have performance history
        if use_adaptive_threshold and hasattr(self, 'adaptive_calibration'):
            # Update alpha based on recent performance
            current_alpha = self.adaptive_calibration.current_alpha
            temp_predictor = type(self.conformal_predictor)(
                current_alpha, self.conformal_predictor.score_type
            )
            if hasattr(self.conformal_predictor, 'quantile'):
                temp_predictor.quantile = self.conformal_predictor.quantile
            return temp_predictor.predict_set(probabilities)
        else:
            return self.conformal_predictor.predict_set(probabilities)
    
    def _auto_optimize_parameters(self) -> None:
        """Automatically optimize parameters based on performance metrics."""
        current_time = time.time()
        if current_time - self.last_optimization_check < self.optimization_interval:
            return
        
        self.last_optimization_check = current_time
        
        try:
            # Get performance statistics
            perf_stats = self.perf_monitor.get_stats()
            resource_stats = self.resource_manager.get_resource_stats()
            
            # Optimize batch size based on resource usage
            new_batch_size = self.resource_manager.suggest_batch_size(self.batch_size)
            if new_batch_size != self.batch_size:
                logger.info(f"Auto-optimizing batch size: {self.batch_size} -> {new_batch_size}")
                self.batch_size = new_batch_size
                self.batch_processor.batch_size = new_batch_size
            
            # Cache optimization
            if self.enable_caching and hasattr(self.encoder, 'get_cache_stats'):
                cache_stats = self.encoder.get_cache_stats()
                if cache_stats['hit_rate'] < 0.3 and cache_stats['cache_size'] < self.cache_size:
                    # Increase cache size if hit rate is low
                    new_cache_size = min(self.cache_size * 2, 5000)
                    logger.info(f"Auto-optimizing cache size: {self.cache_size} -> {new_cache_size}")
                    self.cache_size = new_cache_size
                    self.encoder.cache_size = new_cache_size
            
            # Store optimization stats
            self.optimization_stats = {
                'last_optimization': current_time,
                'performance_stats': perf_stats,
                'resource_stats': resource_stats,
                'current_batch_size': self.batch_size,
                'current_cache_size': self.cache_size
            }
            
        except Exception as e:
            logger.warning(f"Auto-optimization failed: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'batch_size': self.batch_size,
            'cache_enabled': self.enable_caching,
            'gpu_acceleration': self.use_gpu_acceleration,
            'memory_pooling': self.enable_memory_pooling,
            'num_workers': self.num_workers,
            'auto_optimize': self.auto_optimize
        }
        
        # Add performance stats
        stats['performance'] = self.perf_monitor.get_stats()
        stats['resources'] = self.resource_manager.get_resource_stats()
        
        # Add cache stats if available
        if self.enable_caching and hasattr(self.encoder, 'get_cache_stats'):
            stats['cache'] = self.encoder.get_cache_stats()
        
        # Add memory pool stats if available
        if self.memory_pool:
            stats['memory_pool'] = self.memory_pool.get_stats()
        
        return stats
    
    def benchmark(self, 
                 X: Union[torch.Tensor, np.ndarray],
                 num_runs: int = 10,
                 operations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark different operations."""
        if operations is None:
            operations = ['encode', 'predict_proba', 'predict_set']
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        results = {}
        
        for operation in operations:
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                if operation == 'encode':
                    _ = self.encoder.encode(X)
                elif operation == 'predict_proba':
                    _ = self.predict_proba(X)
                elif operation == 'predict_set':
                    _ = self.predict_set(X)
                else:
                    continue
                
                times.append(time.time() - start_time)
            
            results[operation] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput_samples_per_sec': len(X) / np.mean(times)
            }
        
        return results
    
    def clear_caches(self) -> None:
        """Clear all caches and reset optimization state."""
        if self.enable_caching and hasattr(self.encoder, 'clear_cache'):
            self.encoder.clear_cache()
        
        if self.memory_pool:
            self.memory_pool.clear()
        
        self.perf_monitor.reset()
        
        logger.info("Cleared all caches and reset optimization state")


class ScalableAdaptiveConformalHDC(AdaptiveConformalHDC, OptimizedConformalHDC):
    """
    Highly scalable adaptive conformal HDC with all optimizations.
    Combines streaming adaptation with performance optimizations.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract adaptive-specific parameters
        adaptive_params = {
            'window_size': kwargs.pop('window_size', 1000),
            'update_frequency': kwargs.pop('update_frequency', 100),
            'drift_detection': kwargs.pop('drift_detection', True),
            'max_drift_score': kwargs.pop('max_drift_score', 0.1)
        }
        
        # Initialize with optimizations
        OptimizedConformalHDC.__init__(self, *args, **kwargs)
        
        # Add adaptive components
        self.window_size = adaptive_params['window_size']
        self.update_frequency = adaptive_params['update_frequency']
        self.drift_detection = adaptive_params['drift_detection']
        self.max_drift_score = adaptive_params['max_drift_score']
        
        # Adaptive calibration
        self.adaptive_calibration = AdaptiveCalibration(
            target_coverage=1 - self.alpha,
            adaptation_rate=0.1
        )
        
        # Streaming state
        self.drift_warnings = 0
        self.baseline_stats = None
        self.update_count = 0
        
        logger.info("Initialized ScalableAdaptiveConformalHDC with full optimizations")
    
    @timed_operation("adaptive_update")
    def update(self, 
              X: Union[torch.Tensor, np.ndarray], 
              y: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Update with optimized streaming processing."""
        start_time = time.time()
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        if self.validate_inputs:
            self._validate_input_data(X, y, context="streaming update")
        
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        X, y = X.to(self.device), y.to(self.device)
        
        update_stats = {
            'samples_processed': len(X),
            'processing_time': 0.0,
            'drift_detected': False,
            'coverage_updated': False,
            'optimization_applied': False,
            'warnings': []
        }
        
        try:
            # Drift detection
            if self.drift_detection:
                drift_info = self._detect_drift_optimized(X)
                update_stats.update(drift_info)
            
            # Get predictions with batching for large updates
            if len(X) > self.batch_size:
                predictions = self.batch_processor.process_batched(
                    X, lambda batch: self._predict_proba(batch)
                )
            else:
                predictions = self._predict_proba(X)
            
            # Update conformal predictor
            self.conformal_predictor.update(predictions, y)
            self.update_count += len(X)
            
            # Adaptive calibration update
            if self.update_count % self.update_frequency == 0:
                current_coverage = self._estimate_current_coverage(X, y)
                if current_coverage is not None:
                    new_alpha = self.adaptive_calibration.update_alpha(current_coverage)
                    self.alpha = new_alpha
                    update_stats['coverage_updated'] = True
            
            # Auto-optimization check
            if self.auto_optimize and time.time() - self.last_optimization_check > 30:
                self._auto_optimize_parameters()
                update_stats['optimization_applied'] = True
            
            update_stats['processing_time'] = time.time() - start_time
            
        except Exception as e:
            update_stats['warnings'].append(f"Update failed: {e}")
            logger.error(f"Streaming update failed: {e}")
        
        return update_stats
    
    def _detect_drift_optimized(self, X: torch.Tensor) -> Dict[str, Any]:
        """Optimized drift detection using efficient statistics."""
        drift_info = {'drift_detected': False, 'drift_score': 0.0}
        
        try:
            # Compute statistics efficiently
            current_mean = torch.mean(X, dim=0)
            current_std = torch.std(X, dim=0, unbiased=False)
            
            if self.baseline_stats is None:
                self.baseline_stats = {
                    'mean': current_mean.clone(),
                    'std': current_std.clone(),
                    'count': len(X)
                }
            else:
                # Efficient drift score using KL divergence approximation
                mean_diff = torch.norm(current_mean - self.baseline_stats['mean'])
                std_ratio = torch.norm(current_std / (self.baseline_stats['std'] + 1e-8))
                
                drift_score = (mean_diff + abs(std_ratio - 1.0)).item()
                drift_info['drift_score'] = drift_score
                
                if drift_score > self.max_drift_score:
                    drift_info['drift_detected'] = True
                    self.drift_warnings += 1
                    
                    # Exponential moving average update
                    alpha = 0.1
                    self.baseline_stats['mean'] = (1 - alpha) * self.baseline_stats['mean'] + alpha * current_mean
                    self.baseline_stats['std'] = (1 - alpha) * self.baseline_stats['std'] + alpha * current_std
                    
                    logger.warning(f"Distribution drift detected: score={drift_score:.3f}")
        
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
        
        return drift_info
    
    def _estimate_current_coverage(self, X: torch.Tensor, y: torch.Tensor) -> Optional[float]:
        """Estimate current coverage for adaptive calibration."""
        try:
            pred_sets = self.predict_set(X)
            y_np = y.cpu().numpy()
            
            covered = sum(1 for i, pred_set in enumerate(pred_sets) 
                         if y_np[i] in pred_set)
            coverage = covered / len(pred_sets)
            
            return coverage
        except Exception as e:
            logger.warning(f"Coverage estimation failed: {e}")
            return None
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming and optimization statistics."""
        stats = self.get_optimization_stats()
        
        # Add adaptive-specific stats
        stats.update({
            'window_size': self.window_size,
            'update_frequency': self.update_frequency,
            'update_count': self.update_count,
            'drift_warnings': self.drift_warnings,
            'drift_detection_enabled': self.drift_detection,
            'baseline_initialized': self.baseline_stats is not None
        })
        
        # Add adaptive calibration stats
        if hasattr(self, 'adaptive_calibration'):
            stats['adaptive_calibration'] = self.adaptive_calibration.get_stats()
        
        return stats