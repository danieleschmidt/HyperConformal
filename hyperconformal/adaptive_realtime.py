"""
Real-time Adaptive Conformal Prediction for HyperConformal

This module provides real-time adaptive conformal prediction capabilities
with dynamic calibration, concept drift detection, and streaming updates.
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import logging
import time
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field
import threading
import queue
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive conformal prediction."""
    window_size: int = 1000
    update_frequency: int = 100
    min_calibration_samples: int = 50
    drift_detection_threshold: float = 0.05
    adaptation_rate: float = 0.1
    memory_efficient: bool = True
    real_time_processing: bool = True


@dataclass
class StreamingMetrics:
    """Metrics for streaming conformal prediction."""
    coverage_history: List[float] = field(default_factory=list)
    set_size_history: List[float] = field(default_factory=list)
    adaptation_events: List[Dict[str, Any]] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    drift_detections: List[Dict[str, Any]] = field(default_factory=list)


class ConceptDriftDetector:
    """Detect concept drift in streaming data."""
    
    def __init__(
        self,
        window_size: int = 1000,
        detection_method: str = "adwin",
        sensitivity: float = 0.05
    ):
        self.window_size = window_size
        self.detection_method = detection_method
        self.sensitivity = sensitivity
        
        # Sliding window for statistics
        self.error_window = deque(maxlen=window_size)
        self.coverage_window = deque(maxlen=window_size)
        
        # ADWIN parameters
        self.adwin_threshold = sensitivity
        self.adwin_buckets = []
        
        logger.info(f"Drift detector initialized: {detection_method}, sensitivity={sensitivity}")
    
    def update(self, error: float, coverage: float) -> bool:
        """Update drift detector and return True if drift detected."""
        self.error_window.append(error)
        self.coverage_window.append(coverage)
        
        if len(self.error_window) < 50:  # Need minimum samples
            return False
        
        if self.detection_method == "adwin":
            return self._adwin_drift_detection(error)
        elif self.detection_method == "statistical":
            return self._statistical_drift_detection()
        elif self.detection_method == "coverage":
            return self._coverage_drift_detection(coverage)
        else:
            return False
    
    def _adwin_drift_detection(self, error: float) -> bool:
        """ADWIN (Adaptive Windowing) drift detection."""
        # Simplified ADWIN implementation
        recent_errors = list(self.error_window)[-100:]  # Recent window
        historical_errors = list(self.error_window)[:-100]  # Historical window
        
        if len(recent_errors) < 30 or len(historical_errors) < 30:
            return False
        
        recent_mean = np.mean(recent_errors)
        historical_mean = np.mean(historical_errors)
        
        # Statistical test for difference in means
        pooled_std = np.sqrt(
            (np.var(recent_errors) + np.var(historical_errors)) / 2
        )
        
        if pooled_std > 0:
            t_stat = abs(recent_mean - historical_mean) / pooled_std
            drift_detected = t_stat > (1.96 * self.sensitivity)  # 95% confidence
            
            if drift_detected:
                logger.warning(f"ADWIN drift detected: t_stat={t_stat:.4f}")
            
            return drift_detected
        
        return False
    
    def _statistical_drift_detection(self) -> bool:
        """Statistical test for drift in error distribution."""
        if len(self.error_window) < 100:
            return False
        
        # Split window into two halves
        mid_point = len(self.error_window) // 2
        first_half = list(self.error_window)[:mid_point]
        second_half = list(self.error_window)[mid_point:]
        
        # Kolmogorov-Smirnov test (simplified)
        first_sorted = np.sort(first_half)
        second_sorted = np.sort(second_half)
        
        # Compute empirical CDFs and max difference
        max_diff = 0
        for value in np.linspace(0, 1, 100):
            cdf1 = np.mean(first_sorted <= value)
            cdf2 = np.mean(second_sorted <= value)
            max_diff = max(max_diff, abs(cdf1 - cdf2))
        
        # Critical value for KS test
        n1, n2 = len(first_half), len(second_half)
        critical_value = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))
        
        drift_detected = max_diff > critical_value * self.sensitivity
        
        if drift_detected:
            logger.warning(f"Statistical drift detected: KS_stat={max_diff:.4f}")
        
        return drift_detected
    
    def _coverage_drift_detection(self, coverage: float) -> bool:
        """Detect drift based on coverage deviation."""
        if len(self.coverage_window) < 100:
            return False
        
        recent_coverage = np.mean(list(self.coverage_window)[-50:])
        target_coverage = 0.9  # Assume 90% target
        
        coverage_deviation = abs(recent_coverage - target_coverage)
        drift_detected = coverage_deviation > self.sensitivity
        
        if drift_detected:
            logger.warning(f"Coverage drift detected: deviation={coverage_deviation:.4f}")
        
        return drift_detected


class AdaptiveQuantileTracker:
    """Track and adapt quantiles in real-time for conformal prediction."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        adaptation_rate: float = 0.1,
        min_samples: int = 50
    ):
        self.alpha = alpha
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        
        # Current quantile estimate
        self.current_quantile = 0.0
        self.sample_count = 0
        
        # Exponential moving statistics
        self.ema_mean = 0.0
        self.ema_variance = 0.0
        
        logger.info(f"Adaptive quantile tracker initialized: alpha={alpha}")
    
    def update(self, new_score: float) -> float:
        """Update quantile estimate with new nonconformity score."""
        self.sample_count += 1
        
        # Update exponential moving average
        if self.sample_count == 1:
            self.ema_mean = new_score
            self.ema_variance = 0.0
        else:
            delta = new_score - self.ema_mean
            self.ema_mean += self.adaptation_rate * delta
            self.ema_variance = (1 - self.adaptation_rate) * self.ema_variance + \
                              self.adaptation_rate * delta * delta
        
        # Adaptive quantile estimation
        if self.sample_count >= self.min_samples:
            # Use normal approximation for quantile
            std_dev = np.sqrt(self.ema_variance)
            if std_dev > 0:
                # Z-score for (1-alpha) quantile
                z_score = torch.tensor(1 - self.alpha).float()
                z_score = torch.erfinv(2 * z_score - 1) * np.sqrt(2)
                
                self.current_quantile = self.ema_mean + z_score * std_dev
            else:
                self.current_quantile = self.ema_mean
        else:
            # Use simple quantile estimate for small samples
            self.current_quantile = self.ema_mean + 2 * np.sqrt(self.ema_variance)
        
        return float(self.current_quantile)
    
    def get_quantile(self) -> float:
        """Get current quantile estimate."""
        return float(self.current_quantile)
    
    def reset(self):
        """Reset tracker state."""
        self.current_quantile = 0.0
        self.sample_count = 0
        self.ema_mean = 0.0
        self.ema_variance = 0.0


class RealTimeProcessor:
    """Real-time processing engine for streaming conformal prediction."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_queue_size: int = 1000,
        num_workers: int = 2
    ):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        
        # Processing statistics
        self.processed_count = 0
        self.error_count = 0
        self.avg_processing_time = 0.0
        
        logger.info(f"Real-time processor initialized: {num_workers} workers, batch_size={batch_size}")
    
    def start(self, processing_function: Callable):
        """Start real-time processing workers."""
        self.running = True
        self.processing_function = processing_function
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} processing workers")
    
    def stop(self):
        """Stop real-time processing."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.info("Real-time processing stopped")
    
    def submit(self, data: Any) -> bool:
        """Submit data for processing."""
        try:
            self.input_queue.put(data, block=False)
            return True
        except queue.Full:
            logger.warning("Input queue full, dropping data")
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Any]:
        """Get processing result."""
        try:
            return self.output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self, worker_id: int):
        """Worker thread processing loop."""
        batch_buffer = []
        
        while self.running:
            try:
                # Collect batch
                while len(batch_buffer) < self.batch_size and self.running:
                    try:
                        item = self.input_queue.get(timeout=0.1)
                        batch_buffer.append(item)
                    except queue.Empty:
                        break
                
                if batch_buffer:
                    start_time = time.time()
                    
                    # Process batch
                    try:
                        results = self.processing_function(batch_buffer)
                        
                        # Put results in output queue
                        for result in results:
                            try:
                                self.output_queue.put(result, block=False)
                            except queue.Full:
                                logger.warning("Output queue full, dropping result")
                        
                        # Update statistics
                        processing_time = time.time() - start_time
                        self.processed_count += len(batch_buffer)
                        self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time
                        
                    except Exception as e:
                        logger.error(f"Worker {worker_id} processing error: {e}")
                        self.error_count += len(batch_buffer)
                    
                    batch_buffer.clear()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.processed_count),
            'avg_processing_time': self.avg_processing_time,
            'queue_sizes': {
                'input': self.input_queue.qsize(),
                'output': self.output_queue.qsize()
            }
        }


class StreamingAdaptiveConformalHDC:
    """
    Real-time adaptive conformal HDC with concept drift detection
    and streaming calibration updates.
    """
    
    def __init__(
        self,
        encoder,
        num_classes: int,
        config: AdaptiveConfig,
        alpha: float = 0.1
    ):
        self.encoder = encoder
        self.num_classes = num_classes
        self.config = config
        self.alpha = alpha
        
        # Adaptive components
        self.quantile_tracker = AdaptiveQuantileTracker(alpha, config.adaptation_rate)
        self.drift_detector = ConceptDriftDetector(
            config.window_size, "adwin", config.drift_detection_threshold
        )
        
        # Real-time processor
        self.real_time_processor = None
        if config.real_time_processing:
            self.real_time_processor = RealTimeProcessor()
        
        # Streaming data structures
        self.calibration_window = deque(maxlen=config.window_size)
        self.class_prototypes = {}
        self.update_counter = 0
        
        # Metrics tracking
        self.metrics = StreamingMetrics()
        
        # Threading lock for thread-safety
        self.lock = threading.Lock()
        
        logger.info(f"Streaming adaptive conformal HDC initialized")
    
    def initialize_prototypes(self, initial_data: torch.Tensor, initial_labels: torch.Tensor):
        """Initialize class prototypes with initial data."""
        with self.lock:
            unique_classes = torch.unique(initial_labels)
            
            for class_idx in unique_classes:
                class_mask = initial_labels == class_idx
                class_samples = initial_data[class_mask]
                
                if len(class_samples) > 0:
                    # Encode and bundle class samples
                    encoded_samples = self.encoder.encode(class_samples)
                    class_prototype = torch.mean(encoded_samples, dim=0)
                    self.class_prototypes[int(class_idx)] = class_prototype
            
            logger.info(f"Initialized {len(self.class_prototypes)} class prototypes")
    
    def stream_update(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process streaming update with adaptive calibration."""
        start_time = time.time()
        
        with self.lock:
            # Encode input
            encoded_x = self.encoder.encode(x.unsqueeze(0) if x.dim() == 1 else x)
            
            # Generate prediction set
            prediction_set = self._predict_set_internal(encoded_x[0])
            
            # Update calibration if true label available
            if y_true is not None:
                self._update_calibration(encoded_x[0], y_true)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.processing_times.append(processing_time)
            
            # Check for adaptation needs
            adaptation_triggered = self._check_adaptation()
            
            return {
                'prediction_set': prediction_set,
                'processing_time': processing_time,
                'adaptation_triggered': adaptation_triggered,
                'current_quantile': self.quantile_tracker.get_quantile(),
                'update_count': self.update_counter
            }
    
    def _predict_set_internal(self, encoded_x: torch.Tensor) -> List[int]:
        """Internal prediction set generation."""
        if not self.class_prototypes:
            return list(range(self.num_classes))  # Return all classes if no prototypes
        
        prediction_set = []
        current_quantile = self.quantile_tracker.get_quantile()
        
        for class_idx, prototype in self.class_prototypes.items():
            # Compute nonconformity score
            similarity = torch.cosine_similarity(encoded_x, prototype, dim=0)
            nonconformity_score = 1.0 - similarity
            
            # Include class if score <= quantile
            if nonconformity_score <= current_quantile:
                prediction_set.append(class_idx)
        
        # Ensure non-empty prediction set
        if not prediction_set:
            # Fall back to most similar class
            similarities = {}
            for class_idx, prototype in self.class_prototypes.items():
                sim = torch.cosine_similarity(encoded_x, prototype, dim=0)
                similarities[class_idx] = sim
            
            best_class = max(similarities.keys(), key=lambda k: similarities[k])
            prediction_set = [best_class]
        
        return prediction_set
    
    def _update_calibration(self, encoded_x: torch.Tensor, y_true: torch.Tensor):
        """Update calibration with new labeled sample."""
        true_class = int(y_true.item())
        
        # Update class prototype if needed
        if true_class not in self.class_prototypes:
            self.class_prototypes[true_class] = encoded_x.clone()
        else:
            # Exponential moving average update
            current_prototype = self.class_prototypes[true_class]
            self.class_prototypes[true_class] = (
                (1 - self.config.adaptation_rate) * current_prototype +
                self.config.adaptation_rate * encoded_x
            )
        
        # Compute nonconformity score for calibration
        if true_class in self.class_prototypes:
            true_prototype = self.class_prototypes[true_class]
            similarity = torch.cosine_similarity(encoded_x, true_prototype, dim=0)
            nonconformity_score = 1.0 - similarity
            
            # Update quantile tracker
            new_quantile = self.quantile_tracker.update(float(nonconformity_score))
            
            # Add to calibration window
            self.calibration_window.append({
                'score': float(nonconformity_score),
                'class': true_class,
                'timestamp': time.time()
            })
            
            # Compute current coverage for drift detection
            recent_samples = list(self.calibration_window)[-50:]
            if len(recent_samples) >= 10:
                coverage = self._compute_empirical_coverage(recent_samples)
                
                # Update drift detector
                error_rate = 1.0 - coverage if coverage is not None else 0.5
                drift_detected = self.drift_detector.update(error_rate, coverage or 0.5)
                
                if drift_detected:
                    self._handle_concept_drift()
                
                # Record metrics
                if coverage is not None:
                    self.metrics.coverage_history.append(coverage)
        
        self.update_counter += 1
    
    def _compute_empirical_coverage(self, recent_samples: List[Dict]) -> Optional[float]:
        """Compute empirical coverage on recent samples."""
        if len(recent_samples) < 5:
            return None
        
        covered_count = 0
        total_count = 0
        
        for sample in recent_samples:
            # Reconstruct prediction for coverage check
            # This is simplified - in practice, you'd store the prediction sets
            total_count += 1
            # Assume coverage if score <= current quantile (approximation)
            if sample['score'] <= self.quantile_tracker.get_quantile():
                covered_count += 1
        
        return covered_count / total_count if total_count > 0 else None
    
    def _check_adaptation(self) -> bool:
        """Check if adaptation should be triggered."""
        if self.update_counter % self.config.update_frequency == 0:
            if len(self.calibration_window) >= self.config.min_calibration_samples:
                return True
        return False
    
    def _handle_concept_drift(self):
        """Handle detected concept drift."""
        logger.warning("Concept drift detected - triggering adaptation")
        
        # Reset quantile tracker to adapt faster
        self.quantile_tracker.adaptation_rate = min(0.5, self.quantile_tracker.adaptation_rate * 2)
        
        # Record drift event
        drift_event = {
            'timestamp': time.time(),
            'update_count': self.update_counter,
            'action': 'increased_adaptation_rate',
            'new_rate': self.quantile_tracker.adaptation_rate
        }
        self.metrics.drift_detections.append(drift_event)
        
        # Optional: Reset older calibration samples
        if self.config.memory_efficient:
            # Keep only recent samples
            recent_size = min(len(self.calibration_window), self.config.window_size // 2)
            recent_samples = list(self.calibration_window)[-recent_size:]
            self.calibration_window.clear()
            self.calibration_window.extend(recent_samples)
    
    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics."""
        with self.lock:
            metrics_dict = {
                'total_updates': self.update_counter,
                'calibration_window_size': len(self.calibration_window),
                'current_quantile': self.quantile_tracker.get_quantile(),
                'num_prototypes': len(self.class_prototypes),
                'coverage_history': self.metrics.coverage_history[-100:],  # Recent history
                'avg_processing_time': np.mean(self.metrics.processing_times[-100:]) if self.metrics.processing_times else 0,
                'drift_detections': len(self.metrics.drift_detections),
                'adaptation_rate': self.quantile_tracker.adaptation_rate
            }
            
            # Add real-time processor stats if available
            if self.real_time_processor:
                metrics_dict['processor_stats'] = self.real_time_processor.get_stats()
            
            return metrics_dict
    
    def start_real_time_mode(self):
        """Start real-time processing mode."""
        if self.real_time_processor and self.config.real_time_processing:
            processing_fn = lambda batch: [
                self.stream_update(item['x'], item.get('y_true'))
                for item in batch
            ]
            self.real_time_processor.start(processing_fn)
            logger.info("Real-time processing mode started")
    
    def stop_real_time_mode(self):
        """Stop real-time processing mode."""
        if self.real_time_processor:
            self.real_time_processor.stop()
            logger.info("Real-time processing mode stopped")


# Export main classes
__all__ = [
    'AdaptiveConfig',
    'StreamingMetrics',
    'ConceptDriftDetector',
    'AdaptiveQuantileTracker',
    'RealTimeProcessor',
    'StreamingAdaptiveConformalHDC'
]