"""
Advanced streaming and online learning for HyperConformal.

This module provides robust streaming HDC with conformal prediction,
designed for production environments with data drift, concept drift,
and adversarial conditions.
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import logging
import time
import threading
from abc import ABC, abstractmethod
import warnings

from .encoders import BaseEncoder
from .conformal import AdaptiveConformalPredictor
from .hyperconformal import ConformalHDC

logger = logging.getLogger(__name__)


class DriftDetector(ABC):
    """Abstract base class for drift detection algorithms."""
    
    @abstractmethod
    def detect_drift(self, new_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if drift has occurred.
        
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        pass
    
    @abstractmethod
    def update(self, new_data: np.ndarray) -> None:
        """Update internal state with new data."""
        pass


class KLDivergenceDriftDetector(DriftDetector):
    """Drift detection using KL divergence between distributions."""
    
    def __init__(self, reference_window_size: int = 1000, threshold: float = 0.1):
        self.reference_window = deque(maxlen=reference_window_size)
        self.threshold = threshold
        self.reference_stats = None
    
    def _compute_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistical summary of data."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0) + 1e-8,
            'skewness': self._compute_skewness(data),
            'kurtosis': self._compute_kurtosis(data)
        }
    
    def _compute_skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness of data."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return np.mean(((data - mean) / std) ** 3, axis=0)
    
    def _compute_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis of data."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return np.mean(((data - mean) / std) ** 4, axis=0) - 3
    
    def _kl_divergence_gaussian(self, stats1: Dict, stats2: Dict) -> float:
        """Compute KL divergence assuming Gaussian distributions."""
        mu1, sigma1 = stats1['mean'], stats1['std']
        mu2, sigma2 = stats2['mean'], stats2['std']
        
        # KL(P||Q) for multivariate Gaussians (diagonal covariance)
        kl = 0.5 * np.sum(
            np.log(sigma2 / sigma1) + 
            (sigma1**2 + (mu1 - mu2)**2) / sigma2**2 - 1
        )
        
        return float(kl)
    
    def detect_drift(self, new_data: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using KL divergence."""
        if self.reference_stats is None:
            return False, 0.0
        
        new_stats = self._compute_stats(new_data)
        kl_div = self._kl_divergence_gaussian(new_stats, self.reference_stats)
        
        drift_detected = kl_div > self.threshold
        return drift_detected, kl_div
    
    def update(self, new_data: np.ndarray) -> None:
        """Update reference distribution."""
        self.reference_window.extend(new_data)
        
        if len(self.reference_window) >= 100:  # Minimum samples for stable stats
            reference_data = np.array(list(self.reference_window))
            self.reference_stats = self._compute_stats(reference_data)


class AdversarialDetector:
    """Detect adversarial examples using conformal prediction."""
    
    def __init__(self, alpha: float = 0.01, min_set_size_threshold: float = 1.5):
        """
        Initialize adversarial detector.
        
        Args:
            alpha: Significance level for conformal sets
            min_set_size_threshold: Threshold for detecting unusual prediction sets
        """
        self.alpha = alpha
        self.min_set_size_threshold = min_set_size_threshold
        self.baseline_set_sizes = deque(maxlen=1000)
    
    def detect_adversarial(
        self, 
        prediction_sets: List[List[int]], 
        confidence_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect adversarial examples based on conformal prediction behavior.
        
        Args:
            prediction_sets: List of prediction sets for each sample
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Tuple of (adversarial_mask, adversarial_scores)
        """
        set_sizes = np.array([len(pred_set) for pred_set in prediction_sets])
        
        # Compute baseline statistics
        if len(self.baseline_set_sizes) > 100:
            baseline_mean = np.mean(list(self.baseline_set_sizes))
            baseline_std = np.std(list(self.baseline_set_sizes)) + 1e-8
        else:
            baseline_mean = np.mean(set_sizes)
            baseline_std = np.std(set_sizes) + 1e-8
        
        # Detect anomalies in set sizes
        set_size_scores = np.abs(set_sizes - baseline_mean) / baseline_std
        
        # Detect anomalies in confidence scores
        confidence_scores = np.array(confidence_scores)
        conf_mean = np.mean(confidence_scores)
        conf_std = np.std(confidence_scores) + 1e-8
        confidence_anomalies = np.abs(confidence_scores - conf_mean) / conf_std
        
        # Combined adversarial score
        adversarial_scores = set_size_scores + confidence_anomalies
        
        # Binary detection based on threshold
        adversarial_mask = (
            (set_size_scores > 3.0) |  # Set size > 3 std devs
            (confidence_scores < 0.1) |  # Very low confidence
            (set_sizes > self.min_set_size_threshold * baseline_mean)  # Large sets
        )
        
        # Update baseline (only with non-adversarial examples)
        non_adversarial_sizes = set_sizes[~adversarial_mask]
        if len(non_adversarial_sizes) > 0:
            self.baseline_set_sizes.extend(non_adversarial_sizes)
        
        return adversarial_mask, adversarial_scores


class StreamingConformalHDC(ConformalHDC):
    """
    Production-grade streaming HDC with robust drift detection,
    adversarial detection, and adaptive recalibration.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        drift_detector: Optional[DriftDetector] = None,
        adversarial_detector: Optional[AdversarialDetector] = None,
        adaptation_rate: float = 0.01,
        security_level: str = 'medium',
        monitoring_interval: int = 100,
        max_buffer_size: int = 10000,
        **kwargs
    ):
        """
        Initialize streaming conformal HDC.
        
        Args:
            encoder: HDC encoder
            num_classes: Number of classes
            alpha: Miscoverage level
            drift_detector: Drift detection algorithm
            adversarial_detector: Adversarial example detector
            adaptation_rate: Rate of adaptation to new data
            security_level: Security level ('low', 'medium', 'high')
            monitoring_interval: Samples between monitoring checks
            max_buffer_size: Maximum size of streaming buffer
        """
        super().__init__(encoder, num_classes, alpha, **kwargs)
        
        # Streaming components
        self.drift_detector = drift_detector or KLDivergenceDriftDetector()
        self.adversarial_detector = adversarial_detector or AdversarialDetector()
        self.adaptation_rate = adaptation_rate
        self.security_level = security_level
        self.monitoring_interval = monitoring_interval
        
        # Streaming buffers
        self.data_buffer = deque(maxlen=max_buffer_size)
        self.label_buffer = deque(maxlen=max_buffer_size)
        self.prediction_buffer = deque(maxlen=max_buffer_size)
        
        # Statistics tracking
        self.stream_stats = {
            'samples_processed': 0,
            'drift_detections': 0,
            'adversarial_detections': 0,
            'model_updates': 0,
            'processing_times': deque(maxlen=1000),
            'coverage_history': deque(maxlen=1000),
            'accuracy_history': deque(maxlen=1000)
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._adaptation_pending = False
        
        # Security features
        if security_level == 'high':
            self._enable_enhanced_security()
    
    def _enable_enhanced_security(self):
        """Enable enhanced security features."""
        # Rate limiting
        self.max_requests_per_second = 1000
        self.request_timestamps = deque(maxlen=self.max_requests_per_second)
        
        # Input validation
        self.input_validators = [
            self._validate_input_range,
            self._validate_input_statistics,
            self._validate_input_adversarial
        ]
        
        # Logging for security monitoring
        self.security_logger = logging.getLogger(f"{__name__}.security")
        self.security_logger.info("Enhanced security mode enabled")
    
    def _validate_input_range(self, x: np.ndarray) -> bool:
        """Validate input is within expected range."""
        return not (np.any(np.isnan(x)) or np.any(np.isinf(x)))
    
    def _validate_input_statistics(self, x: np.ndarray) -> bool:
        """Validate input statistics are reasonable."""
        # Check for extreme values that might indicate attacks
        z_scores = np.abs((x - np.mean(x)) / (np.std(x) + 1e-8))
        return np.all(z_scores < 10)  # No values > 10 standard deviations
    
    def _validate_input_adversarial(self, x: np.ndarray) -> bool:
        """Basic adversarial input detection."""
        # Simple gradient-based detection (would be more sophisticated in practice)
        if len(x.shape) > 1:
            gradients = np.diff(x, axis=1)
            max_gradient = np.max(np.abs(gradients))
            return max_gradient < 5.0  # Threshold for "natural" gradients
        return True
    
    def _rate_limit_check(self) -> bool:
        """Check if request is within rate limits."""
        if not hasattr(self, 'max_requests_per_second'):
            return True
        
        current_time = time.time()
        self.request_timestamps.append(current_time)
        
        # Count requests in last second
        recent_requests = sum(
            1 for t in self.request_timestamps 
            if current_time - t < 1.0
        )
        
        return recent_requests <= self.max_requests_per_second
    
    def predict_stream(
        self, 
        x: Union[torch.Tensor, np.ndarray], 
        return_diagnostics: bool = False
    ) -> Union[List[List[int]], Tuple[List[List[int]], Dict[str, Any]]]:
        """
        Make predictions on streaming data with full monitoring.
        
        Args:
            x: Input features
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            Prediction sets, optionally with diagnostics
        """
        start_time = time.time()
        
        # Thread safety
        with self._lock:
            # Rate limiting check
            if self.security_level == 'high' and not self._rate_limit_check():
                if return_diagnostics:
                    return [], {'error': 'Rate limit exceeded'}
                else:
                    warnings.warn("Rate limit exceeded", UserWarning)
                    return []
            
            # Input validation
            if isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
            else:
                x_np = x.copy()
            
            # Security validation
            if self.security_level == 'high':
                for validator in self.input_validators:
                    if not validator(x_np):
                        if hasattr(self, 'security_logger'):
                            self.security_logger.warning("Input validation failed")
                        if return_diagnostics:
                            return [], {'error': 'Input validation failed'}
                        else:
                            return []
            
            # Make prediction
            prediction_sets = self.predict_set(x)
            
            # Compute confidence scores for adversarial detection
            probabilities = self.predict_proba(x).cpu().numpy()
            confidence_scores = np.max(probabilities, axis=1)
            
            # Adversarial detection
            adversarial_mask, adversarial_scores = self.adversarial_detector.detect_adversarial(
                prediction_sets, confidence_scores
            )
            
            # Drift detection (sample from batch for efficiency)
            if len(x_np) > 10:
                sample_indices = np.random.choice(len(x_np), size=10, replace=False)
                drift_sample = x_np[sample_indices]
            else:
                drift_sample = x_np
            
            drift_detected, drift_score = self.drift_detector.detect_drift(drift_sample)
            
            # Update statistics
            self.stream_stats['samples_processed'] += len(x_np)
            self.stream_stats['processing_times'].append(time.time() - start_time)
            
            if drift_detected:
                self.stream_stats['drift_detections'] += 1
                logger.warning(f"Drift detected with score: {drift_score:.4f}")
            
            adversarial_count = np.sum(adversarial_mask)
            if adversarial_count > 0:
                self.stream_stats['adversarial_detections'] += adversarial_count
                if hasattr(self, 'security_logger'):
                    self.security_logger.warning(f"Detected {adversarial_count} adversarial examples")
            
            # Store data for potential adaptation
            self.data_buffer.extend(x_np)
            
            # Schedule adaptation if needed
            if (drift_detected or adversarial_count > 0.1 * len(x_np)) and not self._adaptation_pending:
                self._schedule_adaptation()
            
            # Update drift detector
            self.drift_detector.update(drift_sample)
            
            # Prepare diagnostics
            diagnostics = {
                'processing_time': time.time() - start_time,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'adversarial_detected': adversarial_mask,
                'adversarial_scores': adversarial_scores,
                'prediction_confidence': confidence_scores,
                'stream_stats': self.get_stream_statistics()
            }
            
            if return_diagnostics:
                return prediction_sets, diagnostics
            else:
                return prediction_sets
    
    def update_stream(
        self, 
        x: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Update model with new labeled streaming data.
        
        Args:
            x: Input features
            y: True labels
            
        Returns:
            Update statistics
        """
        with self._lock:
            if isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
            else:
                x_np = x.copy()
            
            if isinstance(y, torch.Tensor):
                y_np = y.cpu().numpy()
            else:
                y_np = y.copy()
            
            # Store true labels for accuracy computation
            self.label_buffer.extend(y_np)
            
            # Compute current accuracy
            if len(self.prediction_buffer) > 0 and len(self.label_buffer) > 0:
                recent_predictions = list(self.prediction_buffer)[-len(y_np):]
                recent_labels = list(self.label_buffer)[-len(y_np):]
                
                if len(recent_predictions) == len(recent_labels):
                    # Compute accuracy (fraction of labels in prediction sets)
                    accuracy = np.mean([
                        label in pred_set 
                        for pred_set, label in zip(recent_predictions, recent_labels)
                    ])
                    self.stream_stats['accuracy_history'].append(accuracy)
            
            # Adaptive model update
            if len(self.data_buffer) >= self.monitoring_interval:
                return self._perform_adaptive_update(x_np, y_np)
            
            return {'status': 'buffered', 'buffer_size': len(self.data_buffer)}
    
    def _schedule_adaptation(self):
        """Schedule model adaptation in background thread."""
        if self._adaptation_pending:
            return
        
        self._adaptation_pending = True
        
        def adaptation_worker():
            try:
                time.sleep(0.1)  # Small delay to batch updates
                with self._lock:
                    if len(self.data_buffer) > 50 and len(self.label_buffer) > 50:
                        # Use recent data for adaptation
                        recent_data = np.array(list(self.data_buffer)[-500:])
                        recent_labels = np.array(list(self.label_buffer)[-500:])
                        
                        self._perform_adaptive_update(recent_data, recent_labels)
                
            except Exception as e:
                logger.error(f"Adaptation failed: {e}")
            finally:
                self._adaptation_pending = False
        
        # Start adaptation in background
        thread = threading.Thread(target=adaptation_worker, daemon=True)
        thread.start()
    
    def _perform_adaptive_update(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Perform adaptive model update."""
        start_time = time.time()
        
        try:
            # Incremental prototype update
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float().to(self.device)
            else:
                x_tensor = x
            
            # Encode new data
            encoded_x = self.encoder.encode(x_tensor)
            
            # Update class prototypes using exponential moving average
            for class_idx in range(self.num_classes):
                class_mask = (y == class_idx)
                if class_mask.sum() > 0:
                    new_prototype = encoded_x[class_mask].mean(dim=0)
                    
                    if self.class_prototypes is not None:
                        # EMA update: θ_new = (1-α)θ_old + α*θ_new
                        self.class_prototypes[class_idx] = (
                            (1 - self.adaptation_rate) * self.class_prototypes[class_idx] +
                            self.adaptation_rate * new_prototype
                        )
            
            # Update conformal predictor with new data
            probabilities = self.predict_proba(x_tensor).cpu().numpy()
            
            # Incremental calibration update (simplified)
            if hasattr(self.conformal_predictor, 'update'):
                self.conformal_predictor.update(probabilities, y)
            
            self.stream_stats['model_updates'] += 1
            update_time = time.time() - start_time
            
            logger.info(f"Model updated with {len(x)} samples in {update_time:.3f}s")
            
            return {
                'status': 'updated',
                'samples': len(x),
                'update_time': update_time,
                'adaptation_rate': self.adaptation_rate
            }
            
        except Exception as e:
            logger.error(f"Adaptive update failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        recent_times = list(self.stream_stats['processing_times'])
        recent_coverage = list(self.stream_stats['coverage_history'])
        recent_accuracy = list(self.stream_stats['accuracy_history'])
        
        stats = {
            'samples_processed': self.stream_stats['samples_processed'],
            'drift_detections': self.stream_stats['drift_detections'],
            'adversarial_detections': self.stream_stats['adversarial_detections'],
            'model_updates': self.stream_stats['model_updates'],
            'buffer_size': len(self.data_buffer),
            'adaptation_pending': self._adaptation_pending
        }
        
        if recent_times:
            stats.update({
                'avg_processing_time': np.mean(recent_times),
                'processing_throughput': 1.0 / np.mean(recent_times) if recent_times else 0
            })
        
        if recent_coverage:
            stats.update({
                'recent_coverage': np.mean(recent_coverage),
                'coverage_trend': np.polyfit(range(len(recent_coverage)), recent_coverage, 1)[0]
            })
        
        if recent_accuracy:
            stats.update({
                'recent_accuracy': np.mean(recent_accuracy),
                'accuracy_trend': np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
            })
        
        return stats
    
    def export_monitoring_data(self) -> Dict[str, Any]:
        """Export data for external monitoring systems."""
        return {
            'timestamp': time.time(),
            'model_info': {
                'encoder_type': type(self.encoder).__name__,
                'num_classes': self.num_classes,
                'alpha': self.alpha,
                'hv_dim': self.encoder.hv_dim
            },
            'statistics': self.get_stream_statistics(),
            'health': self.health_check(),
            'security_level': self.security_level,
            'memory_usage': self.memory_footprint()
        }
    
    def reset_statistics(self):
        """Reset all streaming statistics."""
        with self._lock:
            self.stream_stats = {
                'samples_processed': 0,
                'drift_detections': 0,
                'adversarial_detections': 0,
                'model_updates': 0,
                'processing_times': deque(maxlen=1000),
                'coverage_history': deque(maxlen=1000),
                'accuracy_history': deque(maxlen=1000)
            }
            
            # Reset detectors
            if hasattr(self.drift_detector, 'reference_stats'):
                self.drift_detector.reference_stats = None
            
            if hasattr(self.adversarial_detector, 'baseline_set_sizes'):
                self.adversarial_detector.baseline_set_sizes.clear()
    
    def configure_security(self, level: str, **kwargs):
        """Reconfigure security settings."""
        valid_levels = ['low', 'medium', 'high']
        if level not in valid_levels:
            raise ValueError(f"Security level must be one of {valid_levels}")
        
        self.security_level = level
        
        if level == 'high' and not hasattr(self, 'input_validators'):
            self._enable_enhanced_security()
        
        # Update rate limiting
        if 'max_requests_per_second' in kwargs:
            self.max_requests_per_second = kwargs['max_requests_per_second']
        
        logger.info(f"Security level set to {level}")


# Factory function for easy creation
def create_streaming_hdc(
    input_dim: int,
    num_classes: int,
    hv_dim: int = 10000,
    alpha: float = 0.1,
    encoder_type: str = 'random_projection',
    drift_detection: str = 'kl_divergence',
    security_level: str = 'medium',
    **kwargs
) -> StreamingConformalHDC:
    """
    Factory function to create configured streaming HDC.
    
    Args:
        input_dim: Input dimension
        num_classes: Number of classes
        hv_dim: Hypervector dimension
        alpha: Miscoverage level
        encoder_type: Type of encoder ('random_projection', 'level_hdc', 'complex')
        drift_detection: Drift detection method ('kl_divergence')
        security_level: Security level ('low', 'medium', 'high')
        
    Returns:
        Configured StreamingConformalHDC instance
    """
    # Create encoder
    if encoder_type == 'random_projection':
        from .encoders import RandomProjection
        encoder = RandomProjection(input_dim, hv_dim, quantization='binary')
    elif encoder_type == 'level_hdc':
        from .encoders import LevelHDC
        encoder = LevelHDC(input_dim, hv_dim, levels=100)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create drift detector
    if drift_detection == 'kl_divergence':
        drift_detector = KLDivergenceDriftDetector()
    else:
        drift_detector = None
    
    # Create adversarial detector
    adversarial_detector = AdversarialDetector()
    
    return StreamingConformalHDC(
        encoder=encoder,
        num_classes=num_classes,
        alpha=alpha,
        drift_detector=drift_detector,
        adversarial_detector=adversarial_detector,
        security_level=security_level,
        **kwargs
    )