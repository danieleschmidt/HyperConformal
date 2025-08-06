"""
Security features for HyperConformal systems.

This module provides comprehensive security features including:
- Adversarial example detection and mitigation
- Privacy-preserving techniques (differential privacy, federated learning)
- Input validation and sanitization
- Cryptographic verification of model integrity
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import hashlib
import hmac
import secrets
import logging
from abc import ABC, abstractmethod
import warnings
from collections import deque

from .encoders import BaseEncoder
from .hyperconformal import ConformalHDC

logger = logging.getLogger(__name__)

# Optional dependencies for advanced security features
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available. Some security features will be limited.")

try:
    import opacus
    from opacus import PrivacyEngine
    DIFFERENTIAL_PRIVACY_AVAILABLE = True
except ImportError:
    DIFFERENTIAL_PRIVACY_AVAILABLE = False


class AdversarialDefense(ABC):
    """Abstract base class for adversarial defense mechanisms."""
    
    @abstractmethod
    def detect_adversarial(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect adversarial examples.
        
        Args:
            x: Input samples
            
        Returns:
            Tuple of (is_adversarial_mask, adversarial_scores)
        """
        pass
    
    @abstractmethod
    def mitigate_adversarial(self, x: np.ndarray, is_adversarial: np.ndarray) -> np.ndarray:
        """
        Mitigate detected adversarial examples.
        
        Args:
            x: Input samples
            is_adversarial: Boolean mask indicating adversarial examples
            
        Returns:
            Cleaned input samples
        """
        pass


class StatisticalAdversarialDefense(AdversarialDefense):
    """Statistical methods for adversarial detection and mitigation."""
    
    def __init__(
        self, 
        reference_window_size: int = 1000,
        detection_threshold: float = 3.0,
        mitigation_method: str = 'clip'
    ):
        """
        Initialize statistical adversarial defense.
        
        Args:
            reference_window_size: Size of reference distribution window
            detection_threshold: Z-score threshold for detection
            mitigation_method: Method for mitigation ('clip', 'gaussian_noise', 'median')
        """
        self.reference_window = deque(maxlen=reference_window_size)
        self.detection_threshold = detection_threshold
        self.mitigation_method = mitigation_method
        self.reference_stats = None
    
    def _update_reference_stats(self):
        """Update reference distribution statistics."""
        if len(self.reference_window) < 50:
            return
        
        reference_data = np.array(list(self.reference_window))
        self.reference_stats = {
            'mean': np.mean(reference_data, axis=0),
            'std': np.std(reference_data, axis=0) + 1e-8,
            'median': np.median(reference_data, axis=0),
            'mad': np.median(np.abs(reference_data - np.median(reference_data, axis=0)), axis=0) + 1e-8
        }
    
    def detect_adversarial(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect adversarial examples using statistical analysis."""
        if self.reference_stats is None:
            # No reference available, return no detections
            return np.zeros(len(x), dtype=bool), np.zeros(len(x))
        
        # Compute z-scores relative to reference distribution
        z_scores = np.abs(x - self.reference_stats['mean']) / self.reference_stats['std']
        
        # Use maximum z-score across features as adversarial score
        adversarial_scores = np.max(z_scores, axis=1)
        
        # Detect based on threshold
        is_adversarial = adversarial_scores > self.detection_threshold
        
        return is_adversarial, adversarial_scores
    
    def mitigate_adversarial(self, x: np.ndarray, is_adversarial: np.ndarray) -> np.ndarray:
        """Mitigate adversarial examples."""
        if self.reference_stats is None:
            return x.copy()
        
        x_clean = x.copy()
        
        if self.mitigation_method == 'clip':
            # Clip to reasonable range based on reference distribution
            lower_bound = self.reference_stats['mean'] - 3 * self.reference_stats['std']
            upper_bound = self.reference_stats['mean'] + 3 * self.reference_stats['std']
            
            x_clean[is_adversarial] = np.clip(
                x_clean[is_adversarial], 
                lower_bound, 
                upper_bound
            )
            
        elif self.mitigation_method == 'gaussian_noise':
            # Add Gaussian noise to adversarial examples
            noise_scale = 0.1 * self.reference_stats['std']
            noise = np.random.normal(0, noise_scale, x[is_adversarial].shape)
            x_clean[is_adversarial] += noise
            
        elif self.mitigation_method == 'median':
            # Replace with reference median
            x_clean[is_adversarial] = self.reference_stats['median']
        
        return x_clean
    
    def update_reference(self, x: np.ndarray, is_clean: Optional[np.ndarray] = None):
        """Update reference distribution with new clean samples."""
        if is_clean is None:
            # Assume all samples are clean
            self.reference_window.extend(x)
        else:
            # Only use clean samples
            clean_samples = x[is_clean]
            if len(clean_samples) > 0:
                self.reference_window.extend(clean_samples)
        
        # Update statistics
        self._update_reference_stats()


class DifferentialPrivacyMechanism:
    """Differential privacy mechanisms for protecting training data."""
    
    def __init__(
        self, 
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: str = 'gaussian'
    ):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability for (ε,δ)-DP
            sensitivity: Global sensitivity of the function
            mechanism: Noise mechanism ('gaussian', 'laplace')
        """
        if not DIFFERENTIAL_PRIVACY_AVAILABLE:
            logger.warning("Opacus not available. Differential privacy features will be limited.")
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
        # Compute noise scale
        if mechanism == 'gaussian':
            # Gaussian mechanism for (ε,δ)-DP
            self.noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        elif mechanism == 'laplace':
            # Laplace mechanism for ε-DP
            self.noise_scale = self.sensitivity / epsilon
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    
    def add_noise(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Add calibrated noise for differential privacy."""
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.cpu().numpy()
        else:
            device = None
            x_np = x
        
        # Generate noise
        if self.mechanism == 'gaussian':
            noise = np.random.normal(0, self.noise_scale, x_np.shape)
        elif self.mechanism == 'laplace':
            noise = np.random.laplace(0, self.noise_scale, x_np.shape)
        
        # Add noise
        x_private = x_np + noise
        
        # Convert back to original type
        if device is not None:
            return torch.from_numpy(x_private.astype(np.float32)).to(device)
        else:
            return x_private
    
    def privacy_budget_used(self, num_queries: int) -> Dict[str, float]:
        """Compute privacy budget used for given number of queries."""
        if self.mechanism == 'gaussian':
            # Advanced composition for Gaussian mechanism
            sigma = self.noise_scale
            epsilon_used = num_queries * self.epsilon / np.sqrt(num_queries)  # Simplified
            delta_used = num_queries * self.delta
        else:
            # Basic composition for Laplace
            epsilon_used = num_queries * self.epsilon
            delta_used = 0
        
        return {
            'epsilon_used': epsilon_used,
            'delta_used': delta_used,
            'budget_remaining': max(0, self.epsilon - epsilon_used)
        }


class ModelIntegrityVerifier:
    """Cryptographic verification of model integrity."""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize model integrity verifier.
        
        Args:
            secret_key: Secret key for HMAC (if None, generates random key)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for model integrity verification")
        
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.model_hashes = {}
    
    def compute_model_hash(self, model: ConformalHDC) -> bytes:
        """Compute cryptographic hash of model parameters."""
        # Serialize model parameters
        model_data = []
        
        # Include class prototypes
        if model.class_prototypes is not None:
            model_data.append(model.class_prototypes.cpu().numpy().tobytes())
        
        # Include encoder parameters if available
        if hasattr(model.encoder, 'projection_matrix') and model.encoder.projection_matrix is not None:
            model_data.append(model.encoder.projection_matrix.cpu().numpy().tobytes())
        
        # Include conformal predictor state
        if hasattr(model.conformal_predictor, 'quantile') and model.conformal_predictor.quantile is not None:
            model_data.append(str(model.conformal_predictor.quantile).encode())
        
        # Concatenate all data
        combined_data = b''.join(model_data)
        
        # Compute HMAC
        return hmac.new(self.secret_key, combined_data, hashlib.sha256).digest()
    
    def sign_model(self, model: ConformalHDC, model_id: str) -> bytes:
        """Sign model with cryptographic signature."""
        model_hash = self.compute_model_hash(model)
        self.model_hashes[model_id] = model_hash
        return model_hash
    
    def verify_model(self, model: ConformalHDC, model_id: str, expected_signature: bytes) -> bool:
        """Verify model integrity against known signature."""
        current_hash = self.compute_model_hash(model)
        return hmac.compare_digest(current_hash, expected_signature)
    
    def get_model_fingerprint(self, model: ConformalHDC) -> str:
        """Get human-readable fingerprint of model."""
        model_hash = self.compute_model_hash(model)
        return hashlib.sha256(model_hash).hexdigest()[:16]  # First 16 chars


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(
        self,
        feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        max_batch_size: int = 1000,
        rate_limit_per_minute: int = 10000
    ):
        """
        Initialize input validator.
        
        Args:
            feature_ranges: Expected ranges for each feature dimension
            max_batch_size: Maximum allowed batch size
            rate_limit_per_minute: Maximum requests per minute
        """
        self.feature_ranges = feature_ranges or {}
        self.max_batch_size = max_batch_size
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Rate limiting
        self.request_timestamps = deque()
        self.validation_stats = {
            'total_requests': 0,
            'rejected_requests': 0,
            'sanitized_requests': 0
        }
    
    def validate_input(
        self, 
        x: Union[np.ndarray, torch.Tensor], 
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate input data.
        
        Args:
            x: Input data to validate
            strict: Whether to use strict validation
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Convert to numpy for validation
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Rate limiting check
        current_time = time.time()
        self.request_timestamps.append(current_time)
        
        # Remove old timestamps (older than 1 minute)
        cutoff_time = current_time - 60
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()
        
        if len(self.request_timestamps) > self.rate_limit_per_minute:
            errors.append("Rate limit exceeded")
        
        # Basic validation
        if x_np.size == 0:
            errors.append("Empty input")
        
        if len(x_np.shape) != 2:
            errors.append(f"Expected 2D input, got {len(x_np.shape)}D")
        
        if x_np.shape[0] > self.max_batch_size:
            errors.append(f"Batch size {x_np.shape[0]} exceeds maximum {self.max_batch_size}")
        
        # Check for NaN/Inf values
        if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
            errors.append("Input contains NaN or Inf values")
        
        # Feature range validation
        if self.feature_ranges and len(x_np.shape) >= 2:
            for feature_idx, (min_val, max_val) in self.feature_ranges.items():
                if feature_idx < x_np.shape[1]:
                    feature_values = x_np[:, feature_idx]
                    
                    if strict:
                        if np.any(feature_values < min_val) or np.any(feature_values > max_val):
                            errors.append(f"Feature {feature_idx} out of range [{min_val}, {max_val}]")
                    else:
                        # For non-strict mode, just warn about extreme outliers
                        outliers = np.sum((feature_values < min_val * 10) | (feature_values > max_val * 10))
                        if outliers > 0:
                            logger.warning(f"Feature {feature_idx} has {outliers} extreme outliers")
        
        # Statistical validation
        if len(x_np.shape) >= 2 and x_np.shape[1] > 1:
            # Check for unusual statistical properties
            feature_means = np.mean(x_np, axis=0)
            feature_stds = np.std(x_np, axis=0)
            
            # Detect features with zero variance (potential attacks)
            zero_variance_features = np.sum(feature_stds < 1e-10)
            if zero_variance_features > x_np.shape[1] * 0.5:  # >50% zero variance
                errors.append(f"Too many zero-variance features: {zero_variance_features}")
            
            # Detect extreme values (potential adversarial examples)
            z_scores = np.abs((x_np - feature_means) / (feature_stds + 1e-8))
            extreme_values = np.sum(z_scores > 10)  # Values > 10 std devs
            if extreme_values > x_np.size * 0.01:  # >1% extreme values
                if strict:
                    errors.append(f"Too many extreme values: {extreme_values}")
                else:
                    logger.warning(f"Input has {extreme_values} extreme values")
        
        # Update statistics
        self.validation_stats['total_requests'] += 1
        if errors:
            self.validation_stats['rejected_requests'] += 1
        
        return len(errors) == 0, errors
    
    def sanitize_input(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Sanitize input data by clipping and normalization."""
        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None
        
        # Convert to numpy
        x_np = x.cpu().numpy() if is_tensor else x.copy()
        
        # Clip NaN/Inf values
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Feature-wise clipping based on ranges
        if self.feature_ranges and len(x_np.shape) >= 2:
            for feature_idx, (min_val, max_val) in self.feature_ranges.items():
                if feature_idx < x_np.shape[1]:
                    x_np[:, feature_idx] = np.clip(x_np[:, feature_idx], min_val, max_val)
        
        # General outlier clipping (3-sigma rule)
        if len(x_np.shape) >= 2:
            for feature_idx in range(x_np.shape[1]):
                feature_values = x_np[:, feature_idx]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                
                if std_val > 0:
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    x_np[:, feature_idx] = np.clip(feature_values, lower_bound, upper_bound)
        
        # Update statistics
        self.validation_stats['sanitized_requests'] += 1
        
        # Convert back to original type
        if is_tensor:
            return torch.from_numpy(x_np.astype(np.float32)).to(device)
        else:
            return x_np


class SecureConformalHDC(ConformalHDC):
    """Security-hardened version of ConformalHDC with comprehensive protections."""
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        adversarial_defense: Optional[AdversarialDefense] = None,
        differential_privacy: Optional[DifferentialPrivacyMechanism] = None,
        input_validator: Optional[InputValidator] = None,
        model_integrity_verifier: Optional[ModelIntegrityVerifier] = None,
        security_level: str = 'medium',
        **kwargs
    ):
        """
        Initialize secure conformal HDC.
        
        Args:
            encoder: HDC encoder
            num_classes: Number of classes
            alpha: Miscoverage level
            adversarial_defense: Adversarial defense mechanism
            differential_privacy: Differential privacy mechanism
            input_validator: Input validation system
            model_integrity_verifier: Model integrity verification
            security_level: Security level ('low', 'medium', 'high')
        """
        super().__init__(encoder, num_classes, alpha, **kwargs)
        
        self.security_level = security_level
        self.adversarial_defense = adversarial_defense or StatisticalAdversarialDefense()
        self.differential_privacy = differential_privacy
        self.input_validator = input_validator or InputValidator()
        self.model_integrity_verifier = model_integrity_verifier
        
        # Security logging
        self.security_logger = logging.getLogger(f"{__name__}.security")
        self.security_events = deque(maxlen=1000)
        
        # Model signature for integrity verification
        self.model_signature = None
        
        if model_integrity_verifier:
            self.model_signature = model_integrity_verifier.sign_model(self, "primary")
    
    def secure_predict_set(
        self, 
        x: Union[torch.Tensor, np.ndarray],
        validate_input: bool = True,
        detect_adversarial: bool = True
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Make predictions with full security protections.
        
        Args:
            x: Input features
            validate_input: Whether to validate and sanitize input
            detect_adversarial: Whether to detect adversarial examples
            
        Returns:
            Tuple of (prediction_sets, security_info)
        """
        security_info = {
            'input_validated': False,
            'adversarial_detected': False,
            'adversarial_count': 0,
            'privacy_applied': False,
            'model_integrity_verified': False
        }
        
        start_time = time.time()
        
        try:
            # Input validation
            if validate_input:
                is_valid, errors = self.input_validator.validate_input(
                    x, strict=(self.security_level == 'high')
                )
                
                if not is_valid:
                    self._log_security_event('input_validation_failed', {'errors': errors})
                    if self.security_level == 'high':
                        raise ValueError(f"Input validation failed: {errors}")
                    else:
                        logger.warning(f"Input validation warnings: {errors}")
                
                # Sanitize input
                x = self.input_validator.sanitize_input(x)
                security_info['input_validated'] = True
            
            # Adversarial detection
            adversarial_mask = np.zeros(len(x), dtype=bool)
            if detect_adversarial:
                # Convert to numpy for detection
                x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                adversarial_mask, adversarial_scores = self.adversarial_defense.detect_adversarial(x_np)
                
                adversarial_count = np.sum(adversarial_mask)
                security_info['adversarial_detected'] = adversarial_count > 0
                security_info['adversarial_count'] = adversarial_count
                
                if adversarial_count > 0:
                    self._log_security_event('adversarial_detected', {
                        'count': adversarial_count,
                        'total_samples': len(x),
                        'max_score': np.max(adversarial_scores)
                    })
                    
                    # Mitigate adversarial examples
                    x_np_clean = self.adversarial_defense.mitigate_adversarial(x_np, adversarial_mask)
                    
                    # Convert back to tensor if needed
                    if isinstance(x, torch.Tensor):
                        x = torch.from_numpy(x_np_clean.astype(np.float32)).to(x.device)
                    else:
                        x = x_np_clean
            
            # Apply differential privacy if configured
            if self.differential_privacy is not None:
                x = self.differential_privacy.add_noise(x)
                security_info['privacy_applied'] = True
            
            # Verify model integrity
            if self.model_integrity_verifier and self.model_signature:
                is_valid = self.model_integrity_verifier.verify_model(
                    self, "primary", self.model_signature
                )
                security_info['model_integrity_verified'] = is_valid
                
                if not is_valid:
                    self._log_security_event('model_integrity_failed', {})
                    if self.security_level == 'high':
                        raise RuntimeError("Model integrity verification failed")
            
            # Make prediction using parent class
            prediction_sets = super().predict_set(x)
            
            # Update adversarial defense with clean samples
            if hasattr(self.adversarial_defense, 'update_reference'):
                clean_mask = ~adversarial_mask
                if np.any(clean_mask):
                    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    self.adversarial_defense.update_reference(x_np, clean_mask)
            
            # Log successful prediction
            self._log_security_event('secure_prediction', {
                'samples': len(x),
                'processing_time': time.time() - start_time,
                'security_level': self.security_level
            })
            
            return prediction_sets, security_info
            
        except Exception as e:
            self._log_security_event('prediction_error', {
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
    
    def _log_security_event(self, event_type: str, data: Dict[str, Any]):
        """Log security-related events."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data
        }
        
        self.security_events.append(event)
        
        # Log to security logger
        if event_type in ['adversarial_detected', 'input_validation_failed', 'model_integrity_failed']:
            self.security_logger.warning(f"Security event: {event_type}", extra={'event': event})
        else:
            self.security_logger.info(f"Security event: {event_type}", extra={'event': event})
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of security events."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_events = [
            event for event in self.security_events 
            if event['timestamp'] >= cutoff_time
        ]
        
        # Count events by type
        event_counts = {}
        for event in recent_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'time_window_hours': hours,
            'total_events': len(recent_events),
            'event_counts': event_counts,
            'security_level': self.security_level,
            'defenses_active': {
                'adversarial_defense': self.adversarial_defense is not None,
                'differential_privacy': self.differential_privacy is not None,
                'input_validation': self.input_validator is not None,
                'model_integrity': self.model_integrity_verifier is not None
            }
        }
    
    def update_security_configuration(self, **kwargs):
        """Update security configuration."""
        if 'security_level' in kwargs:
            self.security_level = kwargs['security_level']
        
        if 'adversarial_defense' in kwargs:
            self.adversarial_defense = kwargs['adversarial_defense']
        
        if 'differential_privacy' in kwargs:
            self.differential_privacy = kwargs['differential_privacy']
        
        if 'input_validator' in kwargs:
            self.input_validator = kwargs['input_validator']
        
        self._log_security_event('configuration_updated', kwargs)