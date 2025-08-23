"""
Security and monitoring framework for HyperConformal.
Implements secure logging, audit trails, and anomaly detection.
"""

import hashlib
import json
import time
import logging
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np


@dataclass
class SecurityEvent:
    """Represents a security-relevant event."""
    timestamp: float
    event_type: str
    severity: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component: str
    user_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    data_hash: Optional[str] = None


class SecurityLogger:
    """Secure audit logging for HyperConformal operations."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("hyperconformal_audit.log")
        self.logger = logging.getLogger("hyperconformal.security")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler with append mode
        handler = logging.FileHandler(self.log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event with integrity protection."""
        event_dict = asdict(event)
        event_json = json.dumps(event_dict, sort_keys=True)
        
        # Add integrity hash
        integrity_hash = hashlib.sha256(event_json.encode()).hexdigest()
        log_entry = f"INTEGRITY:{integrity_hash} EVENT:{event_json}"
        
        if event.severity == 'CRITICAL':
            self.logger.critical(log_entry)
        elif event.severity == 'ERROR':
            self.logger.error(log_entry)
        elif event.severity == 'WARNING':
            self.logger.warning(log_entry)
        else:
            self.logger.info(log_entry)
    
    def log_model_training(self, num_samples: int, num_features: int, 
                          model_hash: str, user_id: Optional[str] = None) -> None:
        """Log model training event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="MODEL_TRAINING",
            severity="INFO",
            component="ConformalHDC",
            user_id=user_id,
            details={
                "num_samples": num_samples,
                "num_features": num_features,
                "model_hash": model_hash
            }
        )
        self.log_event(event)
    
    def log_prediction_request(self, num_samples: int, features_hash: str,
                              user_id: Optional[str] = None) -> None:
        """Log prediction request."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="PREDICTION_REQUEST",
            severity="INFO", 
            component="ConformalHDC",
            user_id=user_id,
            details={
                "num_samples": num_samples
            },
            data_hash=features_hash
        )
        self.log_event(event)
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any],
                              user_id: Optional[str] = None) -> None:
        """Log security violation."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="SECURITY_VIOLATION",
            severity="CRITICAL",
            component="SecurityMonitor",
            user_id=user_id,
            details={"violation_type": violation_type, **details}
        )
        self.log_event(event)


class AnomalyDetector:
    """Detects anomalous patterns in model usage and data."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prediction_counts = []
        self.feature_stats = {}
        self.model_performance = []
    
    def check_input_anomalies(self, X: torch.Tensor) -> List[str]:
        """Check for anomalous input patterns."""
        anomalies = []
        
        # Check for unusual feature statistics
        feature_means = torch.mean(X, dim=0)
        feature_stds = torch.std(X, dim=0)
        
        # Compare with historical stats
        for i in range(X.shape[1]):
            feat_name = f"feature_{i}"
            if feat_name in self.feature_stats:
                hist_mean, hist_std = self.feature_stats[feat_name]
                
                # Check if current mean is more than 3 std devs from historical
                if abs(feature_means[i] - hist_mean) > 3 * hist_std:
                    anomalies.append(f"Feature {i} mean anomaly: {feature_means[i]:.3f} vs historical {hist_mean:.3f}")
            else:
                # Store initial statistics
                self.feature_stats[feat_name] = (feature_means[i].item(), feature_stds[i].item())
        
        # Check for extreme values
        if torch.any(torch.abs(X) > 100):
            anomalies.append("Extreme feature values detected (|x| > 100)")
        
        # Check for constant features (potential data issues)
        constant_features = (feature_stds == 0).nonzero(as_tuple=True)[0]
        if len(constant_features) > 0:
            anomalies.append(f"Constant features detected: {constant_features.tolist()}")
        
        return anomalies
    
    def check_prediction_anomalies(self, predictions: torch.Tensor) -> List[str]:
        """Check for anomalous prediction patterns."""
        anomalies = []
        
        # Check prediction confidence distribution
        max_probs = torch.max(predictions, dim=1)[0]
        avg_confidence = torch.mean(max_probs)
        
        # Store recent performance
        self.model_performance.append(avg_confidence.item())
        if len(self.model_performance) > self.window_size:
            self.model_performance.pop(0)
        
        # Check for confidence drop
        if len(self.model_performance) > 10:
            recent_avg = np.mean(self.model_performance[-10:])
            historical_avg = np.mean(self.model_performance[:-10])
            
            if recent_avg < historical_avg - 2 * np.std(self.model_performance):
                anomalies.append(f"Significant confidence drop: {recent_avg:.3f} vs {historical_avg:.3f}")
        
        # Check for uniform predictions (potential attack)
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        high_entropy_ratio = (entropy > 0.9 * torch.log(torch.tensor(predictions.shape[1]))).float().mean()
        
        if high_entropy_ratio > 0.8:
            anomalies.append("High proportion of uniform predictions detected")
        
        return anomalies


class SecurityManager:
    """Main security manager for HyperConformal operations."""
    
    def __init__(self, enable_logging: bool = True, enable_anomaly_detection: bool = True):
        self.enable_logging = enable_logging
        self.enable_anomaly_detection = enable_anomaly_detection
        
        self.logger = SecurityLogger() if enable_logging else None
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        
        self.access_counts = {}
        self.rate_limits = {"predictions": 1000, "training": 10}  # per hour
    
    def hash_data(self, data: torch.Tensor) -> str:
        """Create secure hash of data for audit trail."""
        # Convert to bytes for hashing
        if data.dtype != torch.uint8:
            # Normalize to prevent hash changes from numerical precision
            data_normalized = (data * 1000).round().byte()
        else:
            data_normalized = data
        
        data_bytes = data_normalized.cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]  # Truncated hash for logs
    
    def check_rate_limits(self, operation: str, user_id: str = "default") -> bool:
        """Check if operation is within rate limits."""
        current_time = time.time()
        hour_key = f"{user_id}_{operation}_{int(current_time // 3600)}"
        
        if hour_key not in self.access_counts:
            self.access_counts[hour_key] = 0
        
        self.access_counts[hour_key] += 1
        
        if self.access_counts[hour_key] > self.rate_limits.get(operation, 100):
            if self.logger:
                self.logger.log_security_violation(
                    "RATE_LIMIT_EXCEEDED",
                    {"operation": operation, "count": self.access_counts[hour_key]},
                    user_id
                )
            return False
        
        return True
    
    def validate_training_request(self, X: torch.Tensor, y: torch.Tensor, 
                                 user_id: str = "default") -> bool:
        """Validate and log training request."""
        # Check rate limits
        if not self.check_rate_limits("training", user_id):
            warnings.warn("Training rate limit exceeded", UserWarning)
            return False
        
        # Log training request
        if self.logger:
            data_hash = self.hash_data(X)
            self.logger.log_model_training(
                len(X), X.shape[1], data_hash, user_id
            )
        
        # Check for anomalies
        if self.anomaly_detector:
            anomalies = self.anomaly_detector.check_input_anomalies(X)
            if anomalies and self.logger:
                self.logger.log_security_violation(
                    "TRAINING_DATA_ANOMALY", 
                    {"anomalies": anomalies}, 
                    user_id
                )
        
        return True
    
    def validate_prediction_request(self, X: torch.Tensor, 
                                   user_id: str = "default") -> bool:
        """Validate and log prediction request."""
        # Check rate limits
        if not self.check_rate_limits("predictions", user_id):
            warnings.warn("Prediction rate limit exceeded", UserWarning)
            return False
        
        # Log prediction request
        if self.logger:
            data_hash = self.hash_data(X)
            self.logger.log_prediction_request(len(X), data_hash, user_id)
        
        # Check for anomalies
        if self.anomaly_detector:
            anomalies = self.anomaly_detector.check_input_anomalies(X)
            if anomalies:
                warnings.warn(f"Input anomalies detected: {anomalies}", UserWarning)
                if self.logger:
                    self.logger.log_security_violation(
                        "PREDICTION_DATA_ANOMALY",
                        {"anomalies": anomalies},
                        user_id
                    )
        
        return True
    
    def validate_prediction_output(self, predictions: torch.Tensor,
                                  user_id: str = "default") -> None:
        """Validate prediction outputs for anomalies."""
        if self.anomaly_detector:
            anomalies = self.anomaly_detector.check_prediction_anomalies(predictions)
            if anomalies:
                warnings.warn(f"Prediction anomalies detected: {anomalies}", UserWarning)
                if self.logger:
                    self.logger.log_security_violation(
                        "PREDICTION_OUTPUT_ANOMALY",
                        {"anomalies": anomalies},
                        user_id
                    )


# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get or create global security manager."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def enable_security(logging: bool = True, anomaly_detection: bool = True) -> None:
    """Enable security monitoring globally."""
    global _security_manager
    _security_manager = SecurityManager(logging, anomaly_detection)

def disable_security() -> None:
    """Disable security monitoring."""
    global _security_manager
    _security_manager = None