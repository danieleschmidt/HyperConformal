#!/usr/bin/env python3
"""
Security Framework for HyperConformal
Generation 2: Comprehensive security measures and safe execution
"""

import hashlib
import hmac
import time
import os
import sys
import subprocess
import tempfile
from typing import Any, Dict, List, Optional
import logging

class SecurityManager:
    """Centralized security management for HyperConformal operations."""
    
    def __init__(self, max_memory_mb: int = 512, max_execution_time: int = 300):
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.logger = logging.getLogger('HyperConformal.Security')
        self.blocked_patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> List[str]:
        """Load patterns that should be blocked for security."""
        return [
            '__import__',  # Prevent dynamic imports
            'eval(',       # Prevent code evaluation
            'exec(',       # Prevent code execution
            'subprocess',  # Monitor subprocess usage
            'os.system',   # Prevent system commands
            '../',         # Prevent path traversal
            '/etc/',       # Prevent system file access
            '/proc/',      # Prevent process access
        ]
    
    def validate_input_security(self, input_data: Any, context: str = "") -> bool:
        """Validate input for security threats."""
        input_str = str(input_data)
        
        # Check for dangerous patterns
        for pattern in self.blocked_patterns:
            if pattern in input_str:
                self.logger.warning(f"Security threat detected: {pattern} in {context}")
                return False
        
        # Check input size (prevent memory exhaustion)
        if len(input_str) > 1024 * 1024:  # 1MB limit
            self.logger.warning(f"Input too large: {len(input_str)} bytes in {context}")
            return False
        
        return True
    
    def sanitize_file_path(self, path: str) -> Optional[str]:
        """Sanitize file paths to prevent traversal attacks."""
        if not isinstance(path, str):
            return None
        
        # Remove dangerous characters and patterns
        dangerous_patterns = ['../', '../', '..\\', '/', '\\']
        sanitized = path
        
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        # Only allow alphanumeric, dots, underscores, hyphens
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
        if not all(c in allowed_chars for c in sanitized):
            return None
        
        return sanitized
    
    def create_secure_temp_file(self, content: str, suffix: str = '.tmp') -> Optional[str]:
        """Create a secure temporary file with restricted permissions."""
        try:
            # Create temporary file with secure permissions
            fd, temp_path = tempfile.mkstemp(suffix=suffix, text=True)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(temp_path, 0o600)
            
            # Write content
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Secure temp file created: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to create secure temp file: {e}")
            return None
    
    def compute_integrity_hash(self, data: Any) -> str:
        """Compute integrity hash for data verification."""
        data_str = str(data).encode('utf-8')
        return hashlib.sha256(data_str).hexdigest()
    
    def verify_data_integrity(self, data: Any, expected_hash: str) -> bool:
        """Verify data integrity using hash comparison."""
        computed_hash = self.compute_integrity_hash(data)
        return hmac.compare_digest(computed_hash, expected_hash)
    
    def resource_monitor(self):
        """Monitor resource usage for security."""
        try:
            import psutil
            process = psutil.Process()
            
            # Check memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                self.logger.warning(f"Memory usage exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                return False
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > 90:  # High CPU usage
                self.logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
            
            return True
            
        except ImportError:
            # psutil not available, use basic checks
            return True
        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {e}")
            return False

class SecureHyperConformalOperations:
    """Secure wrapper for HyperConformal operations."""
    
    def __init__(self):
        self.security_manager = SecurityManager()
        self.logger = logging.getLogger('HyperConformal.SecureOps')
        
    def secure_encode(self, input_data: List[float], encoder_params: Dict) -> Dict:
        """Securely encode input with validation and monitoring."""
        
        # Security validation
        if not self.security_manager.validate_input_security(input_data, "encode_input"):
            return {"status": "error", "message": "Security validation failed"}
        
        if not self.security_manager.validate_input_security(encoder_params, "encoder_params"):
            return {"status": "error", "message": "Parameter validation failed"}
        
        # Resource monitoring
        if not self.security_manager.resource_monitor():
            return {"status": "error", "message": "Resource limits exceeded"}
        
        try:
            # Simulate secure encoding operation
            self.logger.info(f"Secure encoding: {len(input_data)} dimensions")
            
            # Basic binary encoding (secure implementation)
            threshold = encoder_params.get('threshold', 0.0)
            encoded = [1 if x > threshold else 0 for x in input_data]
            
            # Compute integrity hash
            integrity_hash = self.security_manager.compute_integrity_hash(encoded)
            
            return {
                "status": "success",
                "encoded_data": encoded,
                "integrity_hash": integrity_hash,
                "input_dimensions": len(input_data),
                "output_dimensions": len(encoded)
            }
            
        except Exception as e:
            self.logger.error(f"Secure encoding failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def secure_conformal_predict(self, scores: List[float], alpha: float) -> Dict:
        """Securely perform conformal prediction with validation."""
        
        # Security validation
        if not self.security_manager.validate_input_security(scores, "prediction_scores"):
            return {"status": "error", "message": "Score validation failed"}
        
        if not (0 < alpha < 1):
            return {"status": "error", "message": "Invalid alpha value"}
        
        try:
            # Secure conformal prediction
            self.logger.info(f"Secure conformal prediction: {len(scores)} classes, alpha={alpha}")
            
            # Compute quantile securely
            n = len(scores)
            if n == 0:
                return {"status": "error", "message": "Empty scores array"}
            
            # Safe quantile computation
            import math
            q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
            sorted_scores = sorted(scores)
            quantile = sorted_scores[q_index]
            
            # Generate prediction set
            prediction_set = [i for i, score in enumerate(scores) if score >= quantile]
            
            # Compute integrity hash
            result_data = {
                "quantile": quantile,
                "prediction_set": prediction_set,
                "coverage_level": 1 - alpha
            }
            integrity_hash = self.security_manager.compute_integrity_hash(result_data)
            
            return {
                "status": "success",
                "result": result_data,
                "integrity_hash": integrity_hash
            }
            
        except Exception as e:
            self.logger.error(f"Secure conformal prediction failed: {e}")
            return {"status": "error", "message": str(e)}

# Security testing
if __name__ == "__main__":
    print("🔒 Testing Security Framework")
    print("="*40)
    
    # Test security manager
    security_mgr = SecurityManager()
    
    # Test input validation
    safe_input = [1.0, 2.0, 3.0]
    dangerous_input = "__import__('os').system('ls')"
    
    print(f"Safe input validation: {security_mgr.validate_input_security(safe_input)}")
    print(f"Dangerous input validation: {security_mgr.validate_input_security(dangerous_input)}")
    
    # Test secure operations
    secure_ops = SecureHyperConformalOperations()
    
    # Test secure encoding
    encode_result = secure_ops.secure_encode([0.5, -0.3, 0.8], {'threshold': 0.0})
    print(f"Secure encoding result: {encode_result['status']}")
    
    # Test secure conformal prediction
    predict_result = secure_ops.secure_conformal_predict([0.7, 0.3, 0.9], 0.1)
    print(f"Secure prediction result: {predict_result['status']}")
    
    print("\n🎉 Security framework tests completed!")
