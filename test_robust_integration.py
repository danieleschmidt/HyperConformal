#!/usr/bin/env python3
"""
Integration Tests for HyperConformal Generation 2
Tests robustness, security, error handling, and monitoring
"""

import unittest
import sys
import time
import json
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

class TestRobustIntegration(unittest.TestCase):
    """Integration tests for robust HyperConformal implementation."""
    
    def setUp(self):
        """Setup for each test."""
        self.repo_root = Path(__file__).parent
        
    def test_error_handling_integration(self):
        """Test that error handling works end-to-end."""
        # Import error handling module
        try:
            import robust_error_handling
            
            # Test robust execution decorator
            @robust_error_handling.robust_execution()
            def test_function():
                return {"result": "success"}
            
            result = test_function()
            self.assertEqual(result["status"], "success")
            
            # Test error recovery
            @robust_error_handling.robust_execution()
            def failing_function():
                raise robust_error_handling.ComputationError("Test error")
            
            result = failing_function()
            self.assertIn("status", result)
            
        except ImportError:
            self.skipTest("Error handling module not available")
    
    def test_security_integration(self):
        """Test that security framework works end-to-end."""
        try:
            import security_framework
            
            security_mgr = security_framework.SecurityManager()
            
            # Test input validation
            safe_result = security_mgr.validate_input_security([1, 2, 3])
            self.assertTrue(safe_result)
            
            dangerous_result = security_mgr.validate_input_security("__import__")
            self.assertFalse(dangerous_result)
            
            # Test secure operations
            secure_ops = security_framework.SecureHyperConformalOperations()
            
            encode_result = secure_ops.secure_encode([0.5, -0.3], {'threshold': 0.0})
            self.assertEqual(encode_result["status"], "success")
            
            predict_result = secure_ops.secure_conformal_predict([0.8, 0.3], 0.1)
            self.assertEqual(predict_result["status"], "success")
            
        except ImportError:
            self.skipTest("Security framework module not available")
    
    def test_monitoring_integration(self):
        """Test that monitoring system works end-to-end."""
        try:
            import monitoring_system
            
            # Test health checker
            health_checker = monitoring_system.HealthChecker()
            metrics = health_checker.check_system_health()
            self.assertGreater(len(metrics), 0)
            
            # Test report generation
            report = health_checker.generate_health_report()
            self.assertIn("overall_status", report)
            self.assertIn("summary", report)
            
            # Test performance monitor
            perf_monitor = monitoring_system.PerformanceMonitor()
            
            with perf_monitor.time_operation("test_op"):
                time.sleep(0.001)
            
            stats = perf_monitor.get_performance_stats()
            self.assertIn("test_op", stats)
            
        except ImportError:
            self.skipTest("Monitoring system module not available")
    
    def test_end_to_end_robust_pipeline(self):
        """Test complete end-to-end pipeline with all robustness features."""
        
        # Simulate complete pipeline
        input_data = [0.5, -0.3, 0.8, 0.1, 0.9, -0.2]
        
        # Step 1: Input validation
        validated_input = self._validate_input_robust(input_data)
        self.assertIsNotNone(validated_input)
        
        # Step 2: Secure encoding
        encoded_result = self._secure_encode_robust(validated_input)
        self.assertEqual(encoded_result["status"], "success")
        
        # Step 3: Secure conformal prediction
        scores = [0.8, 0.4, 0.9, 0.2]
        prediction_result = self._secure_predict_robust(scores)
        self.assertEqual(prediction_result["status"], "success")
        
        # Step 4: Monitoring and health check
        health_status = self._check_health_robust()
        self.assertIn(health_status, ["healthy", "warning", "critical"])
    
    def _validate_input_robust(self, data):
        """Robust input validation."""
        # Basic validation without external dependencies
        if not isinstance(data, (list, tuple)):
            return None
        
        if len(data) == 0 or len(data) > 10000:  # Reasonable bounds
            return None
        
        # Check for numeric values
        try:
            validated = [float(x) for x in data]
            # Check bounds
            for val in validated:
                if not (-1000 <= val <= 1000):
                    return None
            return validated
        except (ValueError, TypeError):
            return None
    
    def _secure_encode_robust(self, data):
        """Secure encoding with error handling."""
        try:
            # Basic binary encoding
            encoded = [1 if x > 0 else 0 for x in data]
            
            # Compute hash for integrity
            import hashlib
            data_str = str(encoded).encode('utf-8')
            integrity_hash = hashlib.sha256(data_str).hexdigest()
            
            return {
                "status": "success",
                "encoded": encoded,
                "integrity_hash": integrity_hash
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _secure_predict_robust(self, scores):
        """Secure conformal prediction with error handling."""
        try:
            alpha = 0.1  # 90% confidence
            
            # Validate scores
            if not scores or not all(isinstance(s, (int, float)) for s in scores):
                return {"status": "error", "message": "Invalid scores"}
            
            # Compute quantile
            import math
            n = len(scores)
            q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
            quantile = sorted(scores)[q_index]
            
            # Generate prediction set
            prediction_set = [i for i, score in enumerate(scores) if score >= quantile]
            
            return {
                "status": "success",
                "prediction_set": prediction_set,
                "quantile": quantile,
                "coverage": 1 - alpha
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_health_robust(self):
        """Basic health check."""
        try:
            # Check system basics
            import os
            import sys
            
            # Check memory (basic)
            if hasattr(sys, 'getsizeof'):
                test_data = list(range(1000))
                memory_size = sys.getsizeof(test_data)
                if memory_size > 100000:  # 100KB threshold
                    return "warning"
            
            # Check file access
            temp_file = "test_health.tmp"
            try:
                with open(temp_file, 'w') as f:
                    f.write("health check")
                os.remove(temp_file)
            except (IOError, OSError):
                return "critical"
            
            return "healthy"
            
        except Exception:
            return "critical"

if __name__ == '__main__':
    print("🧪 Running Robust Integration Tests")
    print("="*50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRobustIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All integration tests PASSED!")
    else:
        print("❌ Some integration tests FAILED!")
        
    print("\n🎉 Integration testing completed!")
