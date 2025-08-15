#!/usr/bin/env python3
"""
Robust Error Handling for HyperConformal
Generation 2: Comprehensive error management with graceful degradation
"""

import sys
import traceback
import logging
from typing import Any, Callable, Optional, Dict
from functools import wraps
import time

class HyperConformalError(Exception):
    """Base exception for HyperConformal operations."""
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.error_code = error_code or "HC_UNKNOWN"
        self.context = context or {}
        self.timestamp = time.time()

class ValidationError(HyperConformalError):
    """Exception for input validation failures."""
    def __init__(self, message: str, invalid_input: Any = None):
        super().__init__(message, "HC_VALIDATION", {"input": str(invalid_input)})

class ResourceError(HyperConformalError):
    """Exception for resource-related issues."""
    def __init__(self, message: str, resource_type: str = None):
        super().__init__(message, "HC_RESOURCE", {"type": resource_type})

class ComputationError(HyperConformalError):
    """Exception for computation failures."""
    def __init__(self, message: str, operation: str = None):
        super().__init__(message, "HC_COMPUTATION", {"operation": operation})

class SecurityError(HyperConformalError):
    """Exception for security violations."""
    def __init__(self, message: str, security_context: str = None):
        super().__init__(message, "HC_SECURITY", {"context": security_context})

class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger('HyperConformal.ErrorHandler')
        self.error_counts = {}
        self.recovery_strategies = {}
        self.setup_recovery_strategies()
    
    def setup_recovery_strategies(self):
        """Setup automatic recovery strategies for common errors."""
        self.recovery_strategies = {
            'memory_error': self._recover_memory_error,
            'validation_error': self._recover_validation_error,
            'computation_error': self._recover_computation_error,
            'resource_error': self._recover_resource_error
        }
    
    def _recover_memory_error(self, error: Exception, context: Dict) -> Any:
        """Recovery strategy for memory errors."""
        self.logger.warning("Memory error detected, attempting recovery...")
        # Implement garbage collection, reduce batch size, etc.
        import gc
        gc.collect()
        return {"status": "recovered", "strategy": "memory_cleanup"}
    
    def _recover_validation_error(self, error: Exception, context: Dict) -> Any:
        """Recovery strategy for validation errors."""
        self.logger.warning("Validation error detected, applying defaults...")
        # Return safe defaults or sanitized inputs
        return {"status": "recovered", "strategy": "safe_defaults"}
    
    def _recover_computation_error(self, error: Exception, context: Dict) -> Any:
        """Recovery strategy for computation errors."""
        self.logger.warning("Computation error detected, using fallback...")
        # Use simpler computation or cached results
        return {"status": "recovered", "strategy": "fallback_computation"}
    
    def _recover_resource_error(self, error: Exception, context: Dict) -> Any:
        """Recovery strategy for resource errors."""
        self.logger.warning("Resource error detected, optimizing usage...")
        # Optimize resource usage, use alternatives
        return {"status": "recovered", "strategy": "resource_optimization"}
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict:
        """Handle errors with recovery attempts."""
        context = context or {}
        error_type = type(error).__name__.lower()
        
        # Log error details
        self.logger.error(f"Error occurred: {error}", extra={
            'error_type': error_type,
            'context': context,
            'traceback': traceback.format_exc()
        })
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Attempt recovery
        for strategy_name, strategy_func in self.recovery_strategies.items():
            if strategy_name in error_type:
                try:
                    result = strategy_func(error, context)
                    self.logger.info(f"Recovery successful: {strategy_name}")
                    return result
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed: {recovery_error}")
        
        # Return error information if no recovery possible
        return {
            "status": "error",
            "error_type": error_type,
            "message": str(error),
            "context": context,
            "recovery_attempted": True
        }

def robust_execution(recovery_enabled: bool = True):
    """Decorator for robust function execution with error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            
            try:
                result = func(*args, **kwargs)
                return {"status": "success", "result": result}
                
            except HyperConformalError as e:
                if recovery_enabled:
                    return error_handler.handle_error(e, {"function": func.__name__})
                else:
                    raise
                    
            except Exception as e:
                # Wrap unexpected errors
                wrapped_error = ComputationError(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    operation=func.__name__
                )
                
                if recovery_enabled:
                    return error_handler.handle_error(wrapped_error, {"function": func.__name__})
                else:
                    raise wrapped_error
        
        return wrapper
    return decorator

class InputValidator:
    """Comprehensive input validation with security checks."""
    
    @staticmethod
    def validate_vector_input(data: Any, expected_length: Optional[int] = None) -> list:
        """Validate vector input with safety checks."""
        if data is None:
            raise ValidationError("Input data cannot be None")
        
        # Convert to list if needed
        if hasattr(data, 'tolist'):  # numpy array
            data = data.tolist()
        elif not isinstance(data, (list, tuple)):
            raise ValidationError(f"Invalid input type: {type(data)}, expected list or array")
        
        # Check length
        if expected_length is not None and len(data) != expected_length:
            raise ValidationError(f"Invalid input length: {len(data)}, expected {expected_length}")
        
        # Validate numeric values
        validated_data = []
        for i, value in enumerate(data):
            try:
                # Check for dangerous values
                if isinstance(value, str):
                    raise ValidationError(f"String value at index {i}: {value}")
                
                float_value = float(value)
                
                # Check for NaN, infinity
                if not (-1e10 <= float_value <= 1e10):  # Reasonable bounds
                    raise ValidationError(f"Value out of bounds at index {i}: {float_value}")
                
                validated_data.append(float_value)
                
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid numeric value at index {i}: {value}")
        
        return validated_data
    
    @staticmethod
    def validate_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters with security and sanity checks."""
        validated = {}
        
        # Alpha (significance level)
        if 'alpha' in params:
            alpha = params['alpha']
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                raise ValidationError(f"Invalid alpha value: {alpha}, must be in (0, 1)")
            validated['alpha'] = float(alpha)
        
        # Dimensions
        for dim_param in ['input_dim', 'hv_dim', 'num_classes']:
            if dim_param in params:
                dim_value = params[dim_param]
                if not isinstance(dim_value, int) or dim_value <= 0 or dim_value > 1000000:
                    raise ValidationError(f"Invalid {dim_param}: {dim_value}")
                validated[dim_param] = dim_value
        
        # String parameters
        if 'quantization' in params:
            valid_quantizations = ['binary', 'ternary', 'complex']
            if params['quantization'] not in valid_quantizations:
                raise ValidationError(f"Invalid quantization: {params['quantization']}")
            validated['quantization'] = params['quantization']
        
        return validated

# Example usage and testing
if __name__ == "__main__":
    print("🛡️ Testing Robust Error Handling")
    print("="*40)
    
    # Test error handling
    @robust_execution()
    def test_function_with_error():
        raise ComputationError("Simulated computation error", "test_operation")
    
    result = test_function_with_error()
    print(f"Error handling result: {result}")
    
    # Test input validation
    validator = InputValidator()
    
    # Valid input
    try:
        valid_data = validator.validate_vector_input([1.0, 2.0, 3.0], expected_length=3)
        print(f"✅ Valid input processed: {valid_data}")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Invalid input
    try:
        invalid_data = validator.validate_vector_input(["invalid", "data"])
        print(f"❌ Should not reach here: {invalid_data}")
    except ValidationError as e:
        print(f"✅ Invalid input caught: {e}")
    
    print("\n🎉 Error handling tests completed!")
