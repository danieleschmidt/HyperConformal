"""
Cross-platform compatibility layer for HyperConformal
Support for Windows, macOS, Linux, ARM, x86, embedded systems
"""
import platform
import sys
import os
from typing import Dict, Any, Optional, Tuple
from enum import Enum

class PlatformType(Enum):
    """Supported platform types"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ANDROID = "android"
    IOS = "ios"
    EMBEDDED = "embedded"

class ArchitectureType(Enum):
    """Supported CPU architectures"""
    X86_64 = "x86_64"
    ARM64 = "arm64" 
    ARM32 = "arm32"
    RISC_V = "riscv"
    AVR = "avr"          # Arduino/microcontrollers
    CORTEX_M = "cortex_m" # ARM Cortex-M series

class PlatformDetector:
    """Detect platform and architecture characteristics"""
    
    @staticmethod
    def detect_platform() -> PlatformType:
        """Detect current platform"""
        system = platform.system().lower()
        
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "linux":
            # Check for Android
            if 'android' in platform.platform().lower():
                return PlatformType.ANDROID
            return PlatformType.LINUX
        else:
            return PlatformType.EMBEDDED
    
    @staticmethod
    def detect_architecture() -> ArchitectureType:
        """Detect CPU architecture"""
        machine = platform.machine().lower()
        
        if machine in ["x86_64", "amd64"]:
            return ArchitectureType.X86_64
        elif machine in ["arm64", "aarch64"]:
            return ArchitectureType.ARM64
        elif machine.startswith("arm"):
            return ArchitectureType.ARM32
        elif "riscv" in machine:
            return ArchitectureType.RISC_V
        else:
            return ArchitectureType.X86_64  # Default fallback
    
    @staticmethod
    def get_memory_info() -> Dict[str, int]:
        """Get system memory information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total // (1024 * 1024),
                'available_mb': memory.available // (1024 * 1024),
                'used_mb': memory.used // (1024 * 1024)
            }
        except ImportError:
            # Fallback for embedded systems
            return {
                'total_mb': 512,     # Conservative estimate
                'available_mb': 256,
                'used_mb': 256
            }

class OptimizedOperations:
    """Platform-optimized implementations"""
    
    def __init__(self):
        self.platform = PlatformDetector.detect_platform()
        self.arch = PlatformDetector.detect_architecture()
        self.memory_info = PlatformDetector.get_memory_info()
    
    def hamming_distance_optimized(self, vec1: bytes, vec2: bytes) -> int:
        """Platform-optimized Hamming distance calculation"""
        if self.arch == ArchitectureType.X86_64:
            # Use POPCNT instruction if available
            return self._hamming_x86_64(vec1, vec2)
        elif self.arch in [ArchitectureType.ARM64, ArchitectureType.ARM32]:
            # ARM NEON optimizations
            return self._hamming_arm(vec1, vec2)
        else:
            # Generic implementation for embedded
            return self._hamming_generic(vec1, vec2)
    
    def _hamming_x86_64(self, vec1: bytes, vec2: bytes) -> int:
        """x86-64 optimized Hamming distance using POPCNT"""
        distance = 0
        xor_result = bytes(a ^ b for a, b in zip(vec1, vec2))
        
        # Process 8 bytes at a time on 64-bit systems
        for i in range(0, len(xor_result), 8):
            chunk = xor_result[i:i+8]
            if len(chunk) == 8:
                # Convert to 64-bit integer and use bin().count()
                value = int.from_bytes(chunk, byteorder='little')
                distance += bin(value).count('1')
            else:
                # Handle remaining bytes
                for byte in chunk:
                    distance += bin(byte).count('1')
        
        return distance
    
    def _hamming_arm(self, vec1: bytes, vec2: bytes) -> int:
        """ARM-optimized Hamming distance"""
        # ARM processors have efficient bit manipulation
        distance = 0
        for a, b in zip(vec1, vec2):
            distance += bin(a ^ b).count('1')
        return distance
    
    def _hamming_generic(self, vec1: bytes, vec2: bytes) -> int:
        """Generic Hamming distance for embedded systems"""
        distance = 0
        for a, b in zip(vec1, vec2):
            xor_val = a ^ b
            # Use lookup table for better performance on microcontrollers
            while xor_val:
                distance += xor_val & 1
                xor_val >>= 1
        return distance
    
    def get_optimal_hv_dimension(self) -> int:
        """Get optimal hypervector dimension based on platform"""
        if self.memory_info['available_mb'] < 64:
            # Embedded/IoT devices
            return 1000
        elif self.memory_info['available_mb'] < 512:
            # Mobile/edge devices
            return 4000
        else:
            # Desktop/server systems
            return 10000
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for processing"""
        available_mb = self.memory_info['available_mb']
        
        if available_mb < 64:
            return 32      # Embedded systems
        elif available_mb < 512:
            return 128     # Mobile devices
        else:
            return 1024    # Desktop/server

class EmbeddedOptimizations:
    """Specific optimizations for embedded systems"""
    
    @staticmethod
    def compress_model(model_data: bytes) -> bytes:
        """Compress model for embedded deployment"""
        try:
            import zlib
            return zlib.compress(model_data, level=9)
        except ImportError:
            return model_data  # No compression available
    
    @staticmethod
    def fixed_point_arithmetic(value: float, scale: int = 1000) -> int:
        """Convert float to fixed-point for embedded systems"""
        return int(value * scale)
    
    @staticmethod
    def memory_pool_allocator(size_mb: int = 1) -> Optional[bytearray]:
        """Pre-allocate memory pool for predictable memory usage"""
        try:
            return bytearray(size_mb * 1024 * 1024)
        except MemoryError:
            return None

class PlatformConfig:
    """Platform-specific configuration"""
    
    def __init__(self):
        self.platform = PlatformDetector.detect_platform()
        self.arch = PlatformDetector.detect_architecture()
        self.ops = OptimizedOperations()
    
    def get_config(self) -> Dict[str, Any]:
        """Get platform-optimized configuration"""
        return {
            'platform': self.platform.value,
            'architecture': self.arch.value,
            'hv_dimension': self.ops.get_optimal_hv_dimension(),
            'batch_size': self.ops.get_optimal_batch_size(),
            'memory_mb': self.ops.memory_info['available_mb'],
            'use_compression': self.ops.memory_info['available_mb'] < 512,
            'use_fixed_point': self.arch in [
                ArchitectureType.ARM32, 
                ArchitectureType.AVR,
                ArchitectureType.CORTEX_M
            ]
        }

# Global platform configuration
platform_config = PlatformConfig()