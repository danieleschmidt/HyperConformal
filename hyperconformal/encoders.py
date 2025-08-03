"""
HDC encoder implementations for various quantization schemes.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


class BaseEncoder(ABC, nn.Module):
    """Base class for HDC encoders."""
    
    def __init__(self, input_dim: int, hv_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hypervector."""
        pass
    
    @abstractmethod
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between hypervectors."""
        pass
    
    def memory_footprint(self) -> int:
        """Estimate memory footprint in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())


class RandomProjection(BaseEncoder):
    """Random projection HDC encoder with binary, ternary, or complex quantization."""
    
    def __init__(
        self, 
        input_dim: int, 
        hv_dim: int, 
        quantization: str = 'binary',
        seed: Optional[int] = None
    ):
        super().__init__(input_dim, hv_dim)
        
        if quantization not in ['binary', 'ternary', 'complex']:
            raise ValueError("quantization must be 'binary', 'ternary', or 'complex'")
        
        self.quantization = quantization
        
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize random projection matrix
        if quantization == 'binary':
            # Binary random projection: {-1, +1}
            self.register_buffer(
                'projection_matrix',
                torch.randint(0, 2, (input_dim, hv_dim), dtype=torch.float32) * 2 - 1
            )
        elif quantization == 'ternary':
            # Ternary random projection: {-1, 0, +1}
            self.register_buffer(
                'projection_matrix',
                torch.randint(-1, 2, (input_dim, hv_dim), dtype=torch.float32)
            )
        else:  # complex
            # Complex random projection with unit magnitude
            real = torch.randn(input_dim, hv_dim)
            imag = torch.randn(input_dim, hv_dim)
            complex_proj = torch.complex(real, imag)
            complex_proj = complex_proj / torch.abs(complex_proj)
            self.register_buffer('projection_matrix', complex_proj)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input using random projection."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply projection
        if self.quantization == 'complex':
            # For complex, input should be real
            projected = torch.matmul(x.float(), self.projection_matrix)
        else:
            projected = torch.matmul(x, self.projection_matrix)
        
        # Apply quantization
        if self.quantization == 'binary':
            return torch.sign(projected)
        elif self.quantization == 'ternary':
            # Ternary quantization with threshold
            threshold = 0.5 * torch.std(projected, dim=-1, keepdim=True)
            return torch.sign(projected) * (torch.abs(projected) > threshold).float()
        else:  # complex
            # Phase quantization to unit circle
            return projected / torch.abs(projected)
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between hypervectors."""
        if self.quantization == 'complex':
            # Complex dot product (real part)
            return torch.real(torch.sum(hv1 * torch.conj(hv2), dim=-1))
        else:
            # Hamming similarity for binary/ternary
            return torch.sum(hv1 * hv2, dim=-1) / self.hv_dim


class LevelHDC(BaseEncoder):
    """Level-based HDC encoder for continuous features."""
    
    def __init__(
        self, 
        input_dim: int, 
        hv_dim: int, 
        levels: int = 100,
        circular: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__(input_dim, hv_dim)
        self.levels = levels
        self.circular = circular
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate base hypervectors for each feature and level
        self.register_buffer(
            'base_hvs',
            torch.randn(input_dim, levels, hv_dim)
        )
        
        # Normalize to unit vectors
        self.base_hvs = self.base_hvs / torch.norm(self.base_hvs, dim=-1, keepdim=True)
        
        if circular:
            # For circular encoding, create smooth transitions
            for i in range(input_dim):
                for j in range(levels):
                    angle = 2 * np.pi * j / levels
                    self.base_hvs[i, j] = torch.cos(angle) * self.base_hvs[i, 0] + \
                                         torch.sin(angle) * self.base_hvs[i, 1]
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using level-based approach."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Normalize input to [0, levels-1]
        x_normalized = torch.clamp(x * (self.levels - 1), 0, self.levels - 1)
        
        # Get integer and fractional parts
        level_low = torch.floor(x_normalized).long()
        level_high = torch.clamp(level_low + 1, 0, self.levels - 1)
        alpha = x_normalized - level_low.float()
        
        # Initialize output
        encoded = torch.zeros(batch_size, self.hv_dim, device=x.device)
        
        # Bundle hypervectors for each feature
        for i in range(self.input_dim):
            hv_low = self.base_hvs[i, level_low[:, i]]
            hv_high = self.base_hvs[i, level_high[:, i]]
            
            # Linear interpolation between levels
            feature_hv = (1 - alpha[:, i].unsqueeze(-1)) * hv_low + \
                        alpha[:, i].unsqueeze(-1) * hv_high
            
            # Bundle (element-wise addition)
            encoded += feature_hv
        
        # Normalize
        return encoded / torch.norm(encoded, dim=-1, keepdim=True)
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity."""
        return torch.sum(hv1 * hv2, dim=-1)


class ComplexHDC(BaseEncoder):
    """Complex-valued HDC for signal processing applications."""
    
    def __init__(
        self, 
        input_dim: int,
        hv_dim: int,
        quantization_levels: int = 4,
        seed: Optional[int] = None
    ):
        super().__init__(input_dim, hv_dim)
        self.quantization_levels = quantization_levels
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Complex random projection matrix
        real_part = torch.randn(input_dim, hv_dim)
        imag_part = torch.randn(input_dim, hv_dim)
        projection = torch.complex(real_part, imag_part)
        self.register_buffer('projection_matrix', projection)
        
        # Quantization angles for PSK-style encoding
        angles = torch.linspace(0, 2 * np.pi, quantization_levels + 1)[:-1]
        quantization_points = torch.exp(1j * angles)
        self.register_buffer('quantization_points', quantization_points)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to complex hypervector."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Convert real input to complex if needed
        if not torch.is_complex(x):
            x = x.to(torch.complex64)
        
        # Project to hypervector space
        projected = torch.matmul(x, self.projection_matrix)
        
        # Phase quantization
        phases = torch.angle(projected)
        
        # Find nearest quantization point
        phase_diffs = torch.abs(phases.unsqueeze(-1) - 
                               torch.angle(self.quantization_points).unsqueeze(0).unsqueeze(0))
        nearest_idx = torch.argmin(phase_diffs, dim=-1)
        
        # Quantize to nearest phase
        quantized = self.quantization_points[nearest_idx]
        
        return quantized
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Compute complex similarity (real part of dot product)."""
        dot_product = torch.sum(hv1 * torch.conj(hv2), dim=-1)
        return torch.real(dot_product) / self.hv_dim


def register_encoder(encoder_class):
    """Decorator to register custom encoder classes."""
    # This would be implemented as part of a plugin system
    pass