"""
Tests for HDC encoders.
"""

import pytest
import torch
import numpy as np
from hyperconformal.encoders import RandomProjection, LevelHDC, ComplexHDC


class TestRandomProjection:
    """Test RandomProjection encoder."""
    
    def test_binary_quantization(self):
        """Test binary quantization."""
        encoder = RandomProjection(
            input_dim=100, 
            hv_dim=1000, 
            quantization='binary',
            seed=42
        )
        
        # Test encoding
        x = torch.randn(10, 100)
        encoded = encoder.encode(x)
        
        # Check output shape
        assert encoded.shape == (10, 1000)
        
        # Check binary values
        assert torch.all((encoded == 1) | (encoded == -1))
        
        # Test similarity
        sim = encoder.similarity(encoded[0], encoded[1])
        assert -1 <= sim <= 1
    
    def test_ternary_quantization(self):
        """Test ternary quantization."""
        encoder = RandomProjection(
            input_dim=50, 
            hv_dim=500, 
            quantization='ternary',
            seed=42
        )
        
        x = torch.randn(5, 50)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (5, 500)
        assert torch.all((encoded == 1) | (encoded == 0) | (encoded == -1))
    
    def test_complex_quantization(self):
        """Test complex quantization."""
        encoder = RandomProjection(
            input_dim=20, 
            hv_dim=200, 
            quantization='complex',
            seed=42
        )
        
        x = torch.randn(3, 20)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (3, 200)
        assert torch.is_complex(encoded)
        
        # Check unit magnitude
        magnitudes = torch.abs(encoded)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)
    
    def test_reproducibility(self):
        """Test that encoders are reproducible with same seed."""
        encoder1 = RandomProjection(10, 100, seed=42)
        encoder2 = RandomProjection(10, 100, seed=42)
        
        x = torch.randn(5, 10)
        encoded1 = encoder1.encode(x)
        encoded2 = encoder2.encode(x)
        
        assert torch.allclose(encoded1, encoded2)
    
    def test_memory_footprint(self):
        """Test memory footprint calculation."""
        encoder = RandomProjection(100, 1000, seed=42)
        footprint = encoder.memory_footprint()
        
        assert footprint > 0
        # Should be approximately 100 * 1000 * 4 bytes (float32)
        expected = 100 * 1000 * 4
        assert abs(footprint - expected) < expected * 0.1  # Within 10%


class TestLevelHDC:
    """Test LevelHDC encoder."""
    
    def test_basic_encoding(self):
        """Test basic level encoding."""
        encoder = LevelHDC(
            input_dim=10, 
            hv_dim=100, 
            levels=10,
            seed=42
        )
        
        # Test with normalized input [0, 1]
        x = torch.rand(5, 10)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (5, 100)
        
        # Check normalization
        norms = torch.norm(encoded, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_circular_encoding(self):
        """Test circular level encoding."""
        encoder = LevelHDC(
            input_dim=5, 
            hv_dim=50, 
            levels=8,
            circular=True,
            seed=42
        )
        
        x = torch.rand(3, 5)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (3, 50)
    
    def test_similarity(self):
        """Test similarity computation."""
        encoder = LevelHDC(input_dim=5, hv_dim=50, seed=42)
        
        x = torch.rand(2, 5)
        encoded = encoder.encode(x)
        
        sim = encoder.similarity(encoded[0], encoded[1])
        assert -1 <= sim <= 1


class TestComplexHDC:
    """Test ComplexHDC encoder."""
    
    def test_complex_encoding(self):
        """Test complex HDC encoding."""
        encoder = ComplexHDC(
            input_dim=20,
            hv_dim=100,
            quantization_levels=4,
            seed=42
        )
        
        x = torch.randn(5, 20)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (5, 100)
        assert torch.is_complex(encoded)
        
        # Check unit magnitude
        magnitudes = torch.abs(encoded)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)
    
    def test_psk_quantization(self):
        """Test PSK-style phase quantization."""
        encoder = ComplexHDC(
            input_dim=10,
            hv_dim=50,
            quantization_levels=8,
            seed=42
        )
        
        x = torch.randn(3, 10).to(torch.complex64)
        encoded = encoder.encode(x)
        
        # Check that phases are quantized to specific angles
        phases = torch.angle(encoded)
        unique_phases = torch.unique(phases.round(decimals=4))
        
        # Should have limited number of unique phases
        assert len(unique_phases) <= encoder.quantization_levels
    
    def test_complex_similarity(self):
        """Test complex similarity computation."""
        encoder = ComplexHDC(input_dim=10, hv_dim=50, seed=42)
        
        x = torch.randn(2, 10)
        encoded = encoder.encode(x)
        
        sim = encoder.similarity(encoded[0], encoded[1])
        assert torch.is_tensor(sim)
        assert sim.dtype == torch.float32  # Real-valued similarity