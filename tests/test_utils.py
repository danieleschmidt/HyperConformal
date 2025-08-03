"""
Tests for utility functions.
"""

import pytest
import torch
import numpy as np
from hyperconformal.utils import (
    hamming_distance, hamming_similarity, binary_quantize, ternary_quantize,
    compute_coverage, bundle_hypervectors, bind_hypervectors,
    generate_random_hypervector, entropy_of_prediction_set
)


class TestHammingFunctions:
    """Test Hamming distance and similarity functions."""
    
    def test_hamming_distance_identical(self):
        """Test Hamming distance for identical vectors."""
        hv = torch.tensor([1, -1, 1, -1, 1])
        distance = hamming_distance(hv, hv)
        assert distance == 0
    
    def test_hamming_distance_opposite(self):
        """Test Hamming distance for opposite vectors."""
        hv1 = torch.tensor([1, 1, 1, 1, 1])
        hv2 = torch.tensor([-1, -1, -1, -1, -1])
        distance = hamming_distance(hv1, hv2)
        assert distance == 5
    
    def test_hamming_similarity(self):
        """Test Hamming similarity."""
        hv1 = torch.tensor([1, -1, 1, -1])
        hv2 = torch.tensor([1, -1, -1, 1])  # 2 positions differ
        
        similarity = hamming_similarity(hv1, hv2)
        expected = 1.0 - 2.0/4.0  # 2 differences out of 4
        assert abs(similarity - expected) < 1e-6
    
    def test_batch_hamming(self):
        """Test batch Hamming operations."""
        hv1 = torch.tensor([[1, -1, 1], [-1, 1, -1]])
        hv2 = torch.tensor([1, -1, -1])
        
        distances = hamming_distance(hv1, hv2)
        assert distances.shape == (2,)
        
        similarities = hamming_similarity(hv1, hv2)
        assert similarities.shape == (2,)


class TestQuantization:
    """Test quantization functions."""
    
    def test_binary_quantize(self):
        """Test binary quantization."""
        x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
        quantized = binary_quantize(x, threshold=0.0)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(quantized, expected)
    
    def test_binary_quantize_custom_threshold(self):
        """Test binary quantization with custom threshold."""
        x = torch.tensor([-1.0, 0.5, 1.0, 1.5])
        quantized = binary_quantize(x, threshold=1.0)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        assert torch.allclose(quantized, expected)
    
    def test_ternary_quantize(self):
        """Test ternary quantization."""
        # Use fixed values to avoid randomness in std calculation
        x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
        quantized = ternary_quantize(x, threshold=0.5)
        
        # Should have values in {-1, 0, 1}
        assert torch.all((quantized == -1) | (quantized == 0) | (quantized == 1))
    
    def test_ternary_quantize_zeros(self):
        """Test ternary quantization produces zeros for small values."""
        # Small values around zero
        x = torch.tensor([0.01, -0.01, 0.0, 0.02, -0.02])
        quantized = ternary_quantize(x, threshold=10.0)  # High threshold
        
        # All should be quantized to zero
        assert torch.allclose(quantized, torch.zeros_like(x))


class TestCoverage:
    """Test coverage computation."""
    
    def test_perfect_coverage(self):
        """Test perfect coverage case."""
        prediction_sets = [[0], [1], [2], [0]]
        true_labels = [0, 1, 2, 0]
        
        coverage = compute_coverage(prediction_sets, true_labels)
        assert coverage == 1.0
    
    def test_zero_coverage(self):
        """Test zero coverage case."""
        prediction_sets = [[1], [0], [0], [1]]
        true_labels = [0, 1, 2, 0]
        
        coverage = compute_coverage(prediction_sets, true_labels)
        assert coverage == 0.0
    
    def test_partial_coverage(self):
        """Test partial coverage case."""
        prediction_sets = [[0, 1], [1], [2], [2]]
        true_labels = [0, 1, 2, 0]  # Last one not covered
        
        coverage = compute_coverage(prediction_sets, true_labels)
        assert coverage == 0.75
    
    def test_coverage_with_tensor_labels(self):
        """Test coverage with tensor labels."""
        prediction_sets = [[0], [1], [2]]
        true_labels = torch.tensor([0, 1, 1])  # One mismatch
        
        coverage = compute_coverage(prediction_sets, true_labels)
        assert abs(coverage - 2.0/3.0) < 1e-6
    
    def test_empty_prediction_sets(self):
        """Test coverage with empty prediction sets."""
        prediction_sets = []
        true_labels = []
        
        coverage = compute_coverage(prediction_sets, true_labels)
        assert coverage == 0.0


class TestHypervectorOperations:
    """Test hypervector operations."""
    
    def test_bundle_majority(self):
        """Test majority bundling."""
        hv1 = torch.tensor([1, -1, 1, -1])
        hv2 = torch.tensor([1, 1, 1, -1])
        hv3 = torch.tensor([-1, 1, 1, -1])
        
        bundled = bundle_hypervectors(hv1, hv2, hv3, method='majority')
        expected = torch.tensor([1, 1, 1, -1])  # Majority vote
        assert torch.allclose(bundled, expected)
    
    def test_bundle_sum(self):
        """Test sum bundling."""
        hv1 = torch.tensor([1.0, -1.0, 2.0])
        hv2 = torch.tensor([2.0, 1.0, -1.0])
        
        bundled = bundle_hypervectors(hv1, hv2, method='sum')
        expected = torch.tensor([3.0, 0.0, 1.0])
        assert torch.allclose(bundled, expected)
    
    def test_bundle_xor(self):
        """Test XOR bundling."""
        hv1 = torch.tensor([1, -1, 1])
        hv2 = torch.tensor([1, 1, -1])
        
        bundled = bundle_hypervectors(hv1, hv2, method='xor')
        expected = torch.tensor([1, -1, -1])  # Element-wise product for XOR
        assert torch.allclose(bundled, expected)
    
    def test_bind_xor(self):
        """Test XOR binding."""
        hv1 = torch.tensor([1, -1, 1, -1])
        hv2 = torch.tensor([1, 1, -1, -1])
        
        bound = bind_hypervectors(hv1, hv2, method='xor')
        expected = torch.tensor([1, -1, -1, 1])  # Element-wise product
        assert torch.allclose(bound, expected)
    
    def test_bind_circular_convolution(self):
        """Test circular convolution binding."""
        hv1 = torch.tensor([1.0, 0.0, -1.0, 0.0])
        hv2 = torch.tensor([0.0, 1.0, 0.0, -1.0])
        
        bound = bind_hypervectors(hv1, hv2, method='circular_convolution')
        
        # Should be real-valued result
        assert torch.is_floating_point(bound)
        assert bound.shape == hv1.shape


class TestRandomGeneration:
    """Test random hypervector generation."""
    
    def test_binary_generation(self):
        """Test binary random hypervector generation."""
        hv = generate_random_hypervector(100, quantization='binary', seed=42)
        
        assert hv.shape == (100,)
        assert torch.all((hv == 1) | (hv == -1))
    
    def test_ternary_generation(self):
        """Test ternary random hypervector generation."""
        hv = generate_random_hypervector(50, quantization='ternary', seed=42)
        
        assert hv.shape == (50,)
        assert torch.all((hv == 1) | (hv == 0) | (hv == -1))
    
    def test_gaussian_generation(self):
        """Test Gaussian random hypervector generation."""
        hv = generate_random_hypervector(75, quantization='gaussian', seed=42)
        
        assert hv.shape == (75,)
        assert torch.is_floating_point(hv)
        
        # Should have approximately zero mean and unit variance
        assert abs(hv.mean().item()) < 0.2
        assert abs(hv.std().item() - 1.0) < 0.2
    
    def test_reproducibility(self):
        """Test random generation reproducibility."""
        hv1 = generate_random_hypervector(50, quantization='binary', seed=123)
        hv2 = generate_random_hypervector(50, quantization='binary', seed=123)
        
        assert torch.allclose(hv1, hv2)


class TestUncertaintyMeasures:
    """Test uncertainty measures."""
    
    def test_entropy_singleton_sets(self):
        """Test entropy for singleton prediction sets."""
        prediction_sets = [[0], [1], [2]]
        entropies = entropy_of_prediction_set(prediction_sets, num_classes=3)
        
        # Singleton sets should have zero entropy
        assert all(e == 0.0 for e in entropies)
    
    def test_entropy_full_sets(self):
        """Test entropy for full prediction sets."""
        prediction_sets = [[0, 1, 2], [0, 1, 2]]
        entropies = entropy_of_prediction_set(prediction_sets, num_classes=3)
        
        # Full sets should have maximum entropy
        expected_entropy = np.log(3)
        assert all(abs(e - expected_entropy) < 1e-6 for e in entropies)
    
    def test_entropy_partial_sets(self):
        """Test entropy for partial prediction sets."""
        prediction_sets = [[0, 1], [1, 2], [0]]
        entropies = entropy_of_prediction_set(prediction_sets, num_classes=3)
        
        # Two-element sets should have log(2) entropy
        expected_two = np.log(2)
        assert abs(entropies[0] - expected_two) < 1e-6
        assert abs(entropies[1] - expected_two) < 1e-6
        assert entropies[2] == 0.0  # Singleton
    
    def test_entropy_empty_sets(self):
        """Test entropy for empty prediction sets."""
        prediction_sets = [[], [0]]
        entropies = entropy_of_prediction_set(prediction_sets, num_classes=2)
        
        assert entropies[0] == 0.0  # Empty set
        assert entropies[1] == 0.0  # Singleton