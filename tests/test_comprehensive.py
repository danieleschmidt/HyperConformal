"""
Comprehensive tests for HyperConformal with high coverage.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

import hyperconformal as hc
from hyperconformal.hyperconformal import ValidationError, CalibrationError, HyperConformalError


class TestEncoders:
    """Test all encoder implementations."""
    
    def test_random_projection_binary(self):
        """Test binary random projection encoder."""
        encoder = hc.RandomProjection(input_dim=10, hv_dim=100, quantization='binary', seed=42)
        
        x = torch.randn(5, 10)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (5, 100)
        assert torch.all((encoded == 1) | (encoded == -1))  # Binary values
        
        # Test similarity
        hv1 = torch.randn(1, 100)
        hv2 = torch.randn(1, 100)
        sim = encoder.similarity(hv1, hv2)
        assert sim.shape == (1,)
        assert -1 <= sim.item() <= 1
    
    def test_random_projection_ternary(self):
        """Test ternary random projection encoder."""
        encoder = hc.RandomProjection(input_dim=8, hv_dim=50, quantization='ternary', seed=42)
        
        x = torch.randn(3, 8)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (3, 50)
        unique_vals = torch.unique(encoded)
        assert len(unique_vals) <= 3  # At most ternary values
    
    def test_random_projection_complex(self):
        """Test complex random projection encoder."""
        encoder = hc.RandomProjection(input_dim=6, hv_dim=30, quantization='complex', seed=42)
        
        x = torch.randn(2, 6)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (2, 30)
        assert encoded.dtype == torch.complex64
        
        # Test magnitude is approximately 1 (unit circle)
        magnitudes = torch.abs(encoded)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    
    def test_level_hdc(self):
        """Test level-based HDC encoder."""
        encoder = hc.LevelHDC(input_dim=5, hv_dim=40, levels=10, seed=42)
        
        x = torch.rand(3, 5)  # Values in [0, 1]
        encoded = encoder.encode(x)
        
        assert encoded.shape == (3, 40)
        # Check normalization
        norms = torch.norm(encoded, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)
    
    def test_level_hdc_circular(self):
        """Test circular level HDC encoder."""
        encoder = hc.LevelHDC(input_dim=4, hv_dim=32, levels=8, circular=True, seed=42)
        
        x = torch.rand(2, 4)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (2, 32)
    
    def test_complex_hdc(self):
        """Test complex HDC encoder."""
        encoder = hc.ComplexHDC(input_dim=6, hv_dim=24, quantization_levels=4, seed=42)
        
        x = torch.randn(2, 6)
        encoded = encoder.encode(x)
        
        assert encoded.shape == (2, 24)
        assert encoded.dtype == torch.complex64
    
    def test_encoder_memory_footprint(self):
        """Test encoder memory footprint calculation."""
        encoder = hc.RandomProjection(input_dim=10, hv_dim=100, quantization='binary')
        
        footprint = encoder.memory_footprint()
        assert isinstance(footprint, int)
        assert footprint > 0


class TestConformalPredictors:
    """Test conformal prediction algorithms."""
    
    def test_classification_conformal_predictor_aps(self):
        """Test APS score conformal predictor."""
        predictor = hc.ConformalPredictor(alpha=0.1)
        predictor = hc.ClassificationConformalPredictor(alpha=0.1, score_type='aps')
        
        # Mock predictions and labels
        predictions = torch.softmax(torch.randn(20, 3), dim=1)
        labels = torch.randint(0, 3, (20,))
        
        # Test calibration
        predictor.calibrate(predictions, labels)
        assert predictor.quantile is not None
        
        # Test prediction sets
        test_predictions = torch.softmax(torch.randn(5, 3), dim=1)
        pred_sets = predictor.predict_set(test_predictions)
        
        assert len(pred_sets) == 5
        for pred_set in pred_sets:
            assert len(pred_set) >= 1  # At least one class
            assert all(0 <= cls < 3 for cls in pred_set)
    
    def test_classification_conformal_predictor_margin(self):
        """Test margin score conformal predictor."""
        predictor = hc.ClassificationConformalPredictor(alpha=0.2, score_type='margin')
        
        predictions = torch.softmax(torch.randn(15, 4), dim=1)
        labels = torch.randint(0, 4, (15,))
        
        predictor.calibrate(predictions, labels)
        
        test_predictions = torch.softmax(torch.randn(3, 4), dim=1)
        pred_sets = predictor.predict_set(test_predictions)
        
        assert len(pred_sets) == 3
    
    def test_adaptive_conformal_predictor(self):
        """Test adaptive conformal predictor."""
        predictor = hc.AdaptiveConformalPredictor(
            alpha=0.1, window_size=50, update_frequency=10, score_type='aps'
        )
        
        # Initial data
        predictions = torch.softmax(torch.randn(60, 3), dim=1)
        labels = torch.randint(0, 3, (60,))
        
        # Test initial calibration via update
        predictor.update(predictions[:30], labels[:30])
        
        # Test prediction
        test_predictions = torch.softmax(torch.randn(5, 3), dim=1)
        pred_sets = predictor.predict_set(test_predictions)
        assert len(pred_sets) == 5
        
        # Test streaming updates
        predictor.update(predictions[30:], labels[30:])
        
        # Test coverage estimate
        coverage = predictor.get_current_coverage_estimate()
        assert coverage is None or (0 <= coverage <= 1)


class TestConformalHDC:
    """Test main ConformalHDC implementations."""
    
    def test_conformal_hdc_basic(self):
        """Test basic ConformalHDC functionality."""
        encoder = hc.RandomProjection(input_dim=8, hv_dim=64, quantization='binary', seed=42)
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=3, alpha=0.1)
        
        # Training data
        X = torch.randn(40, 8)
        y = torch.randint(0, 3, (40,))
        
        # Test fitting
        predictor.fit(X, y)
        assert predictor.is_fitted
        assert predictor.training_accuracy is not None
        assert predictor.class_prototypes is not None
        assert predictor.class_prototypes.shape == (3, 64)
        
        # Test predictions
        X_test = torch.randn(10, 8)
        y_test = torch.randint(0, 3, (10,))
        
        predictions = predictor.predict(X_test)
        assert predictions.shape == (10,)
        assert all(0 <= pred < 3 for pred in predictions)
        
        probabilities = predictor.predict_proba(X_test)
        assert probabilities.shape == (10, 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(10), atol=1e-5)
        
        pred_sets = predictor.predict_set(X_test)
        assert len(pred_sets) == 10
        
        # Test coverage
        coverage = predictor.get_empirical_coverage(X_test, y_test)
        assert 0 <= coverage <= 1
        
        # Test average set size
        avg_size = predictor.get_average_set_size(X_test)
        assert avg_size >= 1
    
    def test_conformal_hdc_with_validation(self):
        """Test ConformalHDC with input validation enabled."""
        encoder = hc.RandomProjection(input_dim=5, hv_dim=32, quantization='binary')
        predictor = hc.ConformalHDC(
            encoder=encoder, num_classes=2, alpha=0.15, validate_inputs=True
        )
        
        # Valid data
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,))
        predictor.fit(X, y)
        
        # Test invalid input dimensions
        with pytest.raises(ValidationError):
            X_invalid = torch.randn(10, 7)  # Wrong feature dimension
            predictor.predict(X_invalid)
        
        # Test invalid labels
        with pytest.raises(ValidationError):
            y_invalid = torch.randint(0, 4, (20,))  # Labels outside class range
            predictor.fit(X, y_invalid)
    
    def test_conformal_hdc_different_score_types(self):
        """Test ConformalHDC with different score types."""
        encoder = hc.RandomProjection(input_dim=6, hv_dim=48, quantization='binary')
        
        for score_type in ['aps', 'margin', 'inverse_softmax']:
            predictor = hc.ConformalHDC(
                encoder=encoder, num_classes=3, alpha=0.1, score_type=score_type
            )
            
            X = torch.randn(30, 6)
            y = torch.randint(0, 3, (30,))
            predictor.fit(X, y)
            
            X_test = torch.randn(5, 6)
            pred_sets = predictor.predict_set(X_test)
            assert len(pred_sets) == 5
    
    def test_conformal_hdc_memory_footprint(self):
        """Test memory footprint calculation."""
        encoder = hc.RandomProjection(input_dim=10, hv_dim=100, quantization='binary')
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=4, alpha=0.1)
        
        X = torch.randn(25, 10)
        y = torch.randint(0, 4, (25,))
        predictor.fit(X, y)
        
        footprint = predictor.memory_footprint()
        assert isinstance(footprint, dict)
        assert 'encoder' in footprint
        assert 'prototypes' in footprint
        assert 'total' in footprint
        assert footprint['total'] > 0
    
    def test_conformal_hdc_summary(self):
        """Test model summary generation."""
        encoder = hc.RandomProjection(input_dim=8, hv_dim=64, quantization='ternary')
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=3, alpha=0.2)
        
        X = torch.randn(30, 8)
        y = torch.randint(0, 3, (30,))
        predictor.fit(X, y)
        
        summary = predictor.summary()
        assert isinstance(summary, dict)
        assert 'encoder_type' in summary
        assert 'num_classes' in summary
        assert 'training_accuracy' in summary
        assert summary['encoder_type'] == 'RandomProjection'
        assert summary['num_classes'] == 3
    
    def test_conformal_hdc_health_check(self):
        """Test model health check."""
        encoder = hc.RandomProjection(input_dim=6, hv_dim=32, quantization='binary')
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=2, alpha=0.1)
        
        # Before fitting
        health = predictor.health_check()
        assert not health['is_fitted']
        assert not health['prototypes_valid']
        
        # After fitting
        X = torch.randn(20, 6)
        y = torch.randint(0, 2, (20,))
        predictor.fit(X, y)
        
        health = predictor.health_check()
        assert health['is_fitted']
        assert health['prototypes_valid']
        assert health['encoder_valid']
        assert health['conformal_calibrated']


class TestOptimizedImplementations:
    """Test optimized implementations."""
    
    def test_optimized_conformal_hdc(self):
        """Test OptimizedConformalHDC."""
        encoder = hc.RandomProjection(input_dim=10, hv_dim=80, quantization='binary')
        predictor = hc.OptimizedConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            enable_caching=True,
            cache_size=100,
            batch_size=8,
            use_gpu_acceleration=False,  # CPU for testing
            auto_optimize=False  # Disable for consistent testing
        )
        
        X = torch.randn(32, 10)
        y = torch.randint(0, 3, (32,))
        predictor.fit(X, y)
        
        # Test optimized predictions
        X_test = torch.randn(12, 10)
        pred_sets = predictor.predict_set_optimized(X_test)
        assert len(pred_sets) == 12
        
        # Test batch parallel prediction
        predictions = predictor.predict_batch_parallel(X_test, batch_size=4)
        assert len(predictions) == 12
        
        # Test optimization stats
        opt_stats = predictor.get_optimization_stats()
        assert isinstance(opt_stats, dict)
        assert 'batch_size' in opt_stats
        assert 'cache_enabled' in opt_stats
        
        # Test caching
        if 'cache' in opt_stats:
            cache_stats = opt_stats['cache']
            assert 'hit_rate' in cache_stats
            assert 'cache_size' in cache_stats
    
    def test_scalable_adaptive_conformal_hdc(self):
        """Test ScalableAdaptiveConformalHDC."""
        encoder = hc.RandomProjection(input_dim=8, hv_dim=64, quantization='binary')
        predictor = hc.ScalableAdaptiveConformalHDC(
            encoder=encoder,
            num_classes=2,
            alpha=0.1,
            window_size=20,
            update_frequency=5,
            enable_caching=True,
            batch_size=4,
            auto_optimize=False,
            drift_detection=True
        )
        
        # Initial training
        X = torch.randn(24, 8)
        y = torch.randint(0, 2, (24,))
        predictor.fit(X, y)
        
        # Test streaming updates
        X_stream = torch.randn(8, 8)
        y_stream = torch.randint(0, 2, (8,))
        
        update_stats = predictor.update(X_stream, y_stream)
        assert isinstance(update_stats, dict)
        assert 'samples_processed' in update_stats
        assert 'drift_detected' in update_stats
        
        # Test streaming stats
        streaming_stats = predictor.get_streaming_stats()
        assert isinstance(streaming_stats, dict)
        assert 'window_size' in streaming_stats
        assert 'drift_detection_enabled' in streaming_stats


class TestUtilities:
    """Test utility functions."""
    
    def test_hamming_distance(self):
        """Test Hamming distance computation."""
        hv1 = torch.tensor([[1, -1, 1, -1]])
        hv2 = torch.tensor([[1, 1, -1, -1]])
        
        distance = hc.hamming_distance(hv1, hv2)
        assert distance.item() == 2.0  # 2 positions differ
        
        # Test hamming_similarity (not in __all__ but should be importable)
        from hyperconformal.utils import hamming_similarity
        similarity = hamming_similarity(hv1, hv2)
        assert similarity.item() == 0.5  # 50% similar
    
    def test_binary_quantize(self):
        """Test binary quantization."""
        x = torch.tensor([[-0.5, 0.3, -1.2, 0.8]])
        quantized = hc.binary_quantize(x)
        expected = torch.tensor([[-1, 1, -1, 1]])
        assert torch.equal(quantized, expected.float())
    
    def test_compute_coverage(self):
        """Test coverage computation."""
        pred_sets = [[0, 1], [1], [0, 2], [2]]
        true_labels = [1, 1, 0, 1]
        
        coverage = hc.compute_coverage(pred_sets, true_labels)
        assert coverage == 0.75  # 3 out of 4 covered
        
        # Test with torch tensor
        coverage = hc.compute_coverage(pred_sets, torch.tensor(true_labels))
        assert coverage == 0.75
        
        # Test with numpy array
        coverage = hc.compute_coverage(pred_sets, np.array(true_labels))
        assert coverage == 0.75


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_coverage_score(self):
        """Test coverage score computation."""
        pred_sets = [[0], [1, 2], [0, 1]]
        true_labels = [0, 2, 1]
        
        coverage = hc.coverage_score(pred_sets, true_labels)
        assert coverage == 1.0  # All covered
    
    def test_average_set_size(self):
        """Test average set size computation."""
        pred_sets = [[0], [1, 2], [0, 1, 2]]
        
        avg_size = hc.average_set_size(pred_sets)
        assert avg_size == 2.0  # (1 + 2 + 3) / 3
    
    def test_conditional_coverage(self):
        """Test conditional coverage computation."""
        pred_sets = [[0], [1], [0, 1], [2]]
        true_labels = [0, 1, 0, 2]
        
        cond_cov = hc.conditional_coverage(pred_sets, true_labels, stratify_by='class')
        assert isinstance(cond_cov, dict)
        assert 0 in cond_cov
        assert 1 in cond_cov
        assert 2 in cond_cov


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_encoder_parameters(self):
        """Test invalid encoder parameters."""
        with pytest.raises(ValueError):
            hc.RandomProjection(input_dim=10, hv_dim=100, quantization='invalid')
    
    def test_invalid_conformal_hdc_parameters(self):
        """Test invalid ConformalHDC parameters."""
        encoder = hc.RandomProjection(input_dim=5, hv_dim=32, quantization='binary')
        
        # Invalid alpha
        with pytest.raises(ValidationError):
            hc.ConformalHDC(encoder=encoder, num_classes=3, alpha=1.5, validate_inputs=True)
        
        # Invalid num_classes
        with pytest.raises(ValidationError):
            hc.ConformalHDC(encoder=encoder, num_classes=1, validate_inputs=True)
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        encoder = hc.RandomProjection(input_dim=5, hv_dim=32, quantization='binary')
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=3)
        
        X_test = torch.randn(5, 5)
        
        with pytest.raises(ValueError):
            predictor.predict(X_test)
        
        with pytest.raises(ValueError):
            predictor.predict_set(X_test)
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        # Empty prediction sets
        coverage = hc.compute_coverage([], [])
        assert coverage == 0.0
        
        avg_size = hc.average_set_size([])
        assert avg_size == 0.0
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        encoder = hc.RandomProjection(input_dim=5, hv_dim=32, quantization='binary')
        predictor = hc.ConformalHDC(encoder=encoder, num_classes=2, validate_inputs=False)
        
        # Normal training
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,))
        predictor.fit(X, y)
        
        # Test with NaN values
        X_test = torch.randn(5, 5)
        X_test[0, 0] = float('nan')
        
        # Should handle gracefully (with warnings)
        with pytest.warns(UserWarning):
            predictions = predictor.predict_proba(X_test)
            assert not torch.isnan(predictions).any()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create encoder
        encoder = hc.RandomProjection(input_dim=12, hv_dim=128, quantization='binary', seed=42)
        
        # Create predictor
        predictor = hc.ConformalHDC(
            encoder=encoder,
            num_classes=4,
            alpha=0.1,
            score_type='aps',
            calibration_split=0.2,
            validate_inputs=True,
            random_state=42
        )
        
        # Generate data
        np.random.seed(42)
        torch.manual_seed(42)
        
        X_train = torch.randn(100, 12)
        y_train = torch.randint(0, 4, (100,))
        
        X_test = torch.randn(20, 12)
        y_test = torch.randint(0, 4, (20,))
        
        # Train
        predictor.fit(X_train, y_train)
        
        # Predict
        predictions = predictor.predict(X_test)
        probabilities = predictor.predict_proba(X_test)
        pred_sets = predictor.predict_set(X_test)
        
        # Evaluate
        coverage = predictor.get_empirical_coverage(X_test, y_test)
        avg_set_size = predictor.get_average_set_size(X_test)
        
        # Check results
        assert len(predictions) == 20
        assert probabilities.shape == (20, 4)
        assert len(pred_sets) == 20
        assert 0 <= coverage <= 1
        assert avg_set_size >= 1
        
        # Test metrics
        coverage_metric = hc.coverage_score(pred_sets, y_test)
        size_metric = hc.average_set_size(pred_sets)
        cond_coverage = hc.conditional_coverage(pred_sets, y_test)
        
        assert 0 <= coverage_metric <= 1
        assert size_metric >= 1
        assert isinstance(cond_coverage, dict)
        
        # Test health and summary
        health = predictor.health_check()
        summary = predictor.summary()
        
        assert health['is_fitted']
        assert health['prototypes_valid']
        assert isinstance(summary, dict)
    
    def test_different_encoder_combinations(self):
        """Test different encoder and predictor combinations."""
        encoders = [
            hc.RandomProjection(8, 64, 'binary', seed=42),
            hc.RandomProjection(8, 64, 'ternary', seed=42),
            hc.LevelHDC(8, 64, levels=10, seed=42),
            hc.ComplexHDC(8, 64, quantization_levels=4, seed=42)
        ]
        
        X = torch.randn(30, 8)
        y = torch.randint(0, 3, (30,))
        X_test = torch.randn(10, 8)
        
        for encoder in encoders:
            predictor = hc.ConformalHDC(encoder=encoder, num_classes=3, alpha=0.1)
            predictor.fit(X, y)
            
            pred_sets = predictor.predict_set(X_test)
            assert len(pred_sets) == 10
            
            # Test that all prediction sets are valid
            for pred_set in pred_sets:
                assert len(pred_set) >= 1
                assert all(0 <= cls < 3 for cls in pred_set)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=hyperconformal", "--cov-report=term-missing"])