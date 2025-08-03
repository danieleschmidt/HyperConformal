"""
Tests for the main HyperConformal classes.
"""

import pytest
import torch
import numpy as np
from sklearn.datasets import make_classification
from hyperconformal import ConformalHDC, AdaptiveConformalHDC, RandomProjection
from hyperconformal.metrics import coverage_score, average_set_size


class TestConformalHDC:
    """Test ConformalHDC main class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_classes=3,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        return torch.from_numpy(X).float(), torch.from_numpy(y).long()
    
    def test_initialization(self, sample_data):
        """Test ConformalHDC initialization."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=1000,
            quantization='binary',
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            calibration_split=0.2
        )
        
        assert model.num_classes == 3
        assert model.alpha == 0.1
        assert not model.is_fitted
    
    def test_fitting(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=1000,
            quantization='binary',
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1
        )
        
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.class_prototypes is not None
        assert model.class_prototypes.shape == (3, 1000)
        assert model.training_accuracy is not None
        assert 0 <= model.training_accuracy <= 1
    
    def test_predictions(self, sample_data):
        """Test various prediction methods."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=500,
            quantization='binary',
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1
        )
        
        model.fit(X, y)
        
        # Test subset of data
        X_test = X[:50]
        y_test = y[:50]
        
        # Test class predictions
        pred_classes = model.predict(X_test)
        assert pred_classes.shape == (50,)
        assert np.all((pred_classes >= 0) & (pred_classes < 3))
        
        # Test probabilities
        pred_probs = model.predict_proba(X_test)
        assert pred_probs.shape == (50, 3)
        assert torch.allclose(pred_probs.sum(dim=1), torch.ones(50), atol=1e-6)
        
        # Test prediction sets
        pred_sets = model.predict_set(X_test)
        assert len(pred_sets) == 50
        for pred_set in pred_sets:
            assert len(pred_set) >= 1
            assert all(0 <= cls < 3 for cls in pred_set)
    
    def test_coverage_guarantee(self, sample_data):
        """Test coverage guarantees."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=1000,
            quantization='binary',
            seed=42
        )
        
        alpha = 0.1
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=alpha
        )
        
        # Split data
        split = len(X) // 2
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model.fit(X_train, y_train)
        
        # Test coverage on held-out data
        coverage = model.get_empirical_coverage(X_test, y_test)
        target_coverage = 1 - alpha
        
        # Coverage should be close to target
        assert abs(coverage - target_coverage) < 0.15  # Allow some variance
        
        # Test average set size
        avg_size = model.get_average_set_size(X_test)
        assert 1 <= avg_size <= 3  # Between singleton and full set
    
    def test_different_quantizations(self, sample_data):
        """Test different HDC quantization schemes."""
        X, y = sample_data
        
        quantizations = ['binary', 'ternary']
        
        for quant in quantizations:
            encoder = RandomProjection(
                input_dim=X.shape[1],
                hv_dim=500,
                quantization=quant,
                seed=42
            )
            
            model = ConformalHDC(
                encoder=encoder,
                num_classes=3,
                alpha=0.1
            )
            
            model.fit(X, y)
            
            # Should work for all quantizations
            pred_sets = model.predict_set(X[:10])
            assert len(pred_sets) == 10
    
    def test_memory_footprint(self, sample_data):
        """Test memory footprint calculation."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=1000,
            quantization='binary',
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1
        )
        
        model.fit(X, y)
        
        footprint = model.memory_footprint()
        
        assert 'total' in footprint
        assert 'encoder' in footprint
        assert 'prototypes' in footprint
        assert footprint['total'] > 0
    
    def test_model_summary(self, sample_data):
        """Test model summary."""
        X, y = sample_data
        
        encoder = RandomProjection(
            input_dim=X.shape[1],
            hv_dim=500,
            quantization='binary',
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.15
        )
        
        model.fit(X, y)
        summary = model.summary()
        
        assert summary['encoder_type'] == 'RandomProjection'
        assert summary['hv_dim'] == 500
        assert summary['num_classes'] == 3
        assert summary['alpha'] == 0.15
        assert summary['coverage_guarantee'] == 0.85
        assert summary['is_fitted'] == True


class TestAdaptiveConformalHDC:
    """Test AdaptiveConformalHDC for streaming data."""
    
    @pytest.fixture
    def streaming_data(self):
        """Generate streaming classification data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate multiple batches
        batches = []
        for i in range(5):
            X, y = make_classification(
                n_samples=100,
                n_features=15,
                n_classes=3,
                n_informative=10,
                random_state=42 + i
            )
            batches.append((torch.from_numpy(X).float(), torch.from_numpy(y).long()))
        
        return batches
    
    def test_adaptive_initialization(self):
        """Test adaptive model initialization."""
        encoder = RandomProjection(
            input_dim=15,
            hv_dim=500,
            quantization='binary',
            seed=42
        )
        
        model = AdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            window_size=200,
            update_frequency=50
        )
        
        assert model.window_size == 200
        assert model.update_frequency == 50
    
    def test_streaming_updates(self, streaming_data):
        """Test streaming updates."""
        encoder = RandomProjection(
            input_dim=15,
            hv_dim=500,
            quantization='binary',
            seed=42
        )
        
        model = AdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            window_size=100,
            update_frequency=25
        )
        
        # Fit initial model
        X_init, y_init = streaming_data[0]
        model.fit(X_init, y_init)
        
        # Process streaming batches
        for X_batch, y_batch in streaming_data[1:3]:
            model.update(X_batch, y_batch)
        
        # Should be able to make predictions
        X_test, y_test = streaming_data[-1]
        pred_sets = model.predict_set(X_test[:20])
        
        assert len(pred_sets) == 20
        
        # Check coverage estimate
        coverage_est = model.get_current_coverage_estimate()
        if coverage_est is not None:
            assert 0 <= coverage_est <= 1
    
    def test_adaptive_coverage(self, streaming_data):
        """Test adaptive coverage maintenance."""
        encoder = RandomProjection(
            input_dim=15,
            hv_dim=500,
            quantization='binary',
            seed=42
        )
        
        model = AdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            window_size=150,
            update_frequency=30
        )
        
        # Fit and update with streaming data
        X_init, y_init = streaming_data[0]
        model.fit(X_init, y_init)
        
        for X_batch, y_batch in streaming_data[1:]:
            model.update(X_batch, y_batch)
        
        # Test on final batch
        X_test, y_test = streaming_data[-1]
        coverage = model.get_empirical_coverage(X_test, y_test)
        
        # Should maintain reasonable coverage
        target_coverage = 1 - model.alpha
        assert abs(coverage - target_coverage) < 0.2  # Allow more variance for streaming