"""
Integration tests for HyperConformal library.

These tests verify that different components work together correctly
and test real-world scenarios with actual data flows.
"""

import pytest
import torch
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from hyperconformal import (
    RandomProjection, LevelHDC, ComplexHDC,
    ConformalHDC, AdaptiveConformalHDC,
    coverage_score, average_set_size
)


class TestHDCConformalIntegration:
    """Test integration between HDC encoders and conformal predictors."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=3,
            n_informative=15,
            n_redundant=2,
            random_state=42
        )
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    @pytest.fixture
    def streaming_data(self):
        """Generate streaming data batches."""
        X, y = make_classification(
            n_samples=1000,
            n_features=15,
            n_classes=3,
            n_informative=10,
            random_state=123
        )
        
        # Split into batches
        batch_size = 100
        batches = []
        for i in range(0, len(X), batch_size):
            X_batch = torch.FloatTensor(X[i:i+batch_size])
            y_batch = torch.LongTensor(y[i:i+batch_size])
            batches.append((X_batch, y_batch))
        
        return batches
    
    def test_random_projection_binary_pipeline(self, classification_data):
        """Test complete pipeline with binary RandomProjection."""
        X, y = classification_data
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create encoder and model
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
            score_type='aps'
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(0 <= pred < 3 for pred in predictions)
        
        # Test prediction sets
        pred_sets = model.predict_set(X_test)
        assert len(pred_sets) == len(X_test)
        
        # Check coverage
        coverage = coverage_score(pred_sets, y_test.numpy())
        assert 0.6 <= coverage <= 1.0  # Should have reasonable coverage (statistical guarantees)
        
        # Check efficiency
        avg_size = average_set_size(pred_sets)
        assert 1.0 <= avg_size <= 3.0  # Sets should be reasonably sized
    
    def test_level_hdc_pipeline(self, classification_data):
        """Test complete pipeline with LevelHDC."""
        X, y = classification_data
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create encoder and model
        encoder = LevelHDC(
            input_dim=X.shape[1],
            hv_dim=800,
            levels=50,
            seed=42
        )
        
        model = ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.15,
            score_type='margin'
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Verify model properties
        assert model.is_fitted
        assert model.training_accuracy > 0.0
        assert model.calibration_coverage > 0.0
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X_test)), atol=1e-6)
    
    def test_adaptive_conformal_streaming(self, streaming_data):
        """Test adaptive conformal with streaming data."""
        encoder = RandomProjection(
            input_dim=15,
            hv_dim=500,
            quantization='ternary',
            seed=42
        )
        
        model = AdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            window_size=200,
            update_frequency=50
        )
        
        # Initialize with first batch
        X_init, y_init = streaming_data[0]
        model.fit(X_init, y_init)
        
        # Process streaming batches
        coverages = []
        for i, (X_batch, y_batch) in enumerate(streaming_data[1:4]):  # Use first few batches
            # Get predictions before update
            pred_sets = model.predict_set(X_batch)
            coverage = coverage_score(pred_sets, y_batch.numpy())
            coverages.append(coverage)
            
            # Update model
            model.update(X_batch, y_batch)
        
        # Check that coverage is maintained across batches
        avg_coverage = np.mean(coverages)
        assert 0.5 <= avg_coverage <= 1.0  # Should maintain reasonable coverage


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_multiclass_classification_scenario(self):
        """Test a complete multiclass classification scenario."""
        # Generate data with clear class separation
        X, y = make_classification(
            n_samples=800,
            n_features=25,
            n_classes=5,
            n_informative=20,
            n_clusters_per_class=1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Test multiple encoder types
        encoders = {
            'binary_rp': RandomProjection(25, 1000, 'binary', seed=42),
            'ternary_rp': RandomProjection(25, 1000, 'ternary', seed=42),
            'level_hdc': LevelHDC(25, 800, levels=40, seed=42)
        }
        
        results = {}
        
        for name, encoder in encoders.items():
            model = ConformalHDC(
                encoder=encoder,
                num_classes=5,
                alpha=0.1,
                score_type='aps'
            )
            
            # Train and evaluate
            model.fit(X_train, y_train)
            
            # Collect metrics
            predictions = model.predict(X_test)
            pred_sets = model.predict_set(X_test)
            
            accuracy = (predictions == y_test.numpy()).mean()
            coverage = coverage_score(pred_sets, y_test.numpy())
            efficiency = 1 - (average_set_size(pred_sets) - 1) / 4  # Normalized efficiency
            
            results[name] = {
                'accuracy': accuracy,
                'coverage': coverage,
                'efficiency': efficiency,
                'avg_set_size': average_set_size(pred_sets)
            }
        
        # Verify all models perform reasonably
        for name, metrics in results.items():
            assert metrics['accuracy'] > 0.5, f"{name} has poor accuracy"
            assert metrics['coverage'] > 0.75, f"{name} has poor coverage"
            assert 1.0 <= metrics['avg_set_size'] <= 3.0, f"{name} has poor efficiency"
    
    def test_cross_validation_scenario(self):
        """Test k-fold cross-validation scenario."""
        X, y = make_classification(
            n_samples=400,
            n_features=15,
            n_classes=3,
            n_informative=10,
            random_state=42
        )
        
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Simple 3-fold CV
        n_folds = 3
        fold_size = len(X) // n_folds
        
        fold_results = []
        
        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size
            
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            X_train = torch.cat([X[:test_start], X[test_end:]], dim=0)
            y_train = torch.cat([y[:test_start], y[test_end:]], dim=0)
            
            # Train model
            encoder = RandomProjection(15, 500, 'binary', seed=42 + fold)
            model = ConformalHDC(encoder, 3, alpha=0.1)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            pred_sets = model.predict_set(X_test)
            coverage = coverage_score(pred_sets, y_test.numpy())
            avg_size = average_set_size(pred_sets)
            
            fold_results.append({
                'coverage': coverage,
                'avg_size': avg_size
            })
        
        # Check consistency across folds
        coverages = [r['coverage'] for r in fold_results]
        avg_sizes = [r['avg_size'] for r in fold_results]
        
        assert all(c > 0.7 for c in coverages), "Coverage too low in some folds"
        assert np.std(coverages) < 0.2, "Coverage too variable across folds"
        assert all(1.0 <= s <= 2.5 for s in avg_sizes), "Set sizes out of reasonable range"


class TestRobustnessAndEdgeCases:
    """Test robustness and edge cases."""
    
    def test_small_dataset_robustness(self):
        """Test behavior with very small datasets."""
        # Create minimal dataset
        X = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))
        
        encoder = RandomProjection(10, 200, 'binary', seed=42)
        model = ConformalHDC(encoder, 3, alpha=0.2)
        
        # Should handle small dataset gracefully
        model.fit(X, y)
        
        # Should still make predictions
        predictions = model.predict(X[:5])
        pred_sets = model.predict_set(X[:5])
        
        assert len(predictions) == 5
        assert len(pred_sets) == 5
    
    def test_imbalanced_data_handling(self):
        """Test handling of imbalanced datasets."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=3,
            weights=[0.8, 0.15, 0.05],  # Very imbalanced
            random_state=42
        )
        
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        encoder = RandomProjection(10, 400, 'binary', seed=42)
        model = ConformalHDC(encoder, 3, alpha=0.1)
        
        # Should handle imbalanced data
        model.fit(X, y)
        
        # Check that all classes have learned prototypes
        assert model.class_prototypes.shape == (3, 400)
        
        # Should still provide calibrated predictions
        pred_sets = model.predict_set(X[:50])
        coverage = coverage_score(pred_sets, y[:50].numpy())
        
        # May have lower coverage due to imbalance, but should be reasonable
        assert coverage > 0.6
    
    def test_memory_efficiency(self):
        """Test memory efficiency with larger problems."""
        # Create moderately large problem
        X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_classes=4,
            random_state=42
        )
        
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Test with different hypervector dimensions
        for hv_dim in [500, 1000, 2000]:
            encoder = RandomProjection(50, hv_dim, 'binary', seed=42)
            model = ConformalHDC(encoder, 4, alpha=0.1)
            
            # Should fit without memory issues
            model.fit(X, y)
            
            # Check memory footprint is reasonable
            footprint = model.memory_footprint()
            expected_size = 4 * hv_dim * 4  # 4 classes * hv_dim * 4 bytes per float
            assert footprint >= expected_size  # Should include at least the prototypes


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])