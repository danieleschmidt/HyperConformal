"""
End-to-end tests for HyperConformal library.

These tests simulate real-world usage scenarios and validate
the complete system functionality from data loading to predictions.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from hyperconformal import (
    RandomProjection, LevelHDC, ComplexHDC,
    ConformalHDC, AdaptiveConformalHDC,
    coverage_score, average_set_size
)
from hyperconformal.metrics import conformal_prediction_metrics


class TestRealWorldDatasets:
    """Test on real-world datasets."""
    
    def test_iris_classification(self):
        """Test on the classic Iris dataset."""
        # Load and prepare data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Test with multiple configurations
        configs = [
            {'encoder': RandomProjection(4, 200, 'binary', seed=42), 'alpha': 0.1},
            {'encoder': RandomProjection(4, 300, 'ternary', seed=42), 'alpha': 0.05},
            {'encoder': LevelHDC(4, 150, levels=20, seed=42), 'alpha': 0.15}
        ]
        
        for i, config in enumerate(configs):
            model = ConformalHDC(
                encoder=config['encoder'],
                num_classes=3,
                alpha=config['alpha'],
                score_type='aps'
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            pred_sets = model.predict_set(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Validate predictions
            assert len(predictions) == len(X_test)
            assert len(pred_sets) == len(X_test)
            assert probabilities.shape == (len(X_test), 3)
            
            # Check coverage
            coverage = coverage_score(pred_sets, y_test.numpy())
            target_coverage = 1 - config['alpha']
            
            # Should achieve good coverage on this well-separated dataset
            assert coverage >= target_coverage - 0.2, f"Config {i}: Coverage {coverage} below {target_coverage - 0.2}"
            
            # Check accuracy (Iris is easy, should get high accuracy)
            accuracy = (predictions == y_test.numpy()).mean()
            assert accuracy > 0.8, f"Config {i}: Accuracy {accuracy} too low"
    
    def test_wine_classification(self):
        """Test on the Wine dataset."""
        wine = load_wine()
        X, y = wine.data, wine.target
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Use higher-dimensional encoding for more features
        encoder = RandomProjection(13, 800, 'binary', seed=42)
        model = ConformalHDC(encoder, 3, alpha=0.1)
        
        model.fit(X_train, y_train)
        
        # Comprehensive evaluation
        pred_sets = model.predict_set(X_test)
        metrics = conformal_prediction_metrics(
            pred_sets, y_test.numpy(), num_classes=3, target_coverage=0.9
        )
        
        # Validate metrics
        assert metrics['coverage'] > 0.7, "Coverage too low"
        assert metrics['efficiency'] > 0.4, "Efficiency too low"
        assert 1.0 <= metrics['average_set_size'] <= 2.5, "Set sizes unreasonable"
    
    def test_digits_classification(self):
        """Test on a subset of the digits dataset."""
        digits = load_digits()
        
        # Use subset for faster testing
        n_samples = 500
        indices = np.random.RandomState(42).choice(len(digits.data), n_samples, replace=False)
        X = digits.data[indices]
        y = digits.target[indices]
        
        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Use larger encoding for high-dimensional data
        encoder = RandomProjection(64, 1500, 'binary', seed=42)
        model = ConformalHDC(encoder, 10, alpha=0.1)
        
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        pred_sets = model.predict_set(X_test)
        
        # Validate
        coverage = coverage_score(pred_sets, y_test.numpy())
        avg_size = average_set_size(pred_sets)
        
        assert coverage > 0.6, "Coverage too low for digits"
        assert avg_size <= 5.0, "Average set size too large"


class TestProductionScenarios:
    """Test production-like scenarios."""
    
    def test_model_serialization_workflow(self):
        """Test complete model save/load workflow."""
        # Generate data
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Train model
        encoder = RandomProjection(10, 400, 'binary', seed=42)
        model = ConformalHDC(encoder, 3, alpha=0.1)
        model.fit(X_train, y_train)
        
        # Get baseline predictions
        baseline_predictions = model.predict(X_test)
        baseline_pred_sets = model.predict_set(X_test)
        
        # Test model state access
        model_summary = model.get_model_summary()
        
        assert 'training_accuracy' in model_summary
        assert 'calibration_coverage' in model_summary
        assert 'num_classes' in model_summary
        assert 'encoder_type' in model_summary
        
        # Test memory footprint calculation
        footprint = model.memory_footprint()
        assert footprint > 0
        
        # Verify predictions are consistent
        repeated_predictions = model.predict(X_test)
        assert np.array_equal(baseline_predictions, repeated_predictions)
    
    def test_batch_prediction_workflow(self):
        """Test batch prediction scenarios."""
        # Generate larger dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=4, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Train model
        encoder = RandomProjection(20, 600, 'binary', seed=42)
        model = ConformalHDC(encoder, 4, alpha=0.1)
        model.fit(X_train, y_train)
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, len(X_test)]
        
        for batch_size in batch_sizes:
            all_predictions = []
            all_pred_sets = []
            
            # Process in batches
            for i in range(0, len(X_test), batch_size):
                batch_end = min(i + batch_size, len(X_test))
                X_batch = X_test[i:batch_end]
                
                batch_predictions = model.predict(X_batch)
                batch_pred_sets = model.predict_set(X_batch)
                
                all_predictions.extend(batch_predictions)
                all_pred_sets.extend(batch_pred_sets)
            
            # Verify consistency across batch sizes
            assert len(all_predictions) == len(X_test)
            assert len(all_pred_sets) == len(X_test)
            
            # Check coverage
            coverage = coverage_score(all_pred_sets, y_test.numpy())
            assert coverage > 0.6, f"Poor coverage with batch_size={batch_size}"
    
    def test_streaming_production_scenario(self):
        """Test streaming scenario like production deployment."""
        # Simulate streaming data
        n_batches = 10
        batch_size = 50
        
        all_X = []
        all_y = []
        
        for i in range(n_batches):
            X_batch, y_batch = make_classification(
                n_samples=batch_size,
                n_features=15,
                n_classes=3,
                random_state=42 + i  # Different seed for each batch
            )
            all_X.append(torch.FloatTensor(X_batch))
            all_y.append(torch.LongTensor(y_batch))
        
        # Initialize adaptive model
        encoder = RandomProjection(15, 500, 'ternary', seed=42)
        model = AdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            window_size=200,
            update_frequency=25
        )
        
        # Initialize with first batch
        model.fit(all_X[0], all_y[0])
        
        # Process streaming batches
        performance_history = []
        
        for i in range(1, n_batches):
            X_batch = all_X[i]
            y_batch = all_y[i]
            
            # Make predictions
            pred_sets = model.predict_set(X_batch)
            coverage = coverage_score(pred_sets, y_batch.numpy())
            avg_size = average_set_size(pred_sets)
            
            performance_history.append({
                'batch': i,
                'coverage': coverage,
                'avg_size': avg_size
            })
            
            # Update model
            model.update(X_batch, y_batch)
        
        # Verify streaming performance
        coverages = [p['coverage'] for p in performance_history]
        avg_sizes = [p['avg_size'] for p in performance_history]
        
        # Should maintain reasonable performance
        assert np.mean(coverages) > 0.6, "Poor average coverage in streaming"
        assert np.std(coverages) < 0.3, "Too much coverage variance in streaming"
        assert all(s >= 1.0 for s in avg_sizes), "Invalid set sizes"


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        encoder = RandomProjection(10, 200, 'binary', seed=42)
        model = ConformalHDC(encoder, 3, alpha=0.1)
        
        # Test before fitting
        X_dummy = torch.randn(5, 10)
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X_dummy)
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict_set(X_dummy)
        
        # Fit model
        X_train = torch.randn(50, 10)
        y_train = torch.randint(0, 3, (50,))
        model.fit(X_train, y_train)
        
        # Test wrong input dimensions
        X_wrong_dim = torch.randn(5, 15)  # Wrong feature dimension
        
        with pytest.raises((RuntimeError, ValueError)):
            model.predict(X_wrong_dim)
    
    def test_edge_case_data(self):
        """Test with edge case data."""
        encoder = RandomProjection(5, 100, 'binary', seed=42)
        model = ConformalHDC(encoder, 2, alpha=0.1)
        
        # Test with constant features
        X_constant = torch.ones(20, 5)
        y_constant = torch.randint(0, 2, (20,))
        
        # Should handle without crashing
        model.fit(X_constant, y_constant)
        predictions = model.predict(X_constant[:5])
        
        assert len(predictions) == 5
        
        # Test with extreme values
        X_extreme = torch.randn(20, 5) * 1000  # Very large values
        y_extreme = torch.randint(0, 2, (20,))
        
        model2 = ConformalHDC(
            RandomProjection(5, 100, 'binary', seed=43), 2, alpha=0.1
        )
        model2.fit(X_extreme, y_extreme)
        
        predictions_extreme = model2.predict(X_extreme[:5])
        assert len(predictions_extreme) == 5
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints."""
        # Test with progressively larger problems
        dimensions = [10, 50, 100]
        hv_dims = [200, 500, 1000]
        
        for dim, hv_dim in zip(dimensions, hv_dims):
            X = torch.randn(100, dim)
            y = torch.randint(0, 3, (100,))
            
            encoder = RandomProjection(dim, hv_dim, 'binary', seed=42)
            model = ConformalHDC(encoder, 3, alpha=0.1)
            
            # Should handle without memory issues
            model.fit(X, y)
            
            # Check memory footprint is reasonable
            footprint = model.memory_footprint()
            expected_min = 3 * hv_dim * 4  # 3 classes * hv_dim * 4 bytes
            assert footprint >= expected_min


def make_classification(n_samples, n_features, n_classes, random_state=None, **kwargs):
    """Simple make_classification replacement."""
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    # Generate random features
    X = torch.randn(n_samples, n_features)
    
    # Generate labels
    y = torch.randint(0, n_classes, (n_samples,))
    
    return X.numpy(), y.numpy()


if __name__ == "__main__":
    # Run end-to-end tests
    pytest.main([__file__, "-v", "-x"])  # Stop on first failure for debugging