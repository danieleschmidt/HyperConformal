"""
Tests for conformal prediction algorithms.
"""

import pytest
import torch
import numpy as np
from hyperconformal.conformal import (
    ClassificationConformalPredictor, 
    AdaptiveConformalPredictor,
    RegressionConformalPredictor
)


class TestClassificationConformalPredictor:
    """Test classification conformal predictor."""
    
    def test_aps_scoring(self):
        """Test Adaptive Prediction Sets scoring."""
        predictor = ClassificationConformalPredictor(alpha=0.1, score_type='aps')
        
        # Mock softmax predictions
        predictions = torch.tensor([
            [0.7, 0.2, 0.1],  # Confident prediction
            [0.4, 0.35, 0.25],  # Less confident
            [0.33, 0.33, 0.34]  # Very uncertain
        ])
        labels = torch.tensor([0, 1, 2])
        
        scores = predictor.compute_nonconformity_score(predictions, labels)
        
        assert len(scores) == 3
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)
        
        # More confident predictions should have lower scores
        assert scores[0] < scores[1] < scores[2]
    
    def test_margin_scoring(self):
        """Test margin-based scoring."""
        predictor = ClassificationConformalPredictor(alpha=0.1, score_type='margin')
        
        predictions = torch.tensor([
            [0.8, 0.15, 0.05],  # Large margin
            [0.5, 0.4, 0.1],    # Medium margin
            [0.4, 0.35, 0.25]   # Small margin
        ])
        labels = torch.tensor([0, 0, 0])
        
        scores = predictor.compute_nonconformity_score(predictions, labels)
        
        # Larger margins should give lower scores (better predictions)
        assert scores[0] < scores[1] < scores[2]
    
    def test_calibration_and_prediction(self):
        """Test calibration and prediction set generation."""
        predictor = ClassificationConformalPredictor(alpha=0.2, score_type='aps')
        
        # Generate calibration data
        n_cal = 100
        n_classes = 3
        
        # Random predictions and labels
        torch.manual_seed(42)
        cal_predictions = torch.softmax(torch.randn(n_cal, n_classes), dim=1)
        cal_labels = torch.randint(0, n_classes, (n_cal,))
        
        # Calibrate
        predictor.calibrate(cal_predictions, cal_labels)
        
        # Make predictions
        test_predictions = torch.softmax(torch.randn(20, n_classes), dim=1)
        pred_sets = predictor.predict_set(test_predictions)
        
        assert len(pred_sets) == 20
        for pred_set in pred_sets:
            assert len(pred_set) >= 1  # At least one class
            assert all(0 <= cls < n_classes for cls in pred_set)
    
    def test_coverage_guarantee(self):
        """Test that coverage is approximately correct."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        alpha = 0.1
        predictor = ClassificationConformalPredictor(alpha=alpha, score_type='aps')
        
        n_samples = 1000
        n_classes = 5
        
        # Generate well-calibrated predictions
        cal_predictions = torch.softmax(torch.randn(n_samples, n_classes), dim=1)
        cal_labels = torch.multinomial(cal_predictions, 1).squeeze()
        
        # Split for calibration and testing
        split = n_samples // 2
        predictor.calibrate(cal_predictions[:split], cal_labels[:split])
        
        # Test coverage
        test_predictions = cal_predictions[split:]
        test_labels = cal_labels[split:]
        pred_sets = predictor.predict_set(test_predictions)
        
        coverage = sum(
            1 for i, pred_set in enumerate(pred_sets)
            if test_labels[i].item() in pred_set
        ) / len(pred_sets)
        
        target_coverage = 1 - alpha
        # Coverage should be close to target (within reasonable margin)
        assert abs(coverage - target_coverage) < 0.1


class TestAdaptiveConformalPredictor:
    """Test adaptive conformal predictor."""
    
    def test_initialization(self):
        """Test adaptive predictor initialization."""
        predictor = AdaptiveConformalPredictor(
            alpha=0.1,
            window_size=100,
            update_frequency=10,
            score_type='aps'
        )
        
        assert predictor.window_size == 100
        assert predictor.update_frequency == 10
        assert len(predictor.scores_buffer) == 0
    
    def test_streaming_updates(self):
        """Test streaming updates."""
        predictor = AdaptiveConformalPredictor(
            alpha=0.1,
            window_size=50,
            update_frequency=10
        )
        
        # Generate streaming data
        torch.manual_seed(42)
        for _ in range(5):
            predictions = torch.softmax(torch.randn(10, 3), dim=1)
            labels = torch.randint(0, 3, (10,))
            
            predictor.update(predictions, labels)
        
        # Buffer should have data
        assert len(predictor.scores_buffer) > 0
        
        # Should be able to make predictions
        test_predictions = torch.softmax(torch.randn(5, 3), dim=1)
        pred_sets = predictor.predict_set(test_predictions)
        
        assert len(pred_sets) == 5
    
    def test_coverage_estimation(self):
        """Test current coverage estimation."""
        predictor = AdaptiveConformalPredictor(alpha=0.1, window_size=100)
        
        # Initially no estimate
        assert predictor.get_current_coverage_estimate() is None
        
        # Add some data
        torch.manual_seed(42)
        predictions = torch.softmax(torch.randn(50, 3), dim=1)
        labels = torch.randint(0, 3, (50,))
        predictor.update(predictions, labels)
        
        coverage_est = predictor.get_current_coverage_estimate()
        assert coverage_est is not None
        assert 0 <= coverage_est <= 1


class TestRegressionConformalPredictor:
    """Test regression conformal predictor."""
    
    def test_absolute_scoring(self):
        """Test absolute residual scoring."""
        predictor = RegressionConformalPredictor(alpha=0.1, score_type='absolute')
        
        predictions = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.1, 2.5, 2.8])
        
        scores = predictor.compute_nonconformity_score(predictions, labels)
        expected_scores = torch.abs(predictions - labels)
        
        assert torch.allclose(scores, expected_scores)
    
    def test_normalized_scoring(self):
        """Test normalized residual scoring."""
        predictor = RegressionConformalPredictor(alpha=0.1, score_type='normalized')
        
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = torch.tensor([1.1, 2.2, 2.9, 4.1, 5.2])
        
        scores = predictor.compute_nonconformity_score(predictions, labels)
        
        # Should be normalized by prediction std
        assert torch.all(scores >= 0)
        # Exact values depend on normalization
    
    def test_prediction_intervals(self):
        """Test prediction interval generation."""
        predictor = RegressionConformalPredictor(alpha=0.2, score_type='absolute')
        
        # Calibrate with some data
        cal_predictions = torch.randn(100)
        cal_labels = cal_predictions + 0.1 * torch.randn(100)  # Add noise
        
        predictor.calibrate(cal_predictions, cal_labels)
        
        # Generate intervals
        test_predictions = torch.tensor([1.0, 2.0, 3.0])
        intervals = predictor.predict_set(test_predictions)
        
        assert len(intervals) == 3
        for i, (lower, upper) in enumerate(intervals):
            assert lower < upper
            assert abs((lower + upper) / 2 - test_predictions[i].item()) < 1e-6
    
    def test_coverage_regression(self):
        """Test coverage for regression intervals."""
        torch.manual_seed(42)
        predictor = RegressionConformalPredictor(alpha=0.1)
        
        # Generate synthetic regression data
        n_samples = 500
        true_function = lambda x: x**2 + 0.5 * x
        
        X = torch.randn(n_samples)
        noise = 0.2 * torch.randn(n_samples)
        y_true = true_function(X) + noise
        
        # Use true function as predictor
        predictions = true_function(X)
        
        # Split for calibration and testing
        split = n_samples // 2
        predictor.calibrate(predictions[:split], y_true[:split])
        
        # Test intervals
        test_predictions = predictions[split:]
        test_labels = y_true[split:]
        intervals = predictor.predict_set(test_predictions)
        
        # Check coverage
        coverage = sum(
            1 for i, (lower, upper) in enumerate(intervals)
            if lower <= test_labels[i] <= upper
        ) / len(intervals)
        
        target_coverage = 1 - predictor.alpha
        assert abs(coverage - target_coverage) < 0.1