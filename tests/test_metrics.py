"""
Tests for evaluation metrics.
"""

import pytest
import torch
import numpy as np
from hyperconformal.metrics import (
    coverage_score, average_set_size, conditional_coverage,
    efficiency_score, coverage_width_criterion, calibration_score,
    conformal_prediction_metrics, hdc_similarity_stats
)


class TestBasicMetrics:
    """Test basic evaluation metrics."""
    
    def test_coverage_score_perfect(self):
        """Test perfect coverage score."""
        prediction_sets = [[0], [1], [2], [0]]
        true_labels = [0, 1, 2, 0]
        
        score = coverage_score(prediction_sets, true_labels)
        assert score == 1.0
    
    def test_coverage_score_partial(self):
        """Test partial coverage score."""
        prediction_sets = [[0, 1], [1], [2], [1]]  # Last one wrong
        true_labels = [0, 1, 2, 0]
        
        score = coverage_score(prediction_sets, true_labels)
        assert score == 0.75
    
    def test_average_set_size(self):
        """Test average set size computation."""
        prediction_sets = [[0], [1, 2], [0, 1, 2], [1]]
        
        avg_size = average_set_size(prediction_sets)
        expected = (1 + 2 + 3 + 1) / 4
        assert avg_size == expected
    
    def test_average_set_size_empty(self):
        """Test average set size with empty input."""
        prediction_sets = []
        
        avg_size = average_set_size(prediction_sets)
        assert avg_size == 0.0
    
    def test_efficiency_score(self):
        """Test efficiency score computation."""
        prediction_sets = [[0], [1], [0, 1]]  # Average size = 4/3
        num_classes = 3
        
        efficiency = efficiency_score(prediction_sets, num_classes)
        expected = 1 - (4/3) / 3  # 1 - normalized average size
        assert abs(efficiency - expected) < 1e-6
    
    def test_efficiency_perfect(self):
        """Test perfect efficiency (all singletons)."""
        prediction_sets = [[0], [1], [2]]
        num_classes = 3
        
        efficiency = efficiency_score(prediction_sets, num_classes)
        expected = 1 - 1/3  # 1 - 1/num_classes
        assert abs(efficiency - expected) < 1e-6


class TestConditionalCoverage:
    """Test conditional coverage metrics."""
    
    def test_conditional_coverage_by_class(self):
        """Test conditional coverage stratified by class."""
        prediction_sets = [
            [0],      # Class 0, correct
            [0, 1],   # Class 0, correct
            [1],      # Class 1, correct
            [0],      # Class 1, incorrect
            [2],      # Class 2, correct
        ]
        true_labels = [0, 0, 1, 1, 2]
        
        cond_cov = conditional_coverage(prediction_sets, true_labels, stratify_by='class')
        
        assert cond_cov[0] == 1.0   # Class 0: 2/2 correct
        assert cond_cov[1] == 0.5   # Class 1: 1/2 correct
        assert cond_cov[2] == 1.0   # Class 2: 1/1 correct
    
    def test_conditional_coverage_by_set_size(self):
        """Test conditional coverage stratified by set size."""
        prediction_sets = [
            [0],      # Size 1, correct
            [1],      # Size 1, incorrect
            [0, 1],   # Size 2, correct
            [1, 2],   # Size 2, correct
        ]
        true_labels = [0, 0, 0, 2]
        
        cond_cov = conditional_coverage(prediction_sets, true_labels, stratify_by='set_size')
        
        assert cond_cov['size_1'] == 0.5  # Size 1: 1/2 correct
        assert cond_cov['size_2'] == 1.0  # Size 2: 2/2 correct
    
    def test_conditional_coverage_by_confidence(self):
        """Test conditional coverage stratified by confidence quartiles."""
        # Create data with varying set sizes (inverse confidence)
        prediction_sets = [
            [0] for _ in range(4)  # High confidence (size 1)
        ] + [
            [0, 1] for _ in range(4)  # Medium confidence (size 2)
        ] + [
            [0, 1, 2] for _ in range(4)  # Low confidence (size 3)
        ]
        
        # All correct predictions
        true_labels = [0] * 12
        
        cond_cov = conditional_coverage(prediction_sets, true_labels, stratify_by='confidence')
        
        # All quartiles should have perfect coverage
        for quartile in ['quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']:
            if quartile in cond_cov:
                assert cond_cov[quartile] == 1.0


class TestAdvancedMetrics:
    """Test advanced evaluation metrics."""
    
    def test_coverage_width_criterion(self):
        """Test Coverage-Width Criterion."""
        # Perfect coverage, small sets
        prediction_sets = [[0], [1], [2]]
        true_labels = [0, 1, 2]
        
        cwc = coverage_width_criterion(prediction_sets, true_labels, target_coverage=0.9)
        expected = 1.0  # Average width, no penalty
        assert abs(cwc - expected) < 1e-6
        
        # Undercoverage should add penalty
        prediction_sets = [[1], [0], [1]]  # Wrong predictions
        true_labels = [0, 1, 2]
        
        cwc_penalty = coverage_width_criterion(prediction_sets, true_labels, target_coverage=0.9)
        assert cwc_penalty > 1000  # Should have large penalty
    
    def test_calibration_score(self):
        """Test calibration score (ECE)."""
        # Perfect calibration: larger sets for harder examples
        prediction_sets = [
            [0] for _ in range(10)  # Easy examples
        ] + [
            [0, 1, 2] for _ in range(10)  # Hard examples
        ]
        
        # Easy examples all correct, hard examples have some errors
        true_labels = [0] * 10 + [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
        
        ece = calibration_score(prediction_sets, true_labels, num_bins=2)
        
        # Should be a reasonable calibration score
        assert 0 <= ece <= 1
    
    def test_conformal_prediction_metrics_comprehensive(self):
        """Test comprehensive metrics computation."""
        prediction_sets = [
            [0], [1], [0, 1], [2], [1, 2]
        ]
        true_labels = [0, 1, 0, 2, 2]  # All correct
        num_classes = 3
        
        metrics = conformal_prediction_metrics(
            prediction_sets, true_labels, num_classes, target_coverage=0.9
        )
        
        # Check all expected metrics are present
        expected_keys = [
            'coverage', 'average_set_size', 'efficiency', 
            'coverage_gap', 'cwc', 'calibration_error'
        ]
        for key in expected_keys:
            assert key in metrics
        
        # Check coverage is perfect
        assert metrics['coverage'] == 1.0
        assert metrics['coverage_gap'] == -0.1  # Better than target
        
        # Check class-specific coverage
        assert metrics['coverage_class_0'] == 1.0
        assert metrics['coverage_class_1'] == 1.0
        assert metrics['coverage_class_2'] == 1.0


class TestHDCMetrics:
    """Test HDC-specific metrics."""
    
    def test_hdc_similarity_stats(self):
        """Test HDC similarity statistics."""
        # Mock similarity scores
        similarities = torch.tensor([0.8, 0.6, 0.9, 0.3, 0.7])
        true_labels = torch.tensor([0, 1, 0, 1, 0])
        predicted_labels = torch.tensor([0, 0, 0, 1, 1])  # Some correct, some wrong
        
        stats = hdc_similarity_stats(similarities, true_labels, predicted_labels)
        
        # Check all expected statistics
        assert 'mean_similarity' in stats
        assert 'std_similarity' in stats
        assert 'mean_similarity_correct' in stats
        assert 'mean_similarity_incorrect' in stats
        assert 'similarity_separation' in stats
        
        # Check that correct predictions have higher similarity on average
        correct_mask = (true_labels == predicted_labels).cpu().numpy()
        expected_correct_mean = similarities[correct_mask].mean().item()
        expected_incorrect_mean = similarities[~correct_mask].mean().item()
        
        assert abs(stats['mean_similarity_correct'] - expected_correct_mean) < 1e-6
        assert abs(stats['mean_similarity_incorrect'] - expected_incorrect_mean) < 1e-6
        
        # Separation should be positive (correct > incorrect similarity)
        assert stats['similarity_separation'] > 0
    
    def test_hdc_similarity_stats_all_correct(self):
        """Test similarity stats when all predictions are correct."""
        similarities = torch.tensor([0.8, 0.9, 0.7])
        true_labels = torch.tensor([0, 1, 2])
        predicted_labels = torch.tensor([0, 1, 2])  # All correct
        
        stats = hdc_similarity_stats(similarities, true_labels, predicted_labels)
        
        # All predictions correct
        assert stats['mean_similarity_correct'] == stats['mean_similarity']
        assert stats['mean_similarity_incorrect'] == 0.0
        assert stats['similarity_separation'] == 0.0
    
    def test_hdc_similarity_stats_all_incorrect(self):
        """Test similarity stats when all predictions are incorrect."""
        similarities = torch.tensor([0.3, 0.2, 0.4])
        true_labels = torch.tensor([0, 1, 2])
        predicted_labels = torch.tensor([1, 2, 0])  # All incorrect
        
        stats = hdc_similarity_stats(similarities, true_labels, predicted_labels)
        
        # All predictions incorrect
        assert stats['mean_similarity_incorrect'] == stats['mean_similarity']
        assert stats['mean_similarity_correct'] == 0.0
        assert stats['similarity_separation'] == 0.0


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""
    
    def test_empty_inputs(self):
        """Test metrics with empty inputs."""
        empty_sets = []
        empty_labels = []
        
        # Most metrics should handle empty inputs gracefully
        assert coverage_score(empty_sets, empty_labels) == 0.0
        assert average_set_size(empty_sets) == 0.0
    
    def test_single_example(self):
        """Test metrics with single example."""
        prediction_sets = [[0, 1]]
        true_labels = [0]
        
        coverage = coverage_score(prediction_sets, true_labels)
        assert coverage == 1.0
        
        avg_size = average_set_size(prediction_sets)
        assert avg_size == 2.0
    
    def test_mixed_input_types(self):
        """Test metrics with mixed input types."""
        prediction_sets = [[0], [1]]
        
        # Test with list labels
        list_labels = [0, 1]
        coverage_list = coverage_score(prediction_sets, list_labels)
        
        # Test with tensor labels  
        tensor_labels = torch.tensor([0, 1])
        coverage_tensor = coverage_score(prediction_sets, tensor_labels)
        
        # Test with numpy labels
        numpy_labels = np.array([0, 1])
        coverage_numpy = coverage_score(prediction_sets, numpy_labels)
        
        # All should give same result
        assert coverage_list == coverage_tensor == coverage_numpy == 1.0