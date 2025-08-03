"""
Performance benchmarks for HyperConformal library.

These tests measure and validate performance characteristics
of different components under various conditions.
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List, Tuple

from hyperconformal import (
    RandomProjection, LevelHDC, ComplexHDC,
    ConformalHDC, AdaptiveConformalHDC,
    coverage_score, average_set_size
)


def make_classification(n_samples, n_features, n_classes, random_state=None):
    """Simple classification data generator."""
    if random_state is not None:
        torch.manual_seed(random_state)
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


@pytest.mark.benchmark
class TestEncoderPerformance:
    """Benchmark encoder performance."""
    
    @pytest.mark.parametrize("n_samples,n_features,hv_dim", [
        (100, 20, 500),
        (500, 50, 1000),
        (1000, 100, 2000),
    ])
    def test_random_projection_encoding_speed(self, n_samples, n_features, hv_dim):
        """Test RandomProjection encoding speed."""
        X, _ = make_classification(n_samples, n_features, 3, random_state=42)
        
        # Test different quantizations
        quantizations = ['binary', 'ternary', 'continuous']
        results = {}
        
        for quant in quantizations:
            encoder = RandomProjection(n_features, hv_dim, quant, seed=42)
            
            # Warmup
            _ = encoder.encode(X[:10])
            
            # Benchmark
            with Timer(f"RandomProjection-{quant}") as timer:
                encoded = encoder.encode(X)
            
            results[quant] = {
                'time': timer.elapsed,
                'throughput': n_samples / timer.elapsed,
                'output_shape': encoded.shape
            }
        
        # Validate results
        for quant, metrics in results.items():
            assert metrics['time'] < 5.0, f"{quant} encoding too slow: {metrics['time']:.2f}s"
            assert metrics['throughput'] > 50, f"{quant} throughput too low: {metrics['throughput']:.1f} samples/s"
            
        # Binary should be fastest
        assert results['binary']['time'] <= results['continuous']['time']
        
    @pytest.mark.parametrize("n_samples,levels,hv_dim", [
        (200, 20, 400),
        (500, 50, 800),
        (1000, 100, 1200),
    ])
    def test_level_hdc_encoding_speed(self, n_samples, levels, hv_dim):
        """Test LevelHDC encoding speed."""
        n_features = 15
        X, _ = make_classification(n_samples, n_features, 3, random_state=42)
        
        encoder = LevelHDC(n_features, hv_dim, levels=levels, seed=42)
        
        # Warmup
        _ = encoder.encode(X[:10])
        
        # Benchmark
        with Timer("LevelHDC") as timer:
            encoded = encoder.encode(X)
        
        throughput = n_samples / timer.elapsed
        
        # Validate performance
        assert timer.elapsed < 10.0, f"LevelHDC too slow: {timer.elapsed:.2f}s"
        assert throughput > 20, f"LevelHDC throughput too low: {throughput:.1f} samples/s"
        assert encoded.shape == (n_samples, hv_dim), "Wrong output shape"


@pytest.mark.benchmark
class TestConformalPerformance:
    """Benchmark conformal prediction performance."""
    
    @pytest.mark.parametrize("n_samples,n_features,n_classes", [
        (500, 20, 3),
        (1000, 50, 5),
        (2000, 100, 10),
    ])
    def test_conformal_hdc_training_speed(self, n_samples, n_features, n_classes):
        """Test ConformalHDC training performance."""
        X, y = make_classification(n_samples, n_features, n_classes, random_state=42)
        
        encoder = RandomProjection(n_features, 1000, 'binary', seed=42)
        model = ConformalHDC(encoder, n_classes, alpha=0.1)
        
        # Benchmark training
        with Timer("Training") as timer:
            model.fit(X, y)
        
        training_speed = n_samples / timer.elapsed
        
        # Validate performance
        assert timer.elapsed < 30.0, f"Training too slow: {timer.elapsed:.2f}s"
        assert training_speed > 50, f"Training speed too low: {training_speed:.1f} samples/s"
        assert model.is_fitted, "Model should be fitted"
    
    @pytest.mark.parametrize("n_test,n_features,n_classes", [
        (100, 20, 3),
        (500, 50, 5),
        (1000, 100, 10),
    ])
    def test_prediction_speed(self, n_test, n_features, n_classes):
        """Test prediction speed."""
        # Generate training and test data
        X_train, y_train = make_classification(200, n_features, n_classes, random_state=42)
        X_test, _ = make_classification(n_test, n_features, n_classes, random_state=123)
        
        # Train model
        encoder = RandomProjection(n_features, 800, 'binary', seed=42)
        model = ConformalHDC(encoder, n_classes, alpha=0.1)
        model.fit(X_train, y_train)
        
        # Benchmark predictions
        with Timer("Predictions") as timer:
            predictions = model.predict(X_test)
        
        with Timer("Prediction Sets") as timer2:
            pred_sets = model.predict_set(X_test)
        
        pred_speed = n_test / timer.elapsed
        pred_set_speed = n_test / timer2.elapsed
        
        # Validate performance
        assert timer.elapsed < 5.0, f"Predictions too slow: {timer.elapsed:.2f}s"
        assert timer2.elapsed < 10.0, f"Prediction sets too slow: {timer2.elapsed:.2f}s"
        assert pred_speed > 200, f"Prediction speed too low: {pred_speed:.1f} samples/s"
        assert pred_set_speed > 100, f"Prediction set speed too low: {pred_set_speed:.1f} samples/s"
        
        # Verify correctness
        assert len(predictions) == n_test
        assert len(pred_sets) == n_test


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability with increasing problem sizes."""
    
    def test_memory_scalability(self):
        """Test memory usage scaling."""
        dimensions = [500, 1000, 2000, 4000]
        results = []
        
        for dim in dimensions:
            encoder = RandomProjection(20, dim, 'binary', seed=42)
            model = ConformalHDC(encoder, 3, alpha=0.1)
            
            # Generate minimal training data
            X, y = make_classification(100, 20, 3, random_state=42)
            model.fit(X, y)
            
            footprint = model.memory_footprint()
            results.append({
                'hv_dim': dim,
                'memory': footprint,
                'memory_per_dim': footprint / dim
            })
        
        # Validate linear scaling
        for i in range(1, len(results)):
            prev_mem = results[i-1]['memory']
            curr_mem = results[i]['memory']
            scaling_factor = curr_mem / prev_mem
            dim_factor = results[i]['hv_dim'] / results[i-1]['hv_dim']
            
            # Memory should scale roughly linearly with dimension
            assert 0.8 * dim_factor <= scaling_factor <= 1.5 * dim_factor, \
                f"Memory scaling not linear: {scaling_factor:.2f} vs expected ~{dim_factor:.2f}"
    
    def test_throughput_scaling(self):
        """Test throughput scaling with batch size."""
        batch_sizes = [10, 50, 100, 500, 1000]
        
        # Train a model
        encoder = RandomProjection(30, 1000, 'binary', seed=42)
        model = ConformalHDC(encoder, 4, alpha=0.1)
        
        X_train, y_train = make_classification(200, 30, 4, random_state=42)
        model.fit(X_train, y_train)
        
        throughputs = []
        
        for batch_size in batch_sizes:
            X_test, _ = make_classification(batch_size, 30, 4, random_state=123)
            
            # Warmup
            _ = model.predict(X_test[:min(10, batch_size)])
            
            # Benchmark
            with Timer() as timer:
                predictions = model.predict(X_test)
            
            throughput = batch_size / timer.elapsed
            throughputs.append(throughput)
            
            assert len(predictions) == batch_size
        
        # Throughput should generally increase with batch size (up to a point)
        # At least the larger batches should be more efficient than smallest
        assert throughputs[-1] > throughputs[0], "Throughput should improve with larger batches"


@pytest.mark.benchmark
class TestComparativeBenchmarks:
    """Compare performance across different configurations."""
    
    def test_encoder_comparison(self):
        """Compare different encoder types."""
        n_samples, n_features = 500, 25
        X, y = make_classification(n_samples, n_features, 3, random_state=42)
        
        encoders = {
            'RandomProjection-Binary': RandomProjection(n_features, 1000, 'binary', seed=42),
            'RandomProjection-Ternary': RandomProjection(n_features, 1000, 'ternary', seed=42),
            'LevelHDC': LevelHDC(n_features, 1000, levels=50, seed=42),
        }
        
        results = {}
        
        for name, encoder in encoders.items():
            # Benchmark encoding
            with Timer() as encode_timer:
                encoded = encoder.encode(X)
            
            # Benchmark training
            model = ConformalHDC(encoder, 3, alpha=0.1)
            with Timer() as train_timer:
                model.fit(X, y)
            
            # Benchmark prediction
            with Timer() as pred_timer:
                predictions = model.predict(X[:100])
            
            results[name] = {
                'encode_time': encode_timer.elapsed,
                'train_time': train_timer.elapsed,
                'pred_time': pred_timer.elapsed,
                'encode_throughput': n_samples / encode_timer.elapsed,
                'pred_throughput': 100 / pred_timer.elapsed,
                'total_time': encode_timer.elapsed + train_timer.elapsed + pred_timer.elapsed
            }
        
        # Validate all perform reasonably
        for name, metrics in results.items():
            assert metrics['total_time'] < 20.0, f"{name} total time too high"
            assert metrics['encode_throughput'] > 50, f"{name} encoding too slow"
            assert metrics['pred_throughput'] > 100, f"{name} prediction too slow"
        
        # Binary should generally be fastest
        binary_total = results['RandomProjection-Binary']['total_time']
        ternary_total = results['RandomProjection-Ternary']['total_time']
        
        assert binary_total <= ternary_total * 1.2, "Binary should be comparable or faster than ternary"
    
    def test_score_type_comparison(self):
        """Compare different conformal score types."""
        X, y = make_classification(300, 20, 4, random_state=42)
        encoder = RandomProjection(20, 800, 'binary', seed=42)
        
        score_types = ['aps', 'margin', 'inverse_softmax']
        results = {}
        
        for score_type in score_types:
            model = ConformalHDC(encoder, 4, alpha=0.1, score_type=score_type)
            
            # Benchmark training
            with Timer() as train_timer:
                model.fit(X, y)
            
            # Benchmark prediction sets
            with Timer() as pred_timer:
                pred_sets = model.predict_set(X[:100])
            
            coverage = coverage_score(pred_sets, y[:100].numpy())
            avg_size = average_set_size(pred_sets)
            
            results[score_type] = {
                'train_time': train_timer.elapsed,
                'pred_time': pred_timer.elapsed,
                'coverage': coverage,
                'avg_size': avg_size
            }
        
        # All should perform reasonably
        for score_type, metrics in results.items():
            assert metrics['train_time'] < 10.0, f"{score_type} training too slow"
            assert metrics['pred_time'] < 2.0, f"{score_type} prediction too slow"
            assert metrics['coverage'] > 0.6, f"{score_type} coverage too low"
            assert 1.0 <= metrics['avg_size'] <= 3.0, f"{score_type} set sizes unreasonable"


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-m", "benchmark"])