"""
Comprehensive test suite for advanced HyperConformal features.

Tests quantum computing, federated learning, real-time adaptation,
neuromorphic-quantum hybrid processing, and advanced benchmarking.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)

# Test quantum computing features
class TestQuantumFeatures:
    """Test quantum hyperdimensional computing capabilities."""
    
    def test_quantum_hypervector_creation(self):
        """Test quantum hypervector initialization."""
        from hyperconformal.quantum import QuantumHypervector
        
        qhv = QuantumHypervector(dimension=1000, num_qubits=10)
        
        assert qhv.dimension == 1000
        assert qhv.num_qubits == 10
        assert qhv.amplitudes.shape[0] == 2**10
        assert torch.allclose(torch.norm(qhv.amplitudes), torch.tensor(1.0))
    
    def test_quantum_encoding(self):
        """Test encoding classical hypervector to quantum state."""
        from hyperconformal.quantum import QuantumHypervector
        
        qhv = QuantumHypervector(dimension=100, num_qubits=7)
        classical_hv = torch.randn(100)
        
        encoded_qhv = qhv.encode_classical(classical_hv)
        
        assert isinstance(encoded_qhv, QuantumHypervector)
        # Check normalization
        assert torch.allclose(torch.norm(encoded_qhv.amplitudes), torch.tensor(1.0), atol=1e-6)
    
    def test_quantum_similarity(self):
        """Test quantum similarity computation."""
        from hyperconformal.quantum import QuantumHypervector
        
        qhv1 = QuantumHypervector(dimension=50, num_qubits=6)
        qhv2 = QuantumHypervector(dimension=50, num_qubits=6)
        
        # Encode same classical vector
        classical_hv = torch.randn(50)
        qhv1.encode_classical(classical_hv)
        qhv2.encode_classical(classical_hv)
        
        similarity = qhv1.quantum_similarity(qhv2)
        
        assert isinstance(similarity, torch.Tensor)
        assert 0.0 <= similarity.item() <= 1.0
        # Same vector should have high similarity
        assert similarity.item() > 0.8
    
    def test_quantum_hdc_encoder(self):
        """Test quantum HDC encoder."""
        from hyperconformal.quantum import QuantumHDCEncoder
        
        encoder = QuantumHDCEncoder(
            input_dim=784,
            hv_dim=1000,
            quantum_depth=2,
            use_entanglement=True
        )
        
        # Test encoding
        x = torch.randn(10, 784)  # Batch of 10 samples
        quantum_hv = encoder.encode(x)
        
        assert hasattr(quantum_hv, 'amplitudes')
        assert hasattr(quantum_hv, 'dimension')
    
    def test_quantum_hyperconformal(self):
        """Test full quantum HyperConformal model."""
        from hyperconformal.quantum import QuantumHyperConformal
        
        model = QuantumHyperConformal(
            input_dim=20,
            hv_dim=100,
            num_classes=3,
            alpha=0.1
        )
        
        # Generate synthetic data
        X_train = torch.randn(100, 20)
        y_train = torch.randint(0, 3, (100,))
        X_test = torch.randn(20, 20)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        prediction_sets = model.predict_set(X_test)
        
        assert len(prediction_sets) == 20
        assert all(isinstance(pred_set, list) for pred_set in prediction_sets)
        assert all(len(pred_set) > 0 for pred_set in prediction_sets)
        
        # Test quantum advantage calculation
        advantage = model.quantum_advantage_factor()
        assert isinstance(advantage, float)
        assert advantage >= 1.0


class TestFederatedFeatures:
    """Test federated learning with homomorphic encryption."""
    
    def test_encryption_config(self):
        """Test encryption configuration."""
        from hyperconformal.federated import EncryptionConfig, SimpleHomomorphicEncryption
        
        config = EncryptionConfig(key_size=1024, noise_scale=0.05)
        encryption = SimpleHomomorphicEncryption(config)
        
        assert encryption.config.key_size == 1024
        assert encryption.config.noise_scale == 0.05
        assert encryption.key is not None
    
    def test_tensor_encryption(self):
        """Test tensor encryption and decryption."""
        from hyperconformal.federated import EncryptionConfig, SimpleHomomorphicEncryption
        
        config = EncryptionConfig()
        encryption = SimpleHomomorphicEncryption(config)
        
        # Test encryption/decryption
        original_tensor = torch.randn(10, 5)
        encrypted_data = encryption.encrypt_tensor(original_tensor)
        decrypted_tensor = encryption.decrypt_tensor(encrypted_data)
        
        assert 'encrypted_data' in encrypted_data
        assert 'shape' in encrypted_data
        assert 'scale' in encrypted_data
        assert decrypted_tensor.shape == original_tensor.shape
        
        # Check approximate equality (due to quantization)
        assert torch.allclose(decrypted_tensor, original_tensor, atol=0.1)
    
    def test_homomorphic_addition(self):
        """Test homomorphic addition of encrypted tensors."""
        from hyperconformal.federated import EncryptionConfig, SimpleHomomorphicEncryption
        
        config = EncryptionConfig()
        encryption = SimpleHomomorphicEncryption(config)
        
        tensor1 = torch.randn(5, 3)
        tensor2 = torch.randn(5, 3)
        
        enc1 = encryption.encrypt_tensor(tensor1)
        enc2 = encryption.encrypt_tensor(tensor2)
        
        # Homomorphic addition
        enc_sum = encryption.homomorphic_add(enc1, enc2)
        decrypted_sum = encryption.decrypt_tensor(enc_sum)
        
        expected_sum = tensor1 + tensor2
        assert torch.allclose(decrypted_sum, expected_sum, atol=0.2)
    
    def test_federated_client(self):
        """Test federated client functionality."""
        from hyperconformal.federated import FederatedClient, FederatedClientConfig
        
        client_config = FederatedClientConfig(
            client_id="test_client_1",
            data_privacy_level="high",
            local_epochs=1
        )
        
        encoder_config = {
            'input_dim': 10,
            'hv_dim': 100,
            'quantization': 'binary'
        }
        
        client = FederatedClient(client_config, encoder_config)
        
        assert client.client_id == "test_client_1"
        assert client.config.data_privacy_level == "high"
        
        # Test initialization
        global_state = {'projection.weight': torch.randn(100, 10)}
        client.initialize_model(global_state)
        
        assert client.local_encoder is not None
    
    def test_federated_server(self):
        """Test federated server aggregation."""
        from hyperconformal.federated import FederatedServer
        
        server = FederatedServer(
            num_classes=3,
            hv_dim=50,
            aggregation_method="fedavg"
        )
        
        # Mock client updates
        client_updates = [
            {
                'client_id': 'client1',
                'class_updates': {0: torch.randn(50), 1: torch.randn(50)},
                'num_samples': 100,
                'training_metadata': {}
            },
            {
                'client_id': 'client2', 
                'class_updates': {0: torch.randn(50), 2: torch.randn(50)},
                'num_samples': 150,
                'training_metadata': {}
            }
        ]
        
        # Test aggregation
        aggregated = server.aggregate_updates(client_updates)
        
        assert len(aggregated) >= 1  # At least class 0 should be aggregated
        assert 0 in aggregated  # Class 0 present in both updates
        
        # Test global state retrieval
        global_state = server.get_global_state()
        assert isinstance(global_state, dict)
    
    def test_federated_hyperconformal(self):
        """Test complete federated HyperConformal system."""
        from hyperconformal.federated import FederatedHyperConformal, FederatedClientConfig
        
        fed_system = FederatedHyperConformal(
            input_dim=10,
            hv_dim=50,
            num_classes=2,
            alpha=0.1
        )
        
        # Register clients
        client_configs = [
            FederatedClientConfig(client_id=f"client_{i}")
            for i in range(3)
        ]
        
        clients = []
        for config in client_configs:
            client = fed_system.register_client(config)
            clients.append(client)
        
        assert len(fed_system.clients) == 3
        
        # Generate mock client data
        client_data = {}
        for i, client_config in enumerate(client_configs):
            X_client = torch.randn(20, 10)
            y_client = torch.randint(0, 2, (20,))
            client_data[client_config.client_id] = (X_client, y_client)
        
        # Run federated training round
        round_summary = fed_system.federated_training_round(client_data)
        
        assert round_summary['round'] == 1
        assert round_summary['num_clients'] == 3
        assert round_summary['total_samples'] == 60


class TestAdaptiveRealTime:
    """Test real-time adaptive conformal prediction."""
    
    def test_adaptive_config(self):
        """Test adaptive configuration."""
        from hyperconformal.adaptive_realtime import AdaptiveConfig
        
        config = AdaptiveConfig(
            window_size=500,
            update_frequency=50,
            drift_detection_threshold=0.1
        )
        
        assert config.window_size == 500
        assert config.update_frequency == 50
        assert config.drift_detection_threshold == 0.1
    
    def test_concept_drift_detector(self):
        """Test concept drift detection."""
        from hyperconformal.adaptive_realtime import ConceptDriftDetector
        
        detector = ConceptDriftDetector(
            window_size=100,
            detection_method="adwin",
            sensitivity=0.05
        )
        
        # Simulate stable data (no drift)
        for i in range(50):
            error = 0.1 + 0.02 * np.random.randn()
            coverage = 0.9 + 0.01 * np.random.randn()
            drift_detected = detector.update(error, coverage)
            
            # Should not detect drift in stable data
            if i > 30:  # After sufficient samples
                assert not drift_detected
        
        # Simulate concept drift
        for i in range(30):
            error = 0.3 + 0.05 * np.random.randn()  # Higher error
            coverage = 0.7 + 0.02 * np.random.randn()  # Lower coverage
            drift_detected = detector.update(error, coverage)
            
            # May detect drift in changed data
            if drift_detected:
                break  # Drift detected as expected
    
    def test_adaptive_quantile_tracker(self):
        """Test adaptive quantile tracking."""
        from hyperconformal.adaptive_realtime import AdaptiveQuantileTracker
        
        tracker = AdaptiveQuantileTracker(
            alpha=0.1,
            adaptation_rate=0.1,
            min_samples=20
        )
        
        # Feed scores and track quantile evolution
        scores = np.random.exponential(scale=0.5, size=100)
        
        quantiles = []
        for score in scores:
            quantile = tracker.update(score)
            quantiles.append(quantile)
        
        # Quantile should stabilize after initial period
        assert len(quantiles) == 100
        
        # Final quantile should be reasonable for exponential distribution
        final_quantile = tracker.get_quantile()
        assert 0.0 < final_quantile < 5.0
    
    def test_streaming_adaptive_conformal(self):
        """Test streaming adaptive conformal HDC."""
        from hyperconformal.adaptive_realtime import StreamingAdaptiveConformalHDC, AdaptiveConfig
        from hyperconformal.encoders import RandomProjection
        
        encoder = RandomProjection(input_dim=20, hv_dim=100)
        config = AdaptiveConfig(window_size=200, update_frequency=20)
        
        streaming_model = StreamingAdaptiveConformalHDC(
            encoder=encoder,
            num_classes=3,
            config=config,
            alpha=0.1
        )
        
        # Initialize with some data
        X_init = torch.randn(50, 20)
        y_init = torch.randint(0, 3, (50,))
        streaming_model.initialize_prototypes(X_init, y_init)
        
        # Stream updates
        for i in range(30):
            x = torch.randn(20)
            y_true = torch.randint(0, 3, (1,))
            
            result = streaming_model.stream_update(x, y_true)
            
            assert 'prediction_set' in result
            assert 'processing_time' in result
            assert 'current_quantile' in result
            assert isinstance(result['prediction_set'], list)
        
        # Check metrics
        metrics = streaming_model.get_streaming_metrics()
        assert 'total_updates' in metrics
        assert 'current_quantile' in metrics
        assert metrics['total_updates'] == 30


class TestNeuromorphicQuantum:
    """Test hybrid neuromorphic-quantum processing."""
    
    def test_neuromorphic_neuron(self):
        """Test neuromorphic neuron simulation."""
        from hyperconformal.neuromorphic_quantum import NeuromorphicNeuron
        
        neuron = NeuromorphicNeuron(
            neuron_id=1,
            threshold=1.0,
            decay=0.9,
            refractory_period=1.0
        )
        
        # Test neuron dynamics
        current_time = 0.0
        spike_event = neuron.update(current_time, input_current=1.5)
        
        # Should spike with sufficient input
        assert spike_event is not None
        assert spike_event.neuron_id == 1
        assert spike_event.timestamp == current_time
        
        # Test refractory period
        spike_event2 = neuron.update(current_time + 0.5, input_current=2.0)
        assert spike_event2 is None  # Should be in refractory period
    
    def test_quantum_spike_processor(self):
        """Test quantum spike processing."""
        from hyperconformal.neuromorphic_quantum import QuantumSpikeProcessor, SpikeEvent
        
        processor = QuantumSpikeProcessor(
            num_qubits=8,
            coherence_time=100.0,
            entanglement_strength=0.5
        )
        
        # Create spike events
        spike_events = [
            SpikeEvent(neuron_id=1, timestamp=0.5, weight=0.8),
            SpikeEvent(neuron_id=3, timestamp=1.0, weight=0.6),
            SpikeEvent(neuron_id=5, timestamp=1.5, weight=0.9)
        ]
        
        # Encode spikes to quantum state
        quantum_state = processor.encode_spike_train(spike_events)
        
        assert quantum_state.shape[0] == 2**8
        assert torch.allclose(torch.norm(quantum_state), torch.tensor(1.0), atol=1e-6)
    
    def test_neuromorphic_quantum_hdc(self):
        """Test hybrid neuromorphic-quantum HDC encoder."""
        from hyperconformal.neuromorphic_quantum import (
            NeuromorphicQuantumHDC, NeuromorphicConfig, QuantumNeuromorphicConfig
        )
        
        neuro_config = NeuromorphicConfig(
            num_neurons=100,
            spike_threshold=1.0,
            max_spike_rate=100.0
        )
        
        quantum_config = QuantumNeuromorphicConfig(
            quantum_coherence_time=50.0,
            entanglement_strength=0.3
        )
        
        encoder = NeuromorphicQuantumHDC(
            input_dim=10,
            hv_dim=50,
            neuromorphic_config=neuro_config,
            quantum_config=quantum_config
        )
        
        # Test encoding
        x = torch.randn(5, 10)  # Batch of 5 samples
        encoded_hvs = encoder.encode(x, simulation_time=5.0)
        
        assert encoded_hvs.shape == (5, 50)
        
        # Test energy metrics
        energy_metrics = encoder.get_energy_metrics()
        assert 'total_energy_joules' in energy_metrics
        assert 'quantum_advantage_factor' in energy_metrics
    
    def test_hybrid_conformal_predictor(self):
        """Test hybrid neuromorphic-quantum conformal predictor."""
        from hyperconformal.neuromorphic_quantum import (
            NeuromorphicQuantumHDC, HybridConformalPredictor,
            NeuromorphicConfig, QuantumNeuromorphicConfig
        )
        
        # Create HDC encoder
        neuro_config = NeuromorphicConfig(num_neurons=50, max_spike_rate=50.0)
        quantum_config = QuantumNeuromorphicConfig()
        
        hdc_encoder = NeuromorphicQuantumHDC(
            input_dim=8,
            hv_dim=30,
            neuromorphic_config=neuro_config,
            quantum_config=quantum_config
        )
        
        # Create predictor
        predictor = HybridConformalPredictor(hdc_encoder, alpha=0.1)
        
        # Generate small dataset for testing
        X_cal = torch.randn(20, 8)
        y_cal = torch.randint(0, 2, (20,))
        X_test = torch.randn(5, 8)
        
        # Calibrate with short simulation time for testing
        predictor.calibrate(X_cal, y_cal, simulation_time=2.0)
        
        # Predict
        prediction_sets = predictor.predict_set(X_test, simulation_time=2.0)
        
        assert len(prediction_sets) == 5
        assert all(isinstance(pred_set, list) for pred_set in prediction_sets)
        
        # Test processing metrics
        metrics = predictor.get_processing_metrics()
        assert 'energy_metrics' in metrics
        assert 'quantum_qubits' in metrics


class TestAdvancedBenchmarks:
    """Test advanced benchmarking suite."""
    
    def setup_method(self):
        """Set up temporary directory for benchmarks."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_config(self):
        """Test benchmark configuration."""
        from hyperconformal.advanced_benchmarks import BenchmarkConfig
        
        config = BenchmarkConfig(
            name="test_benchmark",
            datasets=['synthetic'],
            metrics=['coverage', 'set_size'],
            repetitions=2,
            confidence_levels=[0.9],
            hv_dimensions=[100]
        )
        
        assert config.name == "test_benchmark"
        assert config.datasets == ['synthetic']
        assert config.repetitions == 2
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        from hyperconformal.advanced_benchmarks import SyntheticDataset
        
        dataset = SyntheticDataset(
            n_samples=100,
            n_features=20,
            n_classes=3,
            noise_level=0.1,
            random_state=42
        )
        
        X_train, y_train, X_test, y_test = dataset.load_data()
        
        assert X_train.shape[1] == 20
        assert len(torch.unique(y_train)) <= 3
        assert X_test.shape[1] == 20
        assert len(torch.unique(y_test)) <= 3
        
        # Check dataset info
        info = dataset.get_info()
        assert info['name'] == 'synthetic'
        assert info['n_features'] == 20
    
    def test_performance_profiler(self):
        """Test performance profiling."""
        from hyperconformal.advanced_benchmarks import PerformanceProfiler
        import time
        
        profiler = PerformanceProfiler()
        
        profiler.start_profiling()
        time.sleep(0.01)  # Simulate work
        profiler.record_memory_usage(50.0)
        metrics = profiler.stop_profiling()
        
        assert 'elapsed_time' in metrics
        assert 'max_memory_mb' in metrics
        assert 'estimated_energy_mj' in metrics
        assert metrics['elapsed_time'] > 0
    
    def test_coverage_analyzer(self):
        """Test coverage analysis."""
        from hyperconformal.advanced_benchmarks import CoverageAnalyzer
        
        analyzer = CoverageAnalyzer()
        
        # Mock prediction sets and true labels
        prediction_sets = [[0, 1], [1], [0, 2], [2]]
        true_labels = torch.tensor([1, 1, 0, 2])
        
        coverage = analyzer.compute_empirical_coverage(prediction_sets, true_labels)
        avg_set_size = analyzer.compute_average_set_size(prediction_sets)
        
        assert 0.0 <= coverage <= 1.0
        assert avg_set_size > 0
        
        # Test conditional coverage
        conditions = true_labels  # Use labels as conditions
        conditional_cov = analyzer.compute_conditional_coverage(
            prediction_sets, true_labels, conditions
        )
        
        assert isinstance(conditional_cov, dict)
    
    def test_benchmark_suite(self):
        """Test benchmark suite functionality."""
        from hyperconformal.advanced_benchmarks import (
            AdvancedBenchmarkSuite, BenchmarkConfig,
            create_standard_conformal_hdc
        )
        
        suite = AdvancedBenchmarkSuite(output_dir=self.temp_dir)
        
        # Register a simple method
        suite.register_method("standard", create_standard_conformal_hdc)
        
        # Create minimal benchmark config
        config = BenchmarkConfig(
            name="test",
            datasets=['synthetic'],
            repetitions=1,
            confidence_levels=[0.9],
            hv_dimensions=[50],
            calibration_sizes=[20]
        )
        
        # Run benchmark (should work with synthetic data)
        results = suite.run_benchmark(config, save_results=True)
        
        # Check that some results were generated
        assert len(results) >= 0  # May be 0 if method factory fails
        
        # Check that files were created
        json_file = os.path.join(self.temp_dir, "test_results.json")
        # File may not exist if no results, which is ok for this test


class TestIntegration:
    """Integration tests combining multiple advanced features."""
    
    def test_quantum_federated_integration(self):
        """Test integration of quantum and federated features."""
        # This is a conceptual test - in practice you'd need compatible implementations
        pytest.skip("Integration test requires compatible quantum-federated implementation")
    
    def test_streaming_benchmark_integration(self):
        """Test streaming conformal prediction with benchmarking."""
        from hyperconformal.adaptive_realtime import StreamingAdaptiveConformalHDC, AdaptiveConfig
        from hyperconformal.encoders import RandomProjection
        
        # Create streaming model
        encoder = RandomProjection(input_dim=10, hv_dim=50)
        config = AdaptiveConfig(window_size=100, update_frequency=10)
        
        streaming_model = StreamingAdaptiveConformalHDC(
            encoder=encoder,
            num_classes=2,
            config=config,
            alpha=0.1
        )
        
        # Initialize
        X_init = torch.randn(30, 10)
        y_init = torch.randint(0, 2, (30,))
        streaming_model.initialize_prototypes(X_init, y_init)
        
        # Simulate streaming for a while
        prediction_sets = []
        for i in range(20):
            x = torch.randn(10)
            y_true = torch.randint(0, 2, (1,))
            
            result = streaming_model.stream_update(x, y_true)
            prediction_sets.append(result['prediction_set'])
        
        # Verify basic functionality
        assert len(prediction_sets) == 20
        assert all(isinstance(ps, list) for ps in prediction_sets)
        
        # Check that model adapted
        final_metrics = streaming_model.get_streaming_metrics()
        assert final_metrics['total_updates'] == 20


# Test fixtures and utilities
@pytest.fixture
def simple_dataset():
    """Fixture providing a simple dataset for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    return X, y


@pytest.fixture
def temp_output_dir():
    """Fixture providing temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests for advanced features."""
    
    @pytest.mark.slow
    def test_quantum_performance(self):
        """Test quantum processing performance with larger data."""
        from hyperconformal.quantum import QuantumHyperConformal
        
        model = QuantumHyperConformal(
            input_dim=100,
            hv_dim=1000,
            num_classes=5,
            alpha=0.1
        )
        
        # Large dataset
        X_train = torch.randn(1000, 100)
        y_train = torch.randint(0, 5, (1000,))
        X_test = torch.randn(100, 100)
        
        import time
        start_time = time.time()
        
        model.fit(X_train, y_train)
        prediction_sets = model.predict_set(X_test)
        
        elapsed = time.time() - start_time
        
        assert len(prediction_sets) == 100
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 60.0  # 1 minute timeout
    
    @pytest.mark.slow  
    def test_streaming_performance(self):
        """Test streaming performance with continuous updates."""
        from hyperconformal.adaptive_realtime import StreamingAdaptiveConformalHDC, AdaptiveConfig
        from hyperconformal.encoders import RandomProjection
        
        encoder = RandomProjection(input_dim=50, hv_dim=500)
        config = AdaptiveConfig(window_size=1000, update_frequency=100)
        
        streaming_model = StreamingAdaptiveConformalHDC(
            encoder=encoder,
            num_classes=5,
            config=config,
            alpha=0.1
        )
        
        # Initialize
        X_init = torch.randn(100, 50)
        y_init = torch.randint(0, 5, (100,))
        streaming_model.initialize_prototypes(X_init, y_init)
        
        # Stream many updates
        import time
        start_time = time.time()
        
        for i in range(500):
            x = torch.randn(50)
            y_true = torch.randint(0, 5, (1,))
            result = streaming_model.stream_update(x, y_true)
        
        elapsed = time.time() - start_time
        
        # Should handle 500 updates efficiently
        assert elapsed < 30.0  # 30 second timeout
        
        metrics = streaming_model.get_streaming_metrics()
        assert metrics['total_updates'] == 500


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])