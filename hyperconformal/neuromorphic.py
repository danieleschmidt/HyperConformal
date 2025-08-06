"""
Neuromorphic computing extensions for HyperConformal.

This module provides spike-based HDC encoders and conformal predictors
optimized for neuromorphic hardware like Intel Loihi, IBM TrueNorth,
and SpiNNaker.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

from .encoders import BaseEncoder
from .conformal import ConformalPredictor

logger = logging.getLogger(__name__)

try:
    import nengo
    import nengo_loihi
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False
    logger.warning("Neuromorphic backends not available. Install nengo and nengo_loihi for full functionality.")


class SpikeEncoder(ABC):
    """Abstract base class for spike-based encoding schemes."""
    
    @abstractmethod
    def encode_spikes(self, data: np.ndarray, time_steps: int) -> np.ndarray:
        """Encode input data to spike trains."""
        pass
    
    @abstractmethod
    def decode_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Decode spike trains back to vector representation."""
        pass


class RateEncoder(SpikeEncoder):
    """Rate-based spike encoding where spike frequency represents value."""
    
    def __init__(self, max_rate: float = 100.0, dt: float = 0.001):
        """
        Initialize rate encoder.
        
        Args:
            max_rate: Maximum firing rate (Hz)
            dt: Simulation time step (seconds)
        """
        self.max_rate = max_rate
        self.dt = dt
    
    def encode_spikes(self, data: np.ndarray, time_steps: int) -> np.ndarray:
        """Encode data as Poisson spike trains with rate proportional to value."""
        # Normalize data to [0, 1]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Convert to firing rates
        rates = data_norm * self.max_rate
        
        # Generate Poisson spikes
        spikes = np.random.poisson(rates * self.dt, (time_steps, *data.shape))
        return spikes.astype(bool)
    
    def decode_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Decode spike trains by counting spikes and normalizing by time."""
        spike_counts = spikes.sum(axis=0)
        time_steps = spikes.shape[0]
        rates = spike_counts / (time_steps * self.dt)
        return rates / self.max_rate


class TemporalEncoder(SpikeEncoder):
    """Temporal encoding where spike timing represents value."""
    
    def __init__(self, max_delay: float = 0.1, dt: float = 0.001):
        """
        Initialize temporal encoder.
        
        Args:
            max_delay: Maximum delay for encoding (seconds)
            dt: Simulation time step (seconds)
        """
        self.max_delay = max_delay
        self.dt = dt
    
    def encode_spikes(self, data: np.ndarray, time_steps: int) -> np.ndarray:
        """Encode data as precisely timed spikes (smaller values = earlier spikes)."""
        # Normalize and invert (smaller values should spike earlier)
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        delays = (1 - data_norm) * self.max_delay
        
        # Convert delays to time step indices
        spike_times = (delays / self.dt).astype(int)
        spike_times = np.clip(spike_times, 0, time_steps - 1)
        
        # Create spike arrays
        spikes = np.zeros((time_steps, *data.shape), dtype=bool)
        
        # Set spikes at appropriate times
        flat_indices = np.unravel_index(range(data.size), data.shape)
        for i, (time_idx, *spatial_idx) in enumerate(zip(spike_times.flat, *flat_indices)):
            spikes[time_idx, *[idx[i] for idx in spatial_idx]] = True
        
        return spikes
    
    def decode_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """Decode spike trains by finding first spike time."""
        time_steps, *spatial_shape = spikes.shape
        first_spike = np.argmax(spikes, axis=0)
        
        # Handle case where no spikes occurred
        no_spikes = ~spikes.any(axis=0)
        first_spike[no_spikes] = time_steps - 1
        
        # Convert back to normalized values
        delays = first_spike * self.dt
        values = 1 - (delays / self.max_delay)
        return np.clip(values, 0, 1)


class SpikingHDCEncoder(BaseEncoder):
    """HDC encoder using spiking neural networks."""
    
    def __init__(
        self,
        input_dim: int,
        hv_dim: int = 10000,
        spike_encoder: SpikeEncoder = None,
        simulation_time: float = 0.1,
        dt: float = 0.001,
        quantization: str = 'binary'
    ):
        """
        Initialize spiking HDC encoder.
        
        Args:
            input_dim: Input dimension
            hv_dim: Hypervector dimension
            spike_encoder: Spike encoding scheme
            simulation_time: Simulation duration (seconds)
            dt: Time step (seconds)
            quantization: Quantization scheme ('binary', 'ternary', 'continuous')
        """
        super().__init__(input_dim, hv_dim, quantization)
        
        self.spike_encoder = spike_encoder or RateEncoder()
        self.simulation_time = simulation_time
        self.dt = dt
        self.time_steps = int(simulation_time / dt)
        
        # Initialize projection weights (sparse random)
        self.projection_weights = self._init_projection_weights()
        
        # Neuromorphic network (if available)
        self.network = None
        if NEUROMORPHIC_AVAILABLE:
            self._build_nengo_network()
    
    def _init_projection_weights(self) -> np.ndarray:
        """Initialize sparse random projection weights for spiking network."""
        # Use sparse connectivity (typical in neuromorphic hardware)
        sparsity = 0.1  # 10% connectivity
        weights = np.zeros((self.input_dim, self.hv_dim))
        
        # Randomly connect neurons
        for i in range(self.input_dim):
            n_connections = int(self.hv_dim * sparsity)
            target_indices = np.random.choice(self.hv_dim, n_connections, replace=False)
            
            # Binary random weights
            weights[i, target_indices] = np.random.choice([-1, 1], n_connections)
        
        return weights
    
    def _build_nengo_network(self):
        """Build Nengo network for neuromorphic simulation."""
        if not NEUROMORPHIC_AVAILABLE:
            return
        
        self.network = nengo.Network(label="SpikingHDC")
        
        with self.network:
            # Input layer (spike generators)
            self.input_layer = nengo.Node(lambda t: np.zeros(self.input_dim))
            
            # HDC projection layer
            self.hdc_layer = nengo.Ensemble(
                n_neurons=self.hv_dim,
                dimensions=1,  # Each neuron represents one bit
                neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=nengo.dists.Uniform(-0.5, 0.5)
            )
            
            # Connection with projection weights
            self.projection = nengo.Connection(
                self.input_layer, self.hdc_layer,
                transform=self.projection_weights.T,
                synapse=0.005
            )
            
            # Output probe
            self.spike_probe = nengo.Probe(self.hdc_layer.neurons, 'spikes')
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input using spiking HDC."""
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        batch_size = x_np.shape[0]
        encoded_batch = []
        
        for sample in x_np:
            # Generate input spikes
            input_spikes = self.spike_encoder.encode_spikes(sample, self.time_steps)
            
            if NEUROMORPHIC_AVAILABLE and self.network is not None:
                # Use Nengo simulation
                encoded_hv = self._simulate_nengo(input_spikes)
            else:
                # Fallback to numpy simulation
                encoded_hv = self._simulate_numpy(input_spikes, sample)
            
            encoded_batch.append(encoded_hv)
        
        result = torch.tensor(np.array(encoded_batch), dtype=torch.float32)
        if hasattr(x, 'device'):
            result = result.to(x.device)
        
        return result
    
    def _simulate_nengo(self, input_spikes: np.ndarray) -> np.ndarray:
        """Simulate encoding using Nengo network."""
        with nengo.Simulator(self.network, dt=self.dt) as sim:
            # Set input spike pattern
            def input_func(t):
                time_idx = min(int(t / self.dt), self.time_steps - 1)
                return input_spikes[time_idx].astype(float)
            
            # Update input node
            with sim:
                sim.model.params[self.input_layer] = input_func
            
            # Run simulation
            sim.run(self.simulation_time)
            
            # Extract output spikes
            output_spikes = sim.data[self.spike_probe]
            
            # Convert spikes to hypervector
            spike_counts = output_spikes.sum(axis=0)
            
            if self.quantization == 'binary':
                # Threshold at median activity
                threshold = np.median(spike_counts)
                return (spike_counts > threshold).astype(float) * 2 - 1
            elif self.quantization == 'ternary':
                # Three-level quantization
                low_thresh, high_thresh = np.percentile(spike_counts, [33, 67])
                result = np.zeros_like(spike_counts)
                result[spike_counts > high_thresh] = 1
                result[spike_counts < low_thresh] = -1
                return result
            else:
                # Continuous (normalized spike counts)
                return (spike_counts - spike_counts.mean()) / (spike_counts.std() + 1e-8)
    
    def _simulate_numpy(self, input_spikes: np.ndarray, sample: np.ndarray) -> np.ndarray:
        """Fallback numpy-based simulation."""
        # Simple linear projection (approximation of spiking dynamics)
        projected = np.dot(sample, self.projection_weights)
        
        # Add some noise to simulate spiking variability
        noise = np.random.normal(0, 0.1, projected.shape)
        projected += noise
        
        # Apply quantization
        if self.quantization == 'binary':
            return np.sign(projected)
        elif self.quantization == 'ternary':
            # Three-level quantization
            low_thresh, high_thresh = np.percentile(projected, [33, 67])
            result = np.zeros_like(projected)
            result[projected > high_thresh] = 1
            result[projected < low_thresh] = -1
            return result
        else:
            # Continuous (normalized)
            return (projected - projected.mean()) / (projected.std() + 1e-8)
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Compute similarity using spike-based metrics."""
        if self.quantization in ['binary', 'ternary']:
            # Use Hamming distance for discrete representations
            return 1 - torch.mean((hv1 != hv2).float(), dim=-1)
        else:
            # Use cosine similarity for continuous representations
            return torch.nn.functional.cosine_similarity(hv1, hv2, dim=-1)


class SpikingConformalPredictor(ConformalPredictor):
    """Conformal predictor optimized for spiking neural networks."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        score_type: str = 'spike_count',
        energy_model: Optional[str] = None
    ):
        """
        Initialize spiking conformal predictor.
        
        Args:
            alpha: Miscoverage level
            score_type: Score function ('spike_count', 'spike_timing', 'energy_aware')
            energy_model: Energy consumption model ('loihi', 'truenorth', 'measured')
        """
        super().__init__(alpha, score_type)
        self.energy_model = energy_model
        self.energy_consumption = []
    
    def _compute_conformity_scores(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute conformity scores for spiking predictions."""
        if self.score_type == 'spike_count':
            # Use spike count-based scores
            scores = 1 - predictions[np.arange(len(predictions)), labels]
        elif self.score_type == 'spike_timing':
            # Use timing-based scores (simulated)
            # In real implementation, this would use actual spike timing
            timing_scores = np.random.exponential(predictions[np.arange(len(predictions)), labels])
            scores = 1 / (timing_scores + 1e-8)
        elif self.score_type == 'energy_aware':
            # Energy-aware scores
            base_scores = 1 - predictions[np.arange(len(predictions)), labels]
            energy_penalty = self._compute_energy_penalty(predictions)
            scores = base_scores + 0.1 * energy_penalty
        else:
            # Fallback to standard APS
            scores = 1 - predictions[np.arange(len(predictions)), labels]
        
        return scores
    
    def _compute_energy_penalty(self, predictions: np.ndarray) -> np.ndarray:
        """Compute energy penalty based on prediction complexity."""
        if self.energy_model == 'loihi':
            # Intel Loihi energy model (simplified)
            spike_activity = np.sum(predictions > 0.1, axis=1)
            return spike_activity * 23.6e-12  # pJ per spike
        elif self.energy_model == 'truenorth':
            # IBM TrueNorth energy model
            spike_activity = np.sum(predictions > 0.1, axis=1)
            return spike_activity * 45e-12  # pJ per spike
        else:
            # Generic model based on prediction entropy
            entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
            return entropy / np.log(predictions.shape[1])  # Normalized entropy


class NeuromorphicConformalHDC:
    """Complete neuromorphic HDC system with conformal prediction."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hv_dim: int = 10000,
        alpha: float = 0.1,
        neuromorphic_backend: str = 'nengo',
        energy_model: str = 'measured'
    ):
        """
        Initialize neuromorphic conformal HDC.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            hv_dim: Hypervector dimension
            alpha: Miscoverage level
            neuromorphic_backend: Backend ('nengo', 'brian2', 'spinnaker')
            energy_model: Energy consumption model
        """
        self.encoder = SpikingHDCEncoder(
            input_dim=input_dim,
            hv_dim=hv_dim,
            quantization='binary'
        )
        
        self.conformal_predictor = SpikingConformalPredictor(
            alpha=alpha,
            score_type='energy_aware',
            energy_model=energy_model
        )
        
        self.num_classes = num_classes
        self.class_prototypes = None
        self.is_fitted = False
        
        # Energy tracking
        self.total_energy_consumption = 0.0
        self.prediction_count = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuromorphicConformalHDC':
        """Fit the neuromorphic HDC model."""
        logger.info(f"Training neuromorphic HDC with {len(X)} samples")
        
        # Convert to tensor for encoding
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X
        
        # Encode training data
        encoded_X = self.encoder.encode(X_tensor)
        
        # Learn class prototypes
        self.class_prototypes = torch.zeros(self.num_classes, self.encoder.hv_dim)
        for class_idx in range(self.num_classes):
            class_mask = (y == class_idx)
            if class_mask.sum() > 0:
                self.class_prototypes[class_idx] = encoded_X[class_mask].mean(dim=0)
        
        # Compute predictions for calibration
        similarities = self._compute_similarities(encoded_X)
        probabilities = torch.softmax(similarities, dim=1)
        
        # Calibrate conformal predictor
        self.conformal_predictor.calibrate(probabilities.numpy(), y)
        
        self.is_fitted = True
        return self
    
    def _compute_similarities(self, encoded_X: torch.Tensor) -> torch.Tensor:
        """Compute similarities to class prototypes."""
        similarities = torch.zeros(encoded_X.shape[0], self.num_classes)
        
        for class_idx in range(self.num_classes):
            similarities[:, class_idx] = self.encoder.similarity(
                encoded_X, self.class_prototypes[class_idx].unsqueeze(0)
            )
        
        return similarities
    
    def predict_with_energy(
        self, 
        X: np.ndarray
    ) -> Tuple[List[List[int]], float]:
        """Make predictions while tracking energy consumption."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Encode input
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X
        
        encoded_X = self.encoder.encode(X_tensor)
        
        # Compute similarities and probabilities
        similarities = self._compute_similarities(encoded_X)
        probabilities = torch.softmax(similarities, dim=1)
        
        # Generate prediction sets
        prediction_sets = self.conformal_predictor.predict_set(probabilities.numpy())
        
        # Estimate energy consumption (simplified model)
        # In practice, this would interface with neuromorphic hardware
        batch_energy = self._estimate_energy_consumption(X.shape[0])
        self.total_energy_consumption += batch_energy
        self.prediction_count += X.shape[0]
        
        return prediction_sets, batch_energy
    
    def _estimate_energy_consumption(self, batch_size: int) -> float:
        """Estimate energy consumption for batch prediction."""
        # Simplified energy model (would be replaced with hardware-specific models)
        base_energy_per_sample = 0.1e-6  # 0.1 μJ per sample
        
        # Energy scales with hypervector dimension and number of classes
        scaling_factor = (self.encoder.hv_dim / 10000) * (self.num_classes / 10)
        
        return batch_size * base_energy_per_sample * scaling_factor
    
    def get_energy_efficiency(self) -> Dict[str, float]:
        """Get energy efficiency metrics."""
        if self.prediction_count == 0:
            return {'avg_energy_per_prediction': 0.0, 'total_energy': 0.0}
        
        return {
            'avg_energy_per_prediction': self.total_energy_consumption / self.prediction_count,
            'total_energy': self.total_energy_consumption,
            'predictions_made': self.prediction_count
        }
    
    def deploy_to_loihi(self, save_path: str):
        """Deploy model to Intel Loihi neuromorphic processor."""
        if not NEUROMORPHIC_AVAILABLE:
            raise RuntimeError("Neuromorphic backends not available")
        
        # This would contain actual Loihi deployment code
        # For now, save model parameters for deployment
        deployment_data = {
            'encoder_weights': self.encoder.projection_weights,
            'class_prototypes': self.class_prototypes.numpy(),
            'conformal_quantile': self.conformal_predictor.quantile,
            'alpha': self.conformal_predictor.alpha
        }
        
        np.savez_compressed(save_path, **deployment_data)
        logger.info(f"Model deployment data saved to {save_path}")
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'encoder_type': 'SpikingHDC',
            'hv_dim': self.encoder.hv_dim,
            'num_classes': self.num_classes,
            'alpha': self.conformal_predictor.alpha,
            'is_fitted': self.is_fitted,
            'energy_efficiency': self.get_energy_efficiency(),
            'neuromorphic_ready': NEUROMORPHIC_AVAILABLE
        }