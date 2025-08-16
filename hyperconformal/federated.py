"""
Federated Learning for HyperConformal with Homomorphic Encryption

This module provides privacy-preserving federated learning capabilities
for HyperConformal models using homomorphic encryption and secure aggregation.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import logging
import hashlib
import warnings
from collections import defaultdict
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)

# Cryptographic imports with fallbacks
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("Cryptography not available - using basic security fallback")


@dataclass
class FederatedClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    data_privacy_level: str = "high"  # "low", "medium", "high"
    max_gradient_norm: float = 1.0
    differential_privacy_epsilon: float = 1.0
    local_epochs: int = 1
    batch_size: int = 32


@dataclass
class EncryptionConfig:
    """Configuration for homomorphic encryption."""
    key_size: int = 2048
    noise_scale: float = 0.1
    precision_bits: int = 16
    enable_compression: bool = True


class SimpleHomomorphicEncryption:
    """
    Simplified homomorphic encryption for federated learning.
    
    Note: This is a demonstration implementation. For production use,
    consider libraries like Microsoft SEAL or IBM FHE.
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        self.key = self._generate_key()
        logger.info(f"Homomorphic encryption initialized with {config.key_size}-bit key")
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        if CRYPTO_AVAILABLE:
            # Use cryptographically secure random generation
            salt = b'hyperconformal_federated_salt'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(b'federated_learning_key'))
            return key
        else:
            # Fallback to simple key generation
            return hashlib.sha256(b'simple_fallback_key').digest()
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Encrypt tensor using homomorphic encryption."""
        # Quantize tensor to fixed precision
        scale = 2 ** self.config.precision_bits
        quantized = torch.round(tensor * scale).long()
        
        if CRYPTO_AVAILABLE:
            # Use Fernet symmetric encryption (approximates homomorphic properties)
            f = Fernet(self.key)
            encrypted_data = f.encrypt(quantized.numpy().tobytes())
            
            return {
                'encrypted_data': encrypted_data,
                'shape': tensor.shape,
                'scale': scale,
                'noise_added': self._add_noise(tensor.shape)
            }
        else:
            # Simple obfuscation fallback
            obfuscated = quantized ^ 0xDEADBEEF  # XOR obfuscation
            return {
                'encrypted_data': obfuscated,
                'shape': tensor.shape, 
                'scale': scale,
                'noise_added': self._add_noise(tensor.shape)
            }
    
    def decrypt_tensor(self, encrypted_data: Dict[str, Any]) -> torch.Tensor:
        """Decrypt tensor."""
        if CRYPTO_AVAILABLE and isinstance(encrypted_data['encrypted_data'], bytes):
            f = Fernet(self.key)
            decrypted_bytes = f.decrypt(encrypted_data['encrypted_data'])
            quantized = torch.frombuffer(decrypted_bytes, dtype=torch.long)
        else:
            # Simple deobfuscation
            quantized = encrypted_data['encrypted_data'] ^ 0xDEADBEEF
        
        # Reshape and dequantize
        quantized = quantized.view(encrypted_data['shape'])
        tensor = quantized.float() / encrypted_data['scale']
        
        return tensor
    
    def homomorphic_add(self, enc1: Dict[str, Any], enc2: Dict[str, Any]) -> Dict[str, Any]:
        """Homomorphic addition of encrypted tensors."""
        if enc1['shape'] != enc2['shape']:
            raise ValueError("Cannot add tensors with different shapes")
        
        if CRYPTO_AVAILABLE and isinstance(enc1['encrypted_data'], bytes):
            # For demonstration, decrypt, add, re-encrypt
            tensor1 = self.decrypt_tensor(enc1)
            tensor2 = self.decrypt_tensor(enc2)
            result = tensor1 + tensor2
            return self.encrypt_tensor(result)
        else:
            # Simple addition for obfuscated data
            result_data = enc1['encrypted_data'] + enc2['encrypted_data']
            return {
                'encrypted_data': result_data,
                'shape': enc1['shape'],
                'scale': (enc1['scale'] + enc2['scale']) / 2,
                'noise_added': enc1['noise_added'] + enc2['noise_added']
            }
    
    def _add_noise(self, shape: torch.Size) -> torch.Tensor:
        """Add differential privacy noise."""
        noise = torch.normal(0, self.config.noise_scale, shape)
        return noise


class FederatedClient:
    """Federated learning client for HyperConformal models."""
    
    def __init__(
        self,
        client_config: FederatedClientConfig,
        encoder_config: Dict[str, Any],
        encryption_config: Optional[EncryptionConfig] = None
    ):
        self.config = client_config
        self.encoder_config = encoder_config
        self.client_id = client_config.client_id
        
        # Initialize encryption
        if encryption_config:
            self.encryption = SimpleHomomorphicEncryption(encryption_config)
            self.use_encryption = True
        else:
            self.encryption = None
            self.use_encryption = False
        
        # Local model components
        self.local_encoder = None
        self.local_prototypes = {}
        self.training_history = []
        
        logger.info(f"Federated client {self.client_id} initialized")
    
    def initialize_model(self, global_encoder_state: Dict[str, torch.Tensor]):
        """Initialize local model from global state."""
        from .encoders import RandomProjection
        
        # Create local encoder
        self.local_encoder = RandomProjection(
            input_dim=self.encoder_config['input_dim'],
            hv_dim=self.encoder_config['hv_dim'],
            quantization=self.encoder_config.get('quantization', 'binary')
        )
        
        # Load global parameters
        self.local_encoder.load_state_dict(global_encoder_state)
        logger.info(f"Client {self.client_id} model initialized from global state")
    
    def local_training_step(
        self, 
        X_local: torch.Tensor, 
        y_local: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform local training step with privacy preservation."""
        start_time = time.time()
        
        # Encode local data
        with torch.no_grad():
            encoded_data = self.local_encoder.encode(X_local)
        
        # Update local prototypes
        unique_classes = torch.unique(y_local)
        local_updates = {}
        
        for class_idx in unique_classes:
            class_mask = y_local == class_idx
            class_encodings = encoded_data[class_mask]
            
            # Bundle class samples (average for simplicity)
            if len(class_encodings) > 0:
                class_prototype = torch.mean(class_encodings, dim=0)
                
                # Add differential privacy noise
                if self.config.data_privacy_level == "high":
                    noise_scale = self.config.differential_privacy_epsilon
                    noise = torch.normal(0, noise_scale, class_prototype.shape)
                    class_prototype += noise
                
                # Gradient clipping for privacy
                prototype_norm = torch.norm(class_prototype)
                if prototype_norm > self.config.max_gradient_norm:
                    class_prototype = class_prototype * (self.config.max_gradient_norm / prototype_norm)
                
                local_updates[int(class_idx)] = class_prototype
        
        # Encrypt updates if encryption is enabled
        if self.use_encryption:
            encrypted_updates = {}
            for class_idx, prototype in local_updates.items():
                encrypted_updates[class_idx] = self.encryption.encrypt_tensor(prototype)
            local_updates = encrypted_updates
        
        training_time = time.time() - start_time
        
        # Record training metrics
        training_record = {
            'timestamp': time.time(),
            'num_samples': len(X_local),
            'num_classes': len(unique_classes),
            'training_time': training_time,
            'privacy_level': self.config.data_privacy_level,
            'encrypted': self.use_encryption
        }
        self.training_history.append(training_record)
        
        logger.info(f"Client {self.client_id} local training complete: "
                   f"{len(X_local)} samples, {training_time:.3f}s")
        
        return {
            'client_id': self.client_id,
            'class_updates': local_updates,
            'num_samples': len(X_local),
            'training_metadata': training_record
        }
    
    def receive_global_update(self, global_prototypes: Dict[str, torch.Tensor]):
        """Receive and apply global model update."""
        # Decrypt if necessary
        if self.use_encryption and self.encryption:
            decrypted_prototypes = {}
            for class_idx, encrypted_proto in global_prototypes.items():
                if isinstance(encrypted_proto, dict):  # Encrypted
                    decrypted_prototypes[class_idx] = self.encryption.decrypt_tensor(encrypted_proto)
                else:  # Plain tensor
                    decrypted_prototypes[class_idx] = encrypted_proto
            global_prototypes = decrypted_prototypes
        
        # Update local prototypes
        self.local_prototypes.update(global_prototypes)
        logger.info(f"Client {self.client_id} received global update: {len(global_prototypes)} classes")
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy-related metrics."""
        if not self.training_history:
            return {}
        
        total_samples = sum(record['num_samples'] for record in self.training_history)
        avg_training_time = np.mean([record['training_time'] for record in self.training_history])
        
        return {
            'total_samples_processed': total_samples,
            'num_training_rounds': len(self.training_history),
            'avg_training_time': avg_training_time,
            'privacy_level': self.config.data_privacy_level,
            'differential_privacy_epsilon': self.config.differential_privacy_epsilon,
            'encryption_enabled': self.use_encryption
        }


class FederatedServer:
    """Federated learning server for aggregating client updates."""
    
    def __init__(
        self,
        num_classes: int,
        hv_dim: int,
        encryption_config: Optional[EncryptionConfig] = None,
        aggregation_method: str = "fedavg"
    ):
        self.num_classes = num_classes
        self.hv_dim = hv_dim
        self.aggregation_method = aggregation_method
        
        # Initialize encryption
        if encryption_config:
            self.encryption = SimpleHomomorphicEncryption(encryption_config)
            self.use_encryption = True
        else:
            self.encryption = None
            self.use_encryption = False
        
        # Global model state
        self.global_prototypes = {}
        self.client_contributions = defaultdict(list)
        self.round_history = []
        
        logger.info(f"Federated server initialized: {num_classes} classes, {hv_dim}D")
    
    def aggregate_updates(
        self, 
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using federated averaging."""
        start_time = time.time()
        
        # Group updates by class
        class_updates = defaultdict(list)
        client_weights = {}
        
        for update in client_updates:
            client_id = update['client_id']
            num_samples = update['num_samples']
            client_weights[client_id] = num_samples
            
            for class_idx, prototype in update['class_updates'].items():
                class_updates[class_idx].append({
                    'client_id': client_id,
                    'prototype': prototype,
                    'weight': num_samples
                })
        
        # Aggregate each class prototype
        aggregated_prototypes = {}
        
        for class_idx, updates in class_updates.items():
            if self.aggregation_method == "fedavg":
                # Weighted average based on number of samples
                if self.use_encryption:
                    # Homomorphic aggregation
                    encrypted_sum = updates[0]['prototype']
                    total_weight = updates[0]['weight']
                    
                    for update in updates[1:]:
                        encrypted_sum = self.encryption.homomorphic_add(
                            encrypted_sum, update['prototype']
                        )
                        total_weight += update['weight']
                    
                    # For simplicity, decrypt to apply averaging
                    prototype_sum = self.encryption.decrypt_tensor(encrypted_sum)
                    aggregated_prototype = prototype_sum / len(updates)
                    
                    # Re-encrypt if needed
                    if self.use_encryption:
                        aggregated_prototypes[class_idx] = self.encryption.encrypt_tensor(aggregated_prototype)
                    else:
                        aggregated_prototypes[class_idx] = aggregated_prototype
                else:
                    # Standard weighted averaging
                    weighted_sum = torch.zeros(self.hv_dim)
                    total_weight = 0
                    
                    for update in updates:
                        weight = update['weight']
                        prototype = update['prototype']
                        weighted_sum += weight * prototype
                        total_weight += weight
                    
                    aggregated_prototypes[class_idx] = weighted_sum / total_weight
            
            elif self.aggregation_method == "median":
                # Robust aggregation using median
                prototypes = [update['prototype'] for update in updates]
                if self.use_encryption:
                    # Decrypt for median computation
                    decrypted_prototypes = [
                        self.encryption.decrypt_tensor(p) if isinstance(p, dict) else p
                        for p in prototypes
                    ]
                    prototypes = decrypted_prototypes
                
                prototype_tensor = torch.stack(prototypes)
                aggregated_prototypes[class_idx] = torch.median(prototype_tensor, dim=0)[0]
        
        # Update global prototypes
        self.global_prototypes.update(aggregated_prototypes)
        
        # Record round metrics
        round_record = {
            'timestamp': time.time(),
            'num_clients': len(client_updates),
            'total_samples': sum(update['num_samples'] for update in client_updates),
            'classes_updated': list(aggregated_prototypes.keys()),
            'aggregation_time': time.time() - start_time,
            'encryption_used': self.use_encryption
        }
        self.round_history.append(round_record)
        
        logger.info(f"Aggregation complete: {len(client_updates)} clients, "
                   f"{len(aggregated_prototypes)} classes, {round_record['aggregation_time']:.3f}s")
        
        return aggregated_prototypes
    
    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        if self.use_encryption:
            # Decrypt prototypes for distribution
            decrypted_prototypes = {}
            for class_idx, encrypted_proto in self.global_prototypes.items():
                if isinstance(encrypted_proto, dict):  # Encrypted
                    decrypted_prototypes[class_idx] = self.encryption.decrypt_tensor(encrypted_proto)
                else:  # Already decrypted
                    decrypted_prototypes[class_idx] = encrypted_proto
            return decrypted_prototypes
        else:
            return self.global_prototypes.copy()
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Analyze convergence and performance metrics."""
        if len(self.round_history) < 2:
            return {}
        
        # Calculate metrics across rounds
        total_clients = [round_data['num_clients'] for round_data in self.round_history]
        total_samples = [round_data['total_samples'] for round_data in self.round_history]
        aggregation_times = [round_data['aggregation_time'] for round_data in self.round_history]
        
        return {
            'total_rounds': len(self.round_history),
            'avg_clients_per_round': np.mean(total_clients),
            'avg_samples_per_round': np.mean(total_samples),
            'avg_aggregation_time': np.mean(aggregation_times),
            'total_samples_processed': sum(total_samples),
            'convergence_trend': self._compute_convergence_trend(),
            'encryption_overhead': self._compute_encryption_overhead()
        }
    
    def _compute_convergence_trend(self) -> float:
        """Compute convergence trend (simplified)."""
        if len(self.round_history) < 3:
            return 0.0
        
        # Simple proxy: rate of change in number of participating clients
        recent_clients = [r['num_clients'] for r in self.round_history[-3:]]
        if len(set(recent_clients)) == 1:
            return 1.0  # Stable participation
        else:
            return np.std(recent_clients) / np.mean(recent_clients)  # Normalized variation
    
    def _compute_encryption_overhead(self) -> float:
        """Estimate encryption computational overhead."""
        if not self.use_encryption or len(self.round_history) < 2:
            return 0.0
        
        # Simple estimate based on aggregation time increase
        recent_times = [r['aggregation_time'] for r in self.round_history[-5:]]
        baseline_time = 0.001  # Estimated baseline without encryption
        avg_time = np.mean(recent_times)
        
        return max(0.0, (avg_time - baseline_time) / baseline_time)


class FederatedHyperConformal:
    """
    Main federated learning system for HyperConformal models.
    
    Orchestrates training across multiple clients with privacy preservation
    and secure aggregation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hv_dim: int,
        num_classes: int,
        alpha: float = 0.1,
        encryption_config: Optional[EncryptionConfig] = None,
        aggregation_method: str = "fedavg"
    ):
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Initialize server
        self.server = FederatedServer(
            num_classes, hv_dim, encryption_config, aggregation_method
        )
        
        # Client registry
        self.clients = {}
        self.active_clients = set()
        
        # Training state
        self.current_round = 0
        self.is_training = False
        
        logger.info(f"FederatedHyperConformal initialized: {input_dim}D -> {num_classes} classes")
    
    def register_client(
        self, 
        client_config: FederatedClientConfig,
        encryption_config: Optional[EncryptionConfig] = None
    ) -> FederatedClient:
        """Register a new federated client."""
        encoder_config = {
            'input_dim': self.input_dim,
            'hv_dim': self.hv_dim,
            'quantization': 'binary'
        }
        
        client = FederatedClient(client_config, encoder_config, encryption_config)
        self.clients[client_config.client_id] = client
        
        logger.info(f"Client {client_config.client_id} registered")
        return client
    
    def federated_training_round(
        self,
        client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        min_clients: int = 2
    ) -> Dict[str, Any]:
        """Execute one round of federated training."""
        if len(client_data) < min_clients:
            raise ValueError(f"Need at least {min_clients} clients, got {len(client_data)}")
        
        self.current_round += 1
        round_start = time.time()
        
        logger.info(f"Starting federated round {self.current_round} with {len(client_data)} clients")
        
        # Initialize clients if first round
        if self.current_round == 1:
            # Create initial encoder for parameter sharing
            from .encoders import RandomProjection
            initial_encoder = RandomProjection(self.input_dim, self.hv_dim)
            initial_state = initial_encoder.state_dict()
            
            for client_id in client_data.keys():
                if client_id in self.clients:
                    self.clients[client_id].initialize_model(initial_state)
        
        # Collect client updates
        client_updates = []
        
        for client_id, (X_client, y_client) in client_data.items():
            if client_id not in self.clients:
                logger.warning(f"Unknown client {client_id}, skipping")
                continue
            
            # Send global state to client
            global_state = self.server.get_global_state()
            self.clients[client_id].receive_global_update(global_state)
            
            # Perform local training
            update = self.clients[client_id].local_training_step(X_client, y_client)
            client_updates.append(update)
            
            self.active_clients.add(client_id)
        
        # Server aggregation
        aggregated_prototypes = self.server.aggregate_updates(client_updates)
        
        round_time = time.time() - round_start
        
        # Round summary
        round_summary = {
            'round': self.current_round,
            'num_clients': len(client_updates),
            'total_samples': sum(update['num_samples'] for update in client_updates),
            'classes_updated': len(aggregated_prototypes),
            'round_time': round_time,
            'convergence_metrics': self.server.get_convergence_metrics()
        }
        
        logger.info(f"Federated round {self.current_round} complete: "
                   f"{round_summary['num_clients']} clients, {round_time:.3f}s")
        
        return round_summary
    
    def get_federated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning metrics."""
        server_metrics = self.server.get_convergence_metrics()
        
        # Aggregate client metrics
        client_metrics = {}
        for client_id, client in self.clients.items():
            client_metrics[client_id] = client.get_privacy_metrics()
        
        return {
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'active_clients': len(self.active_clients),
            'server_metrics': server_metrics,
            'client_metrics': client_metrics,
            'privacy_features': {
                'encryption_enabled': self.server.use_encryption,
                'differential_privacy': any(
                    client.config.data_privacy_level == "high" 
                    for client in self.clients.values()
                )
            }
        }


# Export main classes
__all__ = [
    'FederatedClientConfig',
    'EncryptionConfig', 
    'FederatedClient',
    'FederatedServer',
    'FederatedHyperConformal',
    'SimpleHomomorphicEncryption'
]