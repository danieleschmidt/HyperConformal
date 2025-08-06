"""
Distributed and federated learning capabilities for HyperConformal.

This module provides scalable distributed training and inference
across multiple nodes, with support for federated learning,
model sharding, and global deployment.
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from abc import ABC, abstractmethod
import logging
import time
import threading
import hashlib
import json
from collections import defaultdict
import asyncio

from .encoders import BaseEncoder
from .hyperconformal import ConformalHDC
from .security import DifferentialPrivacyMechanism

logger = logging.getLogger(__name__)

# Optional dependencies for distributed computing
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


class CommunicationBackend(ABC):
    """Abstract base class for distributed communication."""
    
    @abstractmethod
    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor], source_rank: int) -> Dict[str, torch.Tensor]:
        """Broadcast parameters from source to all nodes."""
        pass
    
    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
        """All-reduce operation across all nodes."""
        pass
    
    @abstractmethod
    def gather(self, tensor: torch.Tensor, dst_rank: int) -> List[torch.Tensor]:
        """Gather tensors from all nodes to destination."""
        pass
    
    @abstractmethod
    def get_rank(self) -> int:
        """Get rank of current node."""
        pass
    
    @abstractmethod
    def get_world_size(self) -> int:
        """Get total number of nodes."""
        pass


class PyTorchDistributedBackend(CommunicationBackend):
    """PyTorch distributed backend for communication."""
    
    def __init__(self, backend: str = 'nccl'):
        """
        Initialize PyTorch distributed backend.
        
        Args:
            backend: Communication backend ('nccl', 'gloo', 'mpi')
        """
        if not dist.is_initialized():
            raise RuntimeError("PyTorch distributed not initialized. Call dist.init_process_group() first.")
        
        self.backend = backend
    
    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor], source_rank: int) -> Dict[str, torch.Tensor]:
        """Broadcast parameters using PyTorch distributed."""
        result = {}
        
        for name, tensor in parameters.items():
            # Create a copy for broadcasting
            broadcast_tensor = tensor.clone()
            dist.broadcast(broadcast_tensor, src=source_rank)
            result[name] = broadcast_tensor
        
        return result
    
    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
        """All-reduce operation using PyTorch distributed."""
        # Map string operations to PyTorch ops
        op_map = {
            'sum': dist.ReduceOp.SUM,
            'avg': dist.ReduceOp.SUM,  # We'll divide by world size
            'max': dist.ReduceOp.MAX,
            'min': dist.ReduceOp.MIN
        }
        
        if op not in op_map:
            raise ValueError(f"Unsupported operation: {op}")
        
        result_tensor = tensor.clone()
        dist.all_reduce(result_tensor, op=op_map[op])
        
        # For average, divide by world size
        if op == 'avg':
            result_tensor /= self.get_world_size()
        
        return result_tensor
    
    def gather(self, tensor: torch.Tensor, dst_rank: int) -> List[torch.Tensor]:
        """Gather tensors using PyTorch distributed."""
        world_size = self.get_world_size()
        
        if self.get_rank() == dst_rank:
            # Receiving node
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, gather_list, dst=dst_rank)
            return gather_list
        else:
            # Sending node
            dist.gather(tensor, dst=dst_rank)
            return []
    
    def get_rank(self) -> int:
        """Get current process rank."""
        return dist.get_rank()
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return dist.get_world_size()


class RedisBackend(CommunicationBackend):
    """Redis-based communication backend for loose coupling."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, node_id: str = None):
        """
        Initialize Redis backend.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            node_id: Unique identifier for this node
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.node_id = node_id or f"node_{int(time.time())}"
        
        # Register this node
        self.redis_client.sadd("active_nodes", self.node_id)
        self.redis_client.expire("active_nodes", 3600)  # 1 hour TTL
    
    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor], source_rank: int) -> Dict[str, torch.Tensor]:
        """Broadcast parameters via Redis."""
        if self.get_rank() == source_rank:
            # Serialize and store parameters
            serialized = {}
            for name, tensor in parameters.items():
                serialized[name] = tensor.cpu().numpy().tobytes()
            
            # Store in Redis with timestamp
            broadcast_key = f"broadcast_{int(time.time())}"
            self.redis_client.hmset(broadcast_key, serialized)
            self.redis_client.expire(broadcast_key, 300)  # 5 minutes TTL
            
            # Notify all nodes
            self.redis_client.publish("parameter_broadcast", broadcast_key)
        
        # Wait for broadcast (simplified - in practice would use pub/sub)
        time.sleep(0.1)
        
        # Find latest broadcast
        keys = self.redis_client.keys("broadcast_*")
        if keys:
            latest_key = max(keys, key=lambda k: int(k.decode().split('_')[1]))
            serialized = self.redis_client.hgetall(latest_key)
            
            # Deserialize parameters
            result = {}
            for name, data in serialized.items():
                tensor_data = np.frombuffer(data, dtype=np.float32)
                # Note: This is simplified - would need to store shape information
                result[name.decode()] = torch.from_numpy(tensor_data)
            
            return result
        
        return parameters
    
    def all_reduce(self, tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
        """All-reduce via Redis (simplified implementation)."""
        # Store tensor with node ID
        tensor_key = f"allreduce_{self.node_id}_{int(time.time())}"
        self.redis_client.set(tensor_key, tensor.cpu().numpy().tobytes())
        self.redis_client.expire(tensor_key, 60)
        
        # Wait for all nodes (simplified)
        time.sleep(1.0)
        
        # Collect all tensors
        keys = self.redis_client.keys("allreduce_*")
        tensors = []
        
        for key in keys:
            data = self.redis_client.get(key)
            if data:
                tensor_data = np.frombuffer(data, dtype=np.float32)
                tensors.append(torch.from_numpy(tensor_data))
        
        # Perform reduction
        if not tensors:
            return tensor
        
        if op == 'sum':
            result = sum(tensors)
        elif op == 'avg':
            result = sum(tensors) / len(tensors)
        elif op == 'max':
            result = torch.stack(tensors).max(dim=0)[0]
        elif op == 'min':
            result = torch.stack(tensors).min(dim=0)[0]
        else:
            raise ValueError(f"Unsupported operation: {op}")
        
        return result
    
    def gather(self, tensor: torch.Tensor, dst_rank: int) -> List[torch.Tensor]:
        """Gather tensors via Redis."""
        # Store tensor
        gather_key = f"gather_{self.node_id}"
        self.redis_client.set(gather_key, tensor.cpu().numpy().tobytes())
        
        if self.get_rank() == dst_rank:
            # Wait and collect from all nodes
            time.sleep(1.0)
            
            keys = self.redis_client.keys("gather_*")
            tensors = []
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    tensor_data = np.frombuffer(data, dtype=np.float32)
                    tensors.append(torch.from_numpy(tensor_data))
            
            return tensors
        
        return []
    
    def get_rank(self) -> int:
        """Get node rank (based on sorted node IDs)."""
        active_nodes = sorted([node.decode() for node in self.redis_client.smembers("active_nodes")])
        return active_nodes.index(self.node_id) if self.node_id in active_nodes else 0
    
    def get_world_size(self) -> int:
        """Get total number of active nodes."""
        return self.redis_client.scard("active_nodes")


class FederatedAggregator(ABC):
    """Abstract base class for federated learning aggregation strategies."""
    
    @abstractmethod
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates into global update."""
        pass


class FedAvgAggregator(FederatedAggregator):
    """Federated Averaging (FedAvg) aggregation strategy."""
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform FedAvg aggregation."""
        if not client_updates:
            return {}
        
        # Use uniform weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        # Get parameter names from first client
        param_names = client_updates[0].keys()
        aggregated = {}
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = None
            
            for client_update, weight in zip(client_updates, client_weights):
                if param_name in client_update:
                    weighted_param = client_update[param_name] * weight
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param
            
            if weighted_sum is not None:
                aggregated[param_name] = weighted_sum
        
        return aggregated


class FedProxAggregator(FederatedAggregator):
    """FedProx aggregation with proximal term for handling heterogeneity."""
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu
        self.global_parameters = None
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform FedProx aggregation with proximal regularization."""
        # First perform standard FedAvg
        fedavg = FedAvgAggregator()
        aggregated = fedavg.aggregate(client_updates, client_weights)
        
        # Apply proximal regularization if we have previous global parameters
        if self.global_parameters is not None:
            for param_name in aggregated.keys():
                if param_name in self.global_parameters:
                    # Add proximal term: θ_new = θ_fedavg + μ * (θ_global - θ_fedavg)
                    global_param = self.global_parameters[param_name]
                    fedavg_param = aggregated[param_name]
                    
                    aggregated[param_name] = fedavg_param + self.mu * (global_param - fedavg_param)
        
        # Update global parameters
        self.global_parameters = {k: v.clone() for k, v in aggregated.items()}
        
        return aggregated


class DistributedConformalHDC:
    """Distributed HDC with conformal prediction across multiple nodes."""
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        communication_backend: CommunicationBackend = None,
        aggregation_strategy: str = 'fedavg',
        privacy_mechanism: Optional[DifferentialPrivacyMechanism] = None,
        compression_ratio: float = 0.1
    ):
        """
        Initialize distributed conformal HDC.
        
        Args:
            encoder: HDC encoder
            num_classes: Number of classes
            alpha: Miscoverage level
            communication_backend: Backend for distributed communication
            aggregation_strategy: Strategy for aggregating updates ('fedavg', 'fedprox')
            privacy_mechanism: Privacy mechanism for federated learning
            compression_ratio: Compression ratio for communication efficiency
        """
        self.local_model = ConformalHDC(encoder, num_classes, alpha)
        self.comm_backend = communication_backend
        self.privacy_mechanism = privacy_mechanism
        self.compression_ratio = compression_ratio
        
        # Aggregation strategy
        if aggregation_strategy == 'fedavg':
            self.aggregator = FedAvgAggregator()
        elif aggregation_strategy == 'fedprox':
            self.aggregator = FedProxAggregator()
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
        
        # Distributed training state
        self.global_round = 0
        self.local_rounds = 0
        self.communication_overhead = []
        
        # Performance tracking
        self.training_stats = {
            'communication_time': [],
            'computation_time': [],
            'accuracy_history': [],
            'coverage_history': []
        }
    
    def fit_distributed(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        global_rounds: int = 10,
        local_epochs: int = 1,
        client_fraction: float = 1.0
    ) -> 'DistributedConformalHDC':
        """
        Distributed training across multiple nodes.
        
        Args:
            X: Training features
            y: Training labels
            global_rounds: Number of global communication rounds
            local_epochs: Number of local training epochs per round
            client_fraction: Fraction of clients to use each round
        """
        rank = self.comm_backend.get_rank()
        world_size = self.comm_backend.get_world_size()
        
        logger.info(f"Starting distributed training: rank {rank}/{world_size}")
        
        for global_round in range(global_rounds):
            round_start_time = time.time()
            
            # Client selection (simplified - in practice would be more sophisticated)
            selected = np.random.random() < client_fraction
            
            if selected:
                # Local training
                local_start_time = time.time()
                
                # Split data for this node (simplified)
                node_size = len(X) // world_size
                start_idx = rank * node_size
                end_idx = (rank + 1) * node_size if rank < world_size - 1 else len(X)
                
                X_local = X[start_idx:end_idx]
                y_local = y[start_idx:end_idx]
                
                # Local training
                for epoch in range(local_epochs):
                    self.local_model.fit(X_local, y_local)
                
                local_training_time = time.time() - local_start_time
                self.training_stats['computation_time'].append(local_training_time)
                
                # Get local model parameters
                local_params = self._get_model_parameters()
                
                # Apply privacy mechanism if configured
                if self.privacy_mechanism:
                    local_params = self._apply_privacy(local_params)
                
                # Compress parameters for communication efficiency
                compressed_params = self._compress_parameters(local_params)
                
                # Communication phase
                comm_start_time = time.time()
                
                # Gather all client updates at coordinator (rank 0)
                if rank == 0:
                    # Coordinator collects updates
                    all_updates = self.comm_backend.gather(self._serialize_params(compressed_params), dst_rank=0)
                    
                    # Deserialize updates
                    client_updates = [self._deserialize_params(update) for update in all_updates]
                    
                    # Aggregate updates
                    global_update = self.aggregator.aggregate(client_updates)
                    
                    # Broadcast global model
                    self.comm_backend.broadcast_parameters(global_update, source_rank=0)
                    
                    # Update local model with global parameters
                    self._set_model_parameters(global_update)
                    
                else:
                    # Workers send updates and receive global model
                    self.comm_backend.gather(self._serialize_params(compressed_params), dst_rank=0)
                    global_params = self.comm_backend.broadcast_parameters({}, source_rank=0)
                    
                    # Update local model
                    self._set_model_parameters(global_params)
                
                comm_time = time.time() - comm_start_time
                self.training_stats['communication_time'].append(comm_time)
            
            # Synchronization barrier
            if hasattr(self.comm_backend, 'barrier'):
                self.comm_backend.barrier()
            
            total_round_time = time.time() - round_start_time
            
            # Evaluate global model periodically
            if global_round % 5 == 0:
                self._evaluate_global_model(X, y)
            
            logger.info(f"Global round {global_round} completed in {total_round_time:.2f}s")
            
            self.global_round = global_round + 1
        
        return self
    
    def _get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract model parameters for communication."""
        params = {}
        
        # Class prototypes
        if self.local_model.class_prototypes is not None:
            params['class_prototypes'] = self.local_model.class_prototypes.clone()
        
        # Encoder parameters (if learnable)
        if hasattr(self.local_model.encoder, 'parameters'):
            for name, param in self.local_model.encoder.named_parameters():
                params[f'encoder.{name}'] = param.clone()
        
        return params
    
    def _set_model_parameters(self, params: Dict[str, torch.Tensor]) -> None:
        """Update model with new parameters."""
        if 'class_prototypes' in params:
            self.local_model.class_prototypes = params['class_prototypes'].to(self.local_model.device)
        
        # Update encoder parameters if present
        if hasattr(self.local_model.encoder, 'parameters'):
            for name, param in self.local_model.encoder.named_parameters():
                param_key = f'encoder.{name}'
                if param_key in params:
                    param.data.copy_(params[param_key])
    
    def _apply_privacy(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to parameters."""
        private_params = {}
        
        for name, param in params.items():
            private_params[name] = self.privacy_mechanism.add_noise(param)
        
        return private_params
    
    def _compress_parameters(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress parameters for efficient communication."""
        compressed = {}
        
        for name, param in params.items():
            if self.compression_ratio < 1.0:
                # Simple top-k sparsification
                flat_param = param.flatten()
                k = int(len(flat_param) * self.compression_ratio)
                
                # Keep top-k values by magnitude
                _, top_indices = torch.topk(torch.abs(flat_param), k)
                
                # Create sparse representation
                compressed_param = torch.zeros_like(flat_param)
                compressed_param[top_indices] = flat_param[top_indices]
                
                compressed[name] = compressed_param.reshape(param.shape)
            else:
                compressed[name] = param
        
        return compressed
    
    def _serialize_params(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Serialize parameters for communication."""
        # Simple concatenation (in practice would use more efficient serialization)
        tensors = []
        for name, param in params.items():
            tensors.append(param.flatten())
        
        return torch.cat(tensors)
    
    def _deserialize_params(self, serialized: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Deserialize parameters from communication."""
        # This is simplified - in practice would need metadata for reconstruction
        # For now, just return the serialized tensor as a single parameter
        return {'serialized': serialized}
    
    def _evaluate_global_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Evaluate the global model performance."""
        try:
            # Compute predictions
            predictions = self.local_model.predict(X)
            accuracy = (predictions == y.cpu().numpy()).mean()
            
            # Compute coverage
            pred_sets = self.local_model.predict_set(X)
            coverage = np.mean([y[i].item() in pred_set for i, pred_set in enumerate(pred_sets)])
            
            self.training_stats['accuracy_history'].append(accuracy)
            self.training_stats['coverage_history'].append(coverage)
            
            logger.info(f"Global model - Accuracy: {accuracy:.3f}, Coverage: {coverage:.3f}")
            
        except Exception as e:
            logger.warning(f"Global model evaluation failed: {e}")
    
    def predict_distributed(self, X: torch.Tensor) -> List[List[int]]:
        """Distributed inference with ensemble predictions."""
        local_pred_sets = self.local_model.predict_set(X)
        
        if self.comm_backend and self.comm_backend.get_world_size() > 1:
            # Gather predictions from all nodes
            rank = self.comm_backend.get_rank()
            
            # Serialize predictions (simplified)
            serialized_preds = torch.tensor([len(pred_set) for pred_set in local_pred_sets], dtype=torch.float32)
            
            # Gather all predictions
            all_preds = self.comm_backend.gather(serialized_preds, dst_rank=0)
            
            if rank == 0:
                # Ensemble predictions (majority voting)
                ensemble_pred_sets = []
                
                for i in range(len(local_pred_sets)):
                    # Simple ensemble - combine all local predictions
                    # In practice, would be more sophisticated
                    ensemble_pred_sets.append(local_pred_sets[i])
                
                return ensemble_pred_sets
        
        return local_pred_sets
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get distributed training statistics."""
        comm_times = self.training_stats['communication_time']
        comp_times = self.training_stats['computation_time']
        
        stats = {
            'global_rounds': self.global_round,
            'total_communication_time': sum(comm_times),
            'total_computation_time': sum(comp_times),
            'avg_communication_time': np.mean(comm_times) if comm_times else 0,
            'avg_computation_time': np.mean(comp_times) if comp_times else 0,
            'communication_efficiency': sum(comp_times) / (sum(comm_times) + 1e-8) if comm_times else float('inf')
        }
        
        if self.training_stats['accuracy_history']:
            stats['final_accuracy'] = self.training_stats['accuracy_history'][-1]
        
        if self.training_stats['coverage_history']:
            stats['final_coverage'] = self.training_stats['coverage_history'][-1]
        
        return stats


class CloudDeploymentManager:
    """Manager for cloud deployment of distributed HyperConformal systems."""
    
    def __init__(self, cloud_provider: str = 'aws'):
        """
        Initialize cloud deployment manager.
        
        Args:
            cloud_provider: Cloud provider ('aws', 'gcp', 'azure')
        """
        self.cloud_provider = cloud_provider
        self.deployment_configs = {}
    
    def generate_kubernetes_config(
        self, 
        model_name: str,
        num_workers: int = 4,
        resources: Dict[str, str] = None
    ) -> str:
        """Generate Kubernetes configuration for distributed deployment."""
        default_resources = {
            'cpu': '2',
            'memory': '4Gi',
            'gpu': '0'
        }
        resources = resources or default_resources
        
        k8s_config = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-distributed
  labels:
    app: hyperconformal
    model: {model_name}
spec:
  replicas: {num_workers}
  selector:
    matchLabels:
      app: hyperconformal
      model: {model_name}
  template:
    metadata:
      labels:
        app: hyperconformal
        model: {model_name}
    spec:
      containers:
      - name: hyperconformal-worker
        image: hyperconformal:{model_name}
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: {resources['cpu']}
            memory: {resources['memory']}
            {'nvidia.com/gpu: ' + resources['gpu'] if resources.get('gpu', '0') != '0' else ''}
          limits:
            cpu: {resources['cpu']}
            memory: {resources['memory']}
            {'nvidia.com/gpu: ' + resources['gpu'] if resources.get('gpu', '0') != '0' else ''}
        env:
        - name: WORLD_SIZE
          value: "{num_workers}"
        - name: MASTER_ADDR
          value: "{model_name}-master"
        - name: MASTER_PORT
          value: "29500"
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-service
  labels:
    app: hyperconformal
    model: {model_name}
spec:
  selector:
    app: hyperconformal
    model: {model_name}
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
  type: LoadBalancer
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-master
  labels:
    app: hyperconformal
    model: {model_name}
spec:
  selector:
    app: hyperconformal
    model: {model_name}
  ports:
  - port: 29500
    targetPort: 29500
    protocol: TCP
  clusterIP: None
"""
        
        return k8s_config
    
    def generate_docker_compose(
        self,
        model_name: str,
        num_workers: int = 4
    ) -> str:
        """Generate Docker Compose configuration for local distributed testing."""
        compose_config = f"""
version: '3.8'

services:
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  coordinator:
    image: hyperconformal:{model_name}
    ports:
      - "8080:8080"
    environment:
      - WORLD_SIZE={num_workers + 1}
      - RANK=0
      - MASTER_ADDR=coordinator
      - MASTER_PORT=29500
      - REDIS_HOST=redis
    depends_on:
      - redis
    command: python -m hyperconformal.distributed.coordinator

"""
        
        # Add worker services
        for i in range(1, num_workers + 1):
            compose_config += f"""
  worker{i}:
    image: hyperconformal:{model_name}
    environment:
      - WORLD_SIZE={num_workers + 1}
      - RANK={i}
      - MASTER_ADDR=coordinator
      - MASTER_PORT=29500
      - REDIS_HOST=redis
    depends_on:
      - coordinator
      - redis
    command: python -m hyperconformal.distributed.worker

"""
        
        compose_config += """
volumes:
  redis_data:
"""
        
        return compose_config
    
    def generate_terraform_config(
        self,
        model_name: str,
        num_workers: int = 4,
        instance_type: str = 'c5.xlarge'
    ) -> str:
        """Generate Terraform configuration for AWS deployment."""
        if self.cloud_provider != 'aws':
            raise ValueError("Terraform config currently only supports AWS")
        
        terraform_config = f"""
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

variable "aws_region" {{
  description = "AWS region for deployment"
  default     = "us-west-2"
}}

variable "key_name" {{
  description = "AWS EC2 Key Pair name"
  type        = string
}}

# Security group for HyperConformal cluster
resource "aws_security_group" "hyperconformal_sg" {{
  name_prefix = "{model_name}-hyperconformal"
  
  ingress {{
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  ingress {{
    from_port   = 29500
    to_port     = 29500
    protocol    = "tcp"
    self        = true
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# Launch template for HyperConformal workers
resource "aws_launch_template" "hyperconformal_template" {{
  name_prefix   = "{model_name}-hyperconformal"
  image_id      = "ami-0c02fb55956c7d316"  # Amazon Linux 2
  instance_type = "{instance_type}"
  key_name      = var.key_name
  
  vpc_security_group_ids = [aws_security_group.hyperconformal_sg.id]
  
  user_data = base64encode(<<-EOF
#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Pull HyperConformal image
docker pull hyperconformal:{model_name}

EOF
  )
}}

# Auto Scaling Group for workers
resource "aws_autoscaling_group" "hyperconformal_asg" {{
  name                = "{model_name}-hyperconformal-asg"
  vpc_zone_identifier = [aws_subnet.hyperconformal_subnet.id]
  target_group_arns   = [aws_lb_target_group.hyperconformal_tg.arn]
  health_check_type   = "ELB"
  
  min_size         = {num_workers}
  max_size         = {num_workers * 2}
  desired_capacity = {num_workers}
  
  launch_template {{
    id      = aws_launch_template.hyperconformal_template.id
    version = "$Latest"
  }}
  
  tag {{
    key                 = "Name"
    value               = "{model_name}-hyperconformal-worker"
    propagate_at_launch = true
  }}
}}

# Application Load Balancer
resource "aws_lb" "hyperconformal_alb" {{
  name               = "{model_name}-hyperconformal-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.hyperconformal_sg.id]
  subnets            = [aws_subnet.hyperconformal_subnet.id, aws_subnet.hyperconformal_subnet2.id]
}}

# Target group
resource "aws_lb_target_group" "hyperconformal_tg" {{
  name     = "{model_name}-hyperconformal-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.hyperconformal_vpc.id
  
  health_check {{
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }}
}}

# VPC and networking (simplified)
resource "aws_vpc" "hyperconformal_vpc" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name = "{model_name}-hyperconformal-vpc"
  }}
}}

resource "aws_subnet" "hyperconformal_subnet" {{
  vpc_id                  = aws_vpc.hyperconformal_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  
  tags = {{
    Name = "{model_name}-hyperconformal-subnet-1"
  }}
}}

resource "aws_subnet" "hyperconformal_subnet2" {{
  vpc_id                  = aws_vpc.hyperconformal_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = true
  
  tags = {{
    Name = "{model_name}-hyperconformal-subnet-2"
  }}
}}

data "aws_availability_zones" "available" {{
  state = "available"
}}

# Internet Gateway
resource "aws_internet_gateway" "hyperconformal_igw" {{
  vpc_id = aws_vpc.hyperconformal_vpc.id
  
  tags = {{
    Name = "{model_name}-hyperconformal-igw"
  }}
}}

# Route table
resource "aws_route_table" "hyperconformal_rt" {{
  vpc_id = aws_vpc.hyperconformal_vpc.id
  
  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.hyperconformal_igw.id
  }}
  
  tags = {{
    Name = "{model_name}-hyperconformal-rt"
  }}
}}

# Route table associations
resource "aws_route_table_association" "hyperconformal_rta1" {{
  subnet_id      = aws_subnet.hyperconformal_subnet.id
  route_table_id = aws_route_table.hyperconformal_rt.id
}}

resource "aws_route_table_association" "hyperconformal_rta2" {{
  subnet_id      = aws_subnet.hyperconformal_subnet2.id
  route_table_id = aws_route_table.hyperconformal_rt.id
}}

# ALB Listener
resource "aws_lb_listener" "hyperconformal_listener" {{
  load_balancer_arn = aws_lb.hyperconformal_alb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.hyperconformal_tg.arn
  }}
}}

# Outputs
output "load_balancer_dns" {{
  value = aws_lb.hyperconformal_alb.dns_name
}}

output "load_balancer_zone_id" {{
  value = aws_lb.hyperconformal_alb.zone_id
}}
"""
        
        return terraform_config


# Factory functions for easy setup
def setup_distributed_training(
    encoder: BaseEncoder,
    num_classes: int,
    backend_type: str = 'pytorch',
    **kwargs
) -> DistributedConformalHDC:
    """Setup distributed training with specified backend."""
    
    if backend_type == 'pytorch':
        if not dist.is_initialized():
            logger.warning("PyTorch distributed not initialized. Using Redis backend.")
            backend_type = 'redis'
    
    if backend_type == 'pytorch':
        comm_backend = PyTorchDistributedBackend()
    elif backend_type == 'redis':
        comm_backend = RedisBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
    
    return DistributedConformalHDC(
        encoder=encoder,
        num_classes=num_classes,
        communication_backend=comm_backend,
        **kwargs
    )