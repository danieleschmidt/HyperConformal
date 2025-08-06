"""
Advanced optimization techniques for HyperConformal systems.

This module provides performance optimizations including:
- GPU acceleration and CUDA kernels
- Memory optimization and caching
- Batch processing optimizations
- Model compression and quantization
- Edge deployment optimizations
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .encoders import BaseEncoder
from .hyperconformal import ConformalHDC

logger = logging.getLogger(__name__)

# Optional dependencies for optimization
try:
    import numba
    from numba import cuda, jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available. Some optimizations will use fallback implementations.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class MemoryOptimizer:
    """Memory optimization utilities for large-scale deployments."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    @lru_cache(maxsize=1000)
    def cached_hamming_distance(self, hv1_hash: str, hv2_hash: str, hv1_data: bytes, hv2_data: bytes) -> int:
        """Cached Hamming distance computation."""
        hv1 = np.frombuffer(hv1_data, dtype=np.uint8)
        hv2 = np.frombuffer(hv2_data, dtype=np.uint8)
        
        # Fast XOR + popcount
        xor_result = np.bitwise_xor(hv1, hv2)
        return int(np.unpackbits(xor_result).sum())
    
    def optimize_tensor_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory usage through quantization and sparsification."""
        # Convert to half precision if beneficial
        if tensor.dtype == torch.float32 and tensor.numel() > 10000:
            if torch.cuda.is_available():
                # Use half precision on GPU
                return tensor.half()
            else:
                # Keep float32 for CPU (half precision slower)
                return tensor
        
        # Sparsify very small values
        if tensor.dtype in [torch.float32, torch.float16]:
            threshold = torch.std(tensor) * 0.01  # 1% of std dev
            mask = torch.abs(tensor) > threshold
            sparse_tensor = tensor * mask.float()
            
            # Only return sparse version if it saves significant memory
            sparsity = 1.0 - mask.float().mean().item()
            if sparsity > 0.7:  # >70% sparse
                return sparse_tensor.to_sparse()
        
        return tensor
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        import psutil
        process = psutil.Process()
        
        memory_info = {
            'rss_gb': process.memory_info().rss / 1024**3,
            'vms_gb': process.memory_info().vms / 1024**3,
            'percent': process.memory_percent()
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_cached_gb': torch.cuda.memory_reserved() / 1024**3
            })
        
        return memory_info
    
    def clear_memory(self, aggressive: bool = False):
        """Clear memory caches and run garbage collection."""
        # Clear Python caches
        self.cached_hamming_distance.cache_clear()
        self.memory_cache.clear()
        
        # Garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if aggressive:
                # Force GPU memory cleanup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        logger.info("Memory cleared")


class CUDAKernels:
    """Custom CUDA kernels for optimized HDC operations."""
    
    def __init__(self):
        if not NUMBA_AVAILABLE or not cuda.is_available():
            raise RuntimeError("CUDA kernels require Numba with CUDA support")
        
        self.block_size = 256
        self.kernels_compiled = False
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile CUDA kernels."""
        
        @cuda.jit
        def binary_encode_kernel(input_data, projection_matrix, output, n_samples, input_dim, hv_dim):
            """CUDA kernel for binary HDC encoding."""
            # Get thread indices
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            
            sample_idx = bx
            
            if sample_idx < n_samples:
                # Each thread processes multiple output dimensions
                threads_per_block = cuda.blockDim.x
                output_per_thread = (hv_dim + threads_per_block - 1) // threads_per_block
                
                for i in range(output_per_thread):
                    hv_idx = tx * output_per_thread + i
                    
                    if hv_idx < hv_dim:
                        result = 0
                        
                        # Compute dot product with projection matrix row
                        for j in range(input_dim):
                            if input_data[sample_idx * input_dim + j] > 0:
                                result ^= projection_matrix[j * hv_dim + hv_idx]
                        
                        output[sample_idx * hv_dim + hv_idx] = result
        
        @cuda.jit
        def hamming_distance_kernel(hv1, hv2, distances, n_pairs, hv_dim):
            """CUDA kernel for batch Hamming distance computation."""
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            
            pair_idx = bx * cuda.blockDim.x + tx
            
            if pair_idx < n_pairs:
                distance = 0
                
                # Compute Hamming distance
                for i in range(hv_dim):
                    if hv1[pair_idx * hv_dim + i] != hv2[pair_idx * hv_dim + i]:
                        distance += 1
                
                distances[pair_idx] = distance
        
        @cuda.jit
        def cosine_similarity_kernel(hv1, hv2, similarities, n_pairs, hv_dim):
            """CUDA kernel for batch cosine similarity computation."""
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            
            pair_idx = bx * cuda.blockDim.x + tx
            
            if pair_idx < n_pairs:
                dot_product = 0.0
                norm1_sq = 0.0
                norm2_sq = 0.0
                
                for i in range(hv_dim):
                    val1 = hv1[pair_idx * hv_dim + i]
                    val2 = hv2[pair_idx * hv_dim + i]
                    
                    dot_product += val1 * val2
                    norm1_sq += val1 * val1
                    norm2_sq += val2 * val2
                
                # Avoid division by zero
                if norm1_sq > 0 and norm2_sq > 0:
                    similarities[pair_idx] = dot_product / (norm1_sq * norm2_sq) ** 0.5
                else:
                    similarities[pair_idx] = 0.0
        
        # Store compiled kernels
        self.binary_encode_kernel = binary_encode_kernel
        self.hamming_distance_kernel = hamming_distance_kernel
        self.cosine_similarity_kernel = cosine_similarity_kernel
        
        self.kernels_compiled = True
        logger.info("CUDA kernels compiled successfully")
    
    def batch_binary_encode(
        self, 
        input_data: torch.Tensor, 
        projection_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Optimized batch binary encoding using CUDA."""
        n_samples, input_dim = input_data.shape
        hv_dim = projection_matrix.shape[1]
        
        # Allocate output
        output = torch.zeros(n_samples, hv_dim, dtype=torch.uint8, device=input_data.device)
        
        # Launch kernel
        blocks_per_grid = n_samples
        threads_per_block = min(self.block_size, hv_dim)
        
        # Convert to CUDA arrays
        d_input = cuda.as_cuda_array(input_data)
        d_projection = cuda.as_cuda_array(projection_matrix)
        d_output = cuda.as_cuda_array(output)
        
        self.binary_encode_kernel[blocks_per_grid, threads_per_block](
            d_input, d_projection, d_output, n_samples, input_dim, hv_dim
        )
        
        return output
    
    def batch_hamming_distance(
        self, 
        hv1: torch.Tensor, 
        hv2: torch.Tensor
    ) -> torch.Tensor:
        """Optimized batch Hamming distance using CUDA."""
        n_pairs = hv1.shape[0]
        hv_dim = hv1.shape[1]
        
        # Allocate output
        distances = torch.zeros(n_pairs, dtype=torch.int32, device=hv1.device)
        
        # Launch kernel
        threads_per_block = min(self.block_size, n_pairs)
        blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block
        
        # Convert to CUDA arrays
        d_hv1 = cuda.as_cuda_array(hv1)
        d_hv2 = cuda.as_cuda_array(hv2)
        d_distances = cuda.as_cuda_array(distances)
        
        self.hamming_distance_kernel[blocks_per_grid, threads_per_block](
            d_hv1, d_hv2, d_distances, n_pairs, hv_dim
        )
        
        return distances


class OptimizedEncoder(BaseEncoder):
    """Highly optimized HDC encoder with multiple acceleration backends."""
    
    def __init__(
        self,
        input_dim: int,
        hv_dim: int = 10000,
        quantization: str = 'binary',
        acceleration: str = 'auto',
        use_memory_mapping: bool = True,
        num_workers: int = None
    ):
        """
        Initialize optimized encoder.
        
        Args:
            input_dim: Input dimension
            hv_dim: Hypervector dimension
            quantization: Quantization scheme
            acceleration: Acceleration backend ('auto', 'cuda', 'cpu', 'numba')
            use_memory_mapping: Whether to use memory mapping for large matrices
            num_workers: Number of worker processes for CPU parallelism
        """
        super().__init__(input_dim, hv_dim, quantization)
        
        self.acceleration = self._select_acceleration_backend(acceleration)
        self.use_memory_mapping = use_memory_mapping
        self.num_workers = num_workers or mp.cpu_count()
        
        # Initialize acceleration-specific components
        if self.acceleration == 'cuda' and NUMBA_AVAILABLE and cuda.is_available():
            self.cuda_kernels = CUDAKernels()
        
        # Initialize projection matrix with optimization
        self._init_optimized_projection()
        
        # Performance tracking
        self.perf_stats = {
            'encode_times': [],
            'similarity_times': [],
            'memory_usage': []
        }
    
    def _select_acceleration_backend(self, requested: str) -> str:
        """Select the best available acceleration backend."""
        if requested == 'auto':
            if torch.cuda.is_available() and NUMBA_AVAILABLE:
                return 'cuda'
            elif NUMBA_AVAILABLE:
                return 'numba'
            else:
                return 'cpu'
        
        # Validate requested backend
        if requested == 'cuda' and not (torch.cuda.is_available() and NUMBA_AVAILABLE):
            logger.warning("CUDA backend requested but not available, falling back to CPU")
            return 'cpu'
        
        if requested == 'numba' and not NUMBA_AVAILABLE:
            logger.warning("Numba backend requested but not available, falling back to CPU")
            return 'cpu'
        
        return requested
    
    def _init_optimized_projection(self):
        """Initialize projection matrix with memory optimizations."""
        if self.use_memory_mapping and self.input_dim * self.hv_dim > 1e8:
            # Use memory-mapped array for very large matrices
            import tempfile
            
            # Create temporary file
            self.projection_file = tempfile.NamedTemporaryFile(delete=False)
            
            # Create memory-mapped array
            self.projection_matrix = np.memmap(
                self.projection_file.name,
                dtype=np.float32,
                mode='w+',
                shape=(self.input_dim, self.hv_dim)
            )
            
            # Initialize with random values
            chunk_size = 10000
            for i in range(0, self.input_dim, chunk_size):
                end_i = min(i + chunk_size, self.input_dim)
                self.projection_matrix[i:end_i] = np.random.randn(
                    end_i - i, self.hv_dim
                ).astype(np.float32)
            
            logger.info(f"Using memory-mapped projection matrix: {self.projection_file.name}")
        else:
            # Standard in-memory matrix
            self.projection_matrix = torch.randn(
                self.input_dim, self.hv_dim, 
                dtype=torch.float32
            )
            
            if torch.cuda.is_available():
                self.projection_matrix = self.projection_matrix.cuda()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized encoding with multiple acceleration backends."""
        start_time = time.time()
        
        batch_size = x.shape[0]
        
        # Select encoding method based on acceleration backend
        if self.acceleration == 'cuda' and hasattr(self, 'cuda_kernels'):
            result = self._cuda_encode(x)
        elif self.acceleration == 'numba':
            result = self._numba_encode(x)
        else:
            result = self._cpu_encode(x)
        
        # Apply quantization
        if self.quantization == 'binary':
            result = torch.sign(result)
        elif self.quantization == 'ternary':
            # Three-level quantization
            std = torch.std(result, dim=1, keepdim=True)
            result = torch.where(result > std, torch.ones_like(result),
                               torch.where(result < -std, -torch.ones_like(result), 
                                          torch.zeros_like(result)))
        
        # Track performance
        encode_time = time.time() - start_time
        self.perf_stats['encode_times'].append(encode_time)
        
        if len(self.perf_stats['encode_times']) % 100 == 0:
            avg_time = np.mean(self.perf_stats['encode_times'][-100:])
            throughput = batch_size / avg_time
            logger.debug(f"Encoding throughput: {throughput:.1f} samples/sec")
        
        return result
    
    def _cuda_encode(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA-accelerated encoding."""
        if isinstance(self.projection_matrix, np.memmap):
            # Convert memmap to CUDA tensor (in chunks to save memory)
            device = x.device
            chunk_size = 10000
            results = []
            
            for i in range(0, self.input_dim, chunk_size):
                end_i = min(i + chunk_size, self.input_dim)
                proj_chunk = torch.from_numpy(
                    self.projection_matrix[i:end_i]
                ).to(device)
                
                x_chunk = x[:, i:end_i]
                result_chunk = torch.mm(x_chunk, proj_chunk)
                results.append(result_chunk)
            
            return torch.cat(results, dim=1)
        else:
            # Direct CUDA matrix multiplication
            return torch.mm(x, self.projection_matrix)
    
    def _numba_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Numba-accelerated encoding."""
        if not NUMBA_AVAILABLE:
            return self._cpu_encode(x)
        
        # Convert to numpy for Numba processing
        x_np = x.cpu().numpy().astype(np.float32)
        
        if isinstance(self.projection_matrix, np.memmap):
            proj_np = self.projection_matrix
        else:
            proj_np = self.projection_matrix.cpu().numpy()
        
        # Use compiled Numba function
        result_np = self._numba_matrix_multiply(x_np, proj_np)
        
        # Convert back to tensor
        return torch.from_numpy(result_np).to(x.device)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _numba_matrix_multiply(x: np.ndarray, proj: np.ndarray) -> np.ndarray:
        """Numba-compiled matrix multiplication."""
        batch_size, input_dim = x.shape
        hv_dim = proj.shape[1]
        
        result = np.zeros((batch_size, hv_dim), dtype=np.float32)
        
        for i in prange(batch_size):
            for j in prange(hv_dim):
                for k in range(input_dim):
                    result[i, j] += x[i, k] * proj[k, j]
        
        return result
    
    def _cpu_encode(self, x: torch.Tensor) -> torch.Tensor:
        """CPU-optimized encoding with parallel processing."""
        if isinstance(self.projection_matrix, np.memmap):
            # Process in parallel chunks for memory-mapped matrices
            def encode_chunk(chunk_data):
                chunk_x, start_idx, end_idx = chunk_data
                proj_chunk = torch.from_numpy(
                    self.projection_matrix[start_idx:end_idx]
                ).to(x.device)
                
                x_chunk = chunk_x[:, start_idx:end_idx]
                return torch.mm(x_chunk, proj_chunk)
            
            # Create chunks
            chunk_size = max(1000, self.input_dim // self.num_workers)
            chunks = []
            
            for i in range(0, self.input_dim, chunk_size):
                end_i = min(i + chunk_size, self.input_dim)
                chunks.append((x, i, end_i))
            
            # Process chunks in parallel
            if len(chunks) > 1 and self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(encode_chunk, chunks))
                
                return torch.cat(results, dim=1)
            else:
                # Sequential processing for small matrices
                return encode_chunk(chunks[0])
        else:
            # Standard PyTorch matrix multiplication
            return torch.mm(x, self.projection_matrix)
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """Optimized similarity computation."""
        start_time = time.time()
        
        if self.acceleration == 'cuda' and hasattr(self, 'cuda_kernels'):
            # Use CUDA kernel for batch similarity
            if hv1.shape[0] == hv2.shape[0] and len(hv1.shape) == 2:
                similarities = self.cuda_kernels.batch_cosine_similarity(hv1, hv2)
            else:
                similarities = F.cosine_similarity(hv1, hv2, dim=-1)
        else:
            # Optimized CPU similarity
            if self.quantization == 'binary':
                # Use Hamming distance for binary vectors
                hamming_dist = torch.sum(hv1 != hv2, dim=-1).float()
                similarities = 1.0 - hamming_dist / hv1.shape[-1]
            else:
                similarities = F.cosine_similarity(hv1, hv2, dim=-1)
        
        # Track performance
        similarity_time = time.time() - start_time
        self.perf_stats['similarity_times'].append(similarity_time)
        
        return similarities
    
    def memory_footprint(self) -> int:
        """Get optimized memory footprint."""
        if isinstance(self.projection_matrix, np.memmap):
            # Memory-mapped matrix doesn't count toward RAM
            return self.input_dim * 4  # Just the memory-map overhead
        else:
            return self.projection_matrix.numel() * self.projection_matrix.element_size()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        encode_times = self.perf_stats['encode_times']
        similarity_times = self.perf_stats['similarity_times']
        
        stats = {
            'acceleration_backend': self.acceleration,
            'total_encodings': len(encode_times),
            'total_similarities': len(similarity_times),
            'memory_mapping_used': isinstance(self.projection_matrix, np.memmap)
        }
        
        if encode_times:
            stats.update({
                'avg_encode_time': np.mean(encode_times),
                'encode_throughput_samples_per_sec': 1.0 / np.mean(encode_times) if encode_times else 0
            })
        
        if similarity_times:
            stats.update({
                'avg_similarity_time': np.mean(similarity_times),
                'similarity_throughput_ops_per_sec': 1.0 / np.mean(similarity_times) if similarity_times else 0
            })
        
        return stats


class BatchProcessor:
    """Optimized batch processing for high-throughput scenarios."""
    
    def __init__(
        self,
        model: ConformalHDC,
        batch_size: int = 1000,
        prefetch_factor: int = 2,
        num_workers: int = 4
    ):
        """
        Initialize batch processor.
        
        Args:
            model: HyperConformal model
            batch_size: Optimal batch size for processing
            prefetch_factor: Number of batches to prefetch
            num_workers: Number of worker threads
        """
        self.model = model
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.memory_optimizer = MemoryOptimizer()
    
    def process_stream(
        self,
        data_stream: Iterator[np.ndarray],
        callback: Callable[[List[List[int]], Dict], None] = None
    ) -> Iterator[Tuple[List[List[int]], Dict]]:
        """Process a stream of data with optimal batching."""
        batch_buffer = []
        
        for sample in data_stream:
            batch_buffer.append(sample)
            
            if len(batch_buffer) >= self.batch_size:
                # Process batch
                batch_array = np.stack(batch_buffer)
                batch_tensor = torch.from_numpy(batch_array).float()
                
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                
                # Process with model
                start_time = time.time()
                prediction_sets = self.model.predict_set(batch_tensor)
                processing_time = time.time() - start_time
                
                # Metadata
                metadata = {
                    'batch_size': len(batch_buffer),
                    'processing_time': processing_time,
                    'throughput': len(batch_buffer) / processing_time,
                    'memory_usage': self.memory_optimizer.get_memory_usage()
                }
                
                # Call callback if provided
                if callback:
                    callback(prediction_sets, metadata)
                
                yield prediction_sets, metadata
                
                # Clear batch
                batch_buffer = []
                
                # Periodic memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Process remaining samples
        if batch_buffer:
            batch_array = np.stack(batch_buffer)
            batch_tensor = torch.from_numpy(batch_array).float()
            
            if torch.cuda.is_available():
                batch_tensor = batch_tensor.cuda()
            
            prediction_sets = self.model.predict_set(batch_tensor)
            metadata = {'batch_size': len(batch_buffer)}
            
            if callback:
                callback(prediction_sets, metadata)
            
            yield prediction_sets, metadata
    
    def benchmark_throughput(
        self,
        test_data: np.ndarray,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark processing throughput."""
        throughputs = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            # Process in optimal batches
            for i in range(0, len(test_data), self.batch_size):
                batch = test_data[i:i + self.batch_size]
                batch_tensor = torch.from_numpy(batch).float()
                
                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()
                
                _ = self.model.predict_set(batch_tensor)
            
            total_time = time.time() - start_time
            throughput = len(test_data) / total_time
            throughputs.append(throughput)
        
        return {
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'min_throughput': np.min(throughputs),
            'std_throughput': np.std(throughputs),
            'optimal_batch_size': self.batch_size
        }


class ModelCompressor:
    """Model compression techniques for edge deployment."""
    
    def __init__(self):
        self.compression_methods = [
            'quantization',
            'pruning',
            'knowledge_distillation',
            'hypervector_compression'
        ]
    
    def quantize_model(
        self,
        model: ConformalHDC,
        quantization_bits: int = 8,
        calibration_data: Optional[torch.Tensor] = None
    ) -> ConformalHDC:
        """Quantize model to reduce memory and computation."""
        logger.info(f"Quantizing model to {quantization_bits} bits")
        
        # Quantize class prototypes
        if model.class_prototypes is not None:
            prototypes = model.class_prototypes
            
            # Compute quantization parameters
            if calibration_data is not None:
                # Use calibration data for better quantization
                encoded_cal = model.encoder.encode(calibration_data)
                proto_min = torch.min(encoded_cal)
                proto_max = torch.max(encoded_cal)
            else:
                proto_min = torch.min(prototypes)
                proto_max = torch.max(prototypes)
            
            # Quantize to specified bits
            scale = (proto_max - proto_min) / (2**quantization_bits - 1)
            zero_point = -proto_min / scale
            
            # Apply quantization
            quantized_prototypes = torch.round(prototypes / scale + zero_point)
            quantized_prototypes = torch.clamp(quantized_prototypes, 0, 2**quantization_bits - 1)
            
            # Store quantization parameters
            model.quantization_scale = scale
            model.quantization_zero_point = zero_point
            model.class_prototypes = quantized_prototypes.to(torch.uint8)
        
        # Quantize encoder if possible
        if hasattr(model.encoder, 'projection_matrix'):
            proj = model.encoder.projection_matrix
            
            # Simple linear quantization for projection matrix
            proj_min = torch.min(proj)
            proj_max = torch.max(proj)
            proj_scale = (proj_max - proj_min) / (2**quantization_bits - 1)
            proj_zero_point = -proj_min / proj_scale
            
            quantized_proj = torch.round(proj / proj_scale + proj_zero_point)
            quantized_proj = torch.clamp(quantized_proj, 0, 2**quantization_bits - 1)
            
            model.encoder.projection_matrix = quantized_proj.to(torch.uint8)
            model.encoder.proj_quantization_scale = proj_scale
            model.encoder.proj_quantization_zero_point = proj_zero_point
        
        logger.info("Model quantization completed")
        return model
    
    def prune_hypervectors(
        self,
        model: ConformalHDC,
        sparsity: float = 0.5,
        structured: bool = False
    ) -> ConformalHDC:
        """Prune hypervectors to reduce model size."""
        if model.class_prototypes is None:
            return model
        
        prototypes = model.class_prototypes
        
        if structured:
            # Structured pruning - remove entire dimensions
            importance_scores = torch.sum(torch.abs(prototypes), dim=0)
            num_keep = int(prototypes.shape[1] * (1 - sparsity))
            
            # Keep most important dimensions
            _, keep_indices = torch.topk(importance_scores, num_keep)
            keep_indices = torch.sort(keep_indices)[0]  # Sort for consistent indexing
            
            # Update prototypes
            model.class_prototypes = prototypes[:, keep_indices]
            
            # Update encoder dimension if applicable
            if hasattr(model.encoder, 'hv_dim'):
                model.encoder.hv_dim = num_keep
            
            logger.info(f"Structured pruning: reduced HV dimension from {prototypes.shape[1]} to {num_keep}")
            
        else:
            # Unstructured pruning - zero out individual elements
            magnitude_threshold = torch.quantile(torch.abs(prototypes), sparsity)
            mask = torch.abs(prototypes) > magnitude_threshold
            
            model.class_prototypes = prototypes * mask.float()
            
            actual_sparsity = (mask == 0).float().mean().item()
            logger.info(f"Unstructured pruning: achieved {actual_sparsity:.1%} sparsity")
        
        return model
    
    def compress_for_edge(
        self,
        model: ConformalHDC,
        target_memory_mb: int = 10,
        target_inference_ms: int = 100
    ) -> Tuple[ConformalHDC, Dict[str, Any]]:
        """Comprehensive compression for edge deployment."""
        original_memory = model.memory_footprint()['total'] / 1024**2  # MB
        
        compression_info = {
            'original_memory_mb': original_memory,
            'target_memory_mb': target_memory_mb,
            'compression_steps': []
        }
        
        # Step 1: Quantization
        if original_memory > target_memory_mb * 2:
            model = self.quantize_model(model, quantization_bits=4)
            compression_info['compression_steps'].append('4-bit quantization')
        elif original_memory > target_memory_mb * 1.5:
            model = self.quantize_model(model, quantization_bits=8)
            compression_info['compression_steps'].append('8-bit quantization')
        
        # Step 2: Pruning if still too large
        current_memory = model.memory_footprint()['total'] / 1024**2
        if current_memory > target_memory_mb:
            sparsity = 1 - (target_memory_mb / current_memory) * 0.8  # Leave 20% margin
            sparsity = max(0, min(0.9, sparsity))  # Clamp to reasonable range
            
            model = self.prune_hypervectors(model, sparsity=sparsity, structured=True)
            compression_info['compression_steps'].append(f'structured pruning ({sparsity:.1%})')
        
        # Step 3: Hypervector dimension reduction if needed
        current_memory = model.memory_footprint()['total'] / 1024**2
        if current_memory > target_memory_mb and hasattr(model.encoder, 'hv_dim'):
            reduction_factor = target_memory_mb / current_memory
            new_hv_dim = int(model.encoder.hv_dim * reduction_factor * 0.9)  # 10% margin
            new_hv_dim = max(1000, new_hv_dim)  # Minimum reasonable dimension
            
            # Reduce hypervector dimension
            if new_hv_dim < model.encoder.hv_dim:
                keep_indices = torch.randperm(model.encoder.hv_dim)[:new_hv_dim]
                keep_indices = torch.sort(keep_indices)[0]
                
                model.class_prototypes = model.class_prototypes[:, keep_indices]
                model.encoder.hv_dim = new_hv_dim
                
                compression_info['compression_steps'].append(f'dimension reduction to {new_hv_dim}')
        
        # Final statistics
        final_memory = model.memory_footprint()['total'] / 1024**2
        compression_ratio = original_memory / final_memory if final_memory > 0 else float('inf')
        
        compression_info.update({
            'final_memory_mb': final_memory,
            'compression_ratio': compression_ratio,
            'memory_reduction_percent': (1 - final_memory / original_memory) * 100
        })
        
        logger.info(f"Edge compression completed: {compression_ratio:.1f}x reduction")
        return model, compression_info


# Factory functions
def create_optimized_model(
    input_dim: int,
    num_classes: int,
    hv_dim: int = 10000,
    acceleration: str = 'auto',
    memory_limit_gb: float = 4.0,
    **kwargs
) -> ConformalHDC:
    """Create optimized HyperConformal model."""
    
    # Use optimized encoder
    encoder = OptimizedEncoder(
        input_dim=input_dim,
        hv_dim=hv_dim,
        acceleration=acceleration,
        use_memory_mapping=(input_dim * hv_dim > 1e8)
    )
    
    # Create model with memory optimization
    model = ConformalHDC(encoder, num_classes, **kwargs)
    
    # Apply memory optimization
    memory_optimizer = MemoryOptimizer(max_memory_gb=memory_limit_gb)
    
    return model