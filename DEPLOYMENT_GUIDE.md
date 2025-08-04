# HyperConformal Deployment Guide

## 🚀 Production Deployment Guide

This guide covers deploying HyperConformal in production environments with optimal performance and security.

## Prerequisites

- Python 3.8+ 
- PyTorch 1.9.0+
- NumPy 1.20.0+
- SciPy 1.7.0+
- scikit-learn 1.0.0+

## Installation

### Standard Installation
```bash
pip install hyperconformal
```

### Development Installation
```bash
git clone https://github.com/terragonlabs/hyperconformal.git
cd hyperconformal
pip install -e .
```

### With Optional Dependencies
```bash
pip install hyperconformal[neuromorphic]  # For neuromorphic hardware support
```

## Performance Optimization

### 1. Choose the Right Implementation

```python
import hyperconformal as hc

# For basic use cases
predictor = hc.ConformalHDC(encoder, num_classes=10)

# For high-performance applications
predictor = hc.OptimizedConformalHDC(
    encoder=encoder,
    num_classes=10,
    enable_caching=True,
    use_gpu_acceleration=True,
    batch_size=64,
    auto_optimize=True
)

# For streaming/online applications
predictor = hc.ScalableAdaptiveConformalHDC(
    encoder=encoder,
    num_classes=10,
    window_size=1000,
    drift_detection=True,
    enable_caching=True
)
```

### 2. Hardware Optimization

#### GPU Acceleration
```python
# Enable GPU acceleration for large datasets
predictor = hc.OptimizedConformalHDC(
    encoder=encoder,
    device=torch.device('cuda'),
    use_gpu_acceleration=True,
    batch_size=128  # Larger batches for GPU
)
```

#### CPU Optimization
```python
# Optimize for multi-core CPU
predictor = hc.OptimizedConformalHDC(
    encoder=encoder,
    num_workers=8,  # Match CPU cores
    batch_size=32,  # Smaller batches for CPU
    enable_memory_pooling=True
)
```

### 3. Memory Management

```python
# For memory-constrained environments
predictor = hc.OptimizedConformalHDC(
    encoder=encoder,
    cache_size=100,  # Smaller cache
    batch_size=16,   # Smaller batches
    enable_memory_pooling=True
)

# Monitor memory usage
memory_stats = predictor.memory_footprint()
print(f"Memory usage: {memory_stats['total'] / 1024 / 1024:.1f} MB")
```

## Production Configurations

### 1. High-Throughput Server

```python
# Configuration for high-throughput prediction server
class ProductionPredictor:
    def __init__(self):
        encoder = hc.RandomProjection(
            input_dim=784, 
            hv_dim=10000, 
            quantization='binary'
        )
        
        self.predictor = hc.OptimizedConformalHDC(
            encoder=encoder,
            num_classes=10,
            alpha=0.1,
            enable_caching=True,
            cache_size=5000,
            batch_size=128,
            use_gpu_acceleration=torch.cuda.is_available(),
            num_workers=8,
            auto_optimize=True
        )
        
    def predict_batch(self, X_batch):
        return self.predictor.predict_batch_parallel(
            X_batch, 
            batch_size=64,
            num_workers=4
        )
```

### 2. Streaming Analytics

```python
# Configuration for real-time streaming
class StreamingPredictor:
    def __init__(self):
        encoder = hc.LevelHDC(
            input_dim=50,
            hv_dim=5000,
            levels=100
        )
        
        self.predictor = hc.ScalableAdaptiveConformalHDC(
            encoder=encoder,
            num_classes=5,
            alpha=0.1,
            window_size=2000,
            update_frequency=100,
            drift_detection=True,
            max_drift_score=0.05,
            enable_caching=True,
            auto_optimize=True
        )
    
    def process_stream(self, data_stream):
        for batch_x, batch_y in data_stream:
            # Get predictions
            pred_sets = self.predictor.predict_set_optimized(batch_x)
            
            # Update model (adaptive learning)
            update_stats = self.predictor.update(batch_x, batch_y)
            
            # Monitor drift
            if update_stats['drift_detected']:
                self.handle_drift(update_stats)
            
            yield pred_sets
    
    def handle_drift(self, stats):
        print(f"Distribution drift detected: {stats['drift_score']:.3f}")
```

### 3. Edge/IoT Deployment

```python
# Lightweight configuration for edge devices
class EdgePredictor:
    def __init__(self):
        # Small, efficient encoder
        encoder = hc.RandomProjection(
            input_dim=20,
            hv_dim=1000,
            quantization='binary'
        )
        
        self.predictor = hc.ConformalHDC(
            encoder=encoder,
            num_classes=3,
            alpha=0.1,
            calibration_split=0.1,  # Less calibration data
            validate_inputs=False   # Skip validation for speed
        )
    
    def predict_single(self, x):
        # Optimized for single predictions
        x = torch.tensor(x).unsqueeze(0)
        pred_set = self.predictor.predict_set(x)[0]
        return pred_set
```

## Monitoring and Logging

### 1. Performance Monitoring

```python
# Enable performance monitoring
predictor = hc.OptimizedConformalHDC(encoder, num_classes=10)

# Get optimization statistics
stats = predictor.get_optimization_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2f}")
print(f"Average processing time: {stats['performance']['prediction_time']['mean']:.3f}s")

# Health check
health = predictor.health_check()
if not health['encoder_valid']:
    print("Warning: Encoder issues detected")
```

### 2. Logging Configuration

```python
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperconformal.log'),
        logging.StreamHandler()
    ]
)

# HyperConformal uses structured logging
logger = logging.getLogger('hyperconformal')
```

## Security Considerations

### 1. Input Validation

```python
# Always enable input validation in production
predictor = hc.ConformalHDC(
    encoder=encoder,
    num_classes=10,
    validate_inputs=True  # CRITICAL for security
)
```

### 2. Resource Limits

```python
# Set resource limits to prevent DoS
MAX_BATCH_SIZE = 1000
MAX_FEATURE_DIM = 10000

def safe_predict(predictor, X):
    if len(X) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size too large: {len(X)} > {MAX_BATCH_SIZE}")
    
    if X.shape[1] > MAX_FEATURE_DIM:
        raise ValueError(f"Feature dimension too large: {X.shape[1]} > {MAX_FEATURE_DIM}")
    
    return predictor.predict_set(X)
```

### 3. Model Serialization

```python
# Secure model saving/loading
import pickle
import hashlib

def save_model_secure(predictor, filepath):
    # Save with checksum
    model_data = predictor.save_model(filepath)
    
    with open(filepath, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    
    with open(f"{filepath}.checksum", 'w') as f:
        f.write(checksum)

def load_model_secure(predictor, filepath):
    # Verify checksum before loading
    with open(f"{filepath}.checksum", 'r') as f:
        expected_checksum = f.read().strip()
    
    with open(filepath, 'rb') as f:
        actual_checksum = hashlib.sha256(f.read()).hexdigest()
    
    if actual_checksum != expected_checksum:
        raise ValueError("Model file integrity check failed")
    
    return predictor.load_model(filepath)
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Install HyperConformal
RUN pip install -e .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "serve.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  hyperconformal-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0  # GPU support
      - OMP_NUM_THREADS=4       # CPU threads
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

## Kubernetes Deployment

### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hyperconformal-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hyperconformal
  template:
    metadata:
      labels:
        app: hyperconformal
    spec:
      containers:
      - name: hyperconformal
        image: hyperconformal:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: BATCH_SIZE
          value: "64"
        - name: CACHE_SIZE
          value: "1000"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Performance Benchmarks

### Expected Performance

| Configuration | Throughput | Latency | Memory |
|---------------|------------|---------|---------|
| Basic CPU | 1K pred/sec | 10ms | 50MB |
| Optimized CPU | 5K pred/sec | 5ms | 100MB |
| GPU Accelerated | 20K pred/sec | 2ms | 200MB |
| Edge Device | 100 pred/sec | 50ms | 10MB |

### Benchmarking Code

```python
def benchmark_predictor(predictor, X_test, num_runs=100):
    """Benchmark predictor performance."""
    results = predictor.benchmark(X_test, num_runs=num_runs)
    
    for operation, stats in results.items():
        print(f"{operation}:")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")
        print(f"  Latency: {stats['mean_time']*1000:.1f}ms")
    
    return results
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```python
   # Reduce cache size and batch size
   predictor = hc.OptimizedConformalHDC(
       encoder=encoder,
       cache_size=100,  # Reduce from default
       batch_size=16    # Reduce from default
   )
   ```

2. **Slow Predictions**
   ```python
   # Enable GPU acceleration and increase batch size
   predictor = hc.OptimizedConformalHDC(
       encoder=encoder,
       use_gpu_acceleration=True,
       batch_size=128
   )
   ```

3. **Poor Coverage**
   ```python
   # Increase calibration data or adjust alpha
   predictor = hc.ConformalHDC(
       encoder=encoder,
       alpha=0.05,  # More conservative
       calibration_split=0.3  # More calibration data
   )
   ```

### Monitoring Commands

```bash
# Monitor memory usage
docker stats hyperconformal-service

# Check logs
kubectl logs -f deployment/hyperconformal-service

# Performance metrics
curl http://localhost:8000/metrics
```

## Best Practices Summary

1. **Performance**: Use OptimizedConformalHDC for production
2. **Security**: Always enable input validation
3. **Monitoring**: Implement health checks and logging
4. **Scaling**: Use batch processing and GPU acceleration
5. **Reliability**: Implement graceful error handling
6. **Maintenance**: Regular model retraining and drift monitoring

## Support

For deployment issues:
- GitHub Issues: https://github.com/terragonlabs/hyperconformal/issues
- Documentation: https://hyperconformal.readthedocs.io
- Email: support@terragonlabs.com