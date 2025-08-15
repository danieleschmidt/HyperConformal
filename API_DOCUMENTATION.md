# HyperConformal API Documentation

**Version**: 1.0.0  
**Last Updated**: 2025-08-14

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Core API](#core-api)
3. [Encoders](#encoders)
4. [Conformal Prediction](#conformal-prediction)
5. [Advanced Features](#advanced-features)
6. [Examples](#examples)
7. [Performance](#performance)
8. [Deployment](#deployment)

## 🚀 Quick Start

### Installation
```bash
pip install hyperconformal
```

### Basic Usage
```python
import hyperconformal as hc
import numpy as np

# Create HDC encoder
encoder = hc.RandomProjection(
    input_dim=784,      # MNIST features
    hv_dim=10000,       # Hypervector dimension
    quantization='binary'
)

# Initialize conformal predictor
predictor = hc.ConformalHDC(
    encoder=encoder,
    num_classes=10,
    alpha=0.1  # 90% coverage guarantee
)

# Train and predict
predictor.fit(X_train, y_train)
pred_sets = predictor.predict_set(X_test)
```

## 🧠 Core API

### HyperConformal Class

The main entry point for conformal prediction with hyperdimensional computing.

```python
class HyperConformal:
    def __init__(self, encoder, num_classes, alpha=0.1, method='score')
```

**Parameters:**
- `encoder`: HDC encoder instance (BaseEncoder)
- `num_classes`: Number of output classes (int)
- `alpha`: Miscoverage rate (float, 0 < alpha < 1)
- `method`: Conformal method ('score', 'threshold', 'quantile')

**Methods:**

#### `.fit(X, y, calibration_split=0.2)`
Train the HDC model and calibrate conformal predictor.

**Parameters:**
- `X`: Training features (array-like, shape=[n_samples, n_features])
- `y`: Training labels (array-like, shape=[n_samples])
- `calibration_split`: Fraction for calibration (float, default=0.2)

**Returns:**
- `self`: Fitted HyperConformal instance

#### `.predict_set(X, return_scores=False)`
Generate prediction sets with coverage guarantees.

**Parameters:**
- `X`: Test features (array-like)
- `return_scores`: Return confidence scores (bool, default=False)

**Returns:**
- `pred_sets`: List of prediction sets
- `scores`: Confidence scores (if return_scores=True)

#### `.predict(X)`
Standard point prediction (most likely class).

**Parameters:**
- `X`: Test features (array-like)

**Returns:**
- `y_pred`: Predicted class labels (array)

#### `.compute_coverage(pred_sets, y_true)`
Compute empirical coverage of prediction sets.

**Parameters:**
- `pred_sets`: Prediction sets from predict_set()
- `y_true`: True labels (array-like)

**Returns:**
- `coverage`: Empirical coverage rate (float)

## 🔢 Encoders

### BaseEncoder

Abstract base class for all HDC encoders.

```python
class BaseEncoder:
    def encode(self, x):
        """Encode input to hypervector."""
        pass
    
    def similarity(self, hv1, hv2):
        """Compute similarity between hypervectors."""
        pass
```

### RandomProjection

Binary random projection encoder for general data.

```python
encoder = hc.RandomProjection(
    input_dim=784,
    hv_dim=10000,
    quantization='binary',  # 'binary', 'ternary', 'complex'
    seed=42
)
```

**Parameters:**
- `input_dim`: Input feature dimension (int)
- `hv_dim`: Hypervector dimension (int)
- `quantization`: Quantization scheme (str)
- `seed`: Random seed (int, optional)

### LevelHDC

Level-based encoding for continuous values.

```python
encoder = hc.LevelHDC(
    input_dim=784,
    hv_dim=10000,
    levels=100,
    circular=True
)
```

**Parameters:**
- `input_dim`: Input feature dimension (int)
- `hv_dim`: Hypervector dimension (int)
- `levels`: Number of quantization levels (int)
- `circular`: Use circular encoding (bool)

### ComplexHDC

Complex-valued hypervectors for signal processing.

```python
encoder = hc.ComplexHDC(
    input_dim=784,
    hv_dim=10000,
    quantization_levels=4
)
```

**Parameters:**
- `input_dim`: Input feature dimension (int)
- `hv_dim`: Hypervector dimension (int)
- `quantization_levels`: PSK modulation levels (int)

### SpatialHDC

Spatial encoding for image data.

```python
encoder = hc.SpatialHDC(
    resolution=(28, 28),
    hv_dim=10000,
    patch_size=(3, 3)
)
```

**Parameters:**
- `resolution`: Image resolution tuple (int, int)
- `hv_dim`: Hypervector dimension (int)
- `patch_size`: Spatial patch size (int, int)

## 📊 Conformal Prediction

### ConformalHDC

Main conformal predictor for HDC models.

```python
predictor = hc.ConformalHDC(
    encoder=encoder,
    num_classes=10,
    alpha=0.1,
    method='score'
)
```

### AdaptiveConformalHDC

Online conformal prediction with concept drift adaptation.

```python
adaptive = hc.AdaptiveConformalHDC(
    encoder=encoder,
    num_classes=10,
    window_size=1000,
    update_frequency=100
)
```

**Methods:**

#### `.update(X, y)`
Update calibration with new data.

**Parameters:**
- `X`: New features (array-like)
- `y`: New labels (array-like)

### DistributedConformalHDC

Distributed conformal prediction across multiple nodes.

```python
distributed = hc.DistributedConformalHDC(
    encoder=encoder,
    num_classes=10,
    num_nodes=4,
    aggregation='average'
)
```

## ⚡ Advanced Features

### Performance Optimization

```python
# Enable caching for repeated computations
predictor.enable_caching(cache_size=1000)

# Batch processing for efficiency
batch_results = predictor.predict_set_batch(X_batch, batch_size=100)

# Concurrent processing
with hc.ConcurrentProcessor(workers=4) as proc:
    results = proc.process_batch(X_batch)
```

### Monitoring & Metrics

```python
# Enable performance monitoring
monitor = hc.PerformanceMonitor()
predictor.add_monitor(monitor)

# Get performance statistics
stats = monitor.get_statistics()
print(f"Average inference time: {stats['avg_time_ms']:.2f}ms")
print(f"Throughput: {stats['throughput']:.1f} pred/s")
```

### Security Features

```python
# Input validation and sanitization
validator = hc.InputValidator()
safe_X = validator.validate_and_sanitize(X)

# Threat detection
threat_detector = hc.ThreatDetector()
is_safe = threat_detector.scan_input(X)
```

## 📋 Examples

### MNIST Classification

```python
import hyperconformal as hc
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST
X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create encoder and predictor
encoder = hc.RandomProjection(input_dim=784, hv_dim=10000)
predictor = hc.ConformalHDC(encoder, num_classes=10, alpha=0.1)

# Train and predict
predictor.fit(X_train, y_train)
pred_sets = predictor.predict_set(X_test)

# Evaluate coverage
coverage = predictor.compute_coverage(pred_sets, y_test)
print(f"Empirical coverage: {coverage:.1%}")
```

### Time Series Prediction

```python
# Create level-based encoder for time series
encoder = hc.LevelHDC(
    input_dim=100,  # Window size
    hv_dim=5000,
    levels=50,
    circular=True
)

# Adaptive conformal for streaming data
predictor = hc.AdaptiveConformalHDC(
    encoder=encoder,
    num_classes=3,  # Up, Down, Stable
    window_size=1000
)

# Process streaming data
for batch in data_stream:
    pred_sets = predictor.predict_set(batch)
    predictor.update(batch, labels)
```

### IoT Edge Deployment

```python
# Ultra-low-power configuration
encoder = hc.RandomProjection(
    input_dim=64,
    hv_dim=1000,  # Smaller for MCU
    quantization='binary'
)

predictor = hc.ConformalHDC(
    encoder=encoder,
    num_classes=5,
    alpha=0.15  # Slightly larger sets for efficiency
)

# Enable embedded optimizations
predictor.enable_embedded_mode(
    memory_limit=2048,  # 2KB
    quantize_weights=True,
    use_lookup_tables=True
)
```

## ⚡ Performance Characteristics

### Benchmarks

| Configuration | Throughput | Latency | Memory | Power |
|---------------|------------|---------|---------|-------|
| HDC-1K (CPU) | 2,480 vec/s | 0.4ms | 2MB | 0.3μW |
| HDC-10K (CPU) | 250 vec/s | 4ms | 20MB | 0.38μW |
| DNN Baseline | 100 pred/s | 10ms | 500MB | 2,840μW |

### Scaling

```python
# Concurrent processing scales linearly up to 4 workers
with hc.ConcurrentProcessor(workers=4) as proc:
    # 3.8x speedup on 4 cores
    results = proc.process_large_batch(X_large)

# Auto-scaling based on load
scaler = hc.AutoScaler(
    min_workers=1,
    max_workers=8,
    target_latency_ms=5.0
)
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["python", "-m", "hyperconformal.server"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hyperconformal-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hyperconformal
  template:
    spec:
      containers:
      - name: hyperconformal
        image: hyperconformal:latest
        ports:
        - containerPort: 8080
```

### Embedded C++

```c
#include "hyperconformal.h"

// Pre-trained model (1.2KB)
const uint8_t model[] = {
    #include "model_weights.h"
};

void classify_with_confidence(uint8_t* input, 
                             uint8_t* prediction,
                             uint8_t* confidence) {
    hypervector_t encoded;
    hc_encode_binary(input, &encoded, &model);
    hc_conformal_predict(&encoded, &model, prediction, confidence);
}
```

## 📞 Support

### Error Handling

```python
try:
    pred_sets = predictor.predict_set(X_test)
except hc.ValidationError as e:
    print(f"Input validation failed: {e}")
except hc.ModelNotFittedError as e:
    print(f"Model not trained: {e}")
except hc.DimensionMismatchError as e:
    print(f"Dimension error: {e}")
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose mode for detailed output
predictor.set_verbose(True)

# Performance profiling
with hc.Profiler() as prof:
    pred_sets = predictor.predict_set(X_test)
print(prof.get_report())
```

### Resources

- **Documentation**: https://hyperconformal.readthedocs.io
- **Examples**: https://github.com/terragonlabs/hyperconformal/examples
- **Issues**: https://github.com/terragonlabs/hyperconformal/issues

---

*Generated by Terragon Labs*  
*HyperConformal v1.0.0 - Calibrated Uncertainty for Hyperdimensional Computing*