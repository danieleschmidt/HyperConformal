# HyperConformal API Reference

## Core Classes

### BaseEncoder

Abstract base class for HDC encoders.

```python
class BaseEncoder(ABC, nn.Module):
    def __init__(self, input_dim: int, hv_dim: int)
    def encode(self, x: torch.Tensor) -> torch.Tensor
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor
    def memory_footprint(self) -> int
```

### RandomProjection

Random projection HDC encoder with multiple quantization schemes.

```python
class RandomProjection(BaseEncoder):
    def __init__(
        self, 
        input_dim: int, 
        hv_dim: int, 
        quantization: str = 'binary',  # 'binary', 'ternary', 'complex'
        seed: Optional[int] = None
    )
```

**Parameters:**
- `input_dim`: Input feature dimension
- `hv_dim`: Hypervector dimension
- `quantization`: Quantization scheme ('binary', 'ternary', 'complex')
- `seed`: Random seed for reproducibility

**Example:**
```python
import hyperconformal as hc

encoder = hc.RandomProjection(
    input_dim=784,
    hv_dim=10000,
    quantization='binary',
    seed=42
)
```

### LevelHDC

Level-based HDC encoder for continuous features.

```python
class LevelHDC(BaseEncoder):
    def __init__(
        self, 
        input_dim: int, 
        hv_dim: int, 
        levels: int = 100,
        circular: bool = False,
        seed: Optional[int] = None
    )
```

**Parameters:**
- `levels`: Number of quantization levels
- `circular`: Whether to use circular encoding

### ComplexHDC

Complex-valued HDC encoder for signal processing.

```python
class ComplexHDC(BaseEncoder):
    def __init__(
        self, 
        input_dim: int,
        hv_dim: int,
        quantization_levels: int = 4,
        seed: Optional[int] = None
    )
```

## Conformal Prediction Classes

### ConformalHDC

Main class combining HDC encoding with conformal prediction.

```python
class ConformalHDC:
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        score_type: str = 'aps',
        calibration_split: float = 0.2,
        device: Optional[torch.device] = None,
        validate_inputs: bool = True,
        random_state: Optional[int] = None
    )
```

**Parameters:**
- `encoder`: HDC encoder instance
- `num_classes`: Number of target classes
- `alpha`: Miscoverage level (1-alpha is target coverage)
- `score_type`: Conformal score type ('aps', 'margin', 'inverse_softmax')
- `calibration_split`: Fraction of training data for calibration
- `validate_inputs`: Whether to validate inputs for safety
- `random_state`: Random seed for reproducibility

**Methods:**

#### fit(X, y)
Train the HDC model and calibrate conformal predictor.

```python
def fit(
    self, 
    X: Union[torch.Tensor, np.ndarray], 
    y: Union[torch.Tensor, np.ndarray]
) -> 'ConformalHDC'
```

**Returns:** Self (for method chaining)

#### predict(X)
Predict class labels.

```python
def predict(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray
```

**Returns:** Predicted class labels

#### predict_proba(X)
Predict class probabilities.

```python
def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor
```

**Returns:** Class probabilities (batch_size, num_classes)

#### predict_set(X)
Generate calibrated prediction sets with coverage guarantees.

```python
def predict_set(self, X: Union[torch.Tensor, np.ndarray]) -> List[List[int]]
```

**Returns:** List of prediction sets (lists of possible class labels)

#### get_empirical_coverage(X, y)
Compute empirical coverage on test data.

```python
def get_empirical_coverage(
    self, 
    X: Union[torch.Tensor, np.ndarray], 
    y: Union[torch.Tensor, np.ndarray]
) -> float
```

**Returns:** Empirical coverage rate [0, 1]

#### memory_footprint()
Estimate memory footprint in bytes.

```python
def memory_footprint(self) -> Dict[str, int]
```

**Returns:** Dictionary with memory usage breakdown

#### health_check()
Perform comprehensive health check of the model.

```python
def health_check(self) -> Dict[str, Any]
```

**Returns:** Health status dictionary

### OptimizedConformalHDC

Optimized version with performance enhancements.

```python
class OptimizedConformalHDC(ConformalHDC):
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        # ... base parameters ...
        enable_caching: bool = True,
        cache_size: int = 1000,
        batch_size: int = 64,
        use_gpu_acceleration: bool = True,
        enable_memory_pooling: bool = True,
        num_workers: int = None,
        auto_optimize: bool = True
    )
```

**Additional Parameters:**
- `enable_caching`: Whether to enable encoder result caching
- `cache_size`: Size of the encoding cache
- `batch_size`: Batch size for processing
- `use_gpu_acceleration`: Whether to use GPU acceleration
- `enable_memory_pooling`: Whether to use memory pooling
- `num_workers`: Number of worker threads
- `auto_optimize`: Whether to automatically optimize parameters

**Additional Methods:**

#### predict_batch_parallel(X, batch_size, num_workers)
Predict labels with parallel processing.

```python
def predict_batch_parallel(
    self, 
    X: Union[torch.Tensor, np.ndarray],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> np.ndarray
```

#### predict_set_optimized(X, use_adaptive_threshold)
Generate prediction sets with optimizations.

```python
def predict_set_optimized(
    self, 
    X: Union[torch.Tensor, np.ndarray],
    use_adaptive_threshold: bool = True
) -> List[List[int]]
```

#### get_optimization_stats()
Get comprehensive optimization statistics.

```python
def get_optimization_stats(self) -> Dict[str, Any]
```

#### benchmark(X, num_runs, operations)
Benchmark different operations.

```python
def benchmark(
    self, 
    X: Union[torch.Tensor, np.ndarray],
    num_runs: int = 10,
    operations: Optional[List[str]] = None
) -> Dict[str, Any]
```

### ScalableAdaptiveConformalHDC

Highly scalable adaptive version for streaming data.

```python
class ScalableAdaptiveConformalHDC(AdaptiveConformalHDC, OptimizedConformalHDC):
    def __init__(
        self,
        # ... base parameters ...
        window_size: int = 1000,
        update_frequency: int = 100,
        drift_detection: bool = True,
        max_drift_score: float = 0.1
    )
```

**Additional Parameters:**
- `window_size`: Size of sliding window for calibration
- `update_frequency`: How often to update calibration
- `drift_detection`: Whether to detect distribution drift
- `max_drift_score`: Maximum allowed drift score

**Additional Methods:**

#### update(X, y)
Update with streaming data.

```python
def update(
    self, 
    X: Union[torch.Tensor, np.ndarray], 
    y: Union[torch.Tensor, np.ndarray]
) -> Dict[str, Any]
```

**Returns:** Update statistics dictionary

#### get_streaming_stats()
Get comprehensive streaming statistics.

```python
def get_streaming_stats(self) -> Dict[str, Any]
```

## Utility Functions

### compute_coverage(prediction_sets, true_labels)
Compute empirical coverage of prediction sets.

```python
def compute_coverage(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float
```

### hamming_distance(hv1, hv2)
Compute Hamming distance between binary hypervectors.

```python
def hamming_distance(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor
```

### binary_quantize(x, threshold)
Quantize values to binary {-1, +1}.

```python
def binary_quantize(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor
```

## Metrics Functions

### coverage_score(prediction_sets, true_labels)
Compute coverage score.

```python
def coverage_score(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float
```

### average_set_size(prediction_sets)
Compute average size of prediction sets.

```python
def average_set_size(prediction_sets: List[List[int]]) -> float
```

### conditional_coverage(prediction_sets, true_labels, stratify_by)
Compute conditional coverage.

```python
def conditional_coverage(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    stratify_by: str = 'class'
) -> Dict[Union[int, str], float]
```

## Exception Classes

### HyperConformalError
Base exception for HyperConformal errors.

### ValidationError
Exception for input validation errors.

### CalibrationError
Exception for calibration errors.

## Examples

### Basic Classification

```python
import hyperconformal as hc
import torch

# Create encoder and predictor
encoder = hc.RandomProjection(input_dim=784, hv_dim=10000, quantization='binary')
predictor = hc.ConformalHDC(encoder=encoder, num_classes=10, alpha=0.1)

# Training data (MNIST-like)
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

# Fit model
predictor.fit(X_train, y_train)

# Test data
X_test = torch.randn(100, 784)
y_test = torch.randint(0, 10, (100,))

# Get predictions
predictions = predictor.predict(X_test)
pred_sets = predictor.predict_set(X_test)
coverage = predictor.get_empirical_coverage(X_test, y_test)

print(f"Coverage: {coverage:.1%}")
print(f"Average set size: {hc.average_set_size(pred_sets):.2f}")
```

### High-Performance Setup

```python
# Optimized for high throughput
encoder = hc.LevelHDC(input_dim=100, hv_dim=5000, levels=50)
predictor = hc.OptimizedConformalHDC(
    encoder=encoder,
    num_classes=5,
    alpha=0.1,
    enable_caching=True,
    cache_size=2000,
    batch_size=128,
    use_gpu_acceleration=True,
    auto_optimize=True
)

# Training
X_train = torch.randn(5000, 100)
y_train = torch.randint(0, 5, (5000,))
predictor.fit(X_train, y_train)

# Batch prediction
X_test = torch.randn(1000, 100)
predictions = predictor.predict_batch_parallel(X_test, batch_size=64)

# Performance stats
stats = predictor.get_optimization_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2f}")
```

### Streaming Analytics

```python
# Adaptive streaming setup
encoder = hc.ComplexHDC(input_dim=50, hv_dim=2000)
predictor = hc.ScalableAdaptiveConformalHDC(
    encoder=encoder,
    num_classes=3,
    alpha=0.1,
    window_size=1000,
    update_frequency=50,
    drift_detection=True,
    enable_caching=True
)

# Initial training
X_init = torch.randn(500, 50)
y_init = torch.randint(0, 3, (500,))
predictor.fit(X_init, y_init)

# Streaming updates
for batch_x, batch_y in data_stream:
    # Get predictions
    pred_sets = predictor.predict_set_optimized(batch_x)
    
    # Update model
    update_stats = predictor.update(batch_x, batch_y)
    
    if update_stats['drift_detected']:
        print(f"Drift detected: {update_stats['drift_score']:.3f}")
```

## Type Hints

All functions include comprehensive type hints for better IDE support:

```python
from typing import List, Optional, Union, Tuple, Dict, Any
import torch
import numpy as np

# Function signature example
def predict_set(
    self, 
    X: Union[torch.Tensor, np.ndarray]
) -> List[List[int]]:
    ...
```

## Device Support

HyperConformal supports multiple device types:

```python
# CPU (default)
predictor = hc.ConformalHDC(encoder, num_classes=10)

# GPU
predictor = hc.ConformalHDC(
    encoder, 
    num_classes=10, 
    device=torch.device('cuda')
)

# Specific GPU
predictor = hc.ConformalHDC(
    encoder, 
    num_classes=10, 
    device=torch.device('cuda:1')
)
```

## Memory Management

All classes provide memory footprint estimation:

```python
# Get memory usage
footprint = predictor.memory_footprint()
print(f"Total memory: {footprint['total'] / 1024 / 1024:.1f} MB")
print(f"Encoder: {footprint['encoder'] / 1024:.1f} KB")
print(f"Prototypes: {footprint['prototypes'] / 1024:.1f} KB")
```

## Error Handling

Comprehensive error handling with specific exception types:

```python
try:
    predictor = hc.ConformalHDC(encoder, num_classes=10, alpha=1.5)
except hc.ValidationError as e:
    print(f"Validation error: {e}")

try:
    predictor.fit(X_invalid, y_invalid)
except hc.CalibrationError as e:
    print(f"Calibration error: {e}")
```