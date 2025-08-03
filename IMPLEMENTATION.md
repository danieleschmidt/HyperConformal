# HyperConformal Implementation Guide

## Overview

This document provides a comprehensive guide to the HyperConformal library implementation, which combines **Hyperdimensional Computing (HDC)** with **Conformal Prediction** for calibrated uncertainty quantification.

## Architecture

### Core Components

1. **HDC Encoders** (`hyperconformal/encoders.py`)
   - `BaseEncoder`: Abstract base class for all HDC encoders
   - `RandomProjection`: Binary/ternary/complex random projection encoder
   - `LevelHDC`: Level-based encoding for continuous features
   - `ComplexHDC`: Complex-valued HDC for signal processing

2. **Conformal Predictors** (`hyperconformal/conformal.py`)
   - `ConformalPredictor`: Base class for conformal prediction
   - `ClassificationConformalPredictor`: For classification tasks
   - `AdaptiveConformalPredictor`: For streaming data
   - `RegressionConformalPredictor`: For regression tasks

3. **Integrated Models** (`hyperconformal/hyperconformal.py`)
   - `ConformalHDC`: Main class combining HDC + conformal prediction
   - `AdaptiveConformalHDC`: Adaptive version for streaming data

4. **Utilities** (`hyperconformal/utils.py`)
   - Hamming distance/similarity functions
   - Quantization utilities
   - Hypervector operations (bundle, bind, permute)
   - Coverage computation functions

5. **Evaluation Metrics** (`hyperconformal/metrics.py`)
   - Coverage score, average set size
   - Conditional coverage analysis
   - Calibration error metrics
   - HDC-specific similarity statistics

## Key Features

### 1. Multiple HDC Quantization Schemes

- **Binary**: `{-1, +1}` - Ultra-efficient for embedded systems
- **Ternary**: `{-1, 0, +1}` - Sparse representation
- **Complex**: Unit magnitude complex numbers for signal processing

### 2. Conformal Prediction Methods

- **APS (Adaptive Prediction Sets)**: Size-adaptive sets with randomization
- **Margin-based**: Using confidence margins
- **Inverse Softmax**: Direct probability thresholding

### 3. Uncertainty Quantification

- **Coverage Guarantees**: Theoretical guarantees of `≥ 1-α` coverage
- **Calibrated Sets**: Prediction sets that contain true labels with high probability
- **Adaptive Calibration**: Online recalibration for streaming data

### 4. Efficient Implementation

- **Memory Efficient**: Binary operations reduce memory footprint
- **Fast Inference**: XOR operations instead of expensive softmax
- **Streaming Support**: Online learning with sliding window calibration

## Usage Examples

### Basic Classification

```python
import hyperconformal as hc

# Create HDC encoder
encoder = hc.RandomProjection(
    input_dim=784,
    hv_dim=10000,
    quantization='binary'
)

# Create conformal predictor
model = hc.ConformalHDC(
    encoder=encoder,
    num_classes=10,
    alpha=0.1  # 90% coverage guarantee
)

# Train
model.fit(X_train, y_train)

# Get prediction sets with guarantees
pred_sets = model.predict_set(X_test)
coverage = hc.compute_coverage(pred_sets, y_test)
```

### Streaming/Adaptive Learning

```python
# Adaptive model for streaming data
model = hc.AdaptiveConformalHDC(
    encoder=encoder,
    num_classes=3,
    alpha=0.1,
    window_size=1000,
    update_frequency=100
)

# Process streaming batches
for X_batch, y_batch in data_stream:
    pred_sets = model.predict_set(X_batch)
    model.update(X_batch, y_batch)  # Online calibration
```

## Implementation Details

### HDC Encoding Process

1. **Random Projection**: Input features are projected to high-dimensional space
2. **Quantization**: Continuous values are quantized (binary/ternary/complex)
3. **Prototype Learning**: Class prototypes are learned by averaging encoded vectors
4. **Similarity Computation**: New samples are compared to prototypes using appropriate metrics

### Conformal Prediction Workflow

1. **Calibration**: Compute nonconformity scores on calibration set
2. **Threshold Selection**: Choose quantile threshold based on α
3. **Prediction Sets**: Include all labels with nonconformity score ≤ threshold
4. **Coverage Guarantee**: Theoretical guarantee of ≥ (1-α) coverage

### Nonconformity Score Functions

#### APS (Adaptive Prediction Sets)
- Score = cumulative probability up to true label + randomization
- Produces adaptive-size prediction sets
- Handles ties with randomization

#### Margin-based
- Score = 1 - (max_prob - second_max_prob)
- Based on confidence margin
- Simple and interpretable

#### Inverse Softmax
- Score = 1 - probability of true class
- Direct probability-based scoring
- Natural for well-calibrated models

## Performance Characteristics

### Memory Footprint
- **HDC Model**: ~1-10KB (vs 100MB+ for DNNs)
- **Binary Quantization**: 1 bit per dimension
- **Calibration Data**: ~4 bytes per calibration sample

### Computational Complexity
- **Encoding**: O(input_dim × hv_dim) XOR operations
- **Classification**: O(num_classes × hv_dim) Hamming distances
- **Conformal Sets**: O(num_classes) for set generation

### Power Consumption
- **10,000x lower** than softmax-based uncertainty (as claimed in README)
- Binary operations are extremely power-efficient
- Suitable for ultra-low-power MCUs

## Testing

The implementation includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Coverage Tests**: Verification of theoretical guarantees
- **Performance Tests**: Memory and computational efficiency

Run tests with:
```bash
python3 validate_setup.py  # Structure validation
```

## Theoretical Guarantees

### Coverage Theorem
For any distribution P and any α ∈ (0,1), the conformal prediction sets satisfy:

P(Y_{n+1} ∈ C_α(X_{n+1})) ≥ 1 - α

This hold under the exchangeability assumption and is distribution-free.

### HDC-Specific Considerations
- Binary quantization introduces approximation errors
- High-dimensional space provides good separation
- Hamming similarity preserves relative distances

## Extensions and Future Work

### Implemented Extensions
- Complex-valued HDC for signal processing
- Adaptive calibration for streaming data
- Multiple HDC architectures (Level, Spatial, N-gram ready for implementation)

### Potential Extensions
- Quantum HDC support
- Neuromorphic hardware integration
- Federated learning with privacy preservation
- Multi-modal HDC encoders

## References

1. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.
2. Angelopoulos, A., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification.
3. Rahimi, A., et al. (2016). Hyperdimensional computing for noninvasive brain–computer interfaces.

## Contributing

The implementation is modular and extensible:

1. **New HDC Encoders**: Inherit from `BaseEncoder`
2. **New Conformal Methods**: Inherit from `ConformalPredictor`  
3. **New Metrics**: Add to `metrics.py`
4. **Hardware Backends**: Extend encoder classes

See examples in `examples/` directory for usage patterns.