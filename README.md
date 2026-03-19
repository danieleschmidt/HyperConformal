# HyperConformal: Conformal Prediction for Hyperdimensional Computing

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)](#tests)

## What This Is

**HyperConformal** combines two ideas that are individually well-studied but rarely combined:

1. **Hyperdimensional Computing (HDC)** — a brain-inspired classification paradigm that
   operates on high-dimensional binary vectors
2. **Conformal Prediction (CP)** — a distribution-free framework for producing prediction
   sets with a **provable coverage guarantee**

The result: a classifier that is fast enough for microcontrollers with <1 MB RAM *and* can
say "I'm not sure" with statistical backing.

---

## What is Hyperdimensional Computing?

Traditional neural networks represent data as real-valued activations and require
floating-point multiply-accumulate (MAC) operations — expensive on embedded hardware.

HDC takes a different approach inspired by how biological neurons work in high-dimensional
spaces:

- **Hypervectors** are vectors of dimension d ≥ 1000 with binary elements (±1)
- A random projection matrix Φ ∈ {−1,+1}^{n_features × d} maps inputs to hypervectors:
  ```
  h = sign(Φᵀ x)
  ```
- Class **prototypes** are created by *superposition* (element-wise majority vote over all
  training examples for that class)
- Classification is argmax **cosine similarity** between the query hypervector and each
  prototype

**Key properties of high-dimensional binary spaces:**
- Random hypervectors are nearly orthogonal with high probability (concentration of measure)
- Arithmetic is dominated by popcount / XOR — single CPU instructions on ARM Cortex-M
- A full classifier (projection matrix + prototypes) for 10-class problems at d=1000
  fits in ~12 KB — well within a typical MCU's SRAM

---

## What Conformal Prediction Adds

HDC gives you a point prediction.  But on safety-critical edge devices (medical wearables,
industrial sensors, autonomous vehicles), you often need to know *when not to trust* the
prediction.

Conformal prediction addresses this with a mathematical guarantee.

### The Coverage Guarantee

Given a **calibration set** {(x₁,y₁), …, (xₙ,yₙ)} held out from training, split
conformal prediction constructs **prediction sets** Ĉ(x) such that:

```
P(y_test ∈ Ĉ(x_test)) ≥ 1 − α
```

This holds **exactly** (not asymptotically) for any α ∈ (0,1), any classifier, and any
data distribution — as long as the calibration and test examples are exchangeable
(i.e., i.i.d. is sufficient).

No assumptions about Gaussian noise, calibrated softmax, or model quality.

### How It Works (Margin-Based Score)

For each calibration example (xᵢ, yᵢ):
1. Compute similarity scores f_c(xᵢ) for all classes c
2. Compute **nonconformity score**:
   ```
   s_i = 1 − (f_{y_i}(xᵢ) − max_{c ≠ y_i} f_c(xᵢ))
   ```
   High score = true class was NOT clearly the best → nonconforming

3. The conformal quantile is:
   ```
   q̂ = ⌈(n+1)(1−α)⌉ / n  empirical quantile of {s₁, …, sₙ}
   ```

At test time, include class c in the prediction set if:
```
1 − (f_c(x) − max_{c'≠c} f_{c'}(x)) ≤ q̂
```

---

## Why This Matters for Edge AI

| Concern | Traditional DNN | HyperConformal |
|---------|----------------|----------------|
| RAM | 100 KB–10 MB | **2–50 KB** |
| Power per inference | mW | **µW** |
| Uncertainty estimate | Softmax (uncalibrated) | **Provable coverage** |
| Training time | Minutes–hours | **Milliseconds** |
| MCU compatibility | Cortex-M33+ | **Cortex-M0+** |

The target hardware is microcontrollers like the Arduino Nano 33 BLE (ARM Cortex-M4F,
256 KB flash, 64 KB SRAM).  A d=1000 binary HDC model with conformal calibration
occupies ~4 KB — leaving the rest of RAM for sensor buffers and application logic.

---

## Quick Start

### Requirements

```
numpy >= 1.21
pytest (for tests)
```

No PyTorch, no scikit-learn, no GPU required.

### Installation

```bash
git clone https://github.com/danieleschmidt/HyperConformal
cd HyperConformal
```

### Run the Demo

```bash
python demo.py
```

Expected output:
```
HyperConformal Demo: HDC + Conformal Prediction
...
Training accuracy  : 0.950
Empirical coverage : 1.000  (✓ ≥ 0.90)
Avg set size       : 1.17
Coverage guarantee holds: True
```

### Code Example

```python
import numpy as np
from hdc.encoder import HyperdimensionalEncoder
from conformal.predictor import ConformalPredictor

# 1. Train HDC classifier
encoder = HyperdimensionalEncoder(d=1000, n_features=2, seed=42)
encoder.fit(X_train, y_train)

# 2. Calibrate conformal predictor (α=0.1 → ≥90% coverage)
cp = ConformalPredictor(classifier=encoder, alpha=0.10)
cp.calibrate(X_cal, y_cal)

# 3. Get prediction sets with coverage guarantee
psets = cp.predict_set(X_test)   # e.g. [[0], [1], [0, 1], [1], ...]

# 4. Verify coverage
print(cp.coverage(X_test, y_test))   # ≥ 0.90
print(cp.avg_set_size(X_test))       # efficiency: ideally close to 1.0
```

---

## Module Reference

### `hdc/encoder.py` — HyperdimensionalEncoder

```
HyperdimensionalEncoder(d=1000, n_features=2, seed=42)
  .fit(X_train, y_train)          → self
  .encode(X)                      → bipolar hypervectors ∈ {−1,+1}^d
  .predict_scores(X)              → similarity scores ∈ [0,1]^n_classes
  .predict(X)                     → point predictions
```

### `conformal/predictor.py` — ConformalPredictor

```
ConformalPredictor(classifier, alpha=0.10)
  .calibrate(X_cal, y_cal)        → self  (sets q̂)
  .predict_set(X)                 → List[List[int]]  (prediction sets)
  .predict(X)                     → point predictions
  .coverage(X_test, y_test)       → float ∈ [0,1]
  .avg_set_size(X_test)           → float (efficiency)
```

---

## Tests

```bash
# Core HDC + conformal tests (35 tests, no external deps beyond numpy)
python -m pytest tests/test_hdc.py tests/test_conformal.py -v

# Full test suite (requires PyTorch for advanced features)
python -m pytest tests/ -v
```

---

## Connection to Research

This project sits at the intersection of two active research areas:

- **HDC for edge AI**: Rahimi & Recht (2007), Kanerva (2009), Imani et al. (2019),
  and the broader brain-inspired computing literature
- **Conformal prediction**: Vovk, Gammerman & Shafer (2005); Angelopoulos & Bates (2022)
  *"A Gentle Introduction to Conformal Prediction"*

The conformal + HDC combination is particularly relevant for **embodied reasoning** (ER)
systems on constrained hardware: when a wearable device must decide whether to alert a
clinician, a prediction set that provably contains the true label is far more useful than
a miscalibrated confidence score.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
