"""
HyperConformal: Calibrated uncertainty quantification for hyperdimensional computing
with conformal prediction.

This package provides:
- Base HDC encoder classes for different quantization schemes
- Conformal prediction algorithms for uncertainty quantification
- Integration between HDC and conformal prediction for calibrated uncertainty
"""

from .encoders import BaseEncoder, RandomProjection, LevelHDC, ComplexHDC
from .conformal import ConformalPredictor, AdaptiveConformalPredictor, ClassificationConformalPredictor
from .hyperconformal import ConformalHDC, AdaptiveConformalHDC
from .optimized import OptimizedConformalHDC, ScalableAdaptiveConformalHDC
from .utils import compute_coverage, hamming_distance, binary_quantize
from .metrics import coverage_score, average_set_size, conditional_coverage

# Neuromorphic extensions
try:
    from .neuromorphic import SpikingHDCEncoder, SpikingConformalPredictor, NeuromorphicConformalHDC
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Terragon Labs"

__all__ = [
    "BaseEncoder",
    "RandomProjection", 
    "LevelHDC",
    "ComplexHDC",
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "ClassificationConformalPredictor",
    "ConformalHDC",
    "AdaptiveConformalHDC",
    "OptimizedConformalHDC",
    "ScalableAdaptiveConformalHDC",
    "compute_coverage",
    "hamming_distance",
    "binary_quantize",
    "coverage_score",
    "average_set_size",
    "conditional_coverage",
]

# Add neuromorphic classes if available
if NEUROMORPHIC_AVAILABLE:
    __all__.extend([
        "SpikingHDCEncoder",
        "SpikingConformalPredictor", 
        "NeuromorphicConformalHDC"
    ])