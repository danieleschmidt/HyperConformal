"""
HyperConformal: Calibrated uncertainty quantification for hyperdimensional computing
with conformal prediction.

This package provides:
- Base HDC encoder classes for different quantization schemes
- Conformal prediction algorithms for uncertainty quantification
- Integration between HDC and conformal prediction for calibrated uncertainty
"""

from .encoders import BaseEncoder, RandomProjection, LevelHDC, ComplexHDC
from .conformal import ConformalPredictor, AdaptiveConformalPredictor
from .hyperconformal import ConformalHDC, AdaptiveConformalHDC
from .utils import compute_coverage, hamming_distance, binary_quantize
from .metrics import coverage_score, average_set_size, conditional_coverage

__version__ = "0.1.0"
__author__ = "Terragon Labs"

__all__ = [
    "BaseEncoder",
    "RandomProjection", 
    "LevelHDC",
    "ComplexHDC",
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "ConformalHDC",
    "AdaptiveConformalHDC",
    "compute_coverage",
    "hamming_distance",
    "binary_quantize",
    "coverage_score",
    "average_set_size",
    "conditional_coverage",
]