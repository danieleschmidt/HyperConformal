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

# Global-first implementation
try:
    from .i18n import _, set_language
    from .compliance import ComplianceManager, DataRegion, ConsentType
    from .cross_platform import PlatformConfig, OptimizedOperations
    GLOBAL_FEATURES_AVAILABLE = True
except ImportError:
    GLOBAL_FEATURES_AVAILABLE = False

if GLOBAL_FEATURES_AVAILABLE:
    __all__.extend([
        "_", "set_language",
        "ComplianceManager", "DataRegion", "ConsentType", 
        "PlatformConfig", "OptimizedOperations"
    ])

# Advanced research extensions
try:
    from .quantum import QuantumHyperConformal, QuantumHDCEncoder, QuantumConformalPredictor
    from .federated import FederatedHyperConformal, FederatedClient, FederatedServer
    from .adaptive_realtime import StreamingAdaptiveConformalHDC, AdaptiveConfig
    from .neuromorphic_quantum import NeuromorphicQuantumHDC, HybridConformalPredictor
    from .advanced_benchmarks import AdvancedBenchmarkSuite, BenchmarkConfig
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

if ADVANCED_FEATURES_AVAILABLE:
    __all__.extend([
        "QuantumHyperConformal",
        "QuantumHDCEncoder", 
        "QuantumConformalPredictor",
        "FederatedHyperConformal",
        "FederatedClient",
        "FederatedServer",
        "StreamingAdaptiveConformalHDC",
        "AdaptiveConfig",
        "NeuromorphicQuantumHDC",
        "HybridConformalPredictor",
        "AdvancedBenchmarkSuite",
        "BenchmarkConfig"
    ])