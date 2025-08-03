"""
Pytest configuration and shared fixtures for HyperConformal tests.

This module provides common test fixtures, utilities, and configuration
that can be shared across all test modules.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import train_test_split

from hyperconformal import RandomProjection, LevelHDC, ConformalHDC


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def make_classification(
    n_samples: int = 100,
    n_features: int = 20,
    n_classes: int = 3,
    n_informative: int = None,
    random_state: int = 42,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate classification dataset as PyTorch tensors.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        n_informative: Number of informative features (default: n_features)
        random_state: Random seed
        **kwargs: Additional arguments for sklearn.make_classification
    
    Returns:
        Tuple of (X, y) as torch tensors
    """
    if n_informative is None:
        n_informative = max(n_features // 2, 1)
    
    X, y = sklearn_make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=min(n_features - n_informative, 2),
        random_state=random_state,
        **kwargs
    )
    
    return torch.FloatTensor(X), torch.LongTensor(y)


@pytest.fixture
def small_classification_data():
    """Small classification dataset for quick tests."""
    return make_classification(n_samples=50, n_features=10, n_classes=3, random_state=42)


@pytest.fixture
def medium_classification_data():
    """Medium classification dataset for integration tests."""
    return make_classification(n_samples=200, n_features=20, n_classes=3, random_state=42)


@pytest.fixture
def large_classification_data():
    """Large classification dataset for performance tests."""
    return make_classification(n_samples=1000, n_features=50, n_classes=5, random_state=42)


@pytest.fixture
def multiclass_data():
    """Multi-class classification dataset."""
    return make_classification(n_samples=500, n_features=25, n_classes=10, random_state=42)


@pytest.fixture
def imbalanced_data():
    """Imbalanced classification dataset."""
    X, y = sklearn_make_classification(
        n_samples=300,
        n_features=15,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],  # Imbalanced
        random_state=42
    )
    return torch.FloatTensor(X), torch.LongTensor(y)


@pytest.fixture
def streaming_data():
    """Streaming data batches for adaptive testing."""
    batches = []
    for i in range(5):
        X, y = make_classification(
            n_samples=100,
            n_features=15,
            n_classes=3,
            random_state=42 + i
        )
        batches.append((X, y))
    return batches


@pytest.fixture
def sample_data():
    """General purpose sample data."""
    return make_classification(n_samples=100, n_features=20, n_classes=3, random_state=42)


@pytest.fixture
def binary_encoder():
    """Binary RandomProjection encoder."""
    return RandomProjection(
        input_dim=20,
        hv_dim=500,
        quantization='binary',
        seed=42
    )


@pytest.fixture
def ternary_encoder():
    """Ternary RandomProjection encoder."""
    return RandomProjection(
        input_dim=20,
        hv_dim=500,
        quantization='ternary',
        seed=42
    )


@pytest.fixture
def level_encoder():
    """LevelHDC encoder."""
    return LevelHDC(
        input_dim=20,
        hv_dim=400,
        levels=50,
        seed=42
    )


@pytest.fixture
def trained_model(binary_encoder, sample_data):
    """Pre-trained ConformalHDC model."""
    X, y = sample_data
    model = ConformalHDC(binary_encoder, num_classes=3, alpha=0.1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        'coverage_tolerance': 0.15,  # Allow 15% deviation from target coverage
        'performance_timeout': 30.0,  # Max 30 seconds for performance tests
        'memory_limit_mb': 1000,  # Max 1GB memory usage
        'min_accuracy': 0.5,  # Minimum acceptable accuracy
        'max_set_size': 5.0,  # Maximum average prediction set size
    }


class TestDataManager:
    """Utility class for managing test data."""
    
    @staticmethod
    def split_data(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.3, random_state: int = 42):
        """Split data into train/test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    @staticmethod
    def create_cv_folds(X: torch.Tensor, y: torch.Tensor, n_folds: int = 3):
        """Create cross-validation folds."""
        n_samples = len(X)
        fold_size = n_samples // n_folds
        indices = torch.randperm(n_samples)
        
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = torch.cat([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            folds.append((X_train, X_test, y_train, y_test))
        
        return folds
    
    @staticmethod
    def add_noise(X: torch.Tensor, noise_level: float = 0.1):
        """Add Gaussian noise to features."""
        noise = torch.randn_like(X) * noise_level
        return X + noise
    
    @staticmethod
    def normalize_features(X: torch.Tensor):
        """Normalize features to zero mean and unit variance."""
        return (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)


@pytest.fixture
def data_manager():
    """Test data manager instance."""
    return TestDataManager()


# Performance tracking fixtures
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    metrics = {}
    
    def track(name: str, value: float):
        if name not in metrics:
            metrics[name] = []
        metrics[name].append(value)
    
    def get_metrics():
        return metrics
    
    # Add methods to the function
    track.get_metrics = get_metrics
    return track


# Custom markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "memory_intensive: Memory intensive tests")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "test_benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif any(fname in str(item.fspath) for fname in ["test_conformal", "test_encoders", "test_metrics", "test_utils", "test_hyperconformal"]):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["large", "performance", "benchmark", "scalability"]):
            item.add_marker(pytest.mark.slow)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup after test if needed


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup for entire test session."""
    print("\n" + "="*60)
    print("Starting HyperConformal Test Suite")
    print("="*60)
    
    # Set environment variables for testing
    import os
    os.environ['HYPERCONFORMAL_TEST_MODE'] = '1'
    
    yield
    
    print("\n" + "="*60)
    print("HyperConformal Test Suite Complete")
    print("="*60)


# Utility functions available to all tests
def assert_coverage_reasonable(coverage: float, target: float, tolerance: float = 0.2):
    """Assert that coverage is within reasonable bounds of target."""
    assert target - tolerance <= coverage <= 1.0, \
        f"Coverage {coverage:.3f} not within reasonable bounds of target {target:.3f} ± {tolerance:.3f}"


def assert_set_sizes_reasonable(avg_size: float, max_classes: int):
    """Assert that average set sizes are reasonable."""
    assert 1.0 <= avg_size <= max_classes, \
        f"Average set size {avg_size:.3f} not reasonable for {max_classes} classes"


def assert_performance_acceptable(time_taken: float, max_time: float, operation: str):
    """Assert that performance is acceptable."""
    assert time_taken <= max_time, \
        f"{operation} took {time_taken:.3f}s, exceeding limit of {max_time:.3f}s"


# Make utility functions available to all test modules
pytest.assert_coverage_reasonable = assert_coverage_reasonable
pytest.assert_set_sizes_reasonable = assert_set_sizes_reasonable
pytest.assert_performance_acceptable = assert_performance_acceptable