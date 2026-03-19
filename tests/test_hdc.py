"""
Unit tests for HyperdimensionalEncoder.
"""

import numpy as np
import pytest
from hdc.encoder import HyperdimensionalEncoder


class TestHyperdimensionalEncoder:
    """Tests for HyperdimensionalEncoder."""

    def test_init_defaults(self):
        enc = HyperdimensionalEncoder()
        assert enc.d == 1000
        assert enc.n_features == 2
        assert enc.projection_matrix.shape == (2, 1000)

    def test_init_custom(self):
        enc = HyperdimensionalEncoder(d=512, n_features=4, seed=0)
        assert enc.d == 512
        assert enc.n_features == 4
        assert enc.projection_matrix.shape == (4, 512)

    def test_init_invalid_d(self):
        with pytest.raises(ValueError, match="d must be"):
            HyperdimensionalEncoder(d=0, n_features=2)

    def test_init_invalid_features(self):
        with pytest.raises(ValueError, match="n_features"):
            HyperdimensionalEncoder(d=10, n_features=0)

    def test_projection_matrix_bipolar(self):
        enc = HyperdimensionalEncoder(d=100, n_features=3, seed=1)
        vals = np.unique(enc.projection_matrix)
        assert set(vals).issubset({-1, 1}), "Projection matrix must be bipolar ±1"

    def test_encode_single_sample(self):
        enc = HyperdimensionalEncoder(d=100, n_features=3, seed=2)
        x = np.array([0.5, -0.3, 1.2])
        hv = enc.encode(x)
        assert hv.shape == (100,)
        assert set(np.unique(hv)).issubset({-1, 1}), "Hypervector must be bipolar"

    def test_encode_batch(self):
        enc = HyperdimensionalEncoder(d=200, n_features=4, seed=3)
        X = np.random.randn(10, 4)
        hvs = enc.encode(X)
        assert hvs.shape == (10, 200)
        assert set(np.unique(hvs)).issubset({-1, 1})

    def test_encode_wrong_features_raises(self):
        enc = HyperdimensionalEncoder(d=50, n_features=3)
        with pytest.raises(ValueError, match="features"):
            enc.encode(np.array([1.0, 2.0]))  # 2 features, expected 3

    def test_encode_reproducible(self):
        enc = HyperdimensionalEncoder(d=100, n_features=2, seed=42)
        x = np.array([1.0, -1.0])
        hv1 = enc.encode(x)
        hv2 = enc.encode(x)
        np.testing.assert_array_equal(hv1, hv2)

    def test_fit_creates_prototypes(self):
        enc = HyperdimensionalEncoder(d=100, n_features=2, seed=5)
        X = np.random.randn(20, 2)
        y = np.array([0] * 10 + [1] * 10)
        enc.fit(X, y)
        assert set(enc.class_prototypes.keys()) == {0, 1}
        for p in enc.class_prototypes.values():
            assert p.shape == (100,)
            assert set(np.unique(p)).issubset({-1, 1})

    def test_fit_returns_self(self):
        enc = HyperdimensionalEncoder(d=50, n_features=2)
        X = np.random.randn(10, 2)
        y = np.zeros(10, dtype=int)
        result = enc.fit(X, y)
        assert result is enc

    def test_predict_scores_shape(self):
        enc = HyperdimensionalEncoder(d=100, n_features=2, seed=6)
        X_train = np.random.randn(20, 2)
        y_train = np.array([0] * 10 + [1] * 10)
        enc.fit(X_train, y_train)
        X_test = np.random.randn(8, 2)
        scores = enc.predict_scores(X_test)
        assert scores.shape == (8, 2)

    def test_predict_scores_range(self):
        enc = HyperdimensionalEncoder(d=1000, n_features=2, seed=7)
        X_train = np.random.randn(40, 2)
        y_train = np.array([0] * 20 + [1] * 20)
        enc.fit(X_train, y_train)
        X_test = np.random.randn(10, 2)
        scores = enc.predict_scores(X_test)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_predict_returns_valid_labels(self):
        enc = HyperdimensionalEncoder(d=200, n_features=2, seed=8)
        X_train = np.vstack([
            np.random.randn(20, 2) + np.array([-3, 0]),
            np.random.randn(20, 2) + np.array([+3, 0]),
        ])
        y_train = np.array([0] * 20 + [1] * 20)
        enc.fit(X_train, y_train)
        X_test = X_train  # just check shapes and label validity
        preds = enc.predict(X_test)
        assert preds.shape == (40,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        enc = HyperdimensionalEncoder(d=50, n_features=2)
        with pytest.raises(RuntimeError, match="not trained"):
            enc.predict_scores(np.array([[1.0, 2.0]]))

    def test_separable_classification_accuracy(self):
        """Well-separated blobs → accuracy should be high."""
        rng = np.random.default_rng(99)
        X0 = rng.standard_normal((100, 2)) + np.array([-5, 0])
        X1 = rng.standard_normal((100, 2)) + np.array([+5, 0])
        X_train = np.vstack([X0, X1])
        y_train = np.array([0] * 100 + [1] * 100)

        enc = HyperdimensionalEncoder(d=1000, n_features=2, seed=0)
        enc.fit(X_train, y_train)
        preds = enc.predict(X_train)
        acc = np.mean(preds == y_train)
        assert acc >= 0.90, f"Expected high accuracy on separable data, got {acc:.3f}"

    def test_repr(self):
        enc = HyperdimensionalEncoder(d=500, n_features=3)
        s = repr(enc)
        assert "500" in s
        assert "n_features=3" in s
