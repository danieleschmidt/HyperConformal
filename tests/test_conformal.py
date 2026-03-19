"""
Unit tests for ConformalPredictor.
"""

import numpy as np
import pytest
from hdc.encoder import HyperdimensionalEncoder
from conformal.predictor import ConformalPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trained_encoder(n_features=2, d=200, seed=0):
    """Create and train an encoder on well-separated blobs."""
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((80, n_features)) + np.array([-3] + [0] * (n_features - 1))
    X1 = rng.standard_normal((80, n_features)) + np.array([+3] + [0] * (n_features - 1))
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * 80 + [1] * 80)
    enc = HyperdimensionalEncoder(d=d, n_features=n_features, seed=seed + 1)
    enc.fit(X_train, y_train)
    return enc, X_train, y_train


def make_cal_test(n_features=2, seed=10):
    """Generate calibration and test sets."""
    rng = np.random.default_rng(seed)
    X0_cal = rng.standard_normal((50, n_features)) + np.array([-3] + [0] * (n_features - 1))
    X1_cal = rng.standard_normal((50, n_features)) + np.array([+3] + [0] * (n_features - 1))
    X_cal = np.vstack([X0_cal, X1_cal])
    y_cal = np.array([0] * 50 + [1] * 50)

    X0_test = rng.standard_normal((50, n_features)) + np.array([-3] + [0] * (n_features - 1))
    X1_test = rng.standard_normal((50, n_features)) + np.array([+3] + [0] * (n_features - 1))
    X_test = np.vstack([X0_test, X1_test])
    y_test = np.array([0] * 50 + [1] * 50)

    return (X_cal, y_cal), (X_test, y_test)


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------

class TestConformalPredictorInit:
    def test_valid_init(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        assert cp.alpha == 0.1
        assert cp.q_hat is None

    def test_invalid_alpha_zero(self):
        enc, _, _ = make_trained_encoder()
        with pytest.raises(ValueError, match="alpha"):
            ConformalPredictor(enc, alpha=0.0)

    def test_invalid_alpha_one(self):
        enc, _, _ = make_trained_encoder()
        with pytest.raises(ValueError, match="alpha"):
            ConformalPredictor(enc, alpha=1.0)

    def test_invalid_classifier(self):
        with pytest.raises(TypeError, match="predict_scores"):
            ConformalPredictor(classifier=object(), alpha=0.1)


# ---------------------------------------------------------------------------
# Tests for calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_sets_q_hat(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), _ = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        assert cp.q_hat is not None
        assert isinstance(cp.q_hat, float)

    def test_calibrate_returns_self(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), _ = make_cal_test()
        result = cp.calibrate(X_cal, y_cal)
        assert result is cp

    def test_q_hat_finite(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.05)
        (X_cal, y_cal), _ = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        assert np.isfinite(cp.q_hat)

    def test_stricter_alpha_higher_q_hat(self):
        """Smaller α → more conservative → higher q̂."""
        enc, _, _ = make_trained_encoder()
        (X_cal, y_cal), _ = make_cal_test()
        cp_strict = ConformalPredictor(enc, alpha=0.01)
        cp_loose = ConformalPredictor(enc, alpha=0.20)
        cp_strict.calibrate(X_cal, y_cal)
        cp_loose.calibrate(X_cal, y_cal)
        # Higher quantile level → higher q̂ → larger prediction sets
        assert cp_strict.q_hat >= cp_loose.q_hat


# ---------------------------------------------------------------------------
# Tests for predict_set
# ---------------------------------------------------------------------------

class TestPredictSet:
    def test_predict_before_calibrate_raises(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_set(np.array([[0.0, 0.0]]))

    def test_output_shape(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, _) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        psets = cp.predict_set(X_test)
        assert len(psets) == len(X_test)

    def test_sets_not_empty(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, _) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        psets = cp.predict_set(X_test)
        for ps in psets:
            assert len(ps) >= 1, "Prediction sets must never be empty"

    def test_sets_contain_valid_classes(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, _) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        psets = cp.predict_set(X_test)
        valid_classes = set(enc.class_prototypes.keys())
        for ps in psets:
            for c in ps:
                assert c in valid_classes


# ---------------------------------------------------------------------------
# Coverage guarantee test (the core property)
# ---------------------------------------------------------------------------

class TestCoverageGuarantee:
    def test_marginal_coverage_holds_alpha_10(self):
        """
        The conformal prediction coverage guarantee:
            P(y ∈ Ĉ(x)) ≥ 1 − α

        With n_cal=100 calibration points and n_test=100 test points,
        empirical coverage should consistently hold for α=0.10 on well-
        separated data.  We average over multiple random seeds to reduce
        test flakiness.
        """
        alpha = 0.10
        target = 1.0 - alpha

        coverages = []
        for seed in range(8):
            rng = np.random.default_rng(seed * 137)

            # Separable blobs (classification near-perfect → conservative sets)
            X0 = rng.standard_normal((100, 2)) + np.array([-4, 0])
            X1 = rng.standard_normal((100, 2)) + np.array([+4, 0])
            X_train = np.vstack([X0, X1])
            y_train = np.array([0] * 100 + [1] * 100)

            X0c = rng.standard_normal((50, 2)) + np.array([-4, 0])
            X1c = rng.standard_normal((50, 2)) + np.array([+4, 0])
            X_cal = np.vstack([X0c, X1c])
            y_cal = np.array([0] * 50 + [1] * 50)

            X0t = rng.standard_normal((50, 2)) + np.array([-4, 0])
            X1t = rng.standard_normal((50, 2)) + np.array([+4, 0])
            X_test = np.vstack([X0t, X1t])
            y_test = np.array([0] * 50 + [1] * 50)

            enc = HyperdimensionalEncoder(d=1000, n_features=2, seed=seed)
            enc.fit(X_train, y_train)

            cp = ConformalPredictor(enc, alpha=alpha)
            cp.calibrate(X_cal, y_cal)
            cov = cp.coverage(X_test, y_test)
            coverages.append(cov)

        mean_cov = np.mean(coverages)
        # Coverage should hold in the vast majority of runs
        # (all runs for separable data with large calibration sets)
        assert mean_cov >= target, (
            f"Mean empirical coverage {mean_cov:.3f} < {target:.2f}. "
            f"Per-seed coverages: {coverages}"
        )

    def test_larger_alpha_smaller_avg_set_size(self):
        """Larger α → less conservative → smaller average prediction sets."""
        enc, X_train, y_train = make_trained_encoder()
        (X_cal, y_cal), (X_test, _) = make_cal_test()

        cp_tight = ConformalPredictor(enc, alpha=0.05)
        cp_loose = ConformalPredictor(enc, alpha=0.30)
        cp_tight.calibrate(X_cal, y_cal)
        cp_loose.calibrate(X_cal, y_cal)

        size_tight = cp_tight.avg_set_size(X_test)
        size_loose = cp_loose.avg_set_size(X_test)
        # Tight (small α) → should have larger or equal sets than loose
        assert size_tight >= size_loose


# ---------------------------------------------------------------------------
# Tests for predict and coverage helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_predict_returns_array(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, _) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        preds = cp.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_coverage_returns_float(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, y_test) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        cov = cp.coverage(X_test, y_test)
        assert 0.0 <= cov <= 1.0

    def test_avg_set_size_range(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        (X_cal, y_cal), (X_test, _) = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        avg = cp.avg_set_size(X_test)
        assert 1.0 <= avg <= 2.0  # binary classification

    def test_repr(self):
        enc, _, _ = make_trained_encoder()
        cp = ConformalPredictor(enc, alpha=0.1)
        s = repr(cp)
        assert "0.1" in s
        assert "not calibrated" in s
        (X_cal, y_cal), _ = make_cal_test()
        cp.calibrate(X_cal, y_cal)
        s2 = repr(cp)
        assert "q_hat" in s2
