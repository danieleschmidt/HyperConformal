"""
ConformalPredictor: wraps any HDC classifier with statistically valid
prediction sets using split conformal prediction.

Mathematical guarantee (Vovk et al., Angelopoulos & Bates 2022):
    For any α ∈ (0,1) and i.i.d. calibration + test data,
        P(y_test ∈ Ĉ(x_test)) ≥ 1 − α
    where the probability is over calibration randomness.

The finite-sample correction uses the (⌈(n+1)(1−α)⌉/n)-th empirical quantile
of calibration nonconformity scores.  This yields *exact* marginal coverage
(not just asymptotic) when data are exchangeable.

Score choice — margin-based nonconformity score:
    s(x, y) = 1 − (score_y(x) − max_{y'≠y} score_{y'}(x))

A high score means the true class was NOT clearly better than competitors.
The calibration quantile q̂ then defines a threshold: include all y with
score(x, y) ≤ q̂ in the prediction set.  This equals including all classes
whose margin from the best competitor is ≥ 1 − q̂, which is equivalent to
LABEL CONDITIONAL coverage when scores are well-calibrated.
"""

import numpy as np
from typing import Any, List, Optional


class ConformalPredictor:
    """
    Split conformal predictor that wraps an HDC classifier.

    Works with any classifier exposing a ``predict_scores(X)`` method that
    returns an array of shape (n_samples, n_classes) with higher = more
    likely.

    Parameters
    ----------
    classifier : object
        Trained HDC classifier with ``predict_scores(X)`` method.
    alpha : float
        Miscoverage level in (0, 1).  The coverage guarantee is ≥ 1 − alpha.
    n_classes : int, optional
        Number of classes.  Inferred from first calibration call if not given.

    Attributes
    ----------
    q_hat : float or None
        Conformal quantile computed during calibration.
    calibration_scores : np.ndarray or None
        Nonconformity scores from the calibration set.
    """

    def __init__(
        self,
        classifier: Any,
        alpha: float = 0.1,
        n_classes: Optional[int] = None,
    ):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not hasattr(classifier, "predict_scores"):
            raise TypeError(
                "classifier must have a predict_scores(X) method"
            )

        self.classifier = classifier
        self.alpha = alpha
        self.n_classes = n_classes
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Nonconformity score
    # ------------------------------------------------------------------

    def _nonconformity_score(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute margin-based nonconformity score for each calibration point.

        For sample i with true class y_i:
            s_i = 1 − (f_{y_i}(x_i) − max_{y ≠ y_i} f_y(x_i))

        where f_y(x) is the similarity score for class y.

        A small s_i means the true class was comfortably ranked first
        (low nonconformity).  A large s_i (≤ 2 in principle, since margin
        ∈ [−1, 1]) means the classifier was unsure or wrong.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        n = scores.shape[0]
        n_cls = scores.shape[1]

        # Score for true class
        true_scores = scores[np.arange(n), labels.astype(int)]

        # Max score among other classes
        # Zero out true class temporarily using a masked array approach
        other_max = np.full(n, -np.inf)
        for c in range(n_cls):
            mask = labels.astype(int) != c
            other_max = np.where(mask & (scores[:, c] > other_max), scores[:, c], other_max)

        margin = true_scores - other_max  # ∈ (−1, 1] when scores ∈ [0,1]
        return 1.0 - margin  # nonconformity: high = more nonconforming

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """
        Calibrate conformal quantile on a held-out calibration set.

        Computes margin-based nonconformity scores for each calibration
        example, then sets the conformal threshold q̂ as the
            ⌈(n+1)(1−α)⌉ / n
        quantile of those scores, which guarantees marginal coverage ≥ 1−α
        for exchangeable test data.

        Parameters
        ----------
        X_cal : np.ndarray, shape (n_cal, n_features)
        y_cal : np.ndarray, shape (n_cal,)
            Integer class labels in {0, 1, ..., n_classes-1}.

        Returns
        -------
        self
        """
        y_cal = np.asarray(y_cal, dtype=int)
        scores = self.classifier.predict_scores(X_cal)  # (n_cal, n_classes)

        if self.n_classes is None:
            self.n_classes = scores.shape[1]

        self.calibration_scores = self._nonconformity_score(scores, y_cal)

        n = len(self.calibration_scores)
        # Finite-sample-corrected level (Angelopoulos & Bates, 2022, Eq. 3)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)  # clamp to 1 for very small n
        self.q_hat = float(np.quantile(self.calibration_scores, level))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_set(self, X: np.ndarray) -> List[List[int]]:
        """
        Produce prediction sets with marginal coverage ≥ 1 − alpha.

        A class y is included in the prediction set Ĉ(x) when its
        margin-based nonconformity score does NOT exceed q̂:
            y ∈ Ĉ(x)  iff  1 − (f_y(x) − max_{y'≠y} f_{y'}(x)) ≤ q̂
        Equivalently: include y if the score for y is within q̂ of the
        best competing class.

        At least one class is always included (the argmax), so sets are
        never empty.

        Parameters
        ----------
        X : np.ndarray, shape (n_test, n_features) or (n_features,)

        Returns
        -------
        List[List[int]]
            One prediction set (list of class indices) per test sample.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")

        X = np.atleast_2d(X)
        scores = self.classifier.predict_scores(X)  # (n_test, n_classes)
        n_test, n_cls = scores.shape

        prediction_sets = []
        for i in range(n_test):
            s = scores[i]  # (n_classes,)
            # Nonconformity score for each candidate class y:
            #   nc(y) = 1 − (s_y − max_{y'≠y} s_{y'})
            nc = np.empty(n_cls)
            for y in range(n_cls):
                other = np.concatenate([s[:y], s[y + 1:]])
                nc[y] = 1.0 - (s[y] - other.max())

            included = list(np.where(nc <= self.q_hat)[0])

            # Safety: always include argmax (never return empty set)
            argmax = int(np.argmax(s))
            if argmax not in included:
                included.append(argmax)

            prediction_sets.append(sorted(included))

        return prediction_sets

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions (argmax of classifier scores)."""
        X = np.atleast_2d(X)
        scores = self.classifier.predict_scores(X)
        classes = sorted(self.classifier.class_prototypes.keys())
        return np.array([classes[i] for i in np.argmax(scores, axis=1)])

    # ------------------------------------------------------------------
    # Coverage evaluation
    # ------------------------------------------------------------------

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute empirical coverage: fraction of test points where the
        true label is in the prediction set.

        Parameters
        ----------
        X : np.ndarray, shape (n_test, n_features)
        y : np.ndarray, shape (n_test,)

        Returns
        -------
        float in [0, 1]
        """
        y = np.asarray(y, dtype=int)
        psets = self.predict_set(X)
        hits = sum(int(y[i]) in psets[i] for i in range(len(y)))
        return hits / len(y)

    def avg_set_size(self, X: np.ndarray) -> float:
        """Average prediction set size (efficiency metric; smaller = better)."""
        psets = self.predict_set(X)
        return np.mean([len(s) for s in psets])

    def __repr__(self) -> str:
        q = f"{self.q_hat:.4f}" if self.q_hat is not None else "not calibrated"
        return (
            f"ConformalPredictor("
            f"alpha={self.alpha}, q_hat={q}, "
            f"n_classes={self.n_classes})"
        )
