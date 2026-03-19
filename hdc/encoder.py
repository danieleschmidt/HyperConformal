"""
HyperdimensionalEncoder: encodes real-valued inputs as binary hypervectors.

Hyperdimensional Computing (HDC) represents data as high-dimensional binary or
bipolar vectors (hypervectors). The key insight: in high dimensions (d≥1000),
random vectors are nearly orthogonal, so the representation space is rich enough
to encode arbitrary data while supporting fast bitwise operations — ideal for
MCUs with <1MB RAM.

Implementation follows the classical HDC literature (Kanerva 2009, Rahimi et al.):
  1. Generate a fixed random projection matrix Φ ∈ {-1,+1}^{d×n_features}
  2. Encode x as  h = sign(Φᵀ x)  (bipolar binary: {-1, +1}^d)
  3. Bundle class prototypes as  P_c = sign(Σ h_i)  for training examples of class c
  4. Classify by argmax cosine similarity  cos(h, P_c)
"""

import numpy as np
from typing import Optional


class HyperdimensionalEncoder:
    """
    Encodes real-valued feature vectors as binary bipolar hypervectors.

    Parameters
    ----------
    d : int
        Hypervector dimension (default 1000; ≥1000 recommended for good
        orthogonality and classification accuracy).
    n_features : int
        Input feature dimension.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    projection_matrix : np.ndarray, shape (n_features, d)
        Fixed random bipolar projection matrix Φ ∈ {-1,+1}^{n_features × d}.
    class_prototypes : dict
        Maps class label → bipolar prototype vector in ℝ^d (real-valued
        accumulator before final binarization for inference).
    """

    def __init__(self, d: int = 1000, n_features: int = 2, seed: Optional[int] = 42):
        if d < 1:
            raise ValueError("Hypervector dimension d must be >= 1")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.d = d
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

        # Random bipolar projection matrix: each entry is ±1
        self.projection_matrix = self.rng.choice(
            [-1, 1], size=(n_features, d)
        ).astype(np.int8)

        # Class prototype accumulators (int to avoid saturation before binarize)
        self.class_prototypes: dict = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single sample or batch of samples into binary hypervectors.

        The encoding is a random projection followed by binarization:
            h = sign(Φᵀ x)   where  Φ ∈ {-1,+1}^{n_features × d}

        Zero-valued components map to +1 (arbitrary tie-breaking).

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (batch, n_features)
            Input feature vector(s). Assumed to be float-valued; no range
            restriction (the projection + sign operation handles scaling).

        Returns
        -------
        np.ndarray of int8, shape (d,) or (batch, d)
            Binary bipolar hypervector(s) with values in {-1, +1}.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.shape[1]}"
            )

        # Matrix multiply: (batch, n_features) @ (n_features, d) → (batch, d)
        projected = x @ self.projection_matrix.astype(np.float64)

        # Binarize: sign(·), with 0 → +1
        hvs = np.where(projected >= 0, np.int8(1), np.int8(-1))

        return hvs[0] if squeeze else hvs

    # ------------------------------------------------------------------
    # Training (accumulate class prototypes)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HyperdimensionalEncoder":
        """
        Train HDC classifier: build one prototype per class via superposition.

        For each class c, accumulate encoded hypervectors:
            accumulator_c += h_i  for all i where y_i == c
        The final prototype is the sign of the accumulator (majority vote).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes = np.unique(y)
        accumulators: dict = {}

        hvs = self.encode(X)  # (n_samples, d)

        for c in classes:
            mask = y == c
            # Sum hypervectors (majority vote / superposition)
            accumulators[c] = hvs[mask].sum(axis=0, dtype=np.int64)

        # Binarize prototypes (ties → +1)
        self.class_prototypes = {
            c: np.where(acc >= 0, np.int8(1), np.int8(-1))
            for c, acc in accumulators.items()
        }
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity scores between encoded inputs and class prototypes.

        Similarity is normalized to [0, 1]:
            score(x, c) = (cos_sim(h, P_c) + 1) / 2  ∈ [0, 1]

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Normalized similarity scores for each class. Rows sum may not
            equal 1 (these are similarities, not probabilities).
        """
        if not self.class_prototypes:
            raise RuntimeError("Model is not trained. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        hvs = self.encode(X)  # (n_samples, d)
        classes = sorted(self.class_prototypes.keys())

        # Stack prototypes: (n_classes, d)
        protos = np.stack(
            [self.class_prototypes[c].astype(np.float64) for c in classes]
        )

        # Cosine similarity: hvs @ protos.T / d  (both are ±1 so |hv|=|p|=√d)
        # Normalise by d (not √d * √d) since |hv| = |p| = √d
        cos_sim = hvs.astype(np.float64) @ protos.T / self.d  # (n_samples, n_classes)

        # Map to [0, 1]
        scores = (cos_sim + 1.0) / 2.0
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using nearest-prototype rule.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        scores = self.predict_scores(X)
        classes = sorted(self.class_prototypes.keys())
        return np.array([classes[i] for i in np.argmax(scores, axis=1)])

    def __repr__(self) -> str:
        n_classes = len(self.class_prototypes)
        return (
            f"HyperdimensionalEncoder("
            f"d={self.d}, n_features={self.n_features}, "
            f"n_classes={n_classes})"
        )
