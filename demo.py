#!/usr/bin/env python3
"""
HyperConformal Demo
===================
Demonstrates HDC classification with conformal prediction coverage guarantees
on a synthetic 2-class dataset.

Pipeline:
  1. Generate 400 samples (2 Gaussian blobs, 2 features)
  2. Split: 200 train | 100 calibration | 100 test
  3. Train HyperdimensionalEncoder (d=1000) on train set
  4. Calibrate ConformalPredictor (α=0.1) on calibration set
  5. Evaluate empirical coverage on test set — must be ≥ 1 − α = 0.90

The coverage guarantee is exact (finite-sample) for exchangeable data,
not merely asymptotic.  This is what makes conformal prediction uniquely
valuable for safety-critical edge-AI deployments.
"""

import sys
import numpy as np
from hdc.encoder import HyperdimensionalEncoder
from conformal.predictor import ConformalPredictor


def make_dataset(n_train=200, n_cal=100, n_test=100, seed=0):
    """
    Generate a balanced 2-class Gaussian dataset.

    Class 0: N([-2, 0], I)
    Class 1: N([+2, 0], I)
    """
    rng = np.random.default_rng(seed)

    def gaussian_blob(center, n):
        return rng.standard_normal((n, 2)) + np.array(center)

    half_train = n_train // 2
    X_tr0 = gaussian_blob([-2, 0], half_train)
    X_tr1 = gaussian_blob([+2, 0], half_train)
    X_train = np.vstack([X_tr0, X_tr1])
    y_train = np.array([0] * half_train + [1] * half_train)

    half_cal = n_cal // 2
    X_c0 = gaussian_blob([-2, 0], half_cal)
    X_c1 = gaussian_blob([+2, 0], half_cal)
    X_cal = np.vstack([X_c0, X_c1])
    y_cal = np.array([0] * half_cal + [1] * half_cal)

    half_test = n_test // 2
    X_t0 = gaussian_blob([-2, 0], half_test)
    X_t1 = gaussian_blob([+2, 0], half_test)
    X_test = np.vstack([X_t0, X_t1])
    y_test = np.array([0] * half_test + [1] * half_test)

    return (X_train, y_train), (X_cal, y_cal), (X_test, y_test)


def main():
    print("=" * 60)
    print("HyperConformal Demo: HDC + Conformal Prediction")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Dataset                                                           #
    # ------------------------------------------------------------------ #
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = make_dataset(
        n_train=200, n_cal=100, n_test=100, seed=42
    )
    print(f"\nDataset splits:")
    print(f"  Train      : {len(X_train)} samples")
    print(f"  Calibration: {len(X_cal)} samples")
    print(f"  Test       : {len(X_test)} samples")

    # ------------------------------------------------------------------ #
    # 2. HDC Encoder                                                       #
    # ------------------------------------------------------------------ #
    d = 1000
    encoder = HyperdimensionalEncoder(d=d, n_features=2, seed=7)
    encoder.fit(X_train, y_train)
    print(f"\nHDC Encoder trained: {encoder}")

    # Training accuracy (sanity check)
    y_train_pred = encoder.predict(X_train)
    train_acc = np.mean(y_train_pred == y_train)
    print(f"  Training accuracy  : {train_acc:.3f}")

    # ------------------------------------------------------------------ #
    # 3. Conformal Predictor                                               #
    # ------------------------------------------------------------------ #
    alpha = 0.10  # target miscoverage (want ≥ 90% empirical coverage)
    cp = ConformalPredictor(classifier=encoder, alpha=alpha)
    cp.calibrate(X_cal, y_cal)
    print(f"\nConformal predictor calibrated:")
    print(f"  α      = {alpha:.2f}  (target coverage ≥ {1-alpha:.0%})")
    print(f"  q̂      = {cp.q_hat:.4f}  (nonconformity quantile)")

    # ------------------------------------------------------------------ #
    # 4. Evaluation on test set                                            #
    # ------------------------------------------------------------------ #
    empirical_coverage = cp.coverage(X_test, y_test)
    avg_size = cp.avg_set_size(X_test)
    y_test_pred = cp.predict(X_test)
    point_accuracy = np.mean(y_test_pred == y_test)

    print(f"\nTest results:")
    print(f"  Point accuracy     : {point_accuracy:.3f}")
    print(f"  Empirical coverage : {empirical_coverage:.3f}  "
          f"({'✓' if empirical_coverage >= 1 - alpha else '✗'} ≥ {1 - alpha:.2f})")
    print(f"  Avg set size       : {avg_size:.2f}  (1.0 = singleton, 2.0 = both classes)")

    # ------------------------------------------------------------------ #
    # 5. Coverage guarantee check                                          #
    # ------------------------------------------------------------------ #
    guarantee_holds = empirical_coverage >= (1.0 - alpha)
    print(f"\nCoverage guarantee holds: {guarantee_holds}")
    if guarantee_holds:
        print(f"  ✓ Empirical coverage {empirical_coverage:.3f} ≥ {1 - alpha:.2f}")
    else:
        print(f"  ✗ Coverage {empirical_coverage:.3f} < {1 - alpha:.2f} "
              f"(can happen by chance with small test sets)")

    # ------------------------------------------------------------------ #
    # 6. Show a few example prediction sets                                #
    # ------------------------------------------------------------------ #
    print(f"\nExample prediction sets (first 8 test samples):")
    psets = cp.predict_set(X_test[:8])
    for i, (ps, yt) in enumerate(zip(psets, y_test[:8])):
        status = "✓" if int(yt) in ps else "✗"
        print(f"  [{i}] true={yt}  set={ps}  {status}")

    # ------------------------------------------------------------------ #
    # 7. Memory footprint estimate                                          #
    # ------------------------------------------------------------------ #
    proto_bytes = sum(p.nbytes for p in encoder.class_prototypes.values())
    proj_bytes = encoder.projection_matrix.nbytes
    total_kb = (proto_bytes + proj_bytes) / 1024
    print(f"\nMemory footprint:")
    print(f"  Projection matrix: {proj_bytes/1024:.1f} KB  (int8, {2}×{d})")
    print(f"  Class prototypes : {proto_bytes/1024:.1f} KB  ({2} classes × {d}D)")
    print(f"  Total            : {total_kb:.1f} KB  (fits in <1 MB MCU RAM)")

    print(f"\n{'=' * 60}")
    print("Demo complete.")
    return guarantee_holds


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
