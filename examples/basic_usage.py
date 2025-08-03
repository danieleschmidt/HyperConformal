#!/usr/bin/env python3
"""
Basic usage example for HyperConformal library.

This example demonstrates how to use HyperConformal for calibrated
uncertainty quantification with hyperdimensional computing.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import HyperConformal components
from hyperconformal import ConformalHDC, RandomProjection
from hyperconformal.metrics import conformal_prediction_metrics


def generate_sample_data():
    """Generate sample classification data."""
    print("Generating sample classification data...")
    
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_classes=5,
        n_informative=40,
        n_redundant=10,
        random_state=42
    )
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    
    return X_train, X_test, y_train, y_test


def main():
    """Run basic HyperConformal example."""
    print("=== HyperConformal Basic Usage Example ===\n")
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_sample_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(torch.unique(y_train))}\n")
    
    # Create HDC encoder
    print("Creating RandomProjection HDC encoder...")
    encoder = RandomProjection(
        input_dim=X_train.shape[1],
        hv_dim=2000,  # Hypervector dimension
        quantization='binary',  # Binary quantization for efficiency
        seed=42
    )
    
    # Create ConformalHDC model
    print("Creating ConformalHDC model...")
    alpha = 0.1  # Target 90% coverage
    model = ConformalHDC(
        encoder=encoder,
        num_classes=5,
        alpha=alpha,
        score_type='aps',  # Adaptive Prediction Sets
        calibration_split=0.2
    )
    
    # Train the model
    print("Training ConformalHDC model...")
    model.fit(X_train, y_train)
    
    # Print model summary
    print("\nModel Summary:")
    summary = model.summary()
    for key, value in summary.items():
        if key != 'memory_footprint':
            print(f"  {key}: {value}")
    
    # Memory footprint
    memory = summary['memory_footprint']
    print(f"  Memory footprint:")
    for component, size in memory.items():
        print(f"    {component}: {size:,} bytes")
    
    # Make predictions
    print(f"\nMaking predictions on {len(X_test)} test examples...")
    
    # Point predictions
    pred_classes = model.predict(X_test)
    accuracy = (pred_classes == y_test.numpy()).mean()
    print(f"Point prediction accuracy: {accuracy:.3f}")
    
    # Prediction sets with uncertainty quantification
    pred_sets = model.predict_set(X_test)
    
    # Compute coverage and efficiency metrics
    metrics = conformal_prediction_metrics(
        pred_sets, y_test, num_classes=5, target_coverage=1-alpha
    )
    
    print(f"\nConformal Prediction Results:")
    print(f"  Target coverage: {1-alpha:.1%}")
    print(f"  Empirical coverage: {metrics['coverage']:.1%}")
    print(f"  Coverage gap: {metrics['coverage_gap']:.3f}")
    print(f"  Average set size: {metrics['average_set_size']:.2f}")
    print(f"  Efficiency score: {metrics['efficiency']:.3f}")
    print(f"  Coverage-Width Criterion: {metrics['cwc']:.3f}")
    
    # Show some example predictions
    print(f"\nExample prediction sets:")
    for i in range(min(10, len(pred_sets))):
        true_label = y_test[i].item()
        pred_set = pred_sets[i]
        covered = "✓" if true_label in pred_set else "✗"
        print(f"  Example {i+1}: True={true_label}, Set={pred_set} {covered}")
    
    # Class-specific coverage
    print(f"\nPer-class coverage:")
    for class_idx in range(5):
        if f'coverage_class_{class_idx}' in metrics:
            cov = metrics[f'coverage_class_{class_idx}']
            print(f"  Class {class_idx}: {cov:.1%}")
    
    print(f"\n=== Summary ===")
    print(f"✓ Successfully implemented calibrated uncertainty quantification")
    print(f"✓ HDC model achieves {accuracy:.1%} accuracy with {metrics['coverage']:.1%} coverage")
    print(f"✓ Average prediction set size: {metrics['average_set_size']:.2f}")
    print(f"✓ Memory footprint: {memory['total']:,} bytes")
    print(f"✓ Coverage guarantee: ≥{1-alpha:.1%} (theoretical)")


if __name__ == "__main__":
    main()