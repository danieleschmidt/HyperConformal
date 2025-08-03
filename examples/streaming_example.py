#!/usr/bin/env python3
"""
Streaming/adaptive example for HyperConformal library.

This example demonstrates adaptive conformal prediction for streaming data
where the model continuously updates its calibration.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification

from hyperconformal import AdaptiveConformalHDC, RandomProjection


def generate_streaming_batches(n_batches=10, batch_size=100):
    """Generate streaming data batches with potential distribution shift."""
    print(f"Generating {n_batches} streaming batches of size {batch_size}...")
    
    batches = []
    for i in range(n_batches):
        # Simulate gradual distribution shift
        shift_factor = i * 0.1
        
        X, y = make_classification(
            n_samples=batch_size,
            n_features=30,
            n_classes=3,
            n_informative=20,
            n_redundant=10,
            flip_y=shift_factor * 0.05,  # Gradually increase label noise
            random_state=42 + i
        )
        
        # Add feature shift
        X = X + shift_factor * np.random.randn(*X.shape) * 0.1
        
        batches.append((
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long()
        ))
    
    return batches


def main():
    """Run streaming HyperConformal example."""
    print("=== HyperConformal Streaming Example ===\n")
    
    # Generate streaming data
    streaming_batches = generate_streaming_batches(n_batches=8, batch_size=150)
    
    # Create adaptive HDC encoder
    print("Creating AdaptiveConformalHDC model...")
    encoder = RandomProjection(
        input_dim=30,
        hv_dim=1000,
        quantization='binary',
        seed=42
    )
    
    # Create adaptive conformal model
    alpha = 0.1  # Target 90% coverage
    model = AdaptiveConformalHDC(
        encoder=encoder,
        num_classes=3,
        alpha=alpha,
        window_size=500,  # Sliding window for calibration
        update_frequency=50,  # Update calibration every 50 samples
        score_type='aps'
    )
    
    # Initial training on first batch
    X_init, y_init = streaming_batches[0]
    print(f"Initial training on batch 0: {X_init.shape[0]} samples")
    model.fit(X_init, y_init)
    
    print(f"\nProcessing streaming batches...")
    
    # Process streaming batches
    coverages = []
    set_sizes = []
    
    for batch_idx, (X_batch, y_batch) in enumerate(streaming_batches[1:], 1):
        print(f"\nBatch {batch_idx}:")
        
        # Make predictions before updating
        pred_sets = model.predict_set(X_batch)
        
        # Compute metrics
        coverage = sum(
            1 for i, pred_set in enumerate(pred_sets)
            if y_batch[i].item() in pred_set
        ) / len(pred_sets)
        
        avg_set_size = np.mean([len(pred_set) for pred_set in pred_sets])
        
        coverages.append(coverage)
        set_sizes.append(avg_set_size)
        
        print(f"  Coverage: {coverage:.1%}")
        print(f"  Avg set size: {avg_set_size:.2f}")
        
        # Get current coverage estimate from model
        current_est = model.get_current_coverage_estimate()
        if current_est is not None:
            print(f"  Model coverage estimate: {current_est:.1%}")
        
        # Update model with new data
        model.update(X_batch, y_batch)
        
        # Show some example predictions
        if batch_idx <= 3:  # Show examples for first few batches
            print(f"  Example predictions:")
            for i in range(min(5, len(pred_sets))):
                true_label = y_batch[i].item()
                pred_set = pred_sets[i]
                covered = "✓" if true_label in pred_set else "✗"
                print(f"    {i+1}: True={true_label}, Set={pred_set} {covered}")
    
    # Summary statistics
    print(f"\n=== Streaming Results Summary ===")
    print(f"Target coverage: {1-alpha:.1%}")
    print(f"Average coverage: {np.mean(coverages):.1%}")
    print(f"Coverage std: {np.std(coverages):.3f}")
    print(f"Coverage range: [{np.min(coverages):.1%}, {np.max(coverages):.1%}]")
    print(f"Average set size: {np.mean(set_sizes):.2f}")
    print(f"Set size std: {np.std(set_sizes):.3f}")
    
    # Show coverage evolution
    print(f"\nCoverage evolution over time:")
    for i, (cov, size) in enumerate(zip(coverages, set_sizes)):
        print(f"  Batch {i+1}: {cov:.1%} coverage, {size:.2f} avg size")
    
    # Model final state
    final_summary = model.summary()
    print(f"\nFinal model state:")
    print(f"  Training accuracy: {final_summary['training_accuracy']:.1%}")
    print(f"  Memory footprint: {final_summary['memory_footprint']['total']:,} bytes")
    
    print(f"\n=== Adaptive Learning Benefits ===")
    print(f"✓ Model adapts to distribution shift in streaming data")
    print(f"✓ Maintains calibrated uncertainty quantification online")
    print(f"✓ Efficient memory usage with sliding window calibration")
    print(f"✓ Real-time coverage monitoring and adjustment")


if __name__ == "__main__":
    main()