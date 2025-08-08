#!/usr/bin/env python3
"""
HyperConformal Demo - Dependency-free implementation
Shows core concepts without requiring NumPy/PyTorch
"""

import random
import math
from typing import List, Tuple

class SimpleHDCEncoder:
    """Simplified HDC encoder for demonstration."""
    
    def __init__(self, input_dim: int, hv_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        random.seed(seed)
        
        # Generate random projection matrix (binary)
        self.projection_matrix = []
        for i in range(input_dim):
            row = [random.choice([0, 1]) for _ in range(hv_dim)]
            self.projection_matrix.append(row)
    
    def encode(self, x: List[float]) -> List[int]:
        """Encode input vector to binary hypervector."""
        if len(x) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: {len(x)} != {self.input_dim}")
        
        # Binary quantization of input
        x_binary = [1 if val > 0 else 0 for val in x]
        
        # Hypervector computation (binary random projection)
        hv = [0] * self.hv_dim
        for i, x_bit in enumerate(x_binary):
            if x_bit == 1:
                for j in range(self.hv_dim):
                    hv[j] ^= self.projection_matrix[i][j]  # XOR binding
        
        return hv

class SimpleConformalPredictor:
    """Simplified conformal predictor for demonstration."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Significance level (1-confidence)
        self.calibration_scores = []
        
    def calibrate(self, scores: List[float]):
        """Calibrate using conformity scores from validation set."""
        self.calibration_scores = sorted(scores)
    
    def get_quantile(self) -> float:
        """Get conformal quantile for prediction sets."""
        if not self.calibration_scores:
            return 0.0
        
        # Compute quantile: (n+1)(1-alpha)/n
        n = len(self.calibration_scores)
        q_index = math.ceil((n + 1) * (1 - self.alpha)) - 1
        q_index = max(0, min(q_index, n - 1))
        
        return self.calibration_scores[q_index]
    
    def predict_set(self, class_scores: List[float]) -> List[int]:
        """Generate prediction set with coverage guarantee."""
        quantile = self.get_quantile()
        
        # Include all classes with score >= quantile
        prediction_set = [i for i, score in enumerate(class_scores) 
                         if score >= quantile]
        
        return prediction_set

def hamming_distance(a: List[int], b: List[int]) -> int:
    """Compute Hamming distance between binary vectors."""
    return sum(x != y for x, y in zip(a, b))

def demo_hyperconformal_basic():
    """Demonstrate HyperConformal concepts with simple example."""
    print("🚀 HyperConformal Demo - Basic Implementation")
    print("="*50)
    
    # Setup
    input_dim = 8
    hv_dim = 64
    num_classes = 3
    
    # Create encoder
    encoder = SimpleHDCEncoder(input_dim, hv_dim)
    print(f"📊 HDC Encoder: {input_dim}D → {hv_dim}D binary hypervectors")
    
    # Sample data points
    sample_inputs = [
        [0.2, 0.8, -0.3, 0.1, 0.9, -0.5, 0.6, 0.4],  # Class 0
        [0.7, 0.1, 0.8, -0.2, 0.3, 0.9, -0.1, 0.5],  # Class 1  
        [-0.3, 0.6, 0.2, 0.8, -0.4, 0.1, 0.7, -0.2], # Class 2
    ]
    
    # Create class prototypes
    class_prototypes = []
    for i, input_vec in enumerate(sample_inputs):
        hv = encoder.encode(input_vec)
        class_prototypes.append(hv)
        print(f"Class {i} prototype: {sum(hv)}/{len(hv)} bits set")
    
    # Test classification
    test_input = [0.1, 0.9, -0.2, 0.3, 0.8, -0.3, 0.5, 0.2]
    test_hv = encoder.encode(test_input)
    
    # Compute similarities (inverse Hamming distance)
    similarities = []
    for i, prototype in enumerate(class_prototypes):
        dist = hamming_distance(test_hv, prototype)
        similarity = 1.0 - (dist / hv_dim)  # Normalize to [0,1]
        similarities.append(similarity)
        print(f"Similarity to class {i}: {similarity:.3f}")
    
    # Conformal prediction
    predictor = SimpleConformalPredictor(alpha=0.2)  # 80% confidence
    
    # Mock calibration scores (would come from validation set)
    calibration_scores = [0.7, 0.8, 0.6, 0.9, 0.5, 0.8, 0.7, 0.6, 0.9, 0.4]
    predictor.calibrate(calibration_scores)
    
    # Generate prediction set
    prediction_set = predictor.predict_set(similarities)
    
    print(f"\n🎯 Prediction Results:")
    print(f"Test similarities: {[f'{s:.3f}' for s in similarities]}")
    print(f"Conformal quantile: {predictor.get_quantile():.3f}")
    print(f"Prediction set: {prediction_set}")
    print(f"Coverage guarantee: ≥{(1-predictor.alpha)*100:.0f}%")
    
    if prediction_set:
        print(f"✅ Predicted classes: {prediction_set}")
    else:
        print("⚠️ Empty prediction set (rare event)")
    
    print("\n🎉 Demo completed successfully!")
    
    return {
        'input_dim': input_dim,
        'hv_dim': hv_dim,
        'similarities': similarities,
        'prediction_set': prediction_set,
        'coverage_level': 1 - predictor.alpha
    }

if __name__ == "__main__":
    demo_hyperconformal_basic()
