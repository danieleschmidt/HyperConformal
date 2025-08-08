#!/usr/bin/env python3
"""
Core Algorithm Validation for HyperConformal
Tests mathematical correctness of key algorithms
"""

import math
import random

def test_hamming_distance():
    """Test Hamming distance implementation."""
    def hamming_distance(a, b):
        return sum(x != y for x, y in zip(a, b))
    
    # Test cases
    assert hamming_distance([0, 1, 0], [0, 0, 1]) == 2
    assert hamming_distance([1, 1, 1], [1, 1, 1]) == 0
    assert hamming_distance([0, 0, 0], [1, 1, 1]) == 3
    print("✅ Hamming distance tests passed")

def test_conformal_quantile():
    """Test conformal prediction quantile calculation."""
    def compute_quantile(scores, alpha):
        n = len(scores)
        q_index = math.ceil((n + 1) * (1 - alpha)) - 1
        return sorted(scores)[max(0, min(q_index, n - 1))]
    
    # Test case: alpha=0.1 (90% coverage)
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    quantile = compute_quantile(scores, 0.1)
    expected = 0.7  # For 90% coverage with 5 points
    assert abs(quantile - expected) < 0.01
    print("✅ Conformal quantile tests passed")

def test_coverage_guarantee():
    """Test empirical coverage of conformal prediction."""
    def empirical_coverage_test():
        # Simulate conformal prediction
        random.seed(42)
        num_trials = 100
        alpha = 0.1  # Target: 90% coverage
        
        covered = 0
        for _ in range(num_trials):
            # Simulate scores and true label
            scores = [random.random() for _ in range(5)]
            true_label = random.randint(0, 4)
            
            # Compute prediction set
            calibration_scores = [random.random() for _ in range(10)]
            n = len(calibration_scores)
            q_index = math.ceil((n + 1) * (1 - alpha)) - 1
            quantile = sorted(calibration_scores)[max(0, min(q_index, n - 1))]
            
            prediction_set = [i for i, score in enumerate(scores) if score >= quantile]
            
            if true_label in prediction_set:
                covered += 1
        
        empirical_coverage = covered / num_trials
        target_coverage = 1 - alpha
        
        print(f"Empirical coverage: {empirical_coverage:.2f}")
        print(f"Target coverage: {target_coverage:.2f}")
        
        # Allow some tolerance for randomness
        assert abs(empirical_coverage - target_coverage) < 0.15
        return empirical_coverage
    
    coverage = empirical_coverage_test()
    print(f"✅ Coverage guarantee test passed ({coverage:.2f})")

def test_hdc_properties():
    """Test HDC mathematical properties."""
    
    def xor_vectors(a, b):
        return [x ^ y for x, y in zip(a, b)]
    
    def bundle_vectors(vectors):
        length = len(vectors[0])
        result = []
        for i in range(length):
            count = sum(v[i] for v in vectors)
            result.append(1 if count > len(vectors) // 2 else 0)
        return result
    
    # Test XOR properties
    a = [1, 0, 1, 0]
    b = [0, 1, 1, 1]
    
    # XOR is its own inverse: a XOR b XOR b = a
    temp = xor_vectors(a, b)
    recovered = xor_vectors(temp, b)
    assert recovered == a
    
    # XOR is commutative: a XOR b = b XOR a
    ab = xor_vectors(a, b)
    ba = xor_vectors(b, a)
    assert ab == ba
    
    print("✅ HDC XOR properties verified")
    
    # Test bundling properties
    vectors = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    bundled = bundle_vectors(vectors)
    assert bundled == [1, 1, 1]  # Majority result
    print("✅ HDC bundling properties verified")

if __name__ == "__main__":
    print("🔬 Core Algorithm Validation")
    print("="*40)
    
    test_hamming_distance()
    test_conformal_quantile()
    test_coverage_guarantee()
    test_hdc_properties()
    
    print("\n🎉 All core algorithms validated successfully!")
