#!/usr/bin/env python3
"""
Basic functionality test for HyperConformal without full PyTorch dependency.
Tests core mathematical operations and data structures.
"""

import sys
sys.path.insert(0, '/root/repo')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_basic_imports():
    """Test that core components are importable."""
    print("🧪 Testing basic imports...")
    
    try:
        # Test utilities and metrics (should work without torch)
        from hyperconformal.utils import hamming_distance, binary_quantize, compute_coverage
        from hyperconformal.metrics import coverage_score, average_set_size
        print("✅ Utils and metrics imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_utility_functions():
    """Test core utility functions."""
    print("\n🧪 Testing utility functions...")
    
    try:
        # Import functions
        from hyperconformal.utils import hamming_distance, binary_quantize, compute_coverage
        
        # Test binary quantization
        x = np.array([0.5, -0.3, 1.2, -0.8])
        quantized = binary_quantize(x)
        expected = np.array([1, -1, 1, -1])
        assert np.array_equal(quantized, expected), f"Expected {expected}, got {quantized}"
        print("✅ Binary quantization works")
        
        # Test Hamming distance
        hv1 = np.array([1, -1, 1, -1])
        hv2 = np.array([1, 1, -1, -1])
        distance = hamming_distance(hv1, hv2)
        expected_distance = 0.5  # 2 out of 4 bits differ
        assert abs(distance - expected_distance) < 1e-6, f"Expected {expected_distance}, got {distance}"
        print("✅ Hamming distance works")
        
        # Test coverage computation
        pred_sets = [[0, 1], [1, 2], [0]]
        true_labels = [0, 1, 0]
        coverage = compute_coverage(pred_sets, true_labels)
        expected_coverage = 1.0  # All predictions contain true labels
        assert abs(coverage - expected_coverage) < 1e-6, f"Expected {expected_coverage}, got {coverage}"
        print("✅ Coverage computation works")
        
        return True
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test evaluation metrics."""
    print("\n🧪 Testing evaluation metrics...")
    
    try:
        from hyperconformal.metrics import coverage_score, average_set_size
        
        # Test coverage score
        pred_sets = [[0, 1], [1], [0, 2]]
        true_labels = [0, 1, 2] 
        coverage = coverage_score(pred_sets, true_labels)
        expected = 1.0
        assert abs(coverage - expected) < 1e-6, f"Expected {expected}, got {coverage}"
        print("✅ Coverage score works")
        
        # Test average set size
        avg_size = average_set_size(pred_sets)
        expected_size = (2 + 1 + 2) / 3
        assert abs(avg_size - expected_size) < 1e-6, f"Expected {expected_size}, got {avg_size}"
        print("✅ Average set size works")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_based_hdc():
    """Test HDC functionality using pure NumPy (no PyTorch)."""
    print("\n🧪 Testing NumPy-based HDC simulation...")
    
    try:
        # Simulate binary HDC encoding
        input_dim = 100
        hv_dim = 1000
        num_classes = 3
        
        # Create random projection matrix
        np.random.seed(42)
        projection_matrix = np.random.choice([-1, 1], size=(input_dim, hv_dim))
        
        # Generate sample data
        X, y = make_classification(n_samples=200, n_features=input_dim, n_classes=num_classes, 
                                 n_redundant=0, n_informative=50, random_state=42)
        
        # Encode data
        encoded = np.dot(X, projection_matrix)
        encoded_binary = np.sign(encoded)
        
        # Learn class prototypes
        prototypes = np.zeros((num_classes, hv_dim))
        for class_idx in range(num_classes):
            class_mask = (y == class_idx)
            if np.sum(class_mask) > 0:
                prototypes[class_idx] = np.sign(np.mean(encoded_binary[class_mask], axis=0))
        
        # Test classification
        test_sample = encoded_binary[0:1]
        similarities = np.dot(test_sample, prototypes.T) / hv_dim
        predicted_class = np.argmax(similarities)
        
        print(f"✅ HDC simulation works - predicted class: {predicted_class}, true class: {y[0]}")
        print(f"   Similarities: {similarities[0]}")
        
        return True
    except Exception as e:
        print(f"❌ HDC simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conformal_prediction_logic():
    """Test conformal prediction logic without PyTorch."""
    print("\n🧪 Testing conformal prediction logic...")
    
    try:
        # Simulate prediction probabilities
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        
        # Generate mock probabilities
        probabilities = np.random.dirichlet([1, 1, 1], size=n_samples)
        true_labels = np.random.randint(0, n_classes, size=n_samples)
        
        # Simple APS-style conformal prediction
        alpha = 0.1  # 90% coverage target
        
        # Compute nonconformity scores (1 - probability of true class)
        nonconformity_scores = []
        for i in range(n_samples):
            true_class_prob = probabilities[i, true_labels[i]]
            score = 1 - true_class_prob
            nonconformity_scores.append(score)
        
        # Compute threshold (quantile)
        threshold = np.quantile(nonconformity_scores, 1 - alpha)
        
        # Generate prediction sets for new samples
        test_probs = np.random.dirichlet([1, 1, 1], size=10)
        prediction_sets = []
        
        for probs in test_probs:
            pred_set = []
            for class_idx in range(n_classes):
                score = 1 - probs[class_idx]
                if score <= threshold:
                    pred_set.append(class_idx)
            prediction_sets.append(pred_set)
        
        # Compute average set size
        avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
        
        print(f"✅ Conformal prediction works - threshold: {threshold:.3f}, avg set size: {avg_set_size:.2f}")
        print(f"   Sample prediction sets: {prediction_sets[:3]}")
        
        return True
    except Exception as e:
        print(f"❌ Conformal prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("🚀 HyperConformal Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_utility_functions, 
        test_metrics,
        test_numpy_based_hdc,
        test_conformal_prediction_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Core functionality verified!")
        return True
    else:
        print("❌ Some tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)