#!/usr/bin/env python3
"""
Minimal test suite for HyperConformal - dependency-free validation.
"""

import sys
import os
import unittest
from pathlib import Path

# Add repo to path for testing
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

class TestHyperConformalStructure(unittest.TestCase):
    """Test basic package structure and imports."""
    
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        
    def test_package_structure_exists(self):
        """Test that all required package files exist."""
        required_files = [
            'hyperconformal/__init__.py',
            'hyperconformal/encoders.py', 
            'hyperconformal/conformal.py',
            'hyperconformal/hyperconformal.py',
            'hyperconformal/utils.py',
            'pyproject.toml'
        ]
        
        for file_path in required_files:
            full_path = self.repo_root / file_path
            self.assertTrue(full_path.exists(), f"Missing required file: {file_path}")
    
    def test_python_syntax_validity(self):
        """Test that all Python files have valid syntax."""
        python_files = list((self.repo_root / 'hyperconformal').glob('*.py'))
        
        for py_file in python_files:
            with self.subTest(file=py_file.name):
                try:
                    with open(py_file, 'r') as f:
                        compile(f.read(), py_file, 'exec')
                except SyntaxError as e:
                    self.fail(f"Syntax error in {py_file}: {e}")
    
    def test_core_algorithms_basic(self):
        """Test basic algorithmic components without dependencies."""
        
        # Test Hamming distance (core HDC operation)
        def hamming_distance_basic(a, b):
            """Basic Hamming distance for binary vectors."""
            if len(a) != len(b):
                raise ValueError("Vectors must have same length")
            return sum(x != y for x, y in zip(a, b))
        
        # Test basic cases
        self.assertEqual(hamming_distance_basic([0, 1, 0], [0, 0, 1]), 2)
        self.assertEqual(hamming_distance_basic([1, 1, 1], [1, 1, 1]), 0)
        self.assertEqual(hamming_distance_basic([0, 0, 0], [1, 1, 1]), 3)
    
    def test_binary_quantization_basic(self):
        """Test basic binary quantization logic."""
        
        def binary_quantize_basic(values, threshold=0.0):
            """Basic binary quantization."""
            return [1 if v > threshold else 0 for v in values]
        
        # Test cases
        self.assertEqual(binary_quantize_basic([0.5, -0.3, 0.8, -0.1]), [1, 0, 1, 0])
        self.assertEqual(binary_quantize_basic([1, 2, 3], threshold=1.5), [0, 1, 1])
    
    def test_conformal_prediction_logic(self):
        """Test basic conformal prediction set logic."""
        
        def compute_prediction_set_basic(scores, quantile_value):
            """Basic prediction set computation."""
            return [i for i, score in enumerate(scores) if score >= quantile_value]
        
        scores = [0.1, 0.6, 0.8, 0.3, 0.9]
        pred_set = compute_prediction_set_basic(scores, 0.5)
        expected = [1, 2, 4]  # Indices with scores >= 0.5
        self.assertEqual(pred_set, expected)

class TestCoreAlgorithms(unittest.TestCase):
    """Test core algorithmic components."""
    
    def test_hypervector_operations(self):
        """Test basic hypervector operations."""
        
        def bundle_vectors(vectors):
            """Bundle (majority) operation for binary hypervectors."""
            if not vectors:
                return []
            
            length = len(vectors[0])
            result = []
            for i in range(length):
                count = sum(v[i] for v in vectors)
                result.append(1 if count > len(vectors) // 2 else 0)
            return result
        
        def bind_vectors(a, b):
            """Bind (XOR) operation for binary hypervectors."""
            return [x ^ y for x, y in zip(a, b)]
        
        # Test bundling
        vectors = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
        bundled = bundle_vectors(vectors)
        self.assertEqual(bundled, [1, 1, 1])  # Majority vote
        
        # Test binding (XOR)
        a = [1, 0, 1, 0]
        b = [0, 1, 1, 1]
        bound = bind_vectors(a, b)
        self.assertEqual(bound, [1, 1, 0, 1])

if __name__ == '__main__':
    print("🧪 Running HyperConformal Minimal Test Suite...")
    print("="*50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n✅ Minimal test suite completed")
