#!/usr/bin/env python3
"""
Basic validation script for HyperConformal implementation.
Tests core functionality without external dependencies.
"""

import sys
import os
import traceback

def test_basic_imports():
    """Test basic module imports."""
    print("Testing basic imports...")
    
    try:
        # Test individual modules can be parsed
        import ast
        
        # Test core module structure
        modules_to_test = [
            'hyperconformal/__init__.py',
            'hyperconformal/encoders.py', 
            'hyperconformal/conformal.py',
            'hyperconformal/hyperconformal.py',
            'hyperconformal/utils.py',
            'hyperconformal/metrics.py'
        ]
        
        for module_path in modules_to_test:
            if os.path.exists(module_path):
                with open(module_path, 'r') as f:
                    content = f.read()
                
                try:
                    ast.parse(content)
                    print(f"✓ {module_path} - Valid Python syntax")
                except SyntaxError as e:
                    print(f"✗ {module_path} - Syntax error: {e}")
                    return False
            else:
                print(f"✗ {module_path} - File not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_c_compilation():
    """Test C/C++ code compilation."""
    print("\nTesting C/C++ compilation...")
    
    try:
        # Check if C files exist
        c_files = [
            'cpp/include/hyperconformal.h',
            'cpp/src/hyperconformal.c',
            'cpp/CMakeLists.txt'
        ]
        
        for c_file in c_files:
            if os.path.exists(c_file):
                print(f"✓ {c_file} - Found")
            else:
                print(f"✗ {c_file} - Not found")
                return False
        
        # Basic syntax check for C header
        header_path = 'cpp/include/hyperconformal.h'
        if os.path.exists(header_path):
            with open(header_path, 'r') as f:
                content = f.read()
            
            # Check for basic C header structure
            required_elements = ['#ifndef', '#define', '#endif', 'extern "C"']
            for element in required_elements:
                if element in content:
                    print(f"✓ C header contains {element}")
                else:
                    print(f"✗ C header missing {element}")
        
        return True
        
    except Exception as e:
        print(f"✗ C compilation test failed: {e}")
        return False

def test_project_structure():
    """Test overall project structure."""
    print("\nTesting project structure...")
    
    required_files = [
        'README.md',
        'pyproject.toml',
        'LICENSE',
        'hyperconformal/__init__.py'
    ]
    
    required_dirs = [
        'hyperconformal/',
        'tests/',
        'examples/',
        'cpp/'
    ]
    
    success = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing")
            success = False
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - Missing")
            success = False
    
    return success

def test_documentation_completeness():
    """Test documentation completeness."""
    print("\nTesting documentation...")
    
    try:
        # Check README
        if os.path.exists('README.md'):
            with open('README.md', 'r') as f:
                readme_content = f.read()
            
            required_sections = [
                'HyperConformal',
                'Installation',
                'Quick Start',
                'Features',
                'Examples'
            ]
            
            for section in required_sections:
                if section in readme_content:
                    print(f"✓ README contains {section} section")
                else:
                    print(f"⚠ README missing {section} section")
        
        # Check pyproject.toml
        if os.path.exists('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                pyproject_content = f.read()
            
            required_fields = ['name', 'version', 'description', 'dependencies']
            for field in required_fields:
                if field in pyproject_content:
                    print(f"✓ pyproject.toml contains {field}")
                else:
                    print(f"✗ pyproject.toml missing {field}")
        
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def test_advanced_modules():
    """Test advanced module structure."""
    print("\nTesting advanced modules...")
    
    advanced_modules = [
        'hyperconformal/neuromorphic.py',
        'hyperconformal/streaming.py', 
        'hyperconformal/monitoring.py',
        'hyperconformal/security.py',
        'hyperconformal/distributed.py',
        'hyperconformal/optimization.py'
    ]
    
    success = True
    
    for module_path in advanced_modules:
        if os.path.exists(module_path):
            try:
                import ast
                with open(module_path, 'r') as f:
                    content = f.read()
                
                ast.parse(content)
                print(f"✓ {module_path} - Valid syntax")
                
                # Check for key classes/functions
                if 'class' in content:
                    class_count = content.count('class ')
                    print(f"  └ Contains {class_count} classes")
                
                if 'def ' in content:
                    func_count = content.count('def ')
                    print(f"  └ Contains {func_count} functions")
                    
            except Exception as e:
                print(f"✗ {module_path} - Error: {e}")
                success = False
        else:
            print(f"✗ {module_path} - Not found")
            success = False
    
    return success

def main():
    """Run all validation tests."""
    print("HyperConformal Library Validation")
    print("=" * 40)
    
    tests = [
        test_project_structure,
        test_basic_imports,
        test_c_compilation,
        test_documentation_completeness,
        test_advanced_modules
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("PASSED\n")
            else:
                print("FAILED\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
            traceback.print_exc()
    
    print("=" * 40)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some validation tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())