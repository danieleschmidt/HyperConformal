#!/usr/bin/env python3
"""
Environment validation and dependency checking for HyperConformal library.
Autonomous SDLC execution - Generation 1: MAKE IT WORK
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path

class EnvironmentValidator:
    """Validates and sets up the HyperConformal development environment."""
    
    def __init__(self):
        self.required_packages = [
            'numpy>=1.20.0',
            'torch>=1.9.0', 
            'scipy>=1.7.0',
            'scikit-learn>=1.0.0'
        ]
        self.optional_packages = [
            'pytest>=6.0',
            'black',
            'flake8',
            'mypy'
        ]
        self.results = {}
        
    def check_python_version(self):
        """Check if Python version meets requirements."""
        version = sys.version_info
        required = (3, 8)
        
        if version >= required:
            self.results['python'] = {
                'status': 'PASS',
                'version': f"{version.major}.{version.minor}.{version.micro}",
                'message': f"Python {version.major}.{version.minor}.{version.micro} meets requirement >=3.8"
            }
            return True
        else:
            self.results['python'] = {
                'status': 'FAIL',
                'version': f"{version.major}.{version.minor}.{version.micro}",
                'message': f"Python {version.major}.{version.minor}.{version.micro} < 3.8 (required)"
            }
            return False
    
    def check_package_availability(self, package_name):
        """Check if a package is available for import."""
        try:
            # Handle package names with version specs
            pkg_name = package_name.split('>=')[0].split('==')[0]
            importlib.import_module(pkg_name)
            return True, f"{pkg_name} available"
        except ImportError:
            return False, f"{pkg_name} not available"
    
    def install_missing_packages(self, use_system=False):
        """Attempt to install missing packages."""
        missing = []
        for pkg in self.required_packages:
            pkg_name = pkg.split('>=')[0].split('==')[0]
            available, msg = self.check_package_availability(pkg_name)
            if not available:
                missing.append(pkg)
        
        if not missing:
            return True, "All packages available"
        
        # Try different installation methods
        methods = [
            ['pip3', 'install'] + missing,
            ['python3', '-m', 'pip', 'install'] + missing,
        ]
        
        if use_system:
            methods.append(['pip3', 'install', '--break-system-packages'] + missing)
        
        for method in methods:
            try:
                result = subprocess.run(method, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    return True, f"Successfully installed: {', '.join(missing)}"
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        return False, f"Failed to install: {', '.join(missing)}"
    
    def validate_hyperconformal_structure(self):
        """Validate the HyperConformal package structure."""
        repo_root = Path(__file__).parent
        expected_files = [
            'hyperconformal/__init__.py',
            'hyperconformal/encoders.py',
            'hyperconformal/conformal.py',
            'hyperconformal/hyperconformal.py',
            'hyperconformal/utils.py',
            'pyproject.toml',
            'README.md'
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not (repo_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results['structure'] = {
                'status': 'FAIL',
                'message': f"Missing files: {', '.join(missing_files)}"
            }
            return False
        else:
            self.results['structure'] = {
                'status': 'PASS',
                'message': "All required files present"
            }
            return True
    
    def test_basic_imports(self):
        """Test basic imports without heavy dependencies."""
        try:
            # Test basic Python functionality that should work
            import os
            import sys
            import json
            import math
            import random
            
            # Test if our package structure can be imported
            sys.path.insert(0, str(Path(__file__).parent))
            
            # Try importing with fallback mechanisms
            import hyperconformal
            
            self.results['imports'] = {
                'status': 'PASS',
                'message': "Basic imports successful"
            }
            return True
            
        except Exception as e:
            self.results['imports'] = {
                'status': 'FAIL',
                'message': f"Import failed: {str(e)}"
            }
            return False
    
    def create_fallback_dependencies(self):
        """Create minimal fallback implementations for missing dependencies."""
        fallback_dir = Path(__file__).parent / 'fallbacks'
        fallback_dir.mkdir(exist_ok=True)
        
        # Create minimal numpy-like module
        numpy_fallback = fallback_dir / 'numpy_fallback.py'
        numpy_fallback.write_text('''
"""Minimal numpy fallback for basic operations."""
import math
import random

class ndarray:
    def __init__(self, data, dtype=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            self.data = [data]
        self.dtype = dtype
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def mean(self):
        return sum(self.data) / len(self.data)
        
    def sum(self):
        return sum(self.data)

def array(data, dtype=None):
    return ndarray(data, dtype)

def zeros(shape):
    if isinstance(shape, int):
        return ndarray([0] * shape)
    else:
        return ndarray([0] * shape[0])

def ones(shape):
    if isinstance(shape, int):
        return ndarray([1] * shape)
    else:
        return ndarray([1] * shape[0])

def random_normal(size):
    return ndarray([random.gauss(0, 1) for _ in range(size)])

# Basic constants
pi = math.pi
e = math.e
''')
        
        return True
    
    def run_comprehensive_validation(self):
        """Run all validation checks."""
        print("🧠 HyperConformal Environment Validation - AUTONOMOUS SDLC")
        print("=" * 60)
        
        # Check Python version
        python_ok = self.check_python_version()
        print(f"Python Version: {self.results['python']['status']} - {self.results['python']['message']}")
        
        # Check package structure
        structure_ok = self.validate_hyperconformal_structure()
        print(f"Package Structure: {self.results['structure']['status']} - {self.results['structure']['message']}")
        
        # Check dependencies
        print("\nDependency Check:")
        all_deps_ok = True
        for pkg in self.required_packages:
            pkg_name = pkg.split('>=')[0]
            available, msg = self.check_package_availability(pkg_name)
            status = "PASS" if available else "FAIL"
            print(f"  {pkg_name}: {status} - {msg}")
            if not available:
                all_deps_ok = False
        
        # Create fallbacks if needed
        if not all_deps_ok:
            print("\n🔧 Creating fallback dependencies...")
            self.create_fallback_dependencies()
            print("✅ Fallback dependencies created")
        
        # Test imports
        import_ok = self.test_basic_imports()
        print(f"\nImport Test: {self.results['imports']['status']} - {self.results['imports']['message']}")
        
        # Overall status
        overall_ok = python_ok and structure_ok and import_ok
        
        print("\n" + "=" * 60)
        if overall_ok:
            print("🚀 ENVIRONMENT READY - Proceeding with autonomous implementation")
        else:
            print("⚠️  ENVIRONMENT ISSUES DETECTED - Implementing with fallbacks")
        
        return overall_ok
    
    def generate_status_report(self):
        """Generate detailed status report."""
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'platform': platform.platform(),
            'python_version': sys.version,
            'validation_results': self.results,
            'next_steps': [
                "Install missing dependencies",
                "Run comprehensive tests", 
                "Validate C++ components",
                "Deploy to production"
            ]
        }
        
        with open('environment_status.json', 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        return report

if __name__ == "__main__":
    validator = EnvironmentValidator()
    success = validator.run_comprehensive_validation()
    validator.generate_status_report()
    
    if success:
        print("\n🎯 AUTONOMOUS EXECUTION: Environment validated - proceeding to Generation 1 implementation")
    else:
        print("\n🛠️  AUTONOMOUS EXECUTION: Environment setup complete with fallbacks - proceeding with cautious implementation")
    
    sys.exit(0 if success else 1)