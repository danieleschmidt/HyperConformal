#!/usr/bin/env python3
"""
Comprehensive test runner for HyperConformal library.

This script runs different types of tests with appropriate configurations
and generates comprehensive reports.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Comprehensive test runner with multiple test suites."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.venv_python = project_root / "venv" / "bin" / "python3"
        
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        end_time = time.time()
        
        status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
        print(f"\n{status} ({end_time - start_time:.2f}s): {description}")
        
        return result.returncode == 0
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        cmd = [
            str(self.venv_python), "-m", "pytest",
            "tests/test_conformal.py",
            "tests/test_encoders.py", 
            "tests/test_hyperconformal.py",
            "tests/test_metrics.py",
            "tests/test_utils.py",
            "-v", "--tb=short",
            "-m", "not slow"
        ]
        return self.run_command(cmd, "Unit Tests")
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        cmd = [
            str(self.venv_python), "-m", "pytest",
            "tests/test_integration.py",
            "-v", "--tb=short"
        ]
        return self.run_command(cmd, "Integration Tests")
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        cmd = [
            str(self.venv_python), "-m", "pytest",
            "tests/test_e2e.py",
            "-v", "--tb=short"
        ]
        return self.run_command(cmd, "End-to-End Tests")
    
    def run_coverage_analysis(self) -> bool:
        """Run comprehensive coverage analysis."""
        # Run tests with coverage
        cmd = [
            str(self.venv_python), "-m", "coverage", "run",
            "--source=hyperconformal",
            "-m", "pytest", "tests/",
            "--tb=no"
        ]
        
        if not self.run_command(cmd, "Coverage Analysis (Test Run)"):
            return False
        
        # Generate coverage report
        cmd = [
            str(self.venv_python), "-m", "coverage", "report",
            "--show-missing"
        ]
        
        return self.run_command(cmd, "Coverage Report")
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        checks = [
            {
                "cmd": [str(self.venv_python), "-m", "black", "--check", "hyperconformal/", "tests/"],
                "desc": "Code Formatting (Black)"
            },
            {
                "cmd": [str(self.venv_python), "-m", "isort", "--check-only", "hyperconformal/", "tests/"],
                "desc": "Import Sorting (isort)"
            },
            {
                "cmd": [str(self.venv_python), "-m", "flake8", "hyperconformal/"],
                "desc": "Code Style (flake8)"
            },
            {
                "cmd": [str(self.venv_python), "-m", "mypy", "hyperconformal/"],
                "desc": "Type Checking (mypy)"
            }
        ]
        
        all_passed = True
        for check in checks:
            if not self.run_command(check["cmd"], check["desc"]):
                all_passed = False
        
        return all_passed
    
    def run_performance_tests(self) -> bool:
        """Run performance benchmark tests."""
        cmd = [
            str(self.venv_python), "-m", "pytest",
            "tests/",
            "-v", "-k", "benchmark or performance",
            "--tb=short"
        ]
        return self.run_command(cmd, "Performance Tests")
    
    def run_all_tests(self) -> bool:
        """Run all test suites."""
        results = {
            "Unit Tests": self.run_unit_tests(),
            "Integration Tests": self.run_integration_tests(),
            "End-to-End Tests": self.run_e2e_tests(),
            "Coverage Analysis": self.run_coverage_analysis(),
            "Code Quality": self.run_code_quality_checks(),
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        all_passed = True
        for suite_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {suite_name}")
            if not passed:
                all_passed = False
        
        print(f"\n{'='*60}")
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"OVERALL RESULT: {overall_status}")
        print(f"{'='*60}")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HyperConformal Test Runner")
    parser.add_argument(
        "--suite", 
        choices=["unit", "integration", "e2e", "coverage", "quality", "performance", "all"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(args.project_root)
    
    # Check if venv python exists
    if not runner.venv_python.exists():
        print(f"❌ Virtual environment not found at {runner.venv_python}")
        print("Please create and activate virtual environment first.")
        sys.exit(1)
    
    # Run selected test suite
    suite_map = {
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "e2e": runner.run_e2e_tests,
        "coverage": runner.run_coverage_analysis,
        "quality": runner.run_code_quality_checks,
        "performance": runner.run_performance_tests,
        "all": runner.run_all_tests
    }
    
    success = suite_map[args.suite]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()