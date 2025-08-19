#!/usr/bin/env python3
"""
Final Quality Gates and Production Deployment for HyperConformal
Comprehensive validation and deployment preparation
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

class FinalQualityGates:
    """
    Final comprehensive quality gates and production readiness validation.
    """
    
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.setup_logging()
        self.validation_results = {}
        self.deployment_artifacts = []
        
    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('hyperconformal_final.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HyperConformalFinal')
        
    def run_comprehensive_tests(self):
        """Run all tests across all generations."""
        self.logger.info("🧪 Running comprehensive test suite...")
        
        test_results = {}
        
        # Test files to run
        test_files = [
            ('Generation 1 - Basic Tests', 'test_minimal.py'),
            ('Generation 1 - Demo', 'demo_basic.py'),
            ('Generation 1 - Algorithm Validation', 'validate_algorithms.py'),
            ('Generation 2 - Error Handling', 'robust_error_handling.py'),
            ('Generation 2 - Security Framework', 'security_framework.py'),
            ('Generation 2 - Monitoring System', 'monitoring_system.py'),
            ('Generation 2 - Integration Tests', 'test_robust_integration.py'),
            ('Generation 3 - Performance Optimization', 'performance_optimization.py'),
            ('Generation 3 - Concurrent Processing', 'concurrent_processing.py'),
            ('Generation 3 - Auto-scaling', 'auto_scaling.py'),
            ('Generation 3 - Benchmarks', 'comprehensive_benchmarks.py')
        ]
        
        passed_tests = 0
        total_tests = len(test_files)
        
        for test_name, test_file in test_files:
            test_path = self.repo_root / test_file
            
            if not test_path.exists():
                test_results[test_name] = {'status': 'MISSING', 'message': f'File {test_file} not found'}
                continue
            
            try:
                result = subprocess.run(
                    [sys.executable, str(test_path)], 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                
                if result.returncode == 0:
                    test_results[test_name] = {'status': 'PASSED', 'output_length': len(result.stdout)}
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name}: PASSED")
                else:
                    test_results[test_name] = {'status': 'FAILED', 'error': result.stderr[:200]}
                    self.logger.warning(f"❌ {test_name}: FAILED")
                    
            except subprocess.TimeoutExpired:
                test_results[test_name] = {'status': 'TIMEOUT', 'message': 'Test timed out'}
                self.logger.warning(f"⏰ {test_name}: TIMEOUT")
            except Exception as e:
                test_results[test_name] = {'status': 'ERROR', 'error': str(e)}
                self.logger.error(f"💥 {test_name}: ERROR - {e}")
        
        # Overall test results
        test_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        self.validation_results['comprehensive_tests'] = {
            'passed': passed_tests,
            'total': total_tests,
            'success_rate': test_success_rate,
            'details': test_results
        }
        
        self.logger.info(f"📊 Test Results: {passed_tests}/{total_tests} passed ({test_success_rate:.1%})")
        
        return test_success_rate >= 0.8  # 80% pass rate required
    
    def validate_security_compliance(self):
        """Validate security compliance across all components."""
        self.logger.info("🔒 Validating security compliance...")
        
        security_checks = []
        
        # Check for common security issues
        python_files = list(self.repo_root.glob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Security pattern checks
                dangerous_patterns = [
                    ('eval(', 'Dynamic code evaluation'),
                    ('exec(', 'Dynamic code execution'),
                    ('__import__', 'Dynamic imports'),
                    ('os.system(', 'System command execution'),
                    ('subprocess.call(', 'Subprocess calls without validation'),
                    ('pickle.loads(', 'Unsafe deserialization'),
                    ('input(', 'Direct user input without validation')
                ]
                
                file_issues = []
                for pattern, description in dangerous_patterns:
                    if pattern in content:
                        file_issues.append(f"{description}: {pattern}")
                
                if not file_issues:
                    security_checks.append({
                        'file': py_file.name,
                        'status': 'SECURE',
                        'issues': []
                    })
                else:
                    security_checks.append({
                        'file': py_file.name,
                        'status': 'ISSUES_FOUND',
                        'issues': file_issues
                    })
                    
            except Exception as e:
                security_checks.append({
                    'file': py_file.name,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # Count secure files
        secure_files = sum(1 for check in security_checks if check['status'] == 'SECURE')
        total_files = len(security_checks)
        security_score = secure_files / total_files if total_files > 0 else 0
        
        self.validation_results['security_compliance'] = {
            'secure_files': secure_files,
            'total_files': total_files,
            'security_score': security_score,
            'details': security_checks
        }
        
        self.logger.info(f"🔒 Security Score: {secure_files}/{total_files} files secure ({security_score:.1%})")
        
        return security_score >= 0.9  # 90% security compliance required
    
    def validate_performance_benchmarks(self):
        """Validate performance meets requirements."""
        self.logger.info("⚡ Validating performance benchmarks...")
        
        # Run comprehensive benchmarks
        benchmark_file = self.repo_root / 'comprehensive_benchmarks.py'
        
        if not benchmark_file.exists():
            self.logger.error("❌ Benchmark file not found")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(benchmark_file)], 
                capture_output=True, 
                text=True, 
                timeout=180
            )
            
            if result.returncode != 0:
                self.logger.error("❌ Benchmark execution failed")
                return False
            
            # Parse benchmark output for key metrics
            output = result.stdout
            
            # Extract performance metrics
            performance_metrics = {
                'hdc_encoding_throughput': 0,
                'conformal_prediction_speed': 0,
                'concurrent_speedup': 0,
                'cache_effectiveness': 0,
                'scaling_efficiency': 0
            }
            
            # Simple parsing of benchmark output with input validation
            lines = output.split('\\n')
            for line in lines:
                # Sanitize line input to prevent injection
                line = line.strip()[:1000]  # Limit line length
                if not line or any(char in line for char in ['<', '>', '&', ';']):
                    continue  # Skip potentially malicious lines
                    
                if 'vectors/s' in line and 'size_100' in line:
                    # Extract HDC encoding throughput with validation
                    try:
                        parts = line.split(',')
                        for part in parts:
                            if 'vectors/s' in part:
                                # Validate numeric extraction
                                numeric_part = part.split()[0]
                                if numeric_part.replace('.', '').replace('-', '').isdigit():
                                    performance_metrics['hdc_encoding_throughput'] = float(numeric_part)
                                break
                    except (ValueError, IndexError):
                        pass
                
                elif 'pred/s' in line and 'classes_10' in line:
                    # Extract conformal prediction speed with validation
                    try:
                        parts = line.split(',')
                        for part in parts:
                            if 'pred/s' in part:
                                # Validate numeric extraction
                                numeric_part = part.split()[0]
                                if numeric_part.replace('.', '').replace('-', '').isdigit():
                                    performance_metrics['conformal_prediction_speed'] = float(numeric_part)
                                break
                    except (ValueError, IndexError):
                        pass
                
                elif 'concurrent_speedup:' in line:
                    try:
                        value_str = line.split(':')[1].strip()
                        if value_str.replace('.', '').replace('-', '').isdigit():
                            performance_metrics['concurrent_speedup'] = float(value_str)
                    except (ValueError, IndexError):
                        pass
                
                elif 'Hit rate:' in line:
                    try:
                        hit_rate_str = line.split(':')[1].strip().replace('%', '')
                        if hit_rate_str.replace('.', '').isdigit():
                            performance_metrics['cache_effectiveness'] = float(hit_rate_str) / 100
                    except (ValueError, IndexError):
                        pass
                
                elif 'scaling_efficiency:' in line:
                    try:
                        value_str = line.split(':')[1].strip()
                        if value_str.replace('.', '').replace('-', '').isdigit():
                            performance_metrics['scaling_efficiency'] = float(value_str)
                    except (ValueError, IndexError):
                        pass
            
            # Define performance requirements
            requirements = {
                'hdc_encoding_throughput': 1000,  # vectors/s
                'conformal_prediction_speed': 100000,  # predictions/s
                'concurrent_speedup': 0.01,  # Any speedup is good
                'cache_effectiveness': 0.5,  # 50% hit rate
                'scaling_efficiency': 0.1  # 10% efficiency
            }
            
            # Check if requirements are met
            requirements_met = 0
            total_requirements = len(requirements)
            
            for metric, requirement in requirements.items():
                actual_value = performance_metrics.get(metric, 0)
                if actual_value >= requirement:
                    requirements_met += 1
                    self.logger.info(f"✅ {metric}: {actual_value} >= {requirement}")
                else:
                    self.logger.warning(f"⚠️ {metric}: {actual_value} < {requirement}")
            
            performance_score = requirements_met / total_requirements
            
            self.validation_results['performance_benchmarks'] = {
                'requirements_met': requirements_met,
                'total_requirements': total_requirements,
                'performance_score': performance_score,
                'metrics': performance_metrics,
                'requirements': requirements
            }
            
            self.logger.info(f"⚡ Performance Score: {requirements_met}/{total_requirements} requirements met ({performance_score:.1%})")
            
            return performance_score >= 0.6  # 60% performance requirements met
            
        except Exception as e:
            self.logger.error(f"❌ Performance validation failed: {e}")
            return False
    
    def validate_code_quality(self):
        """Validate code quality metrics."""
        self.logger.info("📝 Validating code quality...")
        
        quality_metrics = {
            'total_lines': 0,
            'total_files': 0,
            'documented_files': 0,
            'syntax_errors': 0,
            'complexity_score': 0
        }
        
        python_files = list(self.repo_root.glob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\\n')
                
                quality_metrics['total_files'] += 1
                quality_metrics['total_lines'] += len(lines)
                
                # Check for documentation
                if '"""' in content or "'''" in content:
                    quality_metrics['documented_files'] += 1
                
                # Check syntax
                try:
                    compile(content, py_file, 'exec')
                except SyntaxError:
                    quality_metrics['syntax_errors'] += 1
                
                # Simple complexity score (functions + classes)
                function_count = content.count('def ')
                class_count = content.count('class ')
                file_complexity = function_count + class_count
                quality_metrics['complexity_score'] += file_complexity
                
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Calculate quality scores
        documentation_rate = (quality_metrics['documented_files'] / 
                            quality_metrics['total_files'] if quality_metrics['total_files'] > 0 else 0)
        
        syntax_error_rate = (quality_metrics['syntax_errors'] / 
                           quality_metrics['total_files'] if quality_metrics['total_files'] > 0 else 0)
        
        avg_complexity = (quality_metrics['complexity_score'] / 
                         quality_metrics['total_files'] if quality_metrics['total_files'] > 0 else 0)
        
        # Overall quality score
        quality_score = (
            documentation_rate * 0.4 +  # 40% weight on documentation
            (1 - syntax_error_rate) * 0.4 +  # 40% weight on syntax correctness
            min(avg_complexity / 10, 1.0) * 0.2  # 20% weight on complexity (normalized)
        )
        
        self.validation_results['code_quality'] = {
            'metrics': quality_metrics,
            'documentation_rate': documentation_rate,
            'syntax_error_rate': syntax_error_rate,
            'avg_complexity': avg_complexity,
            'quality_score': quality_score
        }
        
        self.logger.info(f"📝 Code Quality Score: {quality_score:.1%}")
        self.logger.info(f"   Documentation: {documentation_rate:.1%}")
        self.logger.info(f"   Syntax Errors: {syntax_error_rate:.1%}")
        self.logger.info(f"   Avg Complexity: {avg_complexity:.1f}")
        
        return quality_score >= 0.7  # 70% quality score required
    
    def prepare_deployment_artifacts(self):
        """Prepare artifacts for production deployment."""
        self.logger.info("📦 Preparing deployment artifacts...")
        
        artifacts = []
        
        # Core implementation files
        core_files = [
            'hyperconformal/__init__.py',
            'hyperconformal/encoders.py',
            'hyperconformal/conformal.py',
            'hyperconformal/hyperconformal.py',
            'hyperconformal/utils.py',
            'hyperconformal/metrics.py'
        ]
        
        # Generation implementations
        generation_files = [
            'validate_environment.py',
            'autonomous_implementation.py',
            'generation_2_robust.py', 
            'generation_3_scale.py',
            'robust_error_handling.py',
            'security_framework.py',
            'monitoring_system.py',
            'performance_optimization.py',
            'concurrent_processing.py',
            'auto_scaling.py'
        ]
        
        # Configuration and deployment files
        config_files = [
            'pyproject.toml',
            'README.md',
            'docker/Dockerfile',
            'docker/docker-compose.yml',
            'k8s/hyperconformal-deployment.yaml'
        ]
        
        # Test and validation files
        test_files = [
            'test_minimal.py',
            'demo_basic.py',
            'validate_algorithms.py',
            'test_robust_integration.py',
            'comprehensive_benchmarks.py'
        ]
        
        all_artifact_files = core_files + generation_files + config_files + test_files
        
        existing_artifacts = []
        missing_artifacts = []
        
        for artifact_file in all_artifact_files:
            artifact_path = self.repo_root / artifact_file
            if artifact_path.exists():
                existing_artifacts.append({
                    'file': artifact_file,
                    'size': artifact_path.stat().st_size,
                    'type': self._classify_artifact(artifact_file)
                })
            else:
                missing_artifacts.append(artifact_file)
        
        # Calculate deployment readiness
        artifact_completeness = len(existing_artifacts) / len(all_artifact_files)
        
        self.deployment_artifacts = existing_artifacts
        
        self.validation_results['deployment_artifacts'] = {
            'existing_artifacts': len(existing_artifacts),
            'missing_artifacts': len(missing_artifacts),
            'total_artifacts': len(all_artifact_files),
            'completeness': artifact_completeness,
            'missing_files': missing_artifacts,
            'artifact_details': existing_artifacts
        }
        
        self.logger.info(f"📦 Deployment Artifacts: {len(existing_artifacts)}/{len(all_artifact_files)} available ({artifact_completeness:.1%})")
        
        if missing_artifacts:
            self.logger.warning(f"Missing artifacts: {', '.join(missing_artifacts[:5])}{'...' if len(missing_artifacts) > 5 else ''}")
        
        return artifact_completeness >= 0.8  # 80% artifact completeness required
    
    def _classify_artifact(self, filename: str) -> str:
        """Classify artifact type."""
        if filename.endswith('.py'):
            if 'test_' in filename or filename in ['demo_basic.py', 'validate_algorithms.py']:
                return 'test'
            elif 'generation_' in filename or filename in ['autonomous_implementation.py']:
                return 'implementation'
            elif filename.startswith('hyperconformal/'):
                return 'core'
            else:
                return 'utility'
        elif filename.endswith(('.yaml', '.yml')):
            return 'configuration'
        elif filename.endswith('.toml'):
            return 'build_config'
        elif filename in ['README.md', 'LICENSE']:
            return 'documentation'
        elif filename.startswith('docker/'):
            return 'containerization'
        else:
            return 'other'
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment readiness report."""
        self.logger.info("📋 Generating deployment readiness report...")
        
        # Calculate overall readiness score
        validation_scores = []
        
        if 'comprehensive_tests' in self.validation_results:
            validation_scores.append(self.validation_results['comprehensive_tests']['success_rate'])
        
        if 'security_compliance' in self.validation_results:
            validation_scores.append(self.validation_results['security_compliance']['security_score'])
        
        if 'performance_benchmarks' in self.validation_results:
            validation_scores.append(self.validation_results['performance_benchmarks']['performance_score'])
        
        if 'code_quality' in self.validation_results:
            validation_scores.append(self.validation_results['code_quality']['quality_score'])
        
        if 'deployment_artifacts' in self.validation_results:
            validation_scores.append(self.validation_results['deployment_artifacts']['completeness'])
        
        overall_readiness = sum(validation_scores) / len(validation_scores) if validation_scores else 0
        
        # Determine deployment recommendation
        if overall_readiness >= 0.8:
            deployment_recommendation = "READY_FOR_PRODUCTION"
        elif overall_readiness >= 0.6:
            deployment_recommendation = "READY_FOR_STAGING"
        elif overall_readiness >= 0.4:
            deployment_recommendation = "READY_FOR_DEVELOPMENT"
        else:
            deployment_recommendation = "NOT_READY"
        
        deployment_report = {
            'timestamp': time.time(),
            'overall_readiness_score': overall_readiness,
            'deployment_recommendation': deployment_recommendation,
            'validation_results': self.validation_results,
            'deployment_artifacts': {
                'total_count': len(self.deployment_artifacts),
                'total_size_mb': sum(a['size'] for a in self.deployment_artifacts) / (1024 * 1024),
                'artifact_types': self._summarize_artifact_types()
            },
            'next_steps': self._generate_next_steps(overall_readiness),
            'risk_assessment': self._assess_deployment_risks(),
            'performance_characteristics': self._summarize_performance()
        }
        
        # Save report
        with open(self.repo_root / 'deployment_readiness_report.json', 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        return deployment_report
    
    def _summarize_artifact_types(self):
        """Summarize artifact types."""
        type_counts = {}
        for artifact in self.deployment_artifacts:
            artifact_type = artifact['type']
            type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
        return type_counts
    
    def _generate_next_steps(self, readiness_score: float) -> List[str]:
        """Generate next steps based on readiness score."""
        if readiness_score >= 0.8:
            return [
                "✅ System is ready for production deployment",
                "Configure production environment variables",
                "Set up monitoring and alerting",
                "Deploy to staging for final validation",
                "Execute production deployment"
            ]
        elif readiness_score >= 0.6:
            return [
                "🔧 Address remaining quality issues",
                "Improve test coverage",
                "Optimize performance bottlenecks",
                "Complete security audit",
                "Deploy to staging environment"
            ]
        else:
            return [
                "🚧 Significant improvements needed",
                "Fix failing tests",
                "Address security vulnerabilities",
                "Improve code quality and documentation",
                "Optimize performance",
                "Re-run quality gates"
            ]
    
    def _assess_deployment_risks(self) -> Dict[str, str]:
        """Assess deployment risks."""
        risks = {}
        
        # Test coverage risk
        if 'comprehensive_tests' in self.validation_results:
            test_score = self.validation_results['comprehensive_tests']['success_rate']
            if test_score < 0.8:
                risks['test_coverage'] = f"Low test success rate: {test_score:.1%}"
        
        # Security risk
        if 'security_compliance' in self.validation_results:
            security_score = self.validation_results['security_compliance']['security_score']
            if security_score < 0.9:
                risks['security'] = f"Security compliance below 90%: {security_score:.1%}"
        
        # Performance risk
        if 'performance_benchmarks' in self.validation_results:
            perf_score = self.validation_results['performance_benchmarks']['performance_score']
            if perf_score < 0.6:
                risks['performance'] = f"Performance requirements not met: {perf_score:.1%}"
        
        return risks
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """Summarize performance characteristics."""
        if 'performance_benchmarks' not in self.validation_results:
            return {"status": "no_data"}
        
        perf_data = self.validation_results['performance_benchmarks']
        return {
            'requirements_met': f"{perf_data['requirements_met']}/{perf_data['total_requirements']}",
            'key_metrics': perf_data.get('metrics', {}),
            'performance_score': perf_data['performance_score']
        }
    
    def execute_final_validation(self):
        """Execute complete final validation process."""
        self.logger.info("🚀 EXECUTING FINAL QUALITY GATES")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        # Run all validation steps
        validation_steps = [
            ("Comprehensive Tests", self.run_comprehensive_tests),
            ("Security Compliance", self.validate_security_compliance),
            ("Performance Benchmarks", self.validate_performance_benchmarks),
            ("Code Quality", self.validate_code_quality),
            ("Deployment Artifacts", self.prepare_deployment_artifacts)
        ]
        
        passed_steps = 0
        total_steps = len(validation_steps)
        
        for step_name, step_func in validation_steps:
            self.logger.info(f"\\n🔍 Executing: {step_name}")
            try:
                success = step_func()
                if success:
                    passed_steps += 1
                    self.logger.info(f"✅ {step_name}: PASSED")
                else:
                    self.logger.warning(f"⚠️ {step_name}: FAILED")
            except Exception as e:
                self.logger.error(f"❌ {step_name}: ERROR - {e}")
        
        # Generate final report
        deployment_report = self.generate_deployment_report()
        
        execution_time = time.time() - start_time
        
        # Final summary
        self.logger.info("\\n" + "="*60)
        self.logger.info("🏆 FINAL QUALITY GATES SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Validation Steps: {passed_steps}/{total_steps} passed")
        self.logger.info(f"Overall Readiness: {deployment_report['overall_readiness_score']:.1%}")
        self.logger.info(f"Recommendation: {deployment_report['deployment_recommendation']}")
        self.logger.info(f"Execution Time: {execution_time:.1f} seconds")
        
        if deployment_report['deployment_recommendation'] in ['READY_FOR_PRODUCTION', 'READY_FOR_STAGING']:
            self.logger.info("🎉 HYPERCONFORMAL IS READY FOR DEPLOYMENT!")
        else:
            self.logger.info("🔧 ADDITIONAL WORK NEEDED BEFORE DEPLOYMENT")
        
        return deployment_report

if __name__ == "__main__":
    quality_gates = FinalQualityGates()
    final_report = quality_gates.execute_final_validation()
    print(f"\\n📋 Deployment Report: {json.dumps(final_report, indent=2)}")