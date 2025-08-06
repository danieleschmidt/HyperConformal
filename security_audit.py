#!/usr/bin/env python3
"""
Security audit script for HyperConformal implementation.
Checks for potential security vulnerabilities and compliance.
"""

import os
import re
import ast
import hashlib
from typing import List, Dict, Any

class SecurityAuditor:
    """Security audit utilities."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def audit_code_injection(self, file_path: str) -> List[str]:
        """Audit for potential code injection vulnerabilities."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for dangerous patterns
            dangerous_patterns = [
                (r'eval\s*\(', 'Use of eval() - potential code injection'),
                (r'exec\s*\(', 'Use of exec() - potential code injection'),
                (r'__import__\s*\(', 'Dynamic imports - potential security risk'),
                (r'subprocess\.(call|run|Popen).*shell=True', 'Shell execution with user input'),
                (r'os\.system\s*\(', 'Direct shell execution')
            ]
            
            for pattern, message in dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"Line {line_num}: {message}")
                    
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
        
        return issues
    
    def audit_input_validation(self, file_path: str) -> List[str]:
        """Audit for proper input validation."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for functions that should validate inputs
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function handles external input
                    has_input_params = any(
                        arg.arg in ['x', 'X', 'input', 'data', 'features'] 
                        for arg in node.args.args
                    )
                    
                    if has_input_params:
                        func_source = ast.get_source_segment(content, node)
                        if func_source:
                            # Check for validation patterns
                            validation_patterns = [
                                'isinstance',
                                'validate',
                                'check',
                                'assert',
                                'raise ValueError',
                                'raise TypeError'
                            ]
                            
                            has_validation = any(
                                pattern in func_source 
                                for pattern in validation_patterns
                            )
                            
                            if not has_validation:
                                line_num = node.lineno
                                issues.append(f"Line {line_num}: Function '{node.name}' lacks input validation")
                                
        except Exception as e:
            issues.append(f"Error analyzing {file_path}: {e}")
        
        return issues
    
    def audit_secrets_exposure(self, file_path: str) -> List[str]:
        """Audit for exposed secrets or keys."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Patterns that might indicate secrets
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
                (r'secret_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret key'),
                (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
                (r'["\'][A-Za-z0-9]{32,}["\']', 'Potential secret string')
            ]
            
            for pattern, message in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    # Exclude test files and examples
                    if 'test' not in file_path.lower() and 'example' not in file_path.lower():
                        issues.append(f"Line {line_num}: {message}")
                    
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
        
        return issues
    
    def audit_privacy_compliance(self, file_path: str) -> List[str]:
        """Audit for privacy compliance (GDPR, CCPA, etc.)."""
        warnings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for data collection without privacy controls
            privacy_concerns = [
                ('logging', 'Logging may contain personal data'),
                ('print.*data', 'Debug prints may expose data'),
                ('save.*data', 'Data persistence without encryption'),
                ('upload.*data', 'Data transmission without encryption')
            ]
            
            for pattern, message in privacy_concerns:
                if re.search(pattern, content, re.IGNORECASE):
                    warnings.append(f"{message} - Consider privacy implications")
                    
        except Exception as e:
            warnings.append(f"Error reading {file_path}: {e}")
        
        return warnings
    
    def audit_crypto_usage(self, file_path: str) -> List[str]:
        """Audit cryptographic implementations."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for weak crypto practices
            crypto_issues = [
                (r'hashlib\.md5\s*\(', 'MD5 is cryptographically broken'),
                (r'hashlib\.sha1\s*\(', 'SHA1 is vulnerable'),
                (r'random\.random\s*\(', 'random.random() not cryptographically secure'),
                (r'random\.seed\s*\(.*\)', 'Predictable random seed'),
                (r'AES.*ECB', 'ECB mode is insecure'),
                (r'iv\s*=\s*b?["\']0+["\']', 'Fixed IV is insecure')
            ]
            
            for pattern, message in crypto_issues:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"Line {line_num}: {message}")
                    
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
        
        return issues
    
    def audit_file_permissions(self, file_path: str) -> List[str]:
        """Audit file permissions."""
        issues = []
        
        try:
            stat_info = os.stat(file_path)
            perms = oct(stat_info.st_mode)[-3:]
            
            # Check for overly permissive files
            if perms.endswith('7') or perms.endswith('6'):
                if not file_path.endswith('.py'):
                    issues.append(f"File has overly permissive permissions: {perms}")
                    
        except Exception as e:
            issues.append(f"Error checking permissions for {file_path}: {e}")
        
        return issues

def audit_security():
    """Run comprehensive security audit."""
    print("HyperConformal Security Audit")
    print("=" * 40)
    
    auditor = SecurityAuditor()
    total_issues = 0
    total_warnings = 0
    
    # Find Python files to audit
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to audit\n")
    
    # Run audits
    audit_functions = [
        ('Code Injection', auditor.audit_code_injection),
        ('Input Validation', auditor.audit_input_validation),
        ('Secrets Exposure', auditor.audit_secrets_exposure),
        ('Cryptographic Usage', auditor.audit_crypto_usage),
        ('File Permissions', auditor.audit_file_permissions)
    ]
    
    for audit_name, audit_func in audit_functions:
        print(f"Auditing {audit_name}...")
        
        issues_found = 0
        for file_path in python_files:
            file_issues = audit_func(file_path)
            if file_issues:
                print(f"\n  {file_path}:")
                for issue in file_issues:
                    print(f"    ✗ {issue}")
                    issues_found += 1
        
        if issues_found == 0:
            print("  ✓ No issues found")
        else:
            print(f"  Found {issues_found} issues")
            
        total_issues += issues_found
        print()
    
    # Privacy compliance check (warnings only)
    print("Checking Privacy Compliance...")
    privacy_warnings = 0
    for file_path in python_files:
        file_warnings = auditor.audit_privacy_compliance(file_path)
        if file_warnings:
            print(f"\n  {file_path}:")
            for warning in file_warnings:
                print(f"    ⚠ {warning}")
                privacy_warnings += 1
    
    if privacy_warnings == 0:
        print("  ✓ No privacy concerns found")
    else:
        print(f"  Found {privacy_warnings} privacy considerations")
        
    total_warnings += privacy_warnings
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Security Audit Results:")
    print(f"  Issues: {total_issues}")
    print(f"  Warnings: {total_warnings}")
    
    if total_issues == 0:
        print("  ✓ No security issues found!")
        return 0
    else:
        print(f"  ✗ {total_issues} security issues need attention")
        return 1

def check_dependencies_security():
    """Check for known vulnerabilities in dependencies."""
    print("\nDependency Security Check")
    print("-" * 30)
    
    try:
        # Read requirements from pyproject.toml
        if os.path.exists('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                content = f.read()
            
            # Extract dependencies
            import re
            deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if deps_match:
                deps_str = deps_match.group(1)
                deps = [dep.strip().strip('"\'') for dep in deps_str.split(',') if dep.strip()]
                
                print(f"Found {len(deps)} dependencies:")
                for dep in deps:
                    print(f"  - {dep}")
                
                # Known vulnerable packages (simplified check)
                known_issues = {
                    'torch<1.12.0': 'CVE-2022-24288 - Arbitrary code execution',
                    'numpy<1.21.0': 'CVE-2021-33430 - Buffer overflow',
                    'pillow<8.3.2': 'CVE-2021-34552 - Buffer overflow'
                }
                
                issues_found = False
                for dep in deps:
                    dep_name = dep.split('>=')[0].split('==')[0]
                    for vuln_pattern, description in known_issues.items():
                        if dep_name in vuln_pattern:
                            print(f"  ⚠ Potential vulnerability in {dep}: {description}")
                            issues_found = True
                
                if not issues_found:
                    print("  ✓ No known vulnerabilities found in declared dependencies")
            else:
                print("  ⚠ Could not parse dependencies from pyproject.toml")
                
    except Exception as e:
        print(f"  ✗ Error checking dependencies: {e}")

def main():
    """Run security audit."""
    result = audit_security()
    check_dependencies_security()
    
    print(f"\nSecurity audit completed with exit code: {result}")
    return result

if __name__ == '__main__':
    import sys
    sys.exit(main())