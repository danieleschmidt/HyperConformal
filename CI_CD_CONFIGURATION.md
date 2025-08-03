# CI/CD Pipeline Configuration for HyperConformal

## GitHub Actions Workflow Configuration

**Note**: This configuration requires `workflows` permission to be added to the repository. A user with admin access should create this file at `.github/workflows/ci.yml`.

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run code quality checks
      run: |
        # Format checking
        black --check hyperconformal/ tests/
        
        # Import sorting
        isort --check-only hyperconformal/ tests/
        
        # Linting
        flake8 hyperconformal/
        
        # Type checking
        mypy hyperconformal/ --ignore-missing-imports
    
    - name: Run unit tests with coverage
      run: |
        python -m pytest tests/test_conformal.py tests/test_encoders.py tests/test_hyperconformal.py tests/test_metrics.py tests/test_utils.py \
          --cov=hyperconformal \
          --cov-report=term-missing \
          --cov-report=xml \
          --cov-fail-under=85 \
          -v
    
    - name: Run integration tests
      run: |
        python -m pytest tests/test_integration.py -v --tb=short
    
    - name: Upload coverage to Codecov
      if: matrix.python-version == '3.9'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  test-extended:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' || github.event.pull_request.merged == true
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run end-to-end tests
      run: |
        python -m pytest tests/test_e2e.py -v --tb=short
    
    - name: Run benchmark tests
      run: |
        python -m pytest tests/test_benchmarks.py -v -m "benchmark"
    
    - name: Generate performance report
      run: |
        echo "## Performance Benchmarks" > performance_report.md
        python -m pytest tests/test_benchmarks.py -v --tb=no | grep -E "(PASSED|FAILED|ERROR|::)" >> performance_report.md || true
    
    - name: Upload performance report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-report
        path: performance_report.md

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install safety bandit
        pip install -e ".[dev]"
    
    - name: Run safety check
      run: |
        safety check --json || true
    
    - name: Run bandit security scan
      run: |
        bandit -r hyperconformal/ -f json || true

  docs:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install sphinx sphinx-rtd-theme
    
    - name: Check documentation
      run: |
        # Placeholder for documentation build
        echo "Documentation check placeholder"
        python -c "import hyperconformal; help(hyperconformal)"
```

## Setup Instructions

1. **Enable Workflows Permission**: A repository admin needs to grant the GitHub App `workflows` permission
2. **Create Workflow File**: Save the above configuration as `.github/workflows/ci.yml`
3. **Configure Secrets**: Add any necessary secrets to the repository settings
4. **Test Pipeline**: Create a pull request to test the CI/CD pipeline

## Local Testing Commands

Before pushing to trigger CI, test locally:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the comprehensive test runner
python scripts/run_tests.py --suite all

# Run specific test suites
python scripts/run_tests.py --suite unit
python scripts/run_tests.py --suite integration
python scripts/run_tests.py --suite quality
```

## Pipeline Features

- **Multi-Python Testing**: Validates compatibility across Python 3.8-3.11
- **Quality Gates**: Automated code formatting, linting, and type checking
- **Coverage Reporting**: Ensures 85%+ test coverage requirement
- **Security Scanning**: Vulnerability detection with bandit and safety
- **Performance Monitoring**: Automated benchmark execution
- **Artifact Collection**: Test reports and performance metrics

The pipeline is designed to prevent regression and maintain code quality standards automatically.