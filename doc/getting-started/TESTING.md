# Testing Guide

This document explains how to run tests and understand test coverage for the Traffic Forecast project.

## Quick Start

### Run All Tests

```bash
# Activate conda environment
conda activate dsp

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=traffic_forecast --cov-report=term --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/test_*.py -v --ignore=tests/test_integration.py

# Integration tests only
python -m pytest tests/test_integration.py -v

# Specific test file
python -m pytest tests/test_features.py -v

# Specific test class
python -m pytest tests/test_features.py::TestTemporalFeatures -v

# Specific test method
python -m pytest tests/test_features.py::TestTemporalFeatures::test_peak_hour_detection -v
```

## Test Structure

### Unit Tests

Located in `tests/`, organized by module:

- `test_google_collector.py` - Google Directions API collector
- `test_features.py` - Feature engineering (temporal, spatial, lag)
- `test_storage.py` - Traffic history storage
- `test_models.py` - Model training and prediction
- `test_baseline.py` - Baseline persistence model
- `test_validation.py` - Data validation schemas
- `test_node_selector.py` - Node selection logic
- `test_feature_pipeline.py` - Feature pipeline

### Integration Tests

Located in `tests/test_integration.py`:

- Data collection pipeline
- Feature engineering pipeline
- Model training pipeline
- Storage pipeline

## Code Coverage

### Generate Coverage Report

```bash
# Terminal report
python -m pytest tests/ --cov=traffic_forecast --cov-report=term-missing

# HTML report (opens in browser)
python -m pytest tests/ --cov=traffic_forecast --cov-report=html
open htmlcov/index.html  # On macOS
start htmlcov/index.html # On Windows
```

### Coverage Goals

- **Overall**: 70%+ coverage
- **Critical modules**: 80%+ coverage
  - Collectors
  - Feature engineering
  - Models
  - Storage

### View Coverage by Module

```bash
coverage run -m pytest tests/
coverage report --include="traffic_forecast/*"
coverage report --include="traffic_forecast/collectors/*"
coverage report --include="traffic_forecast/features/*"
```

## Continuous Integration

Tests are automatically run via GitHub Actions on:

- Every push to `master` or `develop`
- Every pull request
- Manual workflow dispatch

See `.github/workflows/ci.yml` for CI configuration.

## Writing Tests

### Test Conventions

1. **Naming**: All test files start with `test_`
2. **Classes**: Test classes start with `Test`
3. **Methods**: Test methods start with `test_`
4. **Fixtures**: Use `setUp()` and `tearDown()` for test fixtures

### Example Test

```python
import unittest
from traffic_forecast.features.temporal_features import create_temporal_features
import pandas as pd

class TestTemporalFeatures(unittest.TestCase):
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2025-10-27', periods=10, freq='1H')
        })

    def test_hour_extraction(self):
        """Test hour feature extraction."""
        result = create_temporal_features(self.df)

        self.assertIn('hour', result.columns)
        self.assertTrue(all(0 <= result['hour']) and all(result['hour'] <= 23))

    def tearDown(self):
        """Clean up after test."""
        pass
```

### Best Practices

1. **Test one thing** - Each test should verify one specific behavior
2. **Use descriptive names** - Test name should describe what is being tested
3. **Arrange-Act-Assert** - Set up data, perform action, verify result
4. **Mock external dependencies** - Use `unittest.mock` for API calls, file I/O
5. **Clean up resources** - Use `tearDown()` or context managers

## Troubleshooting

### Common Issues

**Import errors:**

```bash
# Make sure you're in the project root
cd /path/to/project

# Make sure conda environment is activated
conda activate dsp
```

**Module not found:**

```bash
# Install project in development mode
pip install -e .
```

**Slow tests:**

```bash
# Skip slow tests
python -m pytest tests/ -m "not slow"
```

**API-dependent tests fail:**

```bash
# Skip tests requiring API keys
python -m pytest tests/ -m "not requires_api"
```

## Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_test():
    pass

@pytest.mark.integration
def test_integration_test():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.requires_api
def test_with_api():
    pass
```

Run specific markers:

```bash
pytest -m unit
pytest -m "not slow"
pytest -m "unit and not requires_api"
```

## Pre-commit Hooks

Install pre-commit hooks to run tests automatically before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
