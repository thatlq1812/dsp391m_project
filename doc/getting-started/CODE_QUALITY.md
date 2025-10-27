# Code Quality Guide

This guide explains the code quality tools and standards used in this project.

## Tools Overview

We use several tools to maintain code quality:

- **Black** - Code formatter
- **isort** - Import sorter
- **Flake8** - Style checker
- **Pylint** - Static analyzer
- **mypy** - Type checker
- **Bandit** - Security scanner
- **pre-commit** - Git hooks for automated checks

## Quick Start

### Install Quality Tools

```bash
# Activate conda environment
conda activate dsp

# Install tools
pip install black isort flake8 pylint mypy bandit pre-commit

# Install pre-commit hooks
pre-commit install
```

### Run Quality Checks

```bash
# Format code with Black
black traffic_forecast/ tests/

# Sort imports
isort traffic_forecast/ tests/

# Check style
flake8 traffic_forecast/ tests/

# Static analysis
pylint traffic_forecast/

# Type checking
mypy traffic_forecast/

# Security scan
bandit -r traffic_forecast/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Tool Configurations

### Black (Code Formatter)

Configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 120
target-version = ['py310']
```

**Usage:**

```bash
# Format files
black traffic_forecast/

# Check without modifying
black --check traffic_forecast/

# Show diff
black --diff traffic_forecast/
```

### isort (Import Sorter)

Configuration in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 120
```

**Usage:**

```bash
# Sort imports
isort traffic_forecast/

# Check without modifying
isort --check traffic_forecast/

# Show diff
isort --diff traffic_forecast/
```

### Flake8 (Style Checker)

Configuration in `.flake8`:

```ini
[flake8]
max-line-length = 120
extend-ignore = E203, W503, E501
```

**Usage:**

```bash
# Check code
flake8 traffic_forecast/

# Show statistics
flake8 traffic_forecast/ --statistics

# Check specific file
flake8 traffic_forecast/collectors/google/collector.py
```

### Pylint (Static Analyzer)

Configuration in `pyproject.toml`:

```toml
[tool.pylint.messages_control]
max-line-length = 120
disable = ["C0111", "C0103", "R0913", "R0914", "W0212"]
```

**Usage:**

```bash
# Analyze code
pylint traffic_forecast/

# Generate report
pylint traffic_forecast/ --output-format=text > pylint_report.txt

# Check specific module
pylint traffic_forecast/features/
```

### mypy (Type Checker)

Configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

**Usage:**

```bash
# Type check
mypy traffic_forecast/

# Strict mode
mypy --strict traffic_forecast/

# Generate HTML report
mypy traffic_forecast/ --html-report mypy-report/
```

### Bandit (Security Scanner)

**Usage:**

```bash
# Security scan
bandit -r traffic_forecast/

# Generate JSON report
bandit -r traffic_forecast/ -f json -o bandit_report.json

# Skip specific tests
bandit -r traffic_forecast/ -s B101,B601
```

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit.

### Configuration

See `.pre-commit-config.yaml` for hook configuration.

### Usage

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run manually on staged files
pre-commit run

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks for a commit (not recommended)
git commit --no-verify -m "message"
```

## Code Style Guidelines

### Line Length

- Maximum 120 characters per line
- Break long lines logically
- Use parentheses for line continuation

### Imports

- Standard library imports first
- Third-party imports second
- Local imports last
- Alphabetically sorted within each group

```python
# Standard library
import os
import sys
from datetime import datetime

# Third-party
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Local
from traffic_forecast.features import create_temporal_features
from traffic_forecast.storage import TrafficHistory
```

### Naming Conventions

- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `PascalCase`
- **Functions**: `lowercase_with_underscores()`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private**: `_leading_underscore`

### Docstrings

Use Google-style docstrings:

```python
def calculate_speed(distance_km: float, duration_sec: float) -> float:
    """
    Calculate speed in km/h from distance and duration.

    Args:
        distance_km: Distance in kilometers
        duration_sec: Duration in seconds

    Returns:
        Speed in kilometers per hour

    Raises:
        ValueError: If duration is zero or negative
    """
    if duration_sec <= 0:
        raise ValueError("Duration must be positive")

    return (distance_km / duration_sec) * 3600
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def process_nodes(
    nodes: List[Dict[str, float]],
    config: Optional[Dict] = None
) -> Tuple[int, List[str]]:
    """Process nodes and return results."""
    pass
```

## Continuous Integration

Quality checks run automatically in CI/CD pipeline:

1. **Black** - Code formatting check
2. **Flake8** - Style violations
3. **Pylint** - Static analysis
4. **Bandit** - Security issues
5. **mypy** - Type errors

See `.github/workflows/ci.yml` for CI configuration.

## IDE Integration

### VS Code

Install extensions:

- Python (Microsoft)
- Pylance
- Black Formatter
- isort
- Flake8

Settings (`.vscode/settings.json`):

```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

1. Enable Black formatter in Preferences
2. Enable Flake8 in Code Inspections
3. Enable mypy type checking
4. Configure isort in External Tools

## Common Issues

### Black vs Flake8 Conflicts

Some Black formatting conflicts with Flake8. We ignore these:

- E203 (whitespace before ':')
- W503 (line break before binary operator)

### Import Order

Use isort with Black profile to avoid conflicts:

```bash
isort --profile black traffic_forecast/
```

### Type Checking Errors

For third-party libraries without type stubs:

```python
import library  # type: ignore
```

Or in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = "library.*"
ignore_missing_imports = true
```

## Best Practices

1. **Run formatters before committing**

   ```bash
   black traffic_forecast/ tests/
   isort traffic_forecast/ tests/
   ```

2. **Fix linting issues promptly**

   ```bash
   flake8 traffic_forecast/
   pylint traffic_forecast/
   ```

3. **Keep code simple** - Avoid complex nested structures

4. **Write readable code** - Clear variable names, logical organization

5. **Document public APIs** - All public functions need docstrings

6. **Use type hints** - Especially for public APIs

7. **Fix security issues** - Never ignore Bandit warnings

## Resources

- [Black documentation](https://black.readthedocs.io/)
- [isort documentation](https://pycqa.github.io/isort/)
- [Flake8 documentation](https://flake8.pycqa.org/)
- [Pylint documentation](https://pylint.pycqa.org/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [Bandit documentation](https://bandit.readthedocs.io/)
- [pre-commit documentation](https://pre-commit.com/)
