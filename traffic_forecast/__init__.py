"""
Top-level package for the traffic forecasting project.
Provides convenient helpers to resolve repository paths.
"""

from pathlib import Path

# Absolute path to the project repository root (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the package itself (useful for package-relative resources)
PACKAGE_ROOT = Path(__file__).resolve().parent

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT"]
