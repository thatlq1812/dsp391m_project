# setup.py - Package setup for distribution

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="traffic-forecast",
    version="4.1.0",
    author="THAT Le Quang",
    author_email="fxlqthat@gmail.com",
    description="Cost-optimized traffic forecasting system for Ho Chi Minh City",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thatlq1812/dsp391m_project",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "doc", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "pylint>=3.0.0",
            "mypy>=1.8.0",
            "bandit>=1.7.6",
            "pre-commit>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-forecast=traffic_forecast.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "traffic_forecast": ["configs/*.yaml"],
    },
    zip_safe=False,
)
