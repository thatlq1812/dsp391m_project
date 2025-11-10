"""
Setup script for STMGT CLI tool
Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="stmgt-cli",
    version="2.0.0",
    description="STMGT Traffic Forecasting CLI Tool",
    author="THAT Le Quang",
    author_email="thatlq1812@github.com",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "stmgt=traffic_forecast.cli:cli",
        ],
    },
    python_requires=">=3.10",
)
