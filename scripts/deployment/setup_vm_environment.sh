#!/bin/bash
# Simple setup script for VM deployment
# This installs only essential packages needed for data collection

set -e

echo "==========================================="
echo "Setting up Traffic Forecast Collector"
echo "==========================================="

# Create conda environment with Python 3.10
echo "Creating conda environment..."
conda create -n dsp python=3.10 -y

# Activate environment
source $(conda info --base)/bin/activate dsp

# Install essential packages via conda (faster and more reliable on Linux)
echo "Installing essential packages via conda..."
conda install -c conda-forge -y \
    requests \
    pyyaml \
    python-dotenv \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn

# Install remaining packages via pip
echo "Installing remaining packages via pip..."
pip install \
    fastapi \
    uvicorn \
    apscheduler \
    geopy

# Install project in editable mode
echo "Installing project..."
pip install -e .

echo ""
echo "==========================================="
echo "✓ Setup completed successfully!"
echo "==========================================="
echo ""
echo "To activate environment:"
echo "  conda activate dsp"
echo ""
echo "To test installation:"
echo "  python -c 'import traffic_forecast; print(\"OK\")'"
echo ""
