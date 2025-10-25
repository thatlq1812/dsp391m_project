#!/bin/bash
# Install Dependencies Script
# For Traffic Forecast System
# Can be run standalone or called from other scripts

set -e

echo "========================================="
echo "Installing Dependencies"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "Cannot detect OS"
    exit 1
fi

print_info "Detected OS: $OS $VER"

# Update package manager
print_step "Updating package manager..."
case $OS in
    ubuntu|debian)
        sudo apt update
        ;;
    centos|rhel|fedora)
        sudo yum update -y
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Install system dependencies
print_step "Installing system packages..."
case $OS in
    ubuntu|debian)
        sudo apt install -y \
            git \
            wget \
            curl \
            vim \
            htop \
            tmux \
            build-essential \
            ca-certificates \
            gnupg \
            lsb-release \
            sqlite3 \
            libsqlite3-dev \
            python3-dev \
            libssl-dev \
            libffi-dev
        ;;
    centos|rhel|fedora)
        sudo yum install -y \
            git \
            wget \
            curl \
            vim \
            htop \
            tmux \
            gcc \
            gcc-c++ \
            make \
            sqlite \
            sqlite-devel \
            python3-devel \
            openssl-devel \
            libffi-devel
        ;;
esac

print_info "System packages installed"

# Install Miniconda if not present
print_step "Checking Miniconda installation..."
if ! command -v conda &> /dev/null; then
    print_info "Installing Miniconda..."
    
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O /tmp/$MINICONDA_INSTALLER
    bash /tmp/$MINICONDA_INSTALLER -b -p $HOME/miniconda3
    rm /tmp/$MINICONDA_INSTALLER
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Source to make conda available
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    # Update conda
    conda update -y conda
    
    print_info "Miniconda installed"
else
    print_info "Miniconda already installed"
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Create/update conda environment
print_step "Setting up conda environment..."
if [ -f "environment.yml" ]; then
    if conda env list | grep -q "^dsp "; then
        print_info "Updating existing environment..."
        conda env update -f environment.yml -n dsp
    else
        print_info "Creating new environment..."
        conda env create -f environment.yml
    fi
else
    echo "environment.yml not found!"
    exit 1
fi

# Verify installation
print_step "Verifying installation..."
conda activate dsp

echo ""
print_info "Python version:"
python --version

print_info "Testing imports..."
python << 'EOF'
import sys
try:
    import yaml
    import pandas as pd
    import numpy as np
    import requests
    import pydantic
    print("  - PyYAML: OK")
    print("  - Pandas: OK")
    print("  - NumPy: OK")
    print("  - Requests: OK")
    print("  - Pydantic: OK")
    print("\nAll dependencies installed successfully!")
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

echo ""
print_step "Installation complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate dsp"
echo ""
