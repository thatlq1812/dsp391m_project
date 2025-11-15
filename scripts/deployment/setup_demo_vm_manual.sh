#!/bin/bash
# Manual VM Setup Script (if auto-deploy fails)
# Run this ON the VM after SSH connection

set -e

echo "==========================================="
echo "Manual VM Setup for Traffic Demo"
echo "==========================================="
echo ""

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "⚠ WARNING: This script is designed for Ubuntu"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Update system
echo "[1/6] Updating system..."
sudo apt-get update
sudo apt-get install -y git wget curl
echo "✓ System updated"
echo ""

# Install Miniconda
echo "[2/6] Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    cd ~
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    echo "✓ Miniconda installed"
else
    echo "✓ Miniconda already installed"
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
echo ""

# Clone repository
echo "[3/6] Cloning repository..."
if [ ! -d "$HOME/traffic-demo" ]; then
    cd ~
    git clone https://github.com/thatlq1812/dsp391m_project.git traffic-demo
    cd traffic-demo
    git checkout master
    echo "✓ Repository cloned"
else
    echo "✓ Repository already exists"
    cd ~/traffic-demo
    git pull
    echo "✓ Repository updated"
fi
echo ""

# Setup Python environment
echo "[4/6] Setting up Python environment..."
export PATH="$HOME/miniconda3/bin:$PATH"

if ! conda env list | grep -q "^dsp "; then
    conda create -n dsp python=3.10 -y
    echo "✓ Environment created"
else
    echo "✓ Environment already exists"
fi

source $HOME/miniconda3/bin/activate dsp

echo "Installing packages..."
pip install --no-cache-dir \
    pandas \
    pyarrow \
    requests \
    python-dotenv

echo "✓ Packages installed"
echo ""

# Create data directory
echo "[5/6] Creating data directory..."
sudo mkdir -p /opt/traffic_data
sudo chown $(whoami):$(whoami) /opt/traffic_data
echo "✓ Data directory created"
echo ""

# Setup .env file
echo "[6/6] Setting up .env file..."
cd ~/traffic-demo

if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Google Maps API Key (REQUIRED)
GOOGLE_MAPS_API_KEY=your_key_here

# OpenWeatherMap API Key (Optional)
OPENWEATHER_API_KEY=your_key_here
EOF
    echo "✓ .env template created"
    echo ""
    echo "⚠ IMPORTANT: Edit .env file with your API keys:"
    echo "   nano .env"
else
    echo "✓ .env file already exists"
fi
echo ""

echo "==========================================="
echo "✓ Manual setup completed!"
echo "==========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file:"
echo "   nano ~/traffic-demo/.env"
echo ""
echo "2. Build topology cache:"
echo "   conda activate dsp"
echo "   cd ~/traffic-demo"
echo "   python scripts/data/01_collection/build_topology.py"
echo ""
echo "3. Test collector:"
echo "   python scripts/deployment/traffic_collector.py"
echo ""
echo "4. Setup cron job (every 15 minutes):"
echo "   crontab -e"
echo "   Add line:"
echo "   */15 * * * * cd /home/$(whoami)/traffic-demo && /home/$(whoami)/miniconda3/envs/dsp/bin/python scripts/deployment/traffic_collector.py >> /opt/traffic_data/collector.log 2>&1"
echo ""
