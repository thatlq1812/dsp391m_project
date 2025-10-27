#!/bin/bash
# Complete GCP VM Setup Script for Traffic Forecast System
# Version: Academic v4.0
# Description: One-command setup for Ubuntu/Debian systems
set +H  # Disable history expansion to avoid "event ! not found" errors

set -e # Exit on error

echo "========================================="
echo "Traffic Forecast - GCP Setup"
echo "Academic v4.0"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="dsp391m_project"
REPO_URL="https://github.com/thatlq1812/dsp391m_project.git"
ENV_NAME="dsp"
INSTALL_DIR="$HOME/$PROJECT_NAME"

# Functions
print_step() {
 echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
 echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
 echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
 if ! command -v $1 &> /dev/null; then
 return 1
 fi
 return 0
}

# Step 1: System Update
print_step "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Step 2: Install Essential Tools
print_step "Installing essential tools..."
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
 lsb-release

print_info "Essential tools installed"

# Step 3: Install Miniconda
print_step "Installing Miniconda..."
if ! check_command conda; then
 MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
 wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -O /tmp/$MINICONDA_INSTALLER
 bash /tmp/$MINICONDA_INSTALLER -b -p $HOME/miniconda3
 rm /tmp/$MINICONDA_INSTALLER
 
 # Initialize conda
 $HOME/miniconda3/bin/conda init bash
 source ~/.bashrc
 
 # Update conda
 $HOME/miniconda3/bin/conda update -y conda
 
 print_info "Miniconda installed successfully"
else
 print_info "Miniconda already installed"
fi

# Step 4: Clone Repository
print_step "Cloning repository..."
if [ -d "$INSTALL_DIR" ]; then
 print_info "Project directory exists, updating..."
 cd $INSTALL_DIR
 git pull
else
 git clone $REPO_URL $INSTALL_DIR
 cd $INSTALL_DIR
 print_info "Repository cloned to $INSTALL_DIR"
fi

# Step 5: Create Conda Environment
print_step "Creating conda environment..."
if conda env list | grep -q "^$ENV_NAME "; then
 print_info "Environment '$ENV_NAME' exists, updating..."
 conda env update -f environment.yml -n $ENV_NAME
else
 conda env create -f environment.yml
 print_info "Environment '$ENV_NAME' created"
fi

# Step 6: Setup Environment Variables
print_step "Setting up environment variables..."
if [ ! -f .env ]; then
 cat > .env << 'EOF'
# Traffic Forecast Environment Configuration
# Academic v4.0

# Google Maps API Key (optional - can use mock API)
# GOOGLE_MAPS_API_KEY=your_api_key_here

# Project Settings
PROJECT_ENV=production
TIMEZONE=Asia/Ho_Chi_Minh

# Paths
DATA_DIR=./data
LOGS_DIR=./logs
CACHE_DIR=./cache

# Database
DB_PATH=./traffic_history.db
DB_RETENTION_DAYS=7

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
EOF
 print_info "Environment file created (.env)"
else
 print_info "Environment file already exists"
fi

# Step 7: Create Required Directories
print_step "Creating required directories..."
mkdir -p data/node
mkdir -p data/processed
mkdir -p data/archive
mkdir -p logs
mkdir -p cache
mkdir -p models
print_info "Directories created"

# Step 8: Verify Installation
print_step "Verifying installation..."

# Activate environment and test
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
print_info "Python version:"
python --version

print_info "Key dependencies:"
python -c "
import yaml
import pandas as pd
import numpy as np
print(' - PyYAML: OK')
print(' - Pandas: OK')
print(' - NumPy: OK')
"

# Step 9: Test Collection
print_step "Testing data collection..."
python scripts/collect_and_render.py --print-schedule

# Step 10: Setup Systemd Service
print_step "Setting up systemd service..."
SERVICE_FILE="/etc/systemd/system/traffic-forecast.service"
TEMP_SERVICE="/tmp/traffic-forecast.service"

cat > $TEMP_SERVICE << EOF
[Unit]
Description=Traffic Forecast Data Collection
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$HOME/miniconda3/envs/$ENV_NAME/bin:/usr/bin"
ExecStart=$HOME/miniconda3/envs/$ENV_NAME/bin/python scripts/collect_and_render.py --adaptive
Restart=always
RestartSec=10
StandardOutput=append:$INSTALL_DIR/logs/service.log
StandardError=append:$INSTALL_DIR/logs/service.error.log

[Install]
WantedBy=multi-user.target
EOF

sudo mv $TEMP_SERVICE $SERVICE_FILE
sudo systemctl daemon-reload
sudo systemctl enable traffic-forecast.service

print_info "Systemd service configured (not started yet)"

# Step 11: Setup Cleanup Cron
print_step "Setting up cleanup cron job..."
CRON_ENTRY="0 2 * * * cd $INSTALL_DIR && bash scripts/cleanup.sh >> logs/cleanup.log 2>&1"
(crontab -l 2>/dev/null | grep -v "cleanup.sh"; echo "$CRON_ENTRY") | crontab -
print_info "Cleanup cron job added (runs daily at 2 AM)"

# Step 12: Create Helper Scripts
print_step "Creating helper scripts..."

# Create start script
cat > $INSTALL_DIR/start.sh << 'EOF'
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp
sudo systemctl start traffic-forecast.service
sudo systemctl status traffic-forecast.service
EOF
chmod +x $INSTALL_DIR/start.sh

# Create stop script
cat > $INSTALL_DIR/stop.sh << 'EOF'
#!/bin/bash
sudo systemctl stop traffic-forecast.service
EOF
chmod +x $INSTALL_DIR/stop.sh

# Create status script
cat > $INSTALL_DIR/status.sh << 'EOF'
#!/bin/bash
echo "Service Status:"
sudo systemctl status traffic-forecast.service

echo ""
echo "Recent Logs:"
tail -20 logs/service.log

echo ""
echo "Disk Usage:"
du -sh data/

echo ""
echo "Database Size:"
du -sh traffic_history.db 2>/dev/null || echo "No database yet"
EOF
chmod +x $INSTALL_DIR/status.sh

print_info "Helper scripts created (start.sh, stop.sh, status.sh)"

# Completion
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Installation Directory: $INSTALL_DIR"
echo "Conda Environment: $ENV_NAME"
echo ""
echo "Next Steps:"
echo ""
echo "1. Configure API key (optional):"
echo " nano .env"
echo " # Add: GOOGLE_MAPS_API_KEY=your_key"
echo ""
echo "2. Start the service:"
echo " ./start.sh"
echo ""
echo "3. Check status:"
echo " ./status.sh"
echo ""
echo "4. View logs:"
echo " tail -f logs/service.log"
echo ""
echo "5. Stop the service:"
echo " ./stop.sh"
echo ""
echo "Configuration:"
echo " - Using mock API: YES (free, for development)"
echo " - Collections/day: 25 (adaptive scheduling)"
echo " - Nodes: 64 (major roads only)"
echo " - Monthly cost: \$0 (mock) or \$720 (real API)"
echo ""
echo "Documentation: $INSTALL_DIR/DEPLOY.md"
echo "========================================="
