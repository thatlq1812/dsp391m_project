#!/bin/bash
# Automated GCP VM Deployment for Demo Data Collection
# Maintainer: THAT Le Quang
# Date: November 15, 2025

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
INSTANCE_NAME="traffic-demo-collector"
MACHINE_TYPE="e2-micro"  # Free tier (0.25-1GB RAM, 2 vCPUs)
DISK_SIZE="30GB"
GITHUB_REPO="https://github.com/thatlq1812/dsp391m_project.git"
GITHUB_BRANCH="master"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Traffic Demo VM Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Project:  $PROJECT_ID"
echo -e "Zone:     $ZONE"
echo -e "Instance: $INSTANCE_NAME"
echo -e "Type:     $MACHINE_TYPE (Free tier)"
echo -e "GitHub:   $GITHUB_REPO"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if VM already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
    echo -e "${YELLOW}⚠ VM '$INSTANCE_NAME' already exists${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting existing VM...${NC}"
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
        echo -e "${GREEN}✓ VM deleted${NC}"
        sleep 5
    else
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Step 1: Create VM
echo -e "${YELLOW}[1/5] Creating VM instance...${NC}"
gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --metadata=startup-script='#!/bin/bash
# Basic setup
apt-get update
apt-get install -y git wget curl
' \
  --tags=http-server,https-server \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo -e "${GREEN}✓ VM created successfully${NC}"
echo ""

# Wait for VM to be ready
echo -e "${YELLOW}Waiting for VM to boot (60 seconds)...${NC}"
sleep 60

# Step 2: Install Miniconda
echo -e "${YELLOW}[2/5] Installing Miniconda...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
cd ~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p \$HOME/miniconda3
rm miniconda.sh
export PATH=\"\$HOME/miniconda3/bin:\$PATH\"
conda init bash

cat > ~/.condarc << 'EOF'
channels:
  - defaults
  - conda-forge
auto_activate_base: false
channel_priority: flexible
EOF

echo '✓ Miniconda installed'
"
echo -e "${GREEN}✓ Miniconda setup completed${NC}"
echo ""

# Step 3: Clone repository
echo -e "${YELLOW}[3/5] Cloning repository...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
source ~/.bashrc
export PATH=\"\$HOME/miniconda3/bin:\$PATH\"

cd ~
git clone $GITHUB_REPO traffic-demo
cd traffic-demo
git checkout $GITHUB_BRANCH

echo '✓ Repository cloned'
"
echo -e "${GREEN}✓ Repository setup completed${NC}"
echo ""

# Step 4: Setup Python environment
echo -e "${YELLOW}[4/5] Setting up Python environment...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e
source ~/.bashrc
export PATH=\"\$HOME/miniconda3/bin:\$PATH\"

cd ~/traffic-demo

# Create minimal conda environment
echo 'Creating conda environment...'
conda create -n dsp python=3.10 -y

# Activate and install packages
source \$HOME/miniconda3/bin/activate dsp

echo 'Installing essential packages...'
pip install --no-cache-dir \
    pandas \
    pyarrow \
    requests \
    python-dotenv

echo '✓ Environment ready'
"
echo -e "${GREEN}✓ Python environment setup completed${NC}"
echo ""

# Step 5: Setup systemd service for cron-like collection
echo -e "${YELLOW}[5/5] Setting up collection service...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
set -e

# Create data directory
mkdir -p /opt/traffic_data

# Create systemd timer unit (runs every 15 minutes)
sudo tee /etc/systemd/system/traffic-collector.timer > /dev/null << 'EOF'
[Unit]
Description=Traffic Data Collection Timer
Requires=traffic-collector.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
AccuracySec=1min

[Install]
WantedBy=timers.target
EOF

# Create systemd service unit
sudo tee /etc/systemd/system/traffic-collector.service > /dev/null << EOF
[Unit]
Description=Traffic Data Collection
After=network.target

[Service]
Type=oneshot
User=\$(whoami)
WorkingDirectory=/home/\$(whoami)/traffic-demo
Environment=\"PATH=/home/\$(whoami)/miniconda3/envs/dsp/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"
ExecStart=/home/\$(whoami)/miniconda3/envs/dsp/bin/python scripts/deployment/traffic_collector.py
StandardOutput=append:/opt/traffic_data/collector.log
StandardError=append:/opt/traffic_data/collector_error.log
EOF

# Reload systemd
sudo systemctl daemon-reload

echo '✓ Service configured (not started yet - needs .env file)'
"
echo -e "${GREEN}✓ Service setup completed${NC}"
echo ""

# Final instructions
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ VM DEPLOYMENT COMPLETED${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}⚠ IMPORTANT: Manual steps required:${NC}"
echo ""
echo -e "${BLUE}1. Set up API keys:${NC}"
echo -e "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo -e "   cd ~/traffic-demo"
echo -e "   nano .env"
echo ""
echo -e "   Add these keys:"
echo -e "   ${GREEN}GOOGLE_MAPS_API_KEY=your_key_here${NC}"
echo -e "   ${GREEN}OPENWEATHER_API_KEY=your_key_here${NC} (optional)"
echo ""
echo -e "${BLUE}2. Build topology cache:${NC}"
echo -e "   cd ~/traffic-demo"
echo -e "   conda activate dsp"
echo -e "   python scripts/data/01_collection/build_topology.py"
echo ""
echo -e "${BLUE}3. Start collection service:${NC}"
echo -e "   sudo systemctl enable traffic-collector.timer"
echo -e "   sudo systemctl start traffic-collector.timer"
echo ""
echo -e "${BLUE}4. Check service status:${NC}"
echo -e "   systemctl status traffic-collector.timer"
echo -e "   tail -f /opt/traffic_data/collector.log"
echo ""
echo -e "${BLUE}5. Download data after 3-5 days:${NC}"
echo -e "   gcloud compute scp $INSTANCE_NAME:/opt/traffic_data/traffic_data_*.parquet ./data/demo/ --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo -e "${YELLOW}Estimated cost: FREE (e2-micro under quota) or ~\$8-15/month${NC}"
echo ""
