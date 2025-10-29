#!/bin/bash
# Automated GCP VM Deployment Script
# Deploys Traffic Forecast Collector from GitHub repository

set -e  # Exit on error

# Configuration
PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
INSTANCE_NAME="traffic-forecast-collector"
MACHINE_TYPE="e2-micro"  # Free tier
DISK_SIZE="30GB"
GITHUB_REPO="https://github.com/thatlq1812/dsp391m_project.git"
GITHUB_BRANCH="master"

echo "=========================================="
echo "Traffic Forecast VM Auto Deployment"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "GitHub: $GITHUB_REPO"
echo "=========================================="

# Step 1: Create VM
echo ""
echo "Step 1: Creating VM instance..."
gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-standard \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --metadata=startup-script='#!/bin/bash
# Startup script to prepare VM
apt-get update
apt-get install -y git wget curl
' \
  --tags=http-server,https-server \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "✓ VM created successfully"

# Wait for VM to be ready
echo ""
echo "Waiting for VM to be ready (60 seconds)..."
sleep 60

# Step 2: Setup environment via SSH
echo ""
echo "Step 2: Setting up environment on VM..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
set -e
echo '=== Installing Miniconda ==='
cd ~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p \$HOME/miniconda3
rm miniconda.sh
export PATH=\"\$HOME/miniconda3/bin:\$PATH\"
conda init bash

# Configure conda
cat > ~/.condarc << EOF
channels:
  - defaults
  - conda-forge
auto_activate_base: false
channel_priority: flexible
EOF

echo '✓ Miniconda installed'
"

echo "✓ Environment setup completed"

# Step 3: Clone repository and setup project
echo ""
echo "Step 3: Cloning repository and setting up project..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
set -e
source ~/.bashrc
export PATH=\"\$HOME/miniconda3/bin:\$PATH\"

echo '=== Cloning GitHub repository ==='
cd ~
git clone $GITHUB_REPO traffic-forecast
cd traffic-forecast
git checkout $GITHUB_BRANCH

echo '✓ Repository cloned'

echo '=== Creating conda environment ==='
conda env create -f environment.yml -n dsp -y

echo '✓ Conda environment created'

echo '=== Installing project in development mode ==='
source \$HOME/miniconda3/bin/activate dsp
pip install -e .

echo '✓ Project installed'
"

echo "✓ Repository and environment setup completed"

# Step 4: Configure environment variables
echo ""
echo "Step 4: Configuring environment variables..."
echo ""
echo "⚠️  IMPORTANT: You need to manually set up the .env file with your API keys"
echo "    Run this command after deployment:"
echo ""
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "    cd ~/traffic-forecast"
echo "    nano .env"
echo ""
echo "    Add your API keys:"
echo "    GOOGLE_MAPS_API_KEY=your_key_here"
echo ""

# Step 5: Setup systemd service
echo ""
echo "Step 5: Setting up systemd service for auto-start..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
set -e

echo '=== Creating systemd service ==='
sudo tee /etc/systemd/system/traffic-collector.service > /dev/null << 'EOF'
[Unit]
Description=Traffic Forecast Adaptive Collector
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/home/$(whoami)/traffic-forecast
Environment=\"PATH=/home/$(whoami)/miniconda3/envs/dsp/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"
ExecStart=/home/$(whoami)/miniconda3/envs/dsp/bin/python scripts/run_adaptive_collection.py
Restart=always
RestartSec=10
StandardOutput=append:/home/$(whoami)/traffic-forecast/logs/service.log
StandardError=append:/home/$(whoami)/traffic-forecast/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p ~/traffic-forecast/logs

echo '✓ Systemd service created'
echo '⚠️  Service NOT started yet - configure .env first'
"

echo "✓ Systemd service configured"

# Step 6: Display summary
echo ""
echo "=========================================="
echo "✓ DEPLOYMENT COMPLETED"
echo "=========================================="
echo ""
echo "VM Information:"
VM_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "  Name: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  External IP: $VM_IP"
echo ""
echo "Next Steps:"
echo ""
echo "1. Configure environment variables:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "   cd ~/traffic-forecast"
echo "   nano .env"
echo "   # Add: GOOGLE_MAPS_API_KEY=your_key_here"
echo ""
echo "2. Start the collector service:"
echo "   sudo systemctl enable traffic-collector"
echo "   sudo systemctl start traffic-collector"
echo ""
echo "3. Check service status:"
echo "   sudo systemctl status traffic-collector"
echo "   tail -f ~/traffic-forecast/logs/adaptive_scheduler.log"
echo ""
echo "4. View collected data:"
echo "   ls -lt ~/traffic-forecast/data/runs/"
echo ""
echo "=========================================="
