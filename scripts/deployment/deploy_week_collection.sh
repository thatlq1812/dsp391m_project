#!/bin/bash
# Automated 1-Week Data Collection Deployment on GCP
# Version: Academic v4.0
# Author: THAT Le Quang (Xiel)
# Date: October 25, 2025
set +H  # Disable history expansion to avoid "event ! not found" errors

set -e # Exit on error

# ============================================================
# CONFIGURATION
# ============================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# GCP Configuration
PROJECT_ID="${GCP_PROJECT_ID:-traffic-forecast-391}"
ZONE="${GCP_ZONE:-asia-southeast1-b}"
REGION="${GCP_REGION:-asia-southeast1}"
INSTANCE_NAME="${INSTANCE_NAME:-traffic-collector-v4}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-2}" # 2 vCPU, 8GB RAM
DISK_SIZE="${DISK_SIZE:-50GB}"
IMAGE_FAMILY="ubuntu-2204-lts" # Ubuntu 22.04 LTS (updated from 20.04)
IMAGE_PROJECT="ubuntu-os-cloud"

# Repository Configuration
REPO_URL="https://github.com/thatlq1812/dsp391m_project.git"
REPO_BRANCH="master"

# Collection Configuration
COLLECTION_DURATION_DAYS=7
USE_REAL_API="${USE_REAL_API:-false}" # Set to true for real Google API

# ============================================================
# HELPER FUNCTIONS
# ============================================================

print_header() {
 echo -e "\n${BLUE}=========================================${NC}"
 echo -e "${BLUE}$1${NC}"
 echo -e "${BLUE}=========================================${NC}\n"
}

print_step() {
 echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
 echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
 echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
 echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# ============================================================
# MAIN DEPLOYMENT STEPS
# ============================================================

print_header "Traffic Forecast - 1 Week Collection Deployment"

# Step 0: Pre-flight checks
print_step "Running pre-flight checks..."

if ! command -v gcloud &> /dev/null; then
 print_error "gcloud CLI not found. Please install Google Cloud SDK."
 exit 1
fi

print_info "gcloud CLI: $(gcloud version | head -1)"

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
 print_error "Not authenticated with gcloud. Run: gcloud auth login"
 exit 1
fi

ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
print_info "Authenticated as: $ACCOUNT"

# Step 1: Set GCP project
print_step "Setting GCP project..."
gcloud config set project $PROJECT_ID
print_info "Project set to: $PROJECT_ID"

# Step 2: Enable required APIs
print_step "Enabling required GCP APIs..."
gcloud services enable compute.googleapis.com --quiet
gcloud services enable logging.googleapis.com --quiet
gcloud services enable monitoring.googleapis.com --quiet
print_info "APIs enabled"

# Step 3: Create firewall rules (if not exists)
print_step "Configuring firewall rules..."
if ! gcloud compute firewall-rules describe allow-ssh --project=$PROJECT_ID &>/dev/null; then
 gcloud compute firewall-rules create allow-ssh \
 --allow tcp:22 \
 --source-ranges 0.0.0.0/0 \
 --description "Allow SSH access" \
 --project=$PROJECT_ID
 print_info "Firewall rule 'allow-ssh' created"
else
 print_info "Firewall rule 'allow-ssh' already exists"
fi

# Step 4: Create VM instance
print_step "Creating VM instance: $INSTANCE_NAME..."

if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
 print_info "Instance '$INSTANCE_NAME' already exists"
 read -p "Delete and recreate? (y/N): " -n 1 -r
 echo
 if [[ $REPLY =~ ^[Yy]$ ]]; then
 print_info "Deleting existing instance..."
 gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
 print_info "Instance deleted"
 else
 print_info "Using existing instance"
 fi
fi

if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
 gcloud compute instances create $INSTANCE_NAME \
 --zone=$ZONE \
 --machine-type=$MACHINE_TYPE \
 --image-family=$IMAGE_FAMILY \
 --image-project=$IMAGE_PROJECT \
 --boot-disk-size=$DISK_SIZE \
 --boot-disk-type=pd-standard \
 --metadata=enable-oslogin=TRUE \
 --scopes=https://www.googleapis.com/auth/cloud-platform \
 --tags=traffic-collector
 
 print_success "VM instance created: $INSTANCE_NAME"
 print_info "Waiting 30 seconds for instance to fully boot..."
 sleep 30
else
 print_info "Using existing instance: $INSTANCE_NAME"
fi

# Step 5: Get instance IP
INSTANCE_IP=$(gcloud compute instances describe $INSTANCE_NAME \
 --zone=$ZONE \
 --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
print_info "Instance IP: $INSTANCE_IP"

# Step 6: Create startup script
print_step "Creating startup script..."
cat > /tmp/setup_script.sh << 'EOFSCRIPT'
#!/bin/bash
set -e

echo "Starting Traffic Forecast setup..."

# Update system
sudo apt-get update -qq
sudo apt-get install -y git wget curl vim htop

# Install Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
 wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
 bash /tmp/miniconda.sh -b -p $HOME/miniconda3
 rm /tmp/miniconda.sh
 export PATH="$HOME/miniconda3/bin:$PATH"
 $HOME/miniconda3/bin/conda init bash
fi

# Source conda
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc || true

# Accept Conda Terms of Service (required for environment creation)
echo "Accepting Conda Terms of Service..."
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Clone repository
if [ ! -d "$HOME/dsp391m_project" ]; then
 git clone REPO_URL_PLACEHOLDER $HOME/dsp391m_project
fi

cd $HOME/dsp391m_project

# Checkout branch
git checkout BRANCH_PLACEHOLDER
git pull

# Create conda environment
if ! conda env list | grep -q "^dsp "; then
 echo "Creating conda environment (this may take 5-10 minutes)..."
 conda env create -f environment.yml
fi

# Create .env file
cat > .env << 'EOF'
PROJECT_ENV=production
TIMEZONE=Asia/Ho_Chi_Minh
USE_MOCK_API=USE_MOCK_API_PLACEHOLDER
GOOGLE_MAPS_API_KEY=API_KEY_PLACEHOLDER
DATA_DIR=./data
LOGS_DIR=./logs
CACHE_DIR=./cache
DB_PATH=./traffic_history.db
DB_RETENTION_DAYS=7
ENABLE_MONITORING=true
LOG_LEVEL=INFO
EOF

# Create directories
mkdir -p data/{node,processed,archive} logs cache models

echo "Setup complete!"
EOFSCRIPT

# Replace placeholders
sed -i "s|REPO_URL_PLACEHOLDER|$REPO_URL|g" /tmp/setup_script.sh
sed -i "s|BRANCH_PLACEHOLDER|$REPO_BRANCH|g" /tmp/setup_script.sh
sed -i "s|USE_MOCK_API_PLACEHOLDER|$USE_REAL_API|g" /tmp/setup_script.sh

# Get API key if using real API
if [ "$USE_REAL_API" = "true" ]; then
 read -p "Enter Google Maps API Key: " GOOGLE_API_KEY
 sed -i "s|API_KEY_PLACEHOLDER|$GOOGLE_API_KEY|g" /tmp/setup_script.sh
else
 sed -i "s|API_KEY_PLACEHOLDER||g" /tmp/setup_script.sh
fi

# Step 7: Upload and run setup script
print_step "Uploading setup script to VM..."
gcloud compute scp /tmp/setup_script.sh $INSTANCE_NAME:setup.sh \
 --zone=$ZONE \
 --strict-host-key-checking=no

print_step "Running setup script on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
 --strict-host-key-checking=no \
 --command="bash ~/setup.sh"
print_success "Setup script completed"

# Step 8: Create systemd service for collection
print_step "Creating systemd service..."
cat > /tmp/traffic-collector.service << 'EOFSERVICE'
[Unit]
Description=Traffic Forecast Data Collection Service
After=network.target

[Service]
Type=simple
User=USER_PLACEHOLDER
WorkingDirectory=/home/USER_PLACEHOLDER/dsp391m_project
Environment="PATH=/home/USER_PLACEHOLDER/miniconda3/envs/dsp/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/USER_PLACEHOLDER/miniconda3/envs/dsp/bin/python scripts/collect_and_render.py --interval 1800 --no-visualize
Restart=always
RestartSec=10
StandardOutput=append:/home/USER_PLACEHOLDER/dsp391m_project/logs/collector.log
StandardError=append:/home/USER_PLACEHOLDER/dsp391m_project/logs/collector.error.log

[Install]
WantedBy=multi-user.target
EOFSERVICE

# Get VM user
VM_USER=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
 --strict-host-key-checking=no \
 --command="whoami")
sed -i "s|USER_PLACEHOLDER|$VM_USER|g" /tmp/traffic-collector.service

gcloud compute scp /tmp/traffic-collector.service $INSTANCE_NAME:/tmp/ \
 --zone=$ZONE \
 --strict-host-key-checking=no

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
 --strict-host-key-checking=no \
 --command="
 sudo mv /tmp/traffic-collector.service /etc/systemd/system/
 sudo systemctl daemon-reload
 sudo systemctl enable traffic-collector
"
print_success "Systemd service created and enabled"

# Step 9: Start collection
print_step "Starting data collection..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
 --strict-host-key-checking=no \
 --command="
 sudo systemctl start traffic-collector
 sleep 3
 sudo systemctl status traffic-collector --no-pager
"
print_success "Data collection started!"

# Step 10: Create monitoring script
print_step "Creating monitoring script..."
cat > /tmp/monitor.sh << 'EOFMON'
#!/bin/bash
echo "=== Traffic Collection Status ==="
sudo systemctl status traffic-collector --no-pager | head -20
echo ""
echo "=== Recent Logs (last 20 lines) ==="
tail -20 ~/dsp391m_project/logs/collector.log
echo ""
echo "=== Disk Usage ==="
df -h | grep -E "(Filesystem|/dev/sda1)"
echo ""
echo "=== Data Files ==="
ls -lh ~/dsp391m_project/data/ | tail -10
EOFMON

gcloud compute scp /tmp/monitor.sh $INSTANCE_NAME:monitor.sh \
 --zone=$ZONE \
 --strict-host-key-checking=no

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
 --strict-host-key-checking=no \
 --command="chmod +x ~/monitor.sh"
print_success "Monitoring script uploaded"

# ============================================================
# DEPLOYMENT SUMMARY
# ============================================================

print_header "Deployment Complete!"

cat << EOF
${GREEN}Deployment Summary:${NC}


${BLUE}Instance Details:${NC}
 • Name: $INSTANCE_NAME
 • Zone: $ZONE
 • IP: $INSTANCE_IP
 • Machine Type: $MACHINE_TYPE
 • Status: RUNNING

${BLUE}Collection Configuration:${NC}
 • Duration: $COLLECTION_DURATION_DAYS days
 • Using Real API: $([ "$USE_REAL_API" = "true" ] && echo "YES" || echo "NO (Mock API)")
 • Nodes: 64 (major intersections)
 • Schedule: Adaptive (peak: 30min, off-peak: 60min, weekend: 90min)

${BLUE}Estimated Costs (if using real API):${NC}
 • Daily: ~\$24/day
 • 7 Days: ~\$168
 • 30 Days: ~\$720/month

${YELLOW}Useful Commands:${NC}
 # SSH to instance
 gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no

 # Check status
 gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no --command="~/monitor.sh"

 # View logs
 gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no --command="tail -f ~/dsp391m_project/logs/collector.log"

 # Stop collection
 gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no --command="sudo systemctl stop traffic-collector"

 # Download collected data
 gcloud compute scp --recurse $INSTANCE_NAME:~/dsp391m_project/data/ ./data_downloaded/ --zone=$ZONE --strict-host-key-checking=no

 # Delete instance (after collection)
 gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE

${GREEN}Next Steps:${NC}
 1. Monitor collection: ./scripts/monitor_collection.sh
 2. Wait 7 days for full dataset
 3. Download data and analyze
 4. Delete VM to stop costs


EOF

# Save deployment info
cat > deployment_info.txt << EOF
Deployment Information
Generated: $(date)

Instance Name: $INSTANCE_NAME
Zone: $ZONE
IP Address: $INSTANCE_IP
Machine Type: $MACHINE_TYPE

Collection Started: $(date)
Duration: $COLLECTION_DURATION_DAYS days
Expected Completion: $(date -d "+$COLLECTION_DURATION_DAYS days")

Using Real API: $USE_REAL_API
Estimated Cost: $([ "$USE_REAL_API" = "true" ] && echo "\$168 for 7 days" || echo "\$0 (Mock API)")

SSH Command:
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no

Monitor Command:
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --strict-host-key-checking=no --command="~/monitor.sh"

Download Data:
gcloud compute scp --recurse $INSTANCE_NAME:~/dsp391m_project/data/ ./data_downloaded/ --zone=$ZONE --strict-host-key-checking=no
EOF

print_success "Deployment info saved to: deployment_info.txt"
print_info "Keep this file for reference!"

echo ""
