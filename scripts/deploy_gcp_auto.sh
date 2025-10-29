#!/bin/bash
#
# GCP VM Auto Deployment Script
# Traffic Forecast v5.1 - 3 Day Data Collection
#
# Cost: $21/day × 3 days = ~$63 total
# Target: 54 collections (18/day adaptive)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VM_NAME="traffic-forecast-collector"
ZONE="asia-southeast1-a"
MACHINE_TYPE="e2-micro"
COLLECTION_DAYS=3
COLLECTIONS_PER_DAY=18

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════"
echo "  TRAFFIC FORECAST - GCP AUTO DEPLOYMENT"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"
echo "Project: Traffic Forecast v5.1"
echo "Duration: ${COLLECTION_DAYS} days"
echo "Cost estimate: \$63 (~\$21/day)"
echo "Target: $((COLLECTION_DAYS * COLLECTIONS_PER_DAY)) collections"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/8] Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "  ✓ gcloud CLI installed"

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}  Not authenticated. Running gcloud auth login...${NC}"
    gcloud auth login
fi

echo "  ✓ Authenticated"

# Check required files
REQUIRED_FILES=(
    "configs/project_config.yaml"
    "cache/overpass_topology.json"
    ".env"
    "scripts/collect_once.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$PROJECT_ROOT/$file" ]; then
        echo -e "${RED}ERROR: Missing required file: $file${NC}"
        exit 1
    fi
done

echo "  ✓ All required files present"
echo ""

# Step 2: Select GCP project
echo -e "${YELLOW}[2/8] Select GCP project${NC}"

# List available projects
PROJECTS=$(gcloud projects list --format="value(projectId)" 2>/dev/null)

if [ -z "$PROJECTS" ]; then
    echo -e "${RED}ERROR: No GCP projects found${NC}"
    echo "Create one at: https://console.cloud.google.com"
    exit 1
fi

echo "Available projects:"
select PROJECT_ID in $PROJECTS; do
    if [ -n "$PROJECT_ID" ]; then
        break
    fi
done

gcloud config set project "$PROJECT_ID"
echo -e "${GREEN}  ✓ Project set: $PROJECT_ID${NC}"
echo ""

# Step 3: Confirm deployment
echo -e "${YELLOW}[3/8] Deployment confirmation${NC}"
echo ""
echo "VM Configuration:"
echo "  Name: $VM_NAME"
echo "  Zone: $ZONE"
echo "  Type: $MACHINE_TYPE (free tier eligible in select regions/accounts)"
echo "    Note: Free tier eligibility for e2-micro depends on region and account status."
echo ""
echo "Collection Settings:"
echo "  Duration: ${COLLECTION_DAYS} days"
echo "  Schedule: Adaptive (~18 collections/day)"
# Calculate data points: edges × collections
EDGES=234  # Set this to your actual edge count
TOTAL_COLLECTIONS=$((COLLECTION_DAYS * COLLECTIONS_PER_DAY))
DATA_POINTS=$((EDGES * TOTAL_COLLECTIONS))
echo "  Data points: ~$DATA_POINTS (${EDGES} edges × ${TOTAL_COLLECTIONS} collections)"
echo "  Data points: ~12,636 (234 edges × 54 collections)"
echo ""
echo "Estimated Cost:"
echo "  Daily: ~\$21"
echo "  Total (3 days): ~\$63"
echo ""

read -p "Proceed with deployment? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ] && [ "$CONFIRM" != "y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""

# Step 4: Create VM
echo -e "${YELLOW}[4/8] Creating VM instance...${NC}"

# Enable Compute Engine API if needed
echo "  Checking Compute Engine API..."
if ! gcloud services list --enabled --filter="name:compute.googleapis.com" --format="value(name)" | grep -q compute; then
    echo "  Enabling Compute Engine API..."
    gcloud services enable compute.googleapis.com
    echo "  Waiting for API to be ready..."
    sleep 10
fi

if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
    echo "  VM already exists. Checking status..."
    STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)")
    
    if [ "$STATUS" != "RUNNING" ]; then
        echo "  Starting existing VM..."
        gcloud compute instances start "$VM_NAME" --zone="$ZONE"
    fi
    
    echo -e "${GREEN}  ✓ VM is running${NC}"
else
    echo "  Creating new VM..."
    if gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=30GB \
        --boot-disk-type=pd-standard \
        --metadata=enable-oslogin=true 2>&1; then
        
        echo "  Waiting for VM to be ready..."
        sleep 30
        
        echo -e "${GREEN}  ✓ VM created${NC}"
    else
        echo -e "${RED}  ✗ VM creation failed${NC}"
        echo "  Common issues:"
        echo "    - Compute Engine API not enabled"
        echo "    - Insufficient permissions"
        echo "    - Quota exceeded"
        exit 1
    fi
fi

echo ""

# Step 5: Upload project files
echo -e "${YELLOW}[5/8] Uploading project files...${NC}"

# Create temp archive (exclude unnecessary files)
TEMP_ARCHIVE="/tmp/traffic-forecast-deploy.tar.gz"

echo "  Creating archive..."
cd "$PROJECT_ROOT"

# Create requirements-linux.txt (exclude Windows-only packages)
grep -v -E "pywin32|mpi4py" requirements.txt > requirements-linux.txt

tar -czf "$TEMP_ARCHIVE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data/downloads/*' \
    --exclude='data/runs/*' \
    --exclude='notebooks' \
    --exclude='doc' \
    --exclude='*.egg-info' \
    --exclude='venv' \
    --exclude='*.log' \
    traffic_forecast/ \
    scripts/ \
    configs/ \
    cache/ \
    .env \
    requirements.txt \
    requirements-linux.txt \
    environment.yml \
    pyproject.toml \
    setup.py

echo "  Uploading to VM..."

# Add SSH key to known hosts first (auto-accept)
echo "  Adding SSH key to known hosts..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo 'SSH connection test'" --quiet 2>/dev/null || true
sleep 2

# Upload archive
if gcloud compute scp "$TEMP_ARCHIVE" "$VM_NAME:traffic-forecast-deploy.tar.gz" --zone="$ZONE" --quiet; then
    # Extract on VM
    echo "  Extracting files..."
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        mkdir -p ~/traffic-forecast
        tar -xzf ~/traffic-forecast-deploy.tar.gz -C ~/traffic-forecast
        rm ~/traffic-forecast-deploy.tar.gz
    " --quiet; then
        rm "$TEMP_ARCHIVE"
        echo -e "${GREEN}  ✓ Files uploaded${NC}"
    else
        echo -e "${RED}  ✗ Failed to extract files${NC}"
        rm "$TEMP_ARCHIVE"
        exit 1
    fi
else
    echo -e "${RED}  ✗ Upload failed${NC}"
    echo "  Try running again - SSH keys may need initialization"
    rm "$TEMP_ARCHIVE"
    exit 1
fi
echo ""

# Step 6: Setup environment
echo -e "${YELLOW}[6/8] Setting up environment...${NC}"

echo "  Installing Miniconda..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='if [ ! -d ~/miniconda3 ]; then wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p ~/miniconda3 && rm ~/miniconda.sh; fi'

echo "  Accepting conda ToS..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='source ~/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true'
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='source ~/miniconda3/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true'

echo "  Creating minimal conda environment (Python 3.10)..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='source ~/miniconda3/etc/profile.d/conda.sh && if ! conda env list | grep -q "^dsp "; then conda create -n dsp python=3.10 -y; else echo "Conda environment already exists"; fi'

echo "  Installing packages with pip (Linux-compatible)..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='source ~/miniconda3/etc/profile.d/conda.sh && conda activate dsp && cd ~/traffic-forecast && pip install -r requirements-linux.txt'

echo "  Installing project package..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='source ~/miniconda3/etc/profile.d/conda.sh && conda activate dsp && cd ~/traffic-forecast && pip install --no-deps -e .'

echo "  Creating log directory..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command='mkdir -p ~/traffic-forecast/logs'

echo -e "${GREEN}  ✓ Environment ready${NC}"
echo ""

# Step 7: Test collection
echo -e "${YELLOW}[7/8] Testing collection...${NC}"

echo "  Running test collection..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    cd ~/traffic-forecast
    
    echo '======================================='
    echo 'Running collection test...'
    echo '======================================='
    
    python scripts/collect_once.py
" || {
    echo -e "${RED}  ✗ Test collection failed${NC}"
    echo "  Check logs and try again"
    exit 1
}

echo -e "${GREEN}  ✓ Test collection successful${NC}"
echo ""

# Step 8: Setup cron job
echo -e "${YELLOW}[8/8] Setting up automated collection...${NC}"

# Create collection script
echo "  Creating collection script..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
cat > ~/traffic-forecast/run_collection.sh << 'SCRIPT_EOF'
cat > ~/traffic-forecast/run_collection.sh << 'SCRIPT_EOF'
#!/bin/bash
cd ~/traffic-forecast
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp

echo "[$(date)] Starting collection..." >> logs/cron.log
python scripts/collect_once.py >> logs/collection.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Collection completed successfully" >> logs/cron.log
else
    echo "[$(date)] Collection FAILED with exit code $EXIT_CODE" >> logs/cron.log
fi
SCRIPT_EOF

chmod +x ~/traffic-forecast/run_collection.sh
"
# Setup cron (hourly for now, can be adjusted for adaptive)
echo "  Adding to crontab..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    (crontab -l 2>/dev/null | grep -v '# traffic-forecast-collection'; echo '0 * * * * ~/traffic-forecast/run_collection.sh # traffic-forecast-collection') | crontab -
"

# Verify crontab
echo "  Verifying cron job..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    echo 'Current crontab:'
    crontab -l
"

echo -e "${GREEN}  ✓ Cron job configured${NC}"
echo ""

# Step 9: Setup auto-stop after 3 days
echo -e "${YELLOW}[BONUS] Setting up auto-stop after ${COLLECTION_DAYS} days...${NC}"

# Calculate stop time (exact hour/minute)
STOP_TIME=$(date -d "+${COLLECTION_DAYS} days" '+%M %H %d %m *')

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    # Create auto-stop script
    cat > ~/auto_stop.sh << 'STOP_SCRIPT'
#!/bin/bash
echo \"[\\$(date)] Auto-stopping VM after ${COLLECTION_DAYS} days\" >> ~/traffic-forecast/logs/cron.log
sudo shutdown -h now
STOP_SCRIPT

    chmod +x ~/auto_stop.sh

    # Add to crontab at the exact stop time
    (crontab -l 2>/dev/null | grep -v auto_stop.sh; echo \"$STOP_TIME ~/auto_stop.sh\") | crontab -
"

echo -e "${GREEN}  ✓ Auto-stop scheduled for ${COLLECTION_DAYS} days${NC}"
echo ""

# Summary
echo -e "${GREEN}"
echo "═══════════════════════════════════════════════════════════"
echo "  DEPLOYMENT SUCCESSFUL!"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "VM Information:"
echo "  Name: $VM_NAME"
echo "  Zone: $ZONE"
echo "  Project: $PROJECT_ID"
echo ""
echo "Collection Status:"
echo "  Schedule: Every hour (adaptive)"
echo "  Duration: ${COLLECTION_DAYS} days (auto-stop enabled)"
echo "  Logs: ~/traffic-forecast/logs/"
echo ""
echo "Monitoring Commands:"
echo "  View logs:     gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f ~/traffic-forecast/logs/collection.log'"
echo "  Check cron:    gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -20 ~/traffic-forecast/logs/cron.log'"
echo "  VM status:     gcloud compute instances describe $VM_NAME --zone=$ZONE"
echo ""
echo "Data Download (after 3 days):"
echo "  gcloud compute scp --recurse $VM_NAME:~/traffic-forecast/data ./data-collected --zone=$ZONE"
echo ""
echo "Stop VM manually:"
echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "Delete VM (after data download):"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  • VM will auto-stop after ${COLLECTION_DAYS} days"
echo "  • Download data before deleting VM"
echo "  • Monitor costs: https://console.cloud.google.com/billing"
echo "  • Estimated total cost: ~\$63"
echo ""
echo -e "${GREEN}Happy collecting! 🚀${NC}"
