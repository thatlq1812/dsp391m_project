#!/bin/bash
# GCP VM Deployment Helper Script
# Run this on your GCP VM to deploy Traffic Forecast v5.0

set -e  # Exit on error

echo "======================================================================"
echo "🚀 TRAFFIC FORECAST v5.0 - GCP VM DEPLOYMENT"
echo "======================================================================"

# Configuration
PROJECT_DIR="$HOME/traffic-forecast"
CONDA_ENV="dsp"
PYTHON_VERSION="3.9"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Step 1: System Update
print_step "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git wget curl build-essential

# Step 2: Install Miniconda
print_step "Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda3
    rm ~/miniconda.sh
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    print_warning "Miniconda already installed"
fi

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Step 3: Create Conda Environment
print_step "Creating conda environment: $CONDA_ENV"
if conda env list | grep -q "^$CONDA_ENV "; then
    print_warning "Environment $CONDA_ENV already exists, updating..."
    conda env update -n $CONDA_ENV -f environment.yml
else
    conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
fi

# Step 4: Activate Environment
print_step "Activating environment..."
conda activate $CONDA_ENV

# Step 5: Install Dependencies
print_step "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: Create Directory Structure
print_step "Creating directory structure..."
mkdir -p $PROJECT_DIR/{data,cache,logs,models}
mkdir -p $PROJECT_DIR/data/downloads

# Step 7: Copy Configuration Files
print_step "Setting up configuration files..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    print_error ".env file not found! Please create it with your API keys."
    echo "Create $PROJECT_DIR/.env with:"
    echo "  GOOGLE_MAPS_API_KEY=your_key_here"
    exit 1
fi

# Step 8: Run Initial Overpass Collection
print_step "Running initial Overpass topology collection..."
python traffic_forecast/collectors/overpass/collector.py

# Check if cache was created
if [ -f "$PROJECT_DIR/cache/overpass_topology.json" ]; then
    echo -e "${GREEN}✅ Topology cache created successfully${NC}"
else
    print_error "Failed to create topology cache"
    exit 1
fi

# Step 9: Test Google API Collection
print_step "Testing Google Directions API (5 edges)..."
export GOOGLE_TEST_LIMIT=5
python traffic_forecast/collectors/google/collector.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Google API test successful${NC}"
else
    print_error "Google API test failed! Check your API key and permissions."
    exit 1
fi

# Step 10: Setup Cron Job
print_step "Setting up cron job for scheduled collection..."

# Create collection script
cat > $PROJECT_DIR/run_collection.sh << 'EOF'
#!/bin/bash
# Automated collection script

cd $(dirname "$0")

# Activate conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dsp

# Run collection
python traffic_forecast/collectors/google/collector.py >> logs/collection.log 2>&1

# Log completion
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Collection completed" >> logs/cron.log
EOF

chmod +x $PROJECT_DIR/run_collection.sh

# Add to crontab (every 15 minutes)
CRON_CMD="*/15 * * * * $PROJECT_DIR/run_collection.sh"
(crontab -l 2>/dev/null | grep -v "run_collection.sh"; echo "$CRON_CMD") | crontab -

echo -e "${GREEN}✅ Cron job added: Collection every 15 minutes${NC}"

# Step 11: Create systemd service (optional, for better reliability)
print_step "Creating systemd service (optional)..."

sudo tee /etc/systemd/system/traffic-collection.service > /dev/null << EOF
[Unit]
Description=Traffic Forecast Data Collection
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/run_collection.sh
StandardOutput=append:$PROJECT_DIR/logs/service.log
StandardError=append:$PROJECT_DIR/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/traffic-collection.timer > /dev/null << EOF
[Unit]
Description=Traffic Collection Timer (every 15 minutes)

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable traffic-collection.timer
sudo systemctl start traffic-collection.timer

echo -e "${GREEN}✅ Systemd timer created and started${NC}"

# Step 12: Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
echo "======================================================================"
echo ""
echo "📋 Next Steps:"
echo "   1. Verify cron job: crontab -l"
echo "   2. Check systemd timer: sudo systemctl status traffic-collection.timer"
echo "   3. Monitor logs: tail -f $PROJECT_DIR/logs/collection.log"
echo "   4. Check collected data: ls -lh $PROJECT_DIR/data/"
echo ""
echo "📊 Collection Schedule:"
echo "   • Frequency: Every 15 minutes"
echo "   • Expected: 96 collections/day"
echo "   • Estimated cost: ~$1.17/day (~$35/month)"
echo ""
echo "🔍 Monitoring Commands:"
echo "   • View recent logs: tail -n 50 $PROJECT_DIR/logs/collection.log"
echo "   • Check cron logs: tail -n 50 $PROJECT_DIR/logs/cron.log"
echo "   • Test collection: cd $PROJECT_DIR && ./run_collection.sh"
echo ""
echo "======================================================================"
