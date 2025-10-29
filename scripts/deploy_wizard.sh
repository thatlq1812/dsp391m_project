#!/bin/bash
#
# Traffic Forecast v5.1 - Interactive Deployment Wizard
# Simple menu-driven deployment for GCP VM
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
VM_NAME="traffic-forecast-collector"
ZONE="asia-southeast1-a"
MACHINE_TYPE="e2-micro"
COLLECTION_DAYS=3

clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     TRAFFIC FORECAST v5.1 - DEPLOYMENT WIZARD                    ║
║                                                                   ║
║     Adaptive Scheduling • Cost Optimized • Production Ready      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""
echo -e "${CYAN}Collection Strategy:${NC}"
echo "  • Peak hours (6-9, 16-19):   15 min intervals"
echo "  • Off-peak (9-16, 19-22):    60 min intervals"
echo "  • Night mode (22-6):        120 min intervals"
echo ""
echo -e "${CYAN}Expected Results:${NC}"
echo "  • Duration:     3 days"
echo "  • Collections:  ~150 total"
echo "  • Data points:  ~35,100 (234 edges × 150)"
echo "  • Cost:         ~$45 (40% savings)"
echo ""
echo -e "${YELLOW}Press Enter to continue...${NC}"
read

function show_menu() {
    clear
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  DEPLOYMENT MENU${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  1) Check Prerequisites"
    echo "  2) Select GCP Project"
    echo "  3) Create VM"
    echo "  4) Upload Code"
    echo "  5) Setup Environment"
    echo "  6) Test Collection"
    echo "  7) Start Adaptive Scheduler"
    echo "  8) Monitor Collection"
    echo "  9) Download Data"
    echo " 10) Stop/Delete VM"
    echo ""
    echo "  A) AUTO: Full Deployment (Steps 1-7)"
    echo "  0) Exit"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

function check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    echo ""
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}✗ gcloud CLI not found${NC}"
        echo "  Install from: https://cloud.google.com/sdk/docs/install"
        return 1
    fi
    echo -e "${GREEN}✓ gcloud CLI installed${NC}"
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        echo -e "${YELLOW}! Not authenticated${NC}"
        echo "  Running gcloud auth login..."
        gcloud auth login
    fi
    echo -e "${GREEN}✓ Authenticated${NC}"
    
    # Check required files
    local required_files=(
        "configs/project_config.yaml"
        "cache/overpass_topology.json"
        ".env"
        "scripts/collect_once.py"
        "scripts/run_adaptive_collection.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}✗ Missing: $file${NC}"
            return 1
        fi
    done
    echo -e "${GREEN}✓ All required files present${NC}"
    
    # Check .env has API key
    if ! grep -q "GOOGLE_MAPS_API_KEY=AIza" .env 2>/dev/null; then
        echo -e "${YELLOW}⚠ GOOGLE_MAPS_API_KEY not found in .env${NC}"
        echo "  Please add your API key to .env file"
        return 1
    fi
    echo -e "${GREEN}✓ Google API key configured${NC}"
    
    echo ""
    echo -e "${GREEN}All prerequisites OK!${NC}"
    return 0
}

function select_project() {
    echo -e "${YELLOW}Selecting GCP project...${NC}"
    echo ""
    
    # List projects
    local projects=$(gcloud projects list --format="value(projectId)" 2>/dev/null)
    
    if [ -z "$projects" ]; then
        echo -e "${RED}✗ No GCP projects found${NC}"
        echo "  Create one at: https://console.cloud.google.com"
        return 1
    fi
    
    echo "Available projects:"
    select PROJECT_ID in $projects; do
        if [ -n "$PROJECT_ID" ]; then
            break
        fi
    done
    
    gcloud config set project "$PROJECT_ID"
    echo ""
    echo -e "${GREEN}✓ Project set: $PROJECT_ID${NC}"
    
    # Enable APIs
    echo ""
    echo -e "${YELLOW}Enabling required APIs...${NC}"
    gcloud services enable compute.googleapis.com --quiet
    echo -e "${GREEN}✓ Compute Engine API enabled${NC}"
    
    return 0
}

function create_vm() {
    echo -e "${YELLOW}Creating VM: $VM_NAME${NC}"
    echo ""
    
    # Check if VM exists
    if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
        echo -e "${YELLOW}! VM already exists${NC}"
        
        local status=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)")
        echo "  Status: $status"
        
        if [ "$status" != "RUNNING" ]; then
            echo "  Starting VM..."
            gcloud compute instances start "$VM_NAME" --zone="$ZONE"
        fi
        
        echo -e "${GREEN}✓ VM is running${NC}"
        return 0
    fi
    
    # Create new VM
    echo "Creating new VM..."
    echo "  Name: $VM_NAME"
    echo "  Zone: $ZONE"
    echo "  Type: $MACHINE_TYPE (free tier eligible)"
    echo ""
    
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=30GB \
        --boot-disk-type=pd-standard \
        --metadata=enable-oslogin=true
    
    echo ""
    echo "Waiting for VM to be ready..."
    sleep 30
    
    echo -e "${GREEN}✓ VM created successfully${NC}"
    return 0
}

function upload_code() {
    echo -e "${YELLOW}Uploading project code...${NC}"
    echo ""
    
    # Create archive in temp directory
    echo "Creating code archive..."
    local temp_archive="/tmp/traffic-forecast-${RANDOM}.tar.gz"
    
    tar -czf "$temp_archive" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='data/runs/*' \
        --exclude='data/downloads/*' \
        --exclude='notebooks' \
        --exclude='*.log' \
        traffic_forecast/ \
        scripts/ \
        configs/ \
        cache/ \
        .env \
        requirements.txt \
        environment.yml \
        pyproject.toml \
        setup.py
    
    echo "Archive created: $temp_archive"
    echo "Size: $(du -h "$temp_archive" | cut -f1)"
    echo ""
    
    echo "Uploading to VM..."
    gcloud compute scp "$temp_archive" "$VM_NAME:traffic-forecast.tar.gz" --zone="$ZONE"
    
    echo "Extracting on VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        mkdir -p ~/traffic-forecast
        tar -xzf ~/traffic-forecast.tar.gz -C ~/traffic-forecast
        rm ~/traffic-forecast.tar.gz
        ls -la ~/traffic-forecast/
    "
    
    rm "$temp_archive"
    
    echo -e "${GREEN}✓ Code uploaded${NC}"
    return 0
}

function setup_environment() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        # Install Miniconda
        if [ ! -d ~/miniconda3 ]; then
            echo 'Installing Miniconda...'
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p ~/miniconda3
            rm ~/miniconda.sh
        fi
        
        # Initialize conda
        source ~/miniconda3/etc/profile.d/conda.sh
        
        # Accept ToS
        conda config --set always_yes yes --set changeps1 no
        
        # Create environment
        if ! conda env list | grep -q '^dsp '; then
            echo 'Creating conda environment...'
            conda create -n dsp python=3.10 -y
        fi
        
        # Activate and install packages
        conda activate dsp
        cd ~/traffic-forecast
        
        echo 'Installing Python packages...'
        pip install -r requirements.txt
        pip install --no-deps -e .
        
        # Create logs directory
        mkdir -p logs
        
        echo 'Environment setup complete!'
    "
    
    echo ""
    echo -e "${GREEN}✓ Environment ready${NC}"
    return 0
}

function test_collection() {
    echo -e "${YELLOW}Running test collection...${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate dsp
        cd ~/traffic-forecast
        
        echo '======================================='
        echo 'Running collection test...'
        echo '======================================='
        
        python scripts/collect_once.py
    "
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Test collection successful${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}✗ Test collection failed${NC}"
        return 1
    fi
}

function start_scheduler() {
    echo -e "${YELLOW}Starting adaptive scheduler service...${NC}"
    echo ""
    
    # Create systemd service
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        # Get current user
        USER=\$(whoami)
        
        # Create service file
        sudo tee /etc/systemd/system/traffic-collection.service > /dev/null << EOF
[Unit]
Description=Traffic Forecast Adaptive Collection v5.1
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=/home/\$USER/traffic-forecast
ExecStart=/home/\$USER/miniconda3/envs/dsp/bin/python scripts/run_adaptive_collection.py
Restart=always
RestartSec=60
StandardOutput=append:/home/\$USER/traffic-forecast/logs/service.log
StandardError=append:/home/\$USER/traffic-forecast/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF
        
        # Reload and start service
        sudo systemctl daemon-reload
        sudo systemctl enable traffic-collection.service
        sudo systemctl start traffic-collection.service
        
        sleep 2
        
        echo 'Service status:'
        sudo systemctl status traffic-collection.service --no-pager -l
    "
    
    echo ""
    echo -e "${GREEN}✓ Adaptive scheduler started${NC}"
    echo ""
    echo "Service will run continuously with adaptive scheduling:"
    echo "  • Peak hours:   15 min intervals"
    echo "  • Off-peak:     60 min intervals"
    echo "  • Night:       120 min intervals"
    
    return 0
}

function monitor_collection() {
    echo -e "${YELLOW}Monitoring collection...${NC}"
    echo ""
    echo "1) View live logs"
    echo "2) Check service status"
    echo "3) View recent collections"
    echo "4) Check disk usage"
    echo "0) Back to main menu"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1)
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                tail -f ~/traffic-forecast/logs/adaptive_scheduler.log
            "
            ;;
        2)
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                sudo systemctl status traffic-collection.service --no-pager -l
            "
            ;;
        3)
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                cd ~/traffic-forecast
                source ~/miniconda3/etc/profile.d/conda.sh
                conda activate dsp
                python scripts/view_collections.py
            "
            ;;
        4)
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
                df -h /
                echo ''
                du -sh ~/traffic-forecast/data/runs
            "
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
}

function download_data() {
    echo -e "${YELLOW}Downloading collected data...${NC}"
    echo ""
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_dir="./data-collected-${timestamp}"
    
    echo "Download to: $output_dir"
    
    gcloud compute scp --recurse "$VM_NAME:~/traffic-forecast/data/runs" "$output_dir" --zone="$ZONE"
    
    echo ""
    echo -e "${GREEN}✓ Data downloaded${NC}"
    echo "Location: $output_dir"
    
    # Show summary
    echo ""
    echo "Downloaded runs:"
    ls -d "$output_dir"/run_* 2>/dev/null | wc -l
    
    read -p "Press Enter to continue..."
}

function stop_delete_vm() {
    echo -e "${YELLOW}VM Management${NC}"
    echo ""
    echo "1) Stop VM (keep data)"
    echo "2) Delete VM (WARNING: all data lost)"
    echo "0) Cancel"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1)
            echo "Stopping VM..."
            gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
            echo -e "${GREEN}✓ VM stopped${NC}"
            ;;
        2)
            echo -e "${RED}WARNING: This will delete the VM and all data!${NC}"
            read -p "Type 'DELETE' to confirm: " confirm
            if [ "$confirm" == "DELETE" ]; then
                echo "Deleting VM..."
                gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
                echo -e "${GREEN}✓ VM deleted${NC}"
            else
                echo "Cancelled"
            fi
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
}

function auto_deploy() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  AUTO DEPLOYMENT - Steps 1-7${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    check_prerequisites || return 1
    echo "" && read -p "Press Enter to continue..."
    
    select_project || return 1
    echo "" && read -p "Press Enter to continue..."
    
    create_vm || return 1
    echo "" && read -p "Press Enter to continue..."
    
    upload_code || return 1
    echo "" && read -p "Press Enter to continue..."
    
    setup_environment || return 1
    echo "" && read -p "Press Enter to continue..."
    
    test_collection || return 1
    echo "" && read -p "Press Enter to continue..."
    
    start_scheduler || return 1
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  DEPLOYMENT COMPLETE!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Your VM is now collecting data with adaptive scheduling."
    echo ""
    echo "Next steps:"
    echo "  • Use option 8 to monitor collection"
    echo "  • Use option 9 to download data after 7 days"
    echo "  • Use option 10 to stop/delete VM when done"
    echo ""
    read -p "Press Enter to return to main menu..."
}

# Main loop
while true; do
    show_menu
    read -p "Select option: " choice
    
    clear
    case $choice in
        1) check_prerequisites; read -p "Press Enter..." ;;
        2) select_project; read -p "Press Enter..." ;;
        3) create_vm; read -p "Press Enter..." ;;
        4) upload_code; read -p "Press Enter..." ;;
        5) setup_environment; read -p "Press Enter..." ;;
        6) test_collection; read -p "Press Enter..." ;;
        7) start_scheduler; read -p "Press Enter..." ;;
        8) monitor_collection ;;
        9) download_data ;;
        10) stop_delete_vm ;;
        A|a) auto_deploy ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
done
