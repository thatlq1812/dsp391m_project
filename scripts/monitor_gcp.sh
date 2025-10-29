#!/bin/bash
#
# Monitor GCP VM Collection Progress
# Quick commands to check collection status
#

set -e

# Configuration
VM_NAME="traffic-forecast-collector"
ZONE="asia-southeast1-a"
TARGET_COLLECTIONS=54  # 3 days × 18/day

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function show_menu() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════"
    echo "  TRAFFIC FORECAST - MONITORING MENU"
    echo "═══════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo ""
    echo "1) Show collection progress"
    echo "2) View latest logs (live)"
    echo "3) View cron log"
    echo "4) Check VM status"
    echo "5) Validate collected data"
    echo "6) Check disk usage"
    echo "7) Download collected data"
    echo "8) Stop VM"
    echo "9) SSH into VM"
    echo "0) Exit"
    echo ""
}

function check_progress() {
    echo -e "${YELLOW}Fetching collection progress...${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd ~/traffic-forecast
        
        echo '═══════════════════════════════════════════════════════════'
        echo ' COLLECTION PROGRESS'
        echo '═══════════════════════════════════════════════════════════'
        echo ''
        
        # Count run directories
        RUN_COUNT=\\\$(ls -1d data/runs/run_* 2>/dev/null | wc -l)
        
        echo \"Total runs: \\\$RUN_COUNT\"
        
        if [ -f logs/cron.log ]; then
            FAILED=\\\$(grep -c 'FAILED' logs/cron.log 2>/dev/null || echo 0)
            echo \"Failed: \\\$FAILED\"
            
            # Timeline from cron log
            echo \"\"
            echo \"Timeline:\"
            echo \"  First: \\\$(head -1 logs/cron.log | grep -o '\\[.*\\]')\"
            echo \"  Last:  \\\$(tail -1 logs/cron.log | grep -o '\\[.*\\]')\"
        fi
        
        # Progress
        PROGRESS=\\\$(( RUN_COUNT * 100 / ${TARGET_COLLECTIONS} ))
        echo \"\"
        echo \"Progress: \\\$RUN_COUNT/${TARGET_COLLECTIONS} (\\\$PROGRESS%)\"
        
        # Estimate
        REMAINING=\\\$(( ${TARGET_COLLECTIONS} - RUN_COUNT ))
        echo \"Remaining: \\\$REMAINING runs\"
        
        # Storage
        echo \"\"
        echo \"Storage:\"
        du -sh data/runs 2>/dev/null || echo 'No data yet'
        
        echo ''
        echo '═══════════════════════════════════════════════════════════'
    "
}

function view_logs() {
    echo -e "${YELLOW}Viewing collection logs (Ctrl+C to exit)...${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        tail -f ~/traffic-forecast/logs/collection.log
    "
}

function view_cron_log() {
    echo -e "${YELLOW}Last 30 cron entries:${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        tail -30 ~/traffic-forecast/logs/cron.log
    "
}

function check_vm_status() {
    echo -e "${YELLOW}VM Status:${NC}"
    echo ""
    
    gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="
        table(
            name,
            status,
            machineType.basename(),
            networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP
        )
    "
    
    echo ""
    echo "Uptime:"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="uptime"
}

function validate_data() {
    echo -e "${YELLOW}Validating collected data...${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        cd ~/traffic-forecast
        
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate dsp
        
        python3 << 'PYTHON_EOF'
import json
from pathlib import Path
import glob

print('═══════════════════════════════════════════════════════════')
print(' DATA VALIDATION')
print('═══════════════════════════════════════════════════════════')
print()

# Count run directories
run_dirs = sorted(glob.glob('data/runs/run_*'))

print(f'Total runs: {len(run_dirs)}')

if run_dirs:
    # Check latest run
    latest_run = run_dirs[-1]
    latest_file = Path(latest_run) / 'traffic_edges.json'
    
    print(f'Latest run: {Path(latest_run).name}')
    print()
    
    if latest_file.exists():
        with open(latest_file) as f:
            data = json.load(f)
        
        print(f'Records in latest: {len(data)}')
        
        if data:
            # Check fields
            sample = data[0]
            required = ['node_a_id', 'node_b_id', 'speed_kmh', 'duration_sec', 'timestamp']
            missing = [f for f in required if f not in sample]
            
            if missing:
                print(f'⚠️  Missing fields: {missing}')
            else:
                print('✓ All required fields present')
            
            # Stats
            speeds = [d['speed_kmh'] for d in data if 'speed_kmh' in d]
            if speeds:
                print(f'✓ Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h')
                print(f'✓ Average speed: {sum(speeds)/len(speeds):.1f} km/h')
            
            # Timestamp
            print(f'')
            print(f'Latest collection timestamp: {data[0][\"timestamp\"]}')
    else:
        print('❌ No traffic data in latest run')
else:
    print('❌ No run directories found')

print()
print('═══════════════════════════════════════════════════════════')
PYTHON_EOF
    "
}

function check_disk() {
    echo -e "${YELLOW}Disk Usage:${NC}"
    echo ""
    
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        echo 'Overall:'
        df -h / | tail -1
        echo ''
        echo 'Project directory:'
        du -sh ~/traffic-forecast
        echo ''
        echo 'Data breakdown:'
        du -sh ~/traffic-forecast/data/* 2>/dev/null || echo 'No data yet'
    "
}

function download_data() {
    echo -e "${YELLOW}Downloading collected data...${NC}"
    echo ""
    
    LOCAL_DIR="./data-collected-$(date +%Y%m%d_%H%M%S)"
    
    echo "Download to: $LOCAL_DIR"
    
    gcloud compute scp --recurse "$VM_NAME:~/traffic-forecast/data/runs" "$LOCAL_DIR" --zone="$ZONE"
    
    echo ""
    echo -e "${GREEN}✓ Data downloaded to: $LOCAL_DIR${NC}"
    
    # Show summary
    echo ""
    echo "Downloaded runs:"
    ls -d "$LOCAL_DIR"/run_* 2>/dev/null | wc -l
}

function stop_vm() {
    echo -e "${YELLOW}Stopping VM...${NC}"
    
    read -p "Are you sure? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" == "yes" ]; then
        gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
        echo -e "${GREEN}✓ VM stopped${NC}"
        echo ""
        echo "To restart: gcloud compute instances start $VM_NAME --zone=$ZONE"
    else
        echo "Cancelled"
    fi
}

function ssh_vm() {
    echo -e "${YELLOW}Connecting to VM...${NC}"
    gcloud compute ssh "$VM_NAME" --zone="$ZONE"
}

# Main menu loop
while true; do
    show_menu
    read -p "Select option: " CHOICE
    echo ""
    
    case $CHOICE in
        1) check_progress ;;
        2) view_logs ;;
        3) view_cron_log ;;
        4) check_vm_status ;;
        5) validate_data ;;
        6) check_disk ;;
        7) download_data ;;
        8) stop_vm ;;
        9) ssh_vm ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
