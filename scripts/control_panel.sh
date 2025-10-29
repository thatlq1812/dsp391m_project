#!/bin/bash
#
# Traffic Forecast v5.1 - Local Control Panel
# Interactive dashboard for local development & testing
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if we're in project root
if [ ! -f "configs/project_config.yaml" ]; then
    echo -e "${RED}Error: Run this script from project root${NC}"
    exit 1
fi

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "dsp" ]; then
    echo -e "${YELLOW}Switching to conda environment 'dsp'...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate dsp
fi

function show_header() {
    clear
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║          TRAFFIC FORECAST v5.1 - CONTROL PANEL                   ║
║                                                                   ║
║          Local Development & Testing Dashboard                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

function show_menu() {
    show_header
    echo ""
    echo -e "${CYAN}═══ DATA COLLECTION ═══${NC}"
    echo "  1) Run Single Collection"
    echo "  2) Run Test Collection (2 runs)"
    echo "  3) Start Adaptive Scheduler (background)"
    echo "  4) Stop Background Scheduler"
    echo ""
    echo -e "${CYAN}═══ DATA MANAGEMENT ═══${NC}"
    echo "  5) View Collections"
    echo "  6) Merge Collections"
    echo "  7) Cleanup Old Runs (>14 days)"
    echo "  8) Export Latest Data"
    echo ""
    echo -e "${CYAN}═══ VISUALIZATION ═══${NC}"
    echo "  9) Visualize Latest Run"
    echo " 10) Start Live Dashboard (FastAPI)"
    echo " 11) View Node Information"
    echo " 12) Show Network Topology"
    echo ""
    echo -e "${CYAN}═══ TESTING & DEBUG ═══${NC}"
    echo " 13) Test Google API"
    echo " 14) Test Weather API"
    echo " 15) Check API Rate Limits"
    echo " 16) Verify Cache Files"
    echo ""
    echo -e "${CYAN}═══ SYSTEM ═══${NC}"
    echo " 17) Check Environment"
    echo " 18) View Logs"
    echo " 19) System Status"
    echo ""
    echo "  0) Exit"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

function run_single_collection() {
    echo -e "${YELLOW}Running single collection...${NC}"
    echo ""
    python scripts/collect_once.py
    echo ""
    echo -e "${GREEN}✓ Collection complete${NC}"
    read -p "Press Enter..."
}

function run_test_collection() {
    echo -e "${YELLOW}Running test collection (2 runs)...${NC}"
    echo ""
    echo "Run 1..."
    python scripts/collect_once.py
    echo ""
    echo "Waiting 30 seconds..."
    sleep 30
    echo ""
    echo "Run 2..."
    python scripts/collect_once.py
    echo ""
    echo -e "${GREEN}✓ Test complete${NC}"
    read -p "Press Enter..."
}

function start_adaptive_scheduler() {
    echo -e "${YELLOW}Starting adaptive scheduler in background...${NC}"
    echo ""
    
    # Check if already running
    if pgrep -f "run_adaptive_collection.py" > /dev/null; then
        echo -e "${YELLOW}⚠ Scheduler already running${NC}"
        echo ""
        pgrep -f "run_adaptive_collection.py" -a
        echo ""
        read -p "Press Enter..."
        return
    fi
    
    # Start in background
    nohup python scripts/run_adaptive_collection.py > logs/adaptive_scheduler.log 2>&1 &
    local pid=$!
    
    echo -e "${GREEN}✓ Scheduler started (PID: $pid)${NC}"
    echo ""
    echo "Logs: logs/adaptive_scheduler.log"
    echo "Stop with: kill $pid"
    echo "Or use option 4 from menu"
    echo ""
    read -p "Press Enter..."
}

function stop_scheduler() {
    echo -e "${YELLOW}Stopping background scheduler...${NC}"
    echo ""
    
    local pids=$(pgrep -f "run_adaptive_collection.py")
    
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}No scheduler running${NC}"
        read -p "Press Enter..."
        return
    fi
    
    echo "Found processes:"
    pgrep -f "run_adaptive_collection.py" -a
    echo ""
    
    read -p "Kill these processes? (y/n): " confirm
    if [ "$confirm" == "y" ]; then
        pkill -f "run_adaptive_collection.py"
        echo -e "${GREEN}✓ Scheduler stopped${NC}"
    else
        echo "Cancelled"
    fi
    
    echo ""
    read -p "Press Enter..."
}

function view_collections() {
    echo -e "${YELLOW}Viewing collections...${NC}"
    echo ""
    python scripts/view_collections.py
    echo ""
    read -p "Press Enter..."
}

function merge_collections() {
    echo -e "${YELLOW}Merging collections...${NC}"
    echo ""
    
    if [ ! -d "data/runs" ] || [ -z "$(ls -A data/runs 2>/dev/null)" ]; then
        echo -e "${RED}No collections found${NC}"
        read -p "Press Enter..."
        return
    fi
    
    local output="data/merged_$(date +%Y%m%d_%H%M%S).json"
    
    python scripts/merge_collections.py --output "$output"
    
    echo ""
    echo -e "${GREEN}✓ Collections merged${NC}"
    echo "Output: $output"
    echo ""
    read -p "Press Enter..."
}

function cleanup_old_runs() {
    echo -e "${YELLOW}Cleaning up old runs...${NC}"
    echo ""
    
    read -p "Delete runs older than how many days? (default: 14): " days
    days=${days:-14}
    
    python scripts/cleanup_runs.py --days "$days"
    
    echo ""
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    read -p "Press Enter..."
}

function export_latest() {
    echo -e "${YELLOW}Exporting latest data...${NC}"
    echo ""
    
    local latest=$(ls -td data/runs/run_* 2>/dev/null | head -1)
    
    if [ -z "$latest" ]; then
        echo -e "${RED}No collections found${NC}"
        read -p "Press Enter..."
        return
    fi
    
    local output="data/export_$(date +%Y%m%d_%H%M%S).zip"
    
    echo "Latest run: $latest"
    echo "Creating archive..."
    
    cd "$latest"
    zip -q "../../$output" *.json
    cd ../..
    
    echo ""
    echo -e "${GREEN}✓ Export complete${NC}"
    echo "Output: $output"
    echo "Size: $(du -h "$output" | cut -f1)"
    echo ""
    read -p "Press Enter..."
}

function visualize_latest() {
    echo -e "${YELLOW}Generating visualization...${NC}"
    echo ""
    python visualize.py
    echo ""
    echo -e "${GREEN}✓ Visualization complete${NC}"
    read -p "Press Enter..."
}

function start_dashboard() {
    echo -e "${YELLOW}Starting live dashboard...${NC}"
    echo ""
    echo "Dashboard will run at: http://localhost:8000"
    echo ""
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""
    sleep 2
    
    python scripts/live_dashboard.py
}

function view_node_info() {
    echo -e "${YELLOW}Node Information${NC}"
    echo ""
    
    read -p "Enter node ID (or press Enter for all): " node_id
    
    if [ -z "$node_id" ]; then
        python tools/export_nodes_info.py
    else
        python tools/show_node_info.py "$node_id"
    fi
    
    echo ""
    read -p "Press Enter..."
}

function show_topology() {
    echo -e "${YELLOW}Network Topology${NC}"
    echo ""
    python tools/visualize_nodes.py
    echo ""
    echo -e "${GREEN}✓ Topology visualization generated${NC}"
    read -p "Press Enter..."
}

function test_google_api() {
    echo -e "${YELLOW}Testing Google Directions API...${NC}"
    echo ""
    python tools/test_google_limited.py
    echo ""
    read -p "Press Enter..."
}

function test_weather_api() {
    echo -e "${YELLOW}Testing Open-Meteo API...${NC}"
    echo ""
    python -c "
from traffic_forecast.collectors.weather_collector import WeatherCollector
from traffic_forecast.config import config
import asyncio

async def test():
    collector = WeatherCollector(config)
    data = await collector.collect()
    print(f'✓ Weather data collected')
    print(f'  Grid points: {len(data)}')
    print(f'  Sample: {list(data.keys())[:3]}')

asyncio.run(test())
"
    echo ""
    read -p "Press Enter..."
}

function check_rate_limits() {
    echo -e "${YELLOW}Checking API rate limits...${NC}"
    echo ""
    python tools/test_rate_limiter.py
    echo ""
    read -p "Press Enter..."
}

function verify_cache() {
    echo -e "${YELLOW}Verifying cache files...${NC}"
    echo ""
    
    local files=(
        "cache/overpass_topology.json"
        "cache/weather_grid.json"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            local size=$(du -h "$file" | cut -f1)
            echo -e "${GREEN}✓${NC} $file ($size)"
        else
            echo -e "${RED}✗${NC} $file (missing)"
        fi
    done
    
    echo ""
    echo "Checking topology cache..."
    python tools/check_edges.py
    
    echo ""
    read -p "Press Enter..."
}

function check_environment() {
    echo -e "${YELLOW}Checking environment...${NC}"
    echo ""
    
    echo "Python version:"
    python --version
    echo ""
    
    echo "Conda environment:"
    echo "  Active: ${CONDA_DEFAULT_ENV:-none}"
    echo ""
    
    echo "Key packages:"
    pip list | grep -E "(requests|pydantic|fastapi|folium)"
    echo ""
    
    echo "API Keys:"
    if grep -q "GOOGLE_MAPS_API_KEY=AIza" .env 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} GOOGLE_MAPS_API_KEY configured"
    else
        echo -e "  ${RED}✗${NC} GOOGLE_MAPS_API_KEY missing"
    fi
    
    echo ""
    read -p "Press Enter..."
}

function view_logs() {
    echo -e "${YELLOW}View Logs${NC}"
    echo ""
    echo "1) Adaptive Scheduler"
    echo "2) Collection Errors"
    echo "3) API Calls"
    echo "4) All logs"
    echo "0) Back"
    echo ""
    read -p "Select: " choice
    
    case $choice in
        1) 
            if [ -f "logs/adaptive_scheduler.log" ]; then
                tail -n 50 logs/adaptive_scheduler.log
            else
                echo "No log file found"
            fi
            ;;
        2)
            if [ -f "logs/collection_errors.log" ]; then
                tail -n 50 logs/collection_errors.log
            else
                echo "No log file found"
            fi
            ;;
        3)
            if [ -f "logs/api_calls.log" ]; then
                tail -n 50 logs/api_calls.log
            else
                echo "No log file found"
            fi
            ;;
        4)
            find logs -name "*.log" -exec echo -e "\n=== {} ===" \; -exec tail -n 20 {} \;
            ;;
    esac
    
    echo ""
    read -p "Press Enter..."
}

function system_status() {
    echo -e "${YELLOW}System Status${NC}"
    echo ""
    
    # Disk usage
    echo "Disk Usage:"
    df -h . | tail -1
    echo ""
    
    # Data directory
    if [ -d "data/runs" ]; then
        local runs=$(ls -d data/runs/run_* 2>/dev/null | wc -l)
        local size=$(du -sh data/runs 2>/dev/null | cut -f1)
        echo "Collections:"
        echo "  Count: $runs"
        echo "  Size: $size"
    else
        echo "Collections: none"
    fi
    echo ""
    
    # Cache status
    echo "Cache:"
    if [ -f "cache/overpass_topology.json" ]; then
        local edges=$(python -c "import json; print(len(json.load(open('cache/overpass_topology.json'))['elements']))" 2>/dev/null)
        echo "  Topology: $edges edges"
    fi
    if [ -f "cache/weather_grid.json" ]; then
        local points=$(python -c "import json; print(len(json.load(open('cache/weather_grid.json'))))" 2>/dev/null)
        echo "  Weather grid: $points points"
    fi
    echo ""
    
    # Running processes
    echo "Background Processes:"
    if pgrep -f "run_adaptive_collection.py" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Adaptive scheduler running"
    else
        echo "  No scheduler running"
    fi
    
    echo ""
    read -p "Press Enter..."
}

# Main loop
while true; do
    show_menu
    read -p "Select option: " choice
    
    clear
    case $choice in
        1) run_single_collection ;;
        2) run_test_collection ;;
        3) start_adaptive_scheduler ;;
        4) stop_scheduler ;;
        5) view_collections ;;
        6) merge_collections ;;
        7) cleanup_old_runs ;;
        8) export_latest ;;
        9) visualize_latest ;;
        10) start_dashboard ;;
        11) view_node_info ;;
        12) show_topology ;;
        13) test_google_api ;;
        14) test_weather_api ;;
        15) check_rate_limits ;;
        16) verify_cache ;;
        17) check_environment ;;
        18) view_logs ;;
        19) system_status ;;
        0) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
done
