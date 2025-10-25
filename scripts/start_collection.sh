#!/bin/bash
# Start Collection Script
# Starts data collection with adaptive scheduling

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if already running
if systemctl is-active --quiet traffic-forecast.service; then
    print_info "Service is already running"
    echo ""
    sudo systemctl status traffic-forecast.service
    exit 0
fi

# Check if using systemd
if command -v systemctl &> /dev/null; then
    print_info "Starting via systemd service..."
    sudo systemctl start traffic-forecast.service
    sleep 2
    
    if systemctl is-active --quiet traffic-forecast.service; then
        print_success "Service started successfully"
        echo ""
        sudo systemctl status traffic-forecast.service
    else
        print_error "Failed to start service"
        echo ""
        echo "Check logs:"
        echo "  sudo journalctl -u traffic-forecast.service -n 50"
        exit 1
    fi
else
    # Fallback to tmux
    print_info "systemd not available, starting in tmux..."
    
    # Check if tmux session exists
    if tmux has-session -t traffic-collect 2>/dev/null; then
        print_info "tmux session 'traffic-collect' already exists"
        echo "Attach with: tmux attach -t traffic-collect"
        exit 0
    fi
    
    # Activate conda and start in tmux
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    
    tmux new-session -d -s traffic-collect \
        "conda activate dsp && python scripts/collect_and_render.py --adaptive"
    
    print_success "Started in tmux session 'traffic-collect'"
    echo ""
    echo "Attach with: tmux attach -t traffic-collect"
    echo "Detach with: Ctrl+B, then D"
fi

echo ""
print_info "View logs:"
echo "  tail -f logs/service.log"
