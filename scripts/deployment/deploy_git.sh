#!/bin/bash
# Git-based Deployment Script
# Deploy latest code from GitHub to GCP VM

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  GIT-BASED DEPLOYMENT TO GCP VM"
echo "======================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're on master branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo -e "${YELLOW}Warning: You're on branch '$CURRENT_BRANCH', not 'master'${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes!${NC}"
    echo "Please commit or stash your changes first."
    git status --short
    exit 1
fi

# Ask for deployment confirmation
echo -e "${YELLOW}This will:${NC}"
echo "  1. Push local commits to GitHub"
echo "  2. Stop traffic-collector service on VM"
echo "  3. Pull latest code from GitHub"
echo "  4. Regenerate topology if config changed"
echo "  5. Restart service"
echo ""
read -p "Continue with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Step 1: Push to GitHub
echo ""
echo -e "${GREEN}[1/5] Pushing to GitHub...${NC}"
git push origin master

# Step 2: Deploy to VM
echo ""
echo -e "${GREEN}[2/5] Connecting to VM: $VM_NAME${NC}"

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
set -e

echo ''
echo '=== Stop Service ==='
sudo systemctl stop traffic-collector

echo ''
echo '=== Pull Latest Code ==='
cd ~/traffic-forecast
git fetch origin
BEFORE_HASH=\$(git rev-parse HEAD)
git reset --hard origin/master
AFTER_HASH=\$(git rev-parse HEAD)

echo \"Before: \$BEFORE_HASH\"
echo \"After:  \$AFTER_HASH\"

# Check if config changed
if git diff \$BEFORE_HASH \$AFTER_HASH --name-only | grep -q 'configs/project_config.yaml'; then
    echo ''
    echo '=== Config Changed - Regenerating Topology ==='
    rm -f cache/overpass_topology.json
    source ~/miniconda3/bin/activate dsp
    python scripts/collect_once.py --force-refresh 2>&1 | tail -30
fi

echo ''
echo '=== Restart Service ==='
sudo systemctl start traffic-collector
sleep 2
sudo systemctl status traffic-collector --no-pager | head -15
"

echo ""
echo -e "${GREEN}[3/5] Verifying deployment...${NC}"

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast

echo ''
echo '=== Git Status ==='
git log -1 --oneline

echo ''
echo '=== Service Status ==='
systemctl is-active traffic-collector && echo 'Service: RUNNING ✓' || echo 'Service: STOPPED ✗'

echo ''
echo '=== Latest Collection ==='
ls -lt data/runs/ | head -3

echo ''
echo '=== Topology Stats ==='
if [ -f cache/overpass_topology.json ]; then
    source ~/miniconda3/bin/activate dsp
    python -c 'import json; data=json.load(open(\"cache/overpass_topology.json\")); print(f\"Nodes: {len(data.get(\\\"nodes\\\", []))}, Edges: {len(data.get(\\\"edges\\\", []))}\")'
else
    echo 'No topology cache found'
fi
"

echo ""
echo -e "${GREEN}======================================================================"
echo "  DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "======================================================================${NC}"
echo ""
echo "Next steps:"
echo "  - Monitor logs: ./scripts/deployment/monitor_logs.sh"
echo "  - Check status: ./scripts/deployment/status.sh"
echo "  - Download data: ./scripts/data/download_latest.sh"
echo ""
