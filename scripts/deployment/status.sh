#!/bin/bash
# Check VM Status and Recent Collections

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  TRAFFIC FORECAST COLLECTOR - STATUS"
echo "======================================================================"
echo ""

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast

echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  SERVICE STATUS                                                ║'
echo '╚════════════════════════════════════════════════════════════════╝'
sudo systemctl status traffic-collector --no-pager | head -15

echo ''
echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  RECENT COLLECTIONS                                            ║'
echo '╚════════════════════════════════════════════════════════════════╝'
ls -lht data/runs/ | head -10

echo ''
echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  DISK USAGE                                                    ║'
echo '╚════════════════════════════════════════════════════════════════╝'
du -sh data/runs/
echo \"Total runs: \$(ls data/runs/ | wc -l)\"

echo ''
echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  TOPOLOGY                                                      ║'
echo '╚════════════════════════════════════════════════════════════════╝'
if [ -f cache/overpass_topology.json ]; then
    source ~/miniconda3/bin/activate dsp
    python -c '
import json
with open(\"cache/overpass_topology.json\") as f:
    data = json.load(f)
    nodes = len(data.get(\"nodes\", []))
    edges = len(data.get(\"edges\", []))
    print(f\"Nodes: {nodes}, Edges: {edges}, Avg: {edges/nodes:.1f} edges/node\")
'
else
    echo 'No topology cache found'
fi

echo ''
echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  RECENT SCHEDULER ACTIVITY                                     ║'
echo '╚════════════════════════════════════════════════════════════════╝'
tail -15 logs/adaptive_scheduler.log

echo ''
echo '╔════════════════════════════════════════════════════════════════╗'
echo '║  GIT INFO                                                      ║'
echo '╚════════════════════════════════════════════════════════════════╝'
git log -1 --oneline
git status --short
"

echo ""
echo "======================================================================"
echo "  For more details:"
echo "    - Full logs: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "    - Deploy: ./scripts/deployment/deploy_git.sh"
echo "======================================================================"
