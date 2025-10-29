#!/bin/bash
# Remote Health Check - Check VM status from local machine

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  REMOTE SYSTEM HEALTH CHECK"
echo "======================================================================"
echo ""

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="./traffic-forecast/scripts/monitoring/health_check.sh"

echo ""
echo "For more details:"
echo "  - View status: ./scripts/deployment/status.sh"
echo "  - View statistics: ./scripts/monitoring/view_stats.sh"
echo "  - Monitor logs: ./scripts/deployment/monitor_logs.sh"
