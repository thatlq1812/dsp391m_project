#!/bin/bash
# Monitor Real-time Logs from VM

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  MONITORING LOGS FROM $VM_NAME"
echo "  Press Ctrl+C to stop"
echo "======================================================================"
echo ""

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="tail -f ~/traffic-forecast/logs/adaptive_scheduler.log"
