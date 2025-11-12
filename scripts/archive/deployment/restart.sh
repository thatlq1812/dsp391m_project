#!/bin/bash
# Restart Traffic Collector Service on VM

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  RESTARTING TRAFFIC COLLECTOR SERVICE"
echo "======================================================================"
echo ""

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
echo '=== Stopping service ==='
sudo systemctl stop traffic-collector

echo ''
echo '=== Starting service ==='
sudo systemctl start traffic-collector

sleep 2

echo ''
echo '=== Service status ==='
sudo systemctl status traffic-collector --no-pager | head -15

echo ''
echo '=== Recent logs ==='
tail -10 ~/traffic-forecast/logs/adaptive_scheduler.log
"

echo ""
echo "Service restarted successfully!"
