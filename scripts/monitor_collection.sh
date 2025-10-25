#!/bin/bash
# Quick monitoring script for GCP traffic collection
# Usage: ./scripts/monitor_collection.sh

INSTANCE_NAME="${1:-traffic-collector-v4}"
ZONE="${2:-asia-southeast1-b}"

echo "Connecting to $INSTANCE_NAME..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE \
    --strict-host-key-checking=no \
    --command="bash ~/monitor.sh"
