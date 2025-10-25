#!/bin/bash
# Cleanup failed deployment
# Usage: ./scripts/cleanup_failed_deployment.sh

INSTANCE_NAME="${1:-traffic-collector-v4}"
ZONE="${2:-asia-southeast1-b}"
PROJECT_ID="${GCP_PROJECT_ID:-shaped-ship-474607-f7}"

echo "Cleaning up failed deployment..."
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Project: $PROJECT_ID"
echo ""

# Check if instance exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
    echo "Instance found. Deleting..."
    gcloud compute instances delete $INSTANCE_NAME \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        --quiet
    echo "✓ Instance deleted"
else
    echo "✓ No instance found (already cleaned up)"
fi

echo ""
echo "Cleanup complete. Ready to deploy again."
