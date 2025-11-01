#!/bin/bash
# Download All Traffic Data from VM
# Downloads all available runs using tar.gz compression

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-thatlq1812@gmail.com}"

# Default download directory
DOWNLOAD_DIR="${1:-./data}"

echo "======================================================================"
echo "  DOWNLOAD ALL TRAFFIC DATA"
echo "======================================================================"
echo ""

# Check gcloud authentication (warn only, don't exit)
echo "Checking gcloud authentication..."
ACTIVE_ACCOUNT=$(gcloud config list account --format="value(core.account)" 2>/dev/null)

if [ -z "$ACTIVE_ACCOUNT" ]; then
    echo "WARNING: Could not verify gcloud account"
    echo "Attempting to continue anyway..."
else
    echo "Active account: $ACTIVE_ACCOUNT"
fi

echo "Project: $PROJECT_ID"
echo ""

echo "Download directory: $DOWNLOAD_DIR"
echo ""

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

echo "Downloading all data..."

# Create archive on VM
echo ""
echo "Creating compressed archive on VM..."

ARCHIVE_NAME="traffic_data_all_$(date +%Y%m%d_%H%M%S).tar.gz"

gcloud compute ssh $VM_NAME \
    --account=$GCLOUD_ACCOUNT \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast/data
echo 'Compressing all runs...'
tar -czf /tmp/$ARCHIVE_NAME runs/
echo 'Archive created: /tmp/$ARCHIVE_NAME'
du -h /tmp/$ARCHIVE_NAME
"

# Download archive
echo ""
echo "Downloading compressed archive..."

gcloud compute scp \
    --account=$GCLOUD_ACCOUNT \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    $VM_NAME:/tmp/$ARCHIVE_NAME \
    /tmp/$ARCHIVE_NAME

# Extract archive
echo ""
echo "Extracting archive..."

mkdir -p "$DOWNLOAD_DIR"
tar -xzf /tmp/$ARCHIVE_NAME -C "$DOWNLOAD_DIR"

# Count extracted runs
RUN_COUNT=$(ls -1 "$DOWNLOAD_DIR/runs" | wc -l)

# Cleanup
echo ""
echo "Cleaning up..."
rm -f /tmp/$ARCHIVE_NAME

gcloud compute ssh $VM_NAME \
    --account=$GCLOUD_ACCOUNT \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="rm -f /tmp/$ARCHIVE_NAME" \
    --quiet

echo ""
echo "======================================================================"
echo "  DOWNLOAD COMPLETED!"
echo "======================================================================"
echo ""
echo "Downloaded to: $DOWNLOAD_DIR/runs"
echo "Total runs: $RUN_COUNT"
echo ""
