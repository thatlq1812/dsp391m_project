#!/bin/bash
# Download Missing Runs from VM (Compressed)
# Compares local runs with VM runs and downloads only missing ones using tar.gz compression

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-thatlq1812@gmail.com}"

# Default download directory
DOWNLOAD_DIR="${1:-./data}"

echo "======================================================================"
echo "  DOWNLOAD MISSING RUNS (COMPRESSED)"
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
mkdir -p "$DOWNLOAD_DIR/runs"

# Get local runs
echo "Checking local runs..."
if [ -d "$DOWNLOAD_DIR/runs" ]; then
    LOCAL_RUNS=$(ls -1 "$DOWNLOAD_DIR/runs" 2>/dev/null | sort || echo "")
    LOCAL_COUNT=$(echo "$LOCAL_RUNS" | grep -c . || echo 0)
    echo "Local runs: $LOCAL_COUNT"
else
    echo "No local runs found"
    LOCAL_RUNS=""
    LOCAL_COUNT=0
fi

# Get VM runs
echo ""
echo "Fetching VM runs list..."

VM_RUNS=$(gcloud compute ssh $VM_NAME \
    --account=$GCLOUD_ACCOUNT \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="ls -1 ~/traffic-forecast/data/runs 2>/dev/null | sort" || echo "")

if [ -z "$VM_RUNS" ]; then
    echo "ERROR: No runs found on VM or cannot connect"
    exit 1
fi

VM_COUNT=$(echo "$VM_RUNS" | grep -c .)
echo "VM runs: $VM_COUNT"

# Find missing runs
echo ""
echo "Comparing runs..."

MISSING_RUNS=""
MISSING_COUNT=0

for run in $VM_RUNS; do
    if ! echo "$LOCAL_RUNS" | grep -q "^$run$"; then
        MISSING_RUNS="$MISSING_RUNS$run
"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ -z "$MISSING_RUNS" ]; then
    echo ""
    echo "======================================================================"
    echo "  ALL RUNS ARE UP TO DATE!"
    echo "======================================================================"
    echo ""
    echo "No missing runs to download."
    exit 0
fi

echo "Missing runs: $MISSING_COUNT"
echo ""
echo "Missing run IDs:"
echo "$MISSING_RUNS" | head -10
if [ $MISSING_COUNT -gt 10 ]; then
    echo "... and $((MISSING_COUNT - 10)) more"
fi

# Create compressed archive on VM with only missing runs
echo ""
echo "======================================================================"
echo "  CREATING COMPRESSED ARCHIVE ON VM"
echo "======================================================================"
echo ""

ARCHIVE_NAME="traffic_data_missing_$(date +%Y%m%d_%H%M%S).tar.gz"

# Create temporary file list on VM
echo "Creating file list..."

gcloud compute ssh $VM_NAME \
    --account=$GCLOUD_ACCOUNT \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast/data/runs
# Create list of missing runs
cat > /tmp/missing_runs.txt << 'EOF'
$MISSING_RUNS
EOF

echo 'Compressing missing runs...'
tar -czf /tmp/$ARCHIVE_NAME -T /tmp/missing_runs.txt
echo 'Archive created: /tmp/$ARCHIVE_NAME'
du -h /tmp/$ARCHIVE_NAME
rm /tmp/missing_runs.txt
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

tar -xzf /tmp/$ARCHIVE_NAME -C "$DOWNLOAD_DIR/runs"

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
echo "Downloaded: $MISSING_COUNT runs"
echo "Total local runs: $((LOCAL_COUNT + MISSING_COUNT))"
echo ""
