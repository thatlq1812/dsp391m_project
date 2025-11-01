#!/bin/bash
# Download Missing Runs from VM (Uncompressed)
# Downloads each missing run individually without compression
# Slower but more reliable for large files or unstable connections

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-thatlq1812@gmail.com}"

# Default download directory
DOWNLOAD_DIR="${1:-./data}"

echo "======================================================================"
echo "  DOWNLOAD MISSING RUNS (UNCOMPRESSED)"
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

# Download each missing run individually
echo ""
echo "======================================================================"
echo "  DOWNLOADING MISSING RUNS (UNCOMPRESSED)"
echo "======================================================================"
echo ""

DOWNLOADED=0
FAILED=0

for run in $MISSING_RUNS; do
    [ -z "$run" ] && continue
    
    echo "[$((DOWNLOADED + FAILED + 1))/$MISSING_COUNT] Downloading: $run"
    
    # Download entire run directory
    if gcloud compute scp \
        --account=$GCLOUD_ACCOUNT \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        --recurse \
        $VM_NAME:~/traffic-forecast/data/runs/$run \
        "$DOWNLOAD_DIR/runs/" \
        --quiet 2>/dev/null; then
        
        DOWNLOADED=$((DOWNLOADED + 1))
        echo "  Success!"
    else
        FAILED=$((FAILED + 1))
        echo "  Failed!"
    fi
    
    echo ""
done

# Summary
echo "======================================================================"
echo "  DOWNLOAD COMPLETED!"
echo "======================================================================"
echo ""
echo "Downloaded: $DOWNLOADED runs"
if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED runs"
fi
echo "Total local runs: $((LOCAL_COUNT + DOWNLOADED))"
echo ""
