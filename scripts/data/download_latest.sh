#!/bin/bash
# Download Latest Traffic Data from VM (with compression)
# Uses tar.gz compression for faster download

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

# Default download directory
DOWNLOAD_DIR="${1:-./data}"

echo "======================================================================"
echo "  DOWNLOAD LATEST TRAFFIC DATA (COMPRESSED)"
echo "======================================================================"
echo ""
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Ask what to download
echo "What do you want to download?"
echo "  1) Latest run only (fastest)"
echo "  2) Last 10 runs"
echo "  3) Last 24 hours"
echo "  4) All data (may be large)"
echo "  5) Custom (specify number of runs)"
echo ""
read -p "Select option (1-5): " -n 1 -r OPTION
echo ""

case $OPTION in
    1)
        echo "Downloading latest run..."
        NUM_RUNS=1
        ;;
    2)
        echo "Downloading last 10 runs..."
        NUM_RUNS=10
        ;;
    3)
        echo "Downloading last 24 hours..."
        NUM_RUNS=96  # ~4 runs/hour * 24h
        ;;
    4)
        echo "Downloading all data..."
        NUM_RUNS=9999
        ;;
    5)
        read -p "Number of runs to download: " NUM_RUNS
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Create archive on VM
echo ""
echo "Creating compressed archive on VM..."

ARCHIVE_NAME="traffic_data_$(date +%Y%m%d_%H%M%S).tar.gz"

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast/data
echo 'Compressing runs...'
ls -t runs/ | head -$NUM_RUNS | tar -czf /tmp/$ARCHIVE_NAME -C runs -T -
echo 'Archive created: /tmp/$ARCHIVE_NAME'
du -h /tmp/$ARCHIVE_NAME
"

# Download archive
echo ""
echo "Downloading compressed archive..."

gcloud compute scp \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    $VM_NAME:/tmp/$ARCHIVE_NAME \
    /tmp/$ARCHIVE_NAME

# Extract archive
echo ""
echo "Extracting archive..."

mkdir -p "$DOWNLOAD_DIR/runs"
tar -xzf /tmp/$ARCHIVE_NAME -C "$DOWNLOAD_DIR/runs"

# Count extracted runs
RUN_COUNT=$(ls -1 "$DOWNLOAD_DIR/runs" | wc -l)

# Cleanup
echo ""
echo "Cleaning up..."
rm -f /tmp/$ARCHIVE_NAME

gcloud compute ssh $VM_NAME \
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

# Show summary
if [ -d "$DOWNLOAD_DIR/runs" ]; then
    LATEST_RUN=$(ls -t "$DOWNLOAD_DIR/runs" | head -1)
    
    if [ -n "$LATEST_RUN" ]; then
        echo "Latest run: $LATEST_RUN"
        
        # Count edges in latest run
        if [ -f "$DOWNLOAD_DIR/runs/$LATEST_RUN/traffic_edges.json" ]; then
            EDGE_COUNT=$(grep -o '"node_a_id"' "$DOWNLOAD_DIR/runs/$LATEST_RUN/traffic_edges.json" 2>/dev/null | wc -l || echo "0")
            echo "Edges collected: $EDGE_COUNT"
        fi
        
        # File sizes
        echo ""
        echo "Latest run files:"
        ls -lh "$DOWNLOAD_DIR/runs/$LATEST_RUN/" 2>/dev/null | tail -n +2
    fi
fi

echo ""
echo "To analyze the data:"
echo "  python scripts/view_collections.py"
echo ""
