#!/bin/bash
# Download Latest Traffic Data from VM
# No authentication required - perfect for team members

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

# Default download directory
DOWNLOAD_DIR="${1:-./downloaded_data}"

echo "======================================================================"
echo "  DOWNLOAD LATEST TRAFFIC DATA"
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
        FILTER="| head -2"
        ;;
    2)
        echo "Downloading last 10 runs..."
        FILTER="| head -11"
        ;;
    3)
        echo "Downloading last 24 hours..."
        FILTER="| head -$(expr 24 \* 4 + 1)"  # ~4 runs/hour
        ;;
    4)
        echo "Downloading all data..."
        FILTER=""
        ;;
    5)
        read -p "Number of runs to download: " NUM_RUNS
        FILTER="| head -$(expr $NUM_RUNS + 1)"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Get list of runs from VM
echo ""
echo "Fetching run list from VM..."

RUN_LIST=$(gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="ls -t ~/traffic-forecast/data/runs/ $FILTER")

# Count runs
RUN_COUNT=$(echo "$RUN_LIST" | wc -l)
echo "Found $RUN_COUNT run(s) to download"
echo ""

# Download each run
COUNTER=0
for RUN in $RUN_LIST; do
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$RUN_COUNT] Downloading $RUN..."
    
    # Create run directory
    mkdir -p "$DOWNLOAD_DIR/$RUN"
    
    # Download run files
    gcloud compute scp \
        --zone=$ZONE \
        --project=$PROJECT_ID \
        --recurse \
        $VM_NAME:~/traffic-forecast/data/runs/$RUN/* \
        "$DOWNLOAD_DIR/$RUN/" \
        --quiet 2>/dev/null || echo "  Warning: Some files may be missing"
done

echo ""
echo "======================================================================"
echo "  DOWNLOAD COMPLETED!"
echo "======================================================================"
echo ""
echo "Downloaded to: $DOWNLOAD_DIR"
echo "Total runs: $RUN_COUNT"
echo ""

# Show summary
if [ -f "$DOWNLOAD_DIR/$(echo $RUN_LIST | head -1)/traffic_edges.json" ]; then
    LATEST_RUN=$(echo $RUN_LIST | head -1)
    echo "Latest run: $LATEST_RUN"
    
    # Count edges in latest run
    EDGE_COUNT=$(grep -o '"node_a_id"' "$DOWNLOAD_DIR/$LATEST_RUN/traffic_edges.json" 2>/dev/null | wc -l || echo "0")
    echo "Edges collected: $EDGE_COUNT"
    
    # File sizes
    echo ""
    echo "File sizes:"
    du -h "$DOWNLOAD_DIR/$LATEST_RUN"/* 2>/dev/null | tail -5
fi

echo ""
echo "To analyze the data:"
echo "  cd $DOWNLOAD_DIR"
echo "  python -m traffic_forecast.cli.visualize"
echo ""
