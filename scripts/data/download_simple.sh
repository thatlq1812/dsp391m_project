#!/bin/bash
# Simple Data Download via HTTP (for team members without gcloud)
# This script will be used after we setup nginx to serve data publicly

DOWNLOAD_DIR="${1:-./traffic_data}"
BASE_URL="http://1.53.74.2:8080/data"  # Public data endpoint

echo "======================================================================"
echo "  DOWNLOAD TRAFFIC DATA (NO AUTH REQUIRED)"
echo "======================================================================"
echo ""

mkdir -p "$DOWNLOAD_DIR"

echo "Fetching available runs..."

# Get list of available runs
RUNS=$(curl -s "$BASE_URL/runs.json" | grep -o '"run_[^"]*"' | tr -d '"' | head -10)

if [ -z "$RUNS" ]; then
    echo "Error: Could not fetch run list"
    echo "Please check if the data server is running on VM"
    exit 1
fi

echo "Latest 10 runs available:"
echo "$RUNS" | nl
echo ""

read -p "Download all 10 runs? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for RUN in $RUNS; do
        echo "Downloading $RUN..."
        mkdir -p "$DOWNLOAD_DIR/$RUN"
        
        # Download each file
        for FILE in nodes.json edges.json traffic_edges.json weather_snapshot.json statistics.json; do
            curl -s -o "$DOWNLOAD_DIR/$RUN/$FILE" "$BASE_URL/$RUN/$FILE" 2>/dev/null && \
                echo "  ✓ $FILE" || \
                echo "  ✗ $FILE (not found)"
        done
    done
    
    echo ""
    echo "Download completed!"
    echo "Data saved to: $DOWNLOAD_DIR"
    echo "Total size: $(du -sh $DOWNLOAD_DIR | cut -f1)"
else
    echo "Download cancelled"
fi

# Setup instructions if server not running
echo ""
echo "======================================================================"
echo "  NOTE FOR ADMIN:"
echo "  If download failed, setup public data server on VM:"
echo "    ssh to VM and run: ./scripts/data/serve_data_public.sh"
echo "======================================================================"
