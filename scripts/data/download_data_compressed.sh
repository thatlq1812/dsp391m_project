#!/bin/bash
# Download Traffic Data with Compression (Fast)
# Uses tar.gz compression to reduce download time significantly
#
# Usage: bash scripts/data/download_data_compressed.sh [num_runs] [output_dir]
#
# Arguments:
#   num_runs - Number of recent runs to download (default: interactive)
#   output_dir - Local directory to save download (default: ./data)

set -e

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

# Parse arguments
NUM_RUNS="${1:-}"
DOWNLOAD_DIR="${2:-./data}"

echo "======================================================================"
echo "  DOWNLOAD TRAFFIC DATA (COMPRESSED - FAST)"
echo "======================================================================"
echo ""
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# Create download directory
mkdir -p "$DOWNLOAD_DIR"

# Interactive mode if NUM_RUNS not specified
if [ -z "$NUM_RUNS" ]; then
    echo "What do you want to download?"
    echo "  1) Latest run only (~140KB compressed)"
    echo "  2) Last 5 runs (~700KB compressed)"
    echo "  3) Last 10 runs (~1.4MB compressed)"
    echo "  4) Last 24 hours (~15MB compressed)"
    echo "  5) All data (varies)"
    echo "  6) Custom (specify number of runs)"
    echo ""
    read -p "Select option (1-6): " -n 1 -r OPTION
    echo ""
    
    case $OPTION in
        1)
            echo "Downloading latest run..."
            NUM_RUNS=1
            ;;
        2)
            echo "Downloading last 5 runs..."
            NUM_RUNS=5
            ;;
        3)
            echo "Downloading last 10 runs..."
            NUM_RUNS=10
            ;;
        4)
            echo "Downloading last 24 hours..."
            NUM_RUNS=96  # ~4 runs/hour * 24h
            ;;
        5)
            echo "Downloading all data..."
            NUM_RUNS=9999
            ;;
        6)
            read -p "Number of runs to download: " NUM_RUNS
            ;;
        *)
            echo "Invalid option. Exiting."
            exit 1
            ;;
    esac
fi

# Create archive on VM
echo ""
echo "Step 1/4: Creating compressed archive on VM..."

ARCHIVE_NAME="traffic_data_$(date +%Y%m%d_%H%M%S).tar.gz"

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast/data
echo 'Listing runs to compress...'
RUNS=\$(ls -t runs/ | head -$NUM_RUNS)
echo \"Found \$(echo \"\$RUNS\" | wc -l) runs\"

echo 'Compressing...'
echo \"\$RUNS\" | tar -czf /tmp/$ARCHIVE_NAME -C runs -T -

echo 'Archive created:'
ls -lh /tmp/$ARCHIVE_NAME
"

# Download archive
echo ""
echo "Step 2/4: Downloading compressed archive..."
echo "  (This will be much faster than downloading individual files)"
echo ""

gcloud compute scp \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    $VM_NAME:/tmp/$ARCHIVE_NAME \
    /tmp/$ARCHIVE_NAME

# Extract archive
echo ""
echo "Step 3/4: Extracting archive..."

mkdir -p "$DOWNLOAD_DIR/runs"
tar -xzf /tmp/$ARCHIVE_NAME -C "$DOWNLOAD_DIR/runs"

# Count extracted runs
RUN_COUNT=$(ls -1 "$DOWNLOAD_DIR/runs" | wc -l)

# Cleanup
echo ""
echo "Step 4/4: Cleaning up temporary files..."
rm -f /tmp/$ARCHIVE_NAME

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="rm -f /tmp/$ARCHIVE_NAME" \
    --quiet 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  ✅ DOWNLOAD COMPLETED!"
echo "======================================================================"
echo ""
echo "📁 Downloaded to: $DOWNLOAD_DIR/runs"
echo "📊 Total runs: $RUN_COUNT"
echo "💾 Total size: $(du -sh $DOWNLOAD_DIR/runs | cut -f1)"
echo ""

# Show summary
if [ -d "$DOWNLOAD_DIR/runs" ]; then
    LATEST_RUN=$(ls -t "$DOWNLOAD_DIR/runs" | head -1)
    
    if [ -n "$LATEST_RUN" ]; then
        echo "📈 Latest run: $LATEST_RUN"
        
        # Count edges in latest run
        if [ -f "$DOWNLOAD_DIR/runs/$LATEST_RUN/traffic_edges.json" ]; then
            EDGE_COUNT=$(grep -o '"node_a_id"' "$DOWNLOAD_DIR/runs/$LATEST_RUN/traffic_edges.json" 2>/dev/null | wc -l || echo "0")
            echo "🚗 Traffic edges: $EDGE_COUNT"
        fi
        
        # Show statistics
        if [ -f "$DOWNLOAD_DIR/runs/$LATEST_RUN/statistics.json" ]; then
            echo ""
            echo "📊 Network Statistics:"
            python -m json.tool "$DOWNLOAD_DIR/runs/$LATEST_RUN/statistics.json" 2>/dev/null | grep -E "total_nodes|total_edges|avg_degree" || cat "$DOWNLOAD_DIR/runs/$LATEST_RUN/statistics.json"
        fi
        
        # File sizes
        echo ""
        echo "📄 Latest run files:"
        ls -lh "$DOWNLOAD_DIR/runs/$LATEST_RUN/" 2>/dev/null | tail -n +2 | awk '{print "  "$9" - "$5}'
    fi
fi

echo ""
echo "======================================================================"
echo "Next Steps:"
echo "  1. View collections:    python scripts/view_collections.py"
echo "  2. Quick summary:       python scripts/analysis/quick_summary.py"
echo "  3. Visualize:           python -m traffic_forecast.cli.visualize"
echo "======================================================================"
echo ""
