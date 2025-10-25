#!/bin/bash
# Download collected data from GCP VM
# Usage: ./scripts/download_data.sh [instance_name] [zone]

INSTANCE_NAME="${1:-traffic-collector-v4}"
ZONE="${2:-asia-southeast1-b}"
OUTPUT_DIR="${3:-./data_downloaded_$(date +%Y%m%d_%H%M%S)}"

echo "Downloading data from $INSTANCE_NAME..."
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Download data directory
echo "Downloading data files..."
gcloud compute scp --recurse $INSTANCE_NAME:~/dsp391m_project/data/ "$OUTPUT_DIR/data/" \
    --zone=$ZONE \
    --strict-host-key-checking=no

# Download database
echo "Downloading database..."
gcloud compute scp $INSTANCE_NAME:~/dsp391m_project/traffic_history.db "$OUTPUT_DIR/" \
    --zone=$ZONE \
    --strict-host-key-checking=no

# Download logs
echo "Downloading logs..."
gcloud compute scp --recurse $INSTANCE_NAME:~/dsp391m_project/logs/ "$OUTPUT_DIR/logs/" \
    --zone=$ZONE \
    --strict-host-key-checking=no

# Create summary
echo "Creating summary..."
cat > "$OUTPUT_DIR/README.md" << EOF
# Downloaded Traffic Data

**Downloaded:** $(date)
**Instance:** $INSTANCE_NAME
**Zone:** $ZONE

## Contents

- \`data/\` - Collected traffic data (nodes, edges, weather)
- \`traffic_history.db\` - SQLite database with historical snapshots
- \`logs/\` - Collection logs

## Data Files

\`\`\`
$(ls -lh "$OUTPUT_DIR/data/" | tail -20)
\`\`\`

## Database Size

\`\`\`
$(du -sh "$OUTPUT_DIR/traffic_history.db")
\`\`\`

## Collection Logs

\`\`\`
$(tail -50 "$OUTPUT_DIR/logs/collector.log" 2>/dev/null || echo "No logs found")
\`\`\`

## Next Steps

1. Analyze data with notebooks
2. Train models
3. Generate visualizations
4. Write results report

EOF

echo ""
echo "✓ Download complete!"
echo "  Location: $OUTPUT_DIR"
echo "  Summary: $OUTPUT_DIR/README.md"
echo ""
echo "Next steps:"
echo "  cd $OUTPUT_DIR"
echo "  cat README.md"
