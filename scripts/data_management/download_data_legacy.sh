#!/bin/bash
# Download collected data from GCP VM
#
# DEPRECATED: This script uses slow recursive scp.
# Please use download_data_compressed.sh instead for faster downloads.
#
# Usage: ./scripts/download_data.sh [instance_name] [zone]
set +H  # Disable history expansion to avoid "event ! not found" errors

echo "======================================================================"
echo "NOTICE: This script is deprecated"
echo "======================================================================"
echo ""
echo "For faster and more reliable downloads, use:"
echo " bash scripts/download_data_compressed.sh"
echo ""
echo "Benefits of compressed download:"
echo " - 5-10x faster (single file vs thousands of small files)"
echo " - More reliable (no partial failures)"
echo " - Resume support (can retry failed downloads)"
echo " - Automatic cleanup"
echo ""
read -p "Continue with old method anyway? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
 echo "Redirecting to compressed download..."
 exec bash scripts/download_data_compressed.sh "$@"
fi
echo ""

INSTANCE_NAME="${1:-traffic-collector-v4}"
ZONE="${2:-asia-southeast1-b}"
OUTPUT_DIR="${3:-./data/downloads/download_$(date +%Y%m%d_%H%M%S)}"

echo "Downloading data from $INSTANCE_NAME..."
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Download data directory
echo "Downloading data files..."
gcloud compute scp --recurse $INSTANCE_NAME:/home/thatlqse183256_fpt_edu_vn/dsp391m_project/data/ "$OUTPUT_DIR/data/" \
 --zone=$ZONE \
 --strict-host-key-checking=no

# Download database
echo "Downloading database..."
gcloud compute scp $INSTANCE_NAME:/home/thatlqse183256_fpt_edu_vn/dsp391m_project/traffic_history.db "$OUTPUT_DIR/" \
 --zone=$ZONE \
 --strict-host-key-checking=no 2>/dev/null || echo " (Database file not found - using file storage)"

# Download logs
echo "Downloading logs..."
gcloud compute scp --recurse $INSTANCE_NAME:/home/thatlqse183256_fpt_edu_vn/dsp391m_project/logs/ "$OUTPUT_DIR/logs/" \
 --zone=$ZONE \
 --strict-host-key-checking=no

# Create summary
echo "Creating summary..."
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Downloaded Traffic Data

**Downloaded at:** $(date)
**Source instance:** traffic-collector-v4
**Zone:** asia-southeast1-b

## Contents

- `data/` - Collected traffic data (nodes, edges, weather)
- `traffic_history.db` - SQLite database with historical snapshots (if exists)
- `logs/` - Collection logs

## Directory Structure

Run the following to inspect the downloaded data:

```bash
# View data files
ls -lh data/

# View recent runs
ls -lht data/ | head -20

# Check log files
ls -lh logs/

# Search for errors in logs
grep -i error logs/collector.log
```

## Next Steps

1. Review logs for collection errors
2. Validate data completeness
3. Check nodes.json and edges.json integrity
4. Analyze collection timing and intervals
5. Use notebooks for data exploration

EOF

echo ""
echo " Download complete!"
echo " Location: $OUTPUT_DIR"
echo " Summary: $OUTPUT_DIR/README.md"
echo ""
echo "Next steps:"
echo " cd $OUTPUT_DIR"
echo " cat README.md"
