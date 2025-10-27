#!/bin/bash
# Download and compress data from GCP VM
#
# This script creates a compressed archive on the VM first,
# then downloads the single archive file instead of many small files.
# Much faster and more reliable than recursive scp.
#
# Usage: bash scripts/download_data_compressed.sh [output_dir] [format]
#
# Arguments:
# output_dir - Local directory to save download (default: data/downloads/)
# format - Archive format: tar.gz or zip (default: tar.gz)
set +H  # Disable history expansion to avoid "event ! not found" errors

set -e

INSTANCE_NAME="${1:-traffic-collector-v4}"
ZONE="${2:-asia-southeast1-b}"
REMOTE_USER="thatlqse183256_fpt_edu_vn"
REMOTE_DIR="/home/${REMOTE_USER}/dsp391m_project"
OUTPUT_DIR="${3:-./data/downloads/download_$(date +%Y%m%d_%H%M%S)}"
FORMAT="${4:-tar.gz}"

echo "======================================================================"
echo "DOWNLOAD COMPRESSED DATA FROM GCP VM"
echo "======================================================================"
echo ""
echo "Instance: ${INSTANCE_NAME}"
echo "Zone: ${ZONE}"
echo "Format: ${FORMAT}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Create compressed archive on VM
echo "Step 1: Creating compressed archive on VM..."
echo "This may take a few minutes for large datasets..."
echo ""

if [ "${FORMAT}" = "zip" ]; then
 ARCHIVE_NAME="data_export_$(date +%Y%m%d_%H%M%S).zip"
 gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 echo 'Creating ZIP archive...'
 zip -r -q ${ARCHIVE_NAME} data/ logs/ *.json 2>/dev/null || true
 echo 'Archive created: ${ARCHIVE_NAME}'
 ls -lh ${ARCHIVE_NAME}
 "
else
 ARCHIVE_NAME="data_export_$(date +%Y%m%d_%H%M%S).tar.gz"
 gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 echo 'Creating TAR.GZ archive...'
 tar -czf ${ARCHIVE_NAME} data/ logs/ *.json 2>/dev/null || true
 echo 'Archive created: ${ARCHIVE_NAME}'
 ls -lh ${ARCHIVE_NAME}
 "
fi

echo ""
echo "[OK] Archive created on VM"
echo ""

# Step 2: Download the archive
echo "Step 2: Downloading archive..."
gcloud compute scp \
 ${INSTANCE_NAME}:${REMOTE_DIR}/${ARCHIVE_NAME} \
 "${OUTPUT_DIR}/${ARCHIVE_NAME}" \
 --zone=${ZONE} \
 --strict-host-key-checking=no

echo ""
echo "[OK] Archive downloaded"
echo ""

# Step 3: Extract archive locally
echo "Step 3: Extracting archive..."
cd "${OUTPUT_DIR}"

if [ "${FORMAT}" = "zip" ]; then
 unzip -q "${ARCHIVE_NAME}"
else
 tar -xzf "${ARCHIVE_NAME}"
fi

echo "[OK] Archive extracted"
echo ""

# Step 4: Cleanup archive on VM
echo "Step 4: Cleaning up archive on VM..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 rm -f ${ARCHIVE_NAME}
 echo 'Archive removed from VM'
"

echo "[OK] VM cleanup complete"
echo ""

# Step 5: Generate summary
echo "Step 5: Generating summary..."

cat > README.md << 'EOF'
# Downloaded Traffic Data

**Downloaded at:** $(date)
**Source instance:** traffic-collector-v4
**Zone:** asia-southeast1-b
**Archive format:** ${FORMAT}

## Contents

- `data/` - Collected traffic data (nodes, edges, weather)
- `logs/` - Collection logs
- Root JSON files - Global nodes.json, edges.json if present

## Directory Structure

```
data/
 node/ # Collection runs by timestamp
 YYYYMMDDHHMMSS/
 collectors/
 overpass/ # Road network topology
 google/ # Traffic conditions 
 open_meteo/ # Weather data
 manifest.json
 ...
 archive/ # Archived data (if any)
 images/ # Generated visualizations (if any)

logs/
 collector.log # Collection service logs
```

## Quick Analysis

```bash
# Count collections
ls -1 data/node/ | wc -l

# Find collections with complete Overpass data
find data/node -name "nodes.json" | wc -l

# Check for errors in logs
grep -i error logs/collector.log | tail -20

# View most recent collection
ls -t data/node/ | head -1
```

## Data Statistics

Run this to get data statistics:

```python
import json
from pathlib import Path
from datetime import datetime

# Count collections and file types
collections = list(Path('data/node').glob('*'))
print(f"Total collections: {len(collections)}")

# Check data completeness
complete = 0
for coll in collections:
 if (coll / 'collectors/overpass/nodes.json').exists():
 complete += 1

print(f"Complete (with Overpass): {complete}/{len(collections)}")
print(f"Completeness: {complete/len(collections)*100:.1f}%")

# Timeline
if collections:
 timestamps = sorted([c.name for c in collections])
 start = datetime.strptime(timestamps[0], '%Y%m%d%H%M%S')
 end = datetime.strptime(timestamps[-1], '%Y%m%d%H%M%S')
 duration = end - start
 print(f"\nCollection period:")
 print(f" Start: {start}")
 print(f" End: {end}")
 print(f" Duration: {duration}")
```

## Next Steps

1. Review logs for collection errors
2. Validate data completeness
3. Run analysis notebooks
4. Train models with collected data

EOF

echo "[OK] Summary created"
echo ""

# Step 6: Local cleanup (remove archive, keep extracted data)
echo "Step 6: Cleaning up local archive..."
rm -f "${ARCHIVE_NAME}"
echo "[OK] Local archive removed"
echo ""

echo "======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================"
echo ""

# Get absolute path for display
ABS_OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

ls -lh "${ABS_OUTPUT_DIR}" | head -20
echo ""
echo "Location: ${ABS_OUTPUT_DIR}"
echo "Summary: ${ABS_OUTPUT_DIR}/README.md"
echo ""
echo "Collection count: $(ls -1 ${ABS_OUTPUT_DIR}/data/node 2>/dev/null | wc -l)"
echo "Total size: $(du -sh ${ABS_OUTPUT_DIR} 2>/dev/null | cut -f1)"
echo ""
