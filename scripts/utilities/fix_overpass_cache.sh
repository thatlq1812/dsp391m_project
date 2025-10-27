#!/bin/bash
# Fix Overpass cache bug on production VM
#
# This script:
# 1. Uploads the fixed cache_utils.py to VM
# 2. Clears corrupted cache files
# 3. Restarts the collection service
# 4. Runs backfill script to fix existing data
#
# Usage: bash scripts/fix_overpass_cache.sh
set +H  # Disable history expansion to avoid "event ! not found" errors

set -e

INSTANCE_NAME="traffic-collector-v4"
ZONE="asia-southeast1-b"
REMOTE_USER="thatlqse183256_fpt_edu_vn"
REMOTE_DIR="/home/${REMOTE_USER}/dsp391m_project"

echo "======================================================================"
echo "FIX OVERPASS CACHE BUG ON PRODUCTION VM"
echo "======================================================================"
echo ""

# Step 1: Upload fixed cache_utils.py
echo "Step 1: Uploading fixed cache_utils.py..."
gcloud compute scp \
 traffic_forecast/collectors/cache_utils.py \
 ${INSTANCE_NAME}:${REMOTE_DIR}/traffic_forecast/collectors/cache_utils.py \
 --zone=${ZONE} \
 --strict-host-key-checking=no

echo " Fixed cache_utils.py uploaded"
echo ""

# Step 2: Upload backfill script
echo "Step 2: Uploading backfill script..."
gcloud compute scp \
 scripts/backfill_overpass_data.py \
 ${INSTANCE_NAME}:${REMOTE_DIR}/scripts/backfill_overpass_data.py \
 --zone=${ZONE} \
 --strict-host-key-checking=no

echo " Backfill script uploaded"
echo ""

# Step 3: Clear corrupted cache
echo "Step 3: Clearing corrupted cache..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 echo 'Removing corrupted cache files...'
 rm -rf cache/*
 echo ' Cache cleared'
"

echo " Corrupted cache cleared"
echo ""

# Step 4: Restart collection service
echo "Step 4: Restarting collection service..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 sudo systemctl restart traffic-collector
 sleep 2
 sudo systemctl status traffic-collector --no-pager | head -15
"

echo " Service restarted"
echo ""

# Step 5: Run backfill script on VM
echo "Step 5: Backfilling existing data..."
echo "This will copy Overpass data to all collections that are missing it."
echo ""
read -p "Run backfill now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
 echo "Running backfill (dry-run first)..."
 gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 source ~/miniconda3/bin/activate dsp
 python scripts/backfill_overpass_data.py --dry-run
 "
 
 echo ""
 read -p "Looks good? Apply changes? (y/n) " -n 1 -r
 echo ""
 
 if [[ $REPLY =~ ^[Yy]$ ]]; then
 echo "Applying backfill..."
 gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
 cd ${REMOTE_DIR}
 source ~/miniconda3/bin/activate dsp
 python scripts/backfill_overpass_data.py
 "
 echo " Backfill complete"
 else
 echo "Skipped backfill. You can run it manually later:"
 echo " gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
 echo " cd ${REMOTE_DIR}"
 echo " conda activate dsp"
 echo " python scripts/backfill_overpass_data.py"
 fi
else
 echo "Skipped backfill. You can run it manually later:"
 echo " gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
 echo " cd ${REMOTE_DIR}"
 echo " conda activate dsp"
 echo " python scripts/backfill_overpass_data.py"
fi

echo ""
echo "======================================================================"
echo "FIX DEPLOYMENT COMPLETE"
echo "======================================================================"
echo ""
echo "Summary:"
echo " Fixed cache_utils.py uploaded"
echo " Backfill script uploaded"
echo " Corrupted cache cleared"
echo " Collection service restarted"
echo ""
echo "Next steps:"
echo " 1. Monitor next collection: bash scripts/monitor_collection.sh"
echo " 2. Verify Overpass data in new collections"
echo " 3. Download updated data: bash scripts/download_data.sh"
echo ""
