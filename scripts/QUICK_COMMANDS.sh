#!/bin/bash
# Quick Reference - Copy/paste these commands

# ====================================
# DEPLOYMENT (ONE COMMAND)
# ====================================

bash scripts/deploy_gcp_auto.sh


# ====================================
# MONITORING (INTERACTIVE MENU)
# ====================================

bash scripts/monitor_gcp.sh


# ====================================
# QUICK CHECKS (NO MENU)
# ====================================

# How many collections so far?
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='grep -c "completed successfully" ~/traffic-forecast/logs/cron.log'

# Live logs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='tail -f ~/traffic-forecast/logs/collection.log'

# Last 10 cron entries
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='tail -10 ~/traffic-forecast/logs/cron.log'

# VM status
gcloud compute instances describe traffic-forecast-collector --zone=asia-southeast1-a \
  --format='value(status)'


# ====================================
# DATA DOWNLOAD
# ====================================

# Download all data
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data \
  ./data-collected-$(date +%Y%m%d) --zone=asia-southeast1-a

# Download logs only
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/logs \
  ./vm-logs-$(date +%Y%m%d) --zone=asia-southeast1-a


# ====================================
# VM CONTROL
# ====================================

# Stop VM (save money, can restart)
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a

# Start VM again
gcloud compute instances start traffic-forecast-collector --zone=asia-southeast1-a

# Delete VM (PERMANENT! Download data first!)
gcloud compute instances delete traffic-forecast-collector --zone=asia-southeast1-a


# ====================================
# SSH ACCESS
# ====================================

# SSH into VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# Once inside VM:
cd ~/traffic-forecast
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp

# Run collection manually
python scripts/collect_once.py

# Check crontab
crontab -l

# View logs
tail -f logs/collection.log


# ====================================
# TROUBLESHOOTING
# ====================================

# Re-run test collection on VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='cd ~/traffic-forecast && source ~/miniconda3/etc/profile.d/conda.sh && conda activate dsp && python scripts/collect_once.py'

# Check if API key is there
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='grep GOOGLE_MAPS_API_KEY ~/traffic-forecast/.env'

# Check disk space
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='df -h /'

# Check conda environment
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='source ~/miniconda3/etc/profile.d/conda.sh && conda env list'


# ====================================
# COST MONITORING
# ====================================

# View billing (opens browser)
gcloud alpha billing accounts list
# Then go to: https://console.cloud.google.com/billing


# ====================================
# EMERGENCY STOP
# ====================================

# If something goes wrong, STOP VM IMMEDIATELY
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a --force
