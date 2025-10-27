#!/bin/bash
# Quick Reference - Cloud Deployment Commands
# Author: THAT Le Quang (Xiel)
# Date: October 25, 2025
set +H  # Disable history expansion to avoid "event ! not found" errors

cat << 'EOF'

 Traffic Forecast - Cloud Deployment Quick Ref 
 Academic v4.0 


DEPLOYMENT


1. Configure GCP Project:
 export GCP_PROJECT_ID="your-project-id"
 gcloud auth login
 gcloud config set project $GCP_PROJECT_ID

2. Deploy (Mock API - FREE):
 ./scripts/deploy_week_collection.sh

3. Deploy (Real API - ~$168/week):
 USE_REAL_API=true ./scripts/deploy_week_collection.sh

MONITORING


Quick Status:
 ./scripts/monitor_collection.sh

SSH to VM:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b

View Logs (Live):
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="tail -f ~/dsp391m_project/logs/collector.log"

Check Service:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="sudo systemctl status traffic-collector"

Count Collections:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="ls -d ~/dsp391m_project/data_runs/run_* | wc -l"

 DATA DOWNLOAD


Download All Data:
 ./scripts/download_data.sh

Download to Custom Directory:
 ./scripts/download_data.sh traffic-collector-v4 asia-southeast1-b ./my_data

Manual Download:
 gcloud compute scp --recurse \
 traffic-collector-v4:~/dsp391m_project/data/ \
 ./data_downloaded/ \
 --zone=asia-southeast1-b

 CONTROL


Stop Collection:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="sudo systemctl stop traffic-collector"

Start Collection:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="sudo systemctl start traffic-collector"

Restart Collection:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="sudo systemctl restart traffic-collector"

 COST MANAGEMENT


Stop VM (Stop Billing):
 gcloud compute instances stop traffic-collector-v4 \
 --zone=asia-southeast1-b

Start VM:
 gcloud compute instances start traffic-collector-v4 \
 --zone=asia-southeast1-b

Delete VM (Permanent):
 gcloud compute instances delete traffic-collector-v4 \
 --zone=asia-southeast1-b

Check Billing:
 gcloud billing projects describe $GCP_PROJECT_ID

TROUBLESHOOTING


View Error Logs:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="tail -50 ~/dsp391m_project/logs/collector.error.log"

Check Disk Space:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="df -h"

Manual Test Collection:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b
 cd ~/dsp391m_project
 conda activate dsp
 python scripts/collect_and_render.py --once --no-visualize

View Systemd Logs:
 gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="journalctl -u traffic-collector -n 100"

 EXPECTED METRICS


Collections per Day:
 - Peak hours (5h): 10 collections @ 30min
 - Off-peak (19h): 19 collections @ 60min
 - Weekend: ~15 collections @ 90min
 - Average: ~34 collections/day

Data Size (7 days):
 - Nodes/Edges: ~47 MB
 - Database: ~100 MB
 - Logs: ~2 MB
 - Total: ~150 MB

Cost Estimate:
 - Mock API: $12 (VM only)
 - Real API: $180 (VM + API)

 DOCUMENTATION


Full Guide:
 cat CLOUD_DEPLOY.md

Deployment Guide:
 cat DEPLOY.md

Quick Reference:
 cat doc/QUICKREF.md



Need help? Read CLOUD_DEPLOY.md or contact:
Email: fxlqthat@gmail.com
Phone: +84 33 863 6369

EOF
