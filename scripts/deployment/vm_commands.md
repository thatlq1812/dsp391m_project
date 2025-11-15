# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# VM Management Commands

Quick reference for managing demo data collection VM.

## Connection

```bash
# SSH into VM
gcloud compute ssh traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Short alias (add to ~/.bashrc locally)
alias vm-ssh='gcloud compute ssh traffic-demo-collector --zone=asia-southeast1-a --project=sonorous-nomad-476606-g3'
```

## Service Management

```bash
# Start collection timer
sudo systemctl start traffic-collector.timer

# Stop collection timer
sudo systemctl stop traffic-collector.timer

# Enable on boot
sudo systemctl enable traffic-collector.timer

# Check timer status
systemctl status traffic-collector.timer

# Check service status
systemctl status traffic-collector.service

# View timer list
systemctl list-timers --all
```

## Monitoring

```bash
# Watch live collection log
tail -f /opt/traffic_data/collector.log

# Check error log
tail -f /opt/traffic_data/collector_error.log

# Check disk usage
df -h /opt/traffic_data

# Count collected records
cd /opt/traffic_data
ls -lh traffic_data_*.parquet
```

## Manual Collection

```bash
# Run collector once (for testing)
cd ~/traffic-demo
conda activate dsp
python scripts/deployment/traffic_collector.py

# Check topology cache
ls -lh cache/overpass_topology.json
cat cache/overpass_topology.json | jq '.edges | length'
```

## Data Management

```bash
# Check data files
ls -lh /opt/traffic_data/

# View parquet file info
conda activate dsp
python -c "
import pandas as pd
df = pd.read_parquet('/opt/traffic_data/traffic_data_202511.parquet')
print(f'Records: {len(df):,}')
print(f'Time range: {df.timestamp.min()} to {df.timestamp.max()}')
print(f'Unique edges: {df.edge_id.nunique()}')
"

# Download data to local machine (run from LOCAL terminal)
gcloud compute scp \
    traffic-demo-collector:/opt/traffic_data/traffic_data_202511.parquet \
    ./data/demo/ \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3
```

## Debugging

```bash
# Check if conda environment exists
conda env list

# Check Python packages
conda activate dsp
pip list | grep -E "pandas|requests|pyarrow"

# Test Google API connection
conda activate dsp
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GOOGLE_MAPS_API_KEY')
print(f'Key exists: {bool(key)}')
print(f'Key length: {len(key) if key else 0}')
"

# Check system resources
free -h
top -bn1 | head -15
```

## VM Lifecycle

```bash
# Stop VM (when not collecting)
gcloud compute instances stop traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Start VM
gcloud compute instances start traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Delete VM (when done)
gcloud compute instances delete traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3 \
    --quiet

# Check VM status (from local)
gcloud compute instances list --project=sonorous-nomad-476606-g3
```

## Cost Monitoring

```bash
# Check billing (from local)
gcloud billing accounts list
gcloud beta billing projects describe sonorous-nomad-476606-g3

# Estimate current month cost
# Go to: https://console.cloud.google.com/billing
```

## Backup Data

```bash
# Create backup before deletion
cd /opt/traffic_data
tar -czf ~/traffic_data_backup_$(date +%Y%m%d).tar.gz *.parquet *.log

# Download backup (from local)
gcloud compute scp \
    traffic-demo-collector:~/traffic_data_backup_*.tar.gz \
    ./backups/ \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3
```

## Quick Health Check

```bash
# One-line health check
systemctl is-active traffic-collector.timer && \
tail -3 /opt/traffic_data/collector.log && \
df -h /opt/traffic_data && \
ls -lh /opt/traffic_data/traffic_data_*.parquet | tail -1

# Expected output:
# active
# [timestamp] INFO - ✓ Collection completed
# /opt/traffic_data  28G   500M  27G   2% /opt
# -rw-r--r-- 1 user user 12M Nov 15 14:30 traffic_data_202511.parquet
```

## Common Issues

### Issue: Timer not running

```bash
# Check timer status
systemctl status traffic-collector.timer

# Restart timer
sudo systemctl restart traffic-collector.timer

# Check logs
journalctl -u traffic-collector.service -n 50
```

### Issue: API key not found

```bash
# Check .env file
cat ~/traffic-demo/.env | grep GOOGLE_MAPS_API_KEY

# Edit .env
nano ~/traffic-demo/.env

# Test after edit
cd ~/traffic-demo
conda activate dsp
python scripts/deployment/traffic_collector.py
```

### Issue: Topology file missing

```bash
# Build topology
cd ~/traffic-demo
conda activate dsp
python scripts/data/01_collection/build_topology.py

# Verify
ls -lh cache/overpass_topology.json
```

### Issue: Disk full

```bash
# Check disk usage
df -h

# Remove old logs
rm /opt/traffic_data/collector.log
rm /opt/traffic_data/collector_error.log

# Compress old data
cd /opt/traffic_data
gzip traffic_data_202510.parquet
```

## Update Code

```bash
# Pull latest changes
cd ~/traffic-demo
git pull origin master

# Restart service
sudo systemctl restart traffic-collector.timer
```

## Author

THAT Le Quang  
November 2025
