# GCP Deployment Scripts

**Quick deployment and monitoring for 3-day data collection**

---

## Overview

Two interactive bash scripts for fully automated GCP deployment:

1. **`deploy_gcp_auto.sh`** - One-command deployment (10 mins)
2. **`monitor_gcp.sh`** - Interactive monitoring dashboard

---

## 1. Deploy to GCP VM

### Requirements

- **gcloud CLI** installed ([install guide](https://cloud.google.com/sdk/docs/install))
- **GCP Project** with billing enabled
- **Local test passed** (run `scripts/collect_once.py` first)

### Deploy (One Command)

```bash
bash scripts/deploy_gcp_auto.sh
```

### What it does

**Automated Steps:**

1. ✓ Check prerequisites (gcloud, files, auth)
2. ✓ Select GCP project (interactive)
3. ✓ Show cost estimate and confirm
4. ✓ Create/start VM (e2-micro in Singapore)
5. ✓ Upload project files (excludes docs/notebooks)
6. ✓ Install Miniconda + conda environment
7. ✓ Run test collection (verify it works)
8. ✓ Setup cron job (hourly collection)
9. ✓ Schedule auto-stop after 3 days

**Time:** ~10 minutes (mostly waiting for VM)

**Output:** VM running, collecting data every hour

---

## 2. Monitor Collection

### Interactive Menu

```bash
bash scripts/monitor_gcp.sh
```

### Menu Options

```
1) Show collection progress   - Total collections, success rate, ETA
2) View latest logs (live)    - Tail collection.log in real-time
3) View cron log              - Last 30 cron entries
4) Check VM status            - VM state, IP, uptime
5) Validate collected data    - Data quality check
6) Check disk usage           - Storage usage
7) Download collected data    - SCP data to local
8) Stop VM                    - Manual stop (saves money)
9) SSH into VM                - Direct terminal access
0) Exit
```

### Quick Commands (without menu)

**Check progress:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='grep -c "completed successfully" ~/traffic-forecast/logs/cron.log'
```

**View logs:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='tail -f ~/traffic-forecast/logs/collection.log'
```

**Download data:**

```bash
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data \
  ./data-collected --zone=asia-southeast1-a
```

**Stop VM:**

```bash
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a
```

---

## Timeline

### Day 0 (Today)

- Run `deploy_gcp_auto.sh`
- Verify first collection successful
- Check logs after 1 hour

### Days 1-2

- Monitor daily: `monitor_gcp.sh` → option 1
- Check for errors in cron log
- Validate data quality

### Day 3 (End)

- VM auto-stops automatically
- Download data: option 7 in monitor menu
- Delete VM to avoid charges

---

## Cost Breakdown

| Item                                        | Cost     |
| ------------------------------------------- | -------- |
| VM (e2-micro, 72 hours)                     | ~$2-3    |
| Storage (30GB, 3 days)                      | ~$0.30   |
| **Google API (54 collections × 234 edges)** | **~$60** |
| **Total**                                   | **~$63** |

**Note:** 95% of cost is Google Directions API. VM is almost free.

---

## Expected Results

**After 3 days:**

- ~54 collections (18/day adaptive schedule)
- ~12,636 traffic measurements (234 edges × 54)
- Data files: `traffic_edges.json`, `weather_snapshot.json`, `nodes.json`
- Logs: Collection logs, cron logs, error logs

**Success criteria:**

- ✓ 100% success rate (like local test)
- ✓ No missing hours
- ✓ Speed values 10-60 km/h range
- ✓ All 234 edges per collection

---

## Troubleshooting

### Deployment fails

**Error: gcloud not found**

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

**Error: Not authenticated**

```bash
gcloud auth login
gcloud auth application-default login
```

**Error: VM creation fails**

```bash
# Check quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Enable Compute Engine API
gcloud services enable compute.googleapis.com
```

### Collection fails

**Check logs on VM:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
tail -100 logs/collection.log
```

**Re-run test collection:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='cd ~/traffic-forecast && source ~/miniconda3/etc/profile.d/conda.sh && conda activate dsp && python scripts/collect_once.py'
```

**Check API key:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='cd ~/traffic-forecast && cat .env | grep GOOGLE_MAPS_API_KEY'
```

### VM won't stop

**Force stop:**

```bash
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a --force
```

**Check auto-stop cron:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='crontab -l | grep auto_stop'
```

---

## Manual Operations

### Change collection frequency

**Edit cron on VM:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# Edit crontab
crontab -e

# Examples:
# Every 30 min:  */30 * * * * ~/traffic-forecast/run_collection.sh
# Every hour:    0 * * * * ~/traffic-forecast/run_collection.sh
# Every 2 hours: 0 */2 * * * ~/traffic-forecast/run_collection.sh
```

### Extend collection beyond 3 days

**Remove auto-stop:**

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command='crontab -l | grep -v auto_stop.sh | crontab -'
```

### Download specific files only

**Just traffic data:**

```bash
gcloud compute scp traffic-forecast-collector:~/traffic-forecast/data/traffic_edges.json \
  ./traffic_edges.json --zone=asia-southeast1-a
```

**Just logs:**

```bash
gcloud compute scp --recurse traffic-forecast-collector:~/traffic-forecast/logs \
  ./vm-logs --zone=asia-southeast1-a
```

---

## Cleanup

### After data download

```bash
# Stop VM (can restart later)
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a

# Delete VM (permanent, frees all resources)
gcloud compute instances delete traffic-forecast-collector --zone=asia-southeast1-a
```

**Note:** Always download data BEFORE deleting VM!

---

## Advanced: Adaptive Scheduling

To enable v5.1 adaptive scheduling (18 collections/day instead of 24):

### Edit on VM

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast

# Edit run_collection.sh to use adaptive scheduler
nano run_collection.sh
```

**Replace with:**

```bash
#!/bin/bash
cd ~/traffic-forecast
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp

# Use adaptive scheduler instead of fixed cron
python -c "
from traffic_forecast.scheduler.adaptive import AdaptiveScheduler
scheduler = AdaptiveScheduler()
scheduler.run_once()
" >> logs/collection.log 2>&1

echo "[$(date)] Adaptive collection completed" >> logs/cron.log
```

**Update crontab to check every 15 min:**

```bash
crontab -e
# Change to: */15 * * * * ~/traffic-forecast/run_collection.sh
```

This will run collection only during peak hours (7-9, 17-19) and off-peak (10-16), skipping night (20-6).

---

## Support

**Issues:**

- Check logs first: `monitor_gcp.sh` → option 2
- Validate data: `monitor_gcp.sh` → option 5
- SSH for debugging: `monitor_gcp.sh` → option 9

**Contact:**

- Project repo: https://github.com/thatlq1812/dsp391m_project
- Documentation: `doc/` directory

---

**Last updated:** October 29, 2025  
**Version:** v5.1 (3-day collection)
