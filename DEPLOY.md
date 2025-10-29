# 🚀 DEPLOY TO GCP - QUICK START

## One-Command Deployment

```bash
bash scripts/deploy_gcp_auto.sh
```

**What happens:**

1. Checks your gcloud setup
2. Lists your GCP projects → you select one
3. Shows cost estimate (~$147 for 7 days with adaptive scheduling)
4. Asks "yes/no" to confirm
5. Creates VM, uploads code, installs environment
6. Runs test collection to verify
7. Sets up hourly cron job
8. Schedules auto-stop after 3 days

**Time:** ~10 minutes  
**Output:** VM collecting data with adaptive scheduling (peak/off-peak/night modes) for 7 days

---

## Monitor Progress

```bash
bash scripts/monitor_gcp.sh
```

**Interactive menu:**

- Option 1: Progress (how many collections done)
- Option 2: Live logs
- Option 7: Download data
- Option 8: Stop VM

---

## After 7 Days (or anytime)

**Download data:**

```bash
bash scripts/monitor_gcp.sh
# Select option 7
```

**Stop VM (auto-stopped anyway):**

```bash
gcloud compute instances stop traffic-forecast-collector --zone=asia-southeast1-a
```

**Delete VM (after download!):**

```bash
gcloud compute instances delete traffic-forecast-collector --zone=asia-southeast1-a
```

---

## Expected Results

- **Collections:** ~800+ over 7 days (adaptive scheduling: peak 5min, off-peak 15min, night 30min)
- **Data points:** ~187,200 (234 edges × 800)
- **Files:** traffic_edges.json, weather_snapshot.json, nodes.json
- **Cost:** ~$147 total (25% savings with v5.1 optimizations)

---

## Prerequisites

✓ gcloud CLI installed  
✓ GCP project with billing  
✓ Local test passed (`scripts/collect_once.py` works)

---

## Full Documentation

- **Detailed guide:** `scripts/GCP_DEPLOYMENT_README.md`
- **Quick commands:** `scripts/QUICK_COMMANDS.sh`

---

**Ready?** → `bash scripts/deploy_gcp_auto.sh`
