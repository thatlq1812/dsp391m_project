# Operations Guide - Traffic Forecast v5.1

Daily operations, monitoring, and maintenance guide.

## 📊 Daily Operations

### Check Collection Status

```bash
# View service status
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl status traffic-collection.service"

# View recent logs (last 50 lines)
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -50 ~/traffic-forecast/logs/service.log"

# Count collections
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls ~/traffic-forecast/data/runs/ | wc -l"
```

### View Live Logs

```bash
# Follow logs in real-time
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -f ~/traffic-forecast/logs/service.log"

# Follow error logs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -f ~/traffic-forecast/logs/service_error.log"
```

### List Recent Collections

```bash
# List last 10 collections
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls -ltr ~/traffic-forecast/data/runs/ | tail -10"

# View latest collection files
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    latest=\$(ls -t ~/traffic-forecast/data/runs/ | head -1)
    ls -lh ~/traffic-forecast/data/runs/\$latest/
  "
```

## 🔄 Service Management

### Restart Service

```bash
# Restart collection service
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl restart traffic-collection.service"

# Check if restart successful
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl status traffic-collection.service"
```

### Stop Service

```bash
# Stop collection (temporary)
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl stop traffic-collection.service"
```

### Start Service

```bash
# Start collection
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl start traffic-collection.service"
```

### Disable Service (Prevent Auto-start)

```bash
# Disable service
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl disable traffic-collection.service"

# Enable service
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl enable traffic-collection.service"
```

## 💾 Data Management

### Download Data

```bash
# Download all collections
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs \
  ./data-backup-$(date +%Y%m%d) \
  --zone=asia-southeast1-a

# Download specific run
RUN_ID="run_20251029_150000"
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs/$RUN_ID \
  ./$RUN_ID \
  --zone=asia-southeast1-a
```

### Cleanup Old Data

```bash
# Remove collections older than 14 days
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/cleanup_runs.py --days 14
  "

# Remove specific run
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="rm -rf ~/traffic-forecast/data/runs/run_YYYYMMDD_HHMMSS"
```

### Check Disk Usage

```bash
# Check overall disk usage
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="df -h /"

# Check data directory size
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="du -sh ~/traffic-forecast/data/runs"

# Check size per run
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="du -sh ~/traffic-forecast/data/runs/*"
```

## 🔍 Monitoring & Debugging

### View Collection Summary

```bash
# Using view_collections.py script
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/view_collections.py
  "
```

### Test Single Collection

```bash
# Run manual collection test
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/collect_once.py
  "
```

### Check API Status

```bash
# Test Google Directions API
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python tools/test_google_limited.py
  "

# Test Weather API
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python -c 'from traffic_forecast.collectors.weather_collector import WeatherCollector; import asyncio; asyncio.run(WeatherCollector(None).collect())'
  "
```

### View System Resources

```bash
# CPU and memory usage
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="top -bn1 | head -20"

# Service resource usage
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="systemctl status traffic-collection.service -l"
```

## 💰 Cost Monitoring

### Check Current Costs

```bash
# View project costs (via web console)
# https://console.cloud.google.com/billing

# Estimate API usage
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    collections=\$(ls ~/traffic-forecast/data/runs/ | wc -l)
    api_calls=\$((collections * 234))
    cost=\$(echo \"scale=2; \$api_calls * 0.005\" | bc)
    echo \"Collections: \$collections\"
    echo \"API calls: \$api_calls\"
    echo \"Estimated cost: \$\$cost\"
  "
```

### Set Budget Alert

1. Go to [GCP Billing Budgets](https://console.cloud.google.com/billing/budgets)
2. Create Budget & Alerts
3. Set amount: $50 (for 3 days) or $200 (for 7 days)
4. Set email notifications at 50%, 90%, 100%

## 📈 Performance Optimization

### Verify Adaptive Scheduling

```bash
# Check scheduler logs for interval patterns
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="grep 'interval' ~/traffic-forecast/logs/service.log | tail -20"

# View schedule mode distribution
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="grep -E '(peak|offpeak|night)' ~/traffic-forecast/logs/service.log | tail -30"
```

### Check Cache Status

```bash
# Verify cache files exist
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls -lh ~/traffic-forecast/cache/"

# Check topology cache
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python tools/check_edges.py
  "
```

## 🚨 Troubleshooting

### Service Keeps Restarting

```bash
# View recent errors
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -100 ~/traffic-forecast/logs/service_error.log"

# Check service journal
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo journalctl -u traffic-collection.service -n 50"

# Common fixes:
# 1. Verify .env file exists and has API key
# 2. Check cache files are present
# 3. Verify conda environment is activated
# 4. Check disk space
```

### Collections Failing

```bash
# Check last collection error
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="grep ERROR ~/traffic-forecast/logs/service.log | tail -20"

# Test collection manually
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/collect_once.py 2>&1
  "
```

### API Rate Limiting

```bash
# Check for rate limit errors
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="grep -i 'rate' ~/traffic-forecast/logs/service_error.log"

# If rate limited:
# 1. Increase intervals in project_config.yaml
# 2. Restart service after updating config
```

### Disk Full

```bash
# Clean up old data
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/cleanup_runs.py --days 7
  "

# Clear pip cache
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="pip cache purge"

# Clear conda cache
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="conda clean --all -y"
```

## 🔄 Update Code on VM

```bash
# Create new archive locally
cd /path/to/project
tar -czf traffic-forecast-update.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='data/runs/*' \
  traffic_forecast/ scripts/ configs/

# Upload to VM
gcloud compute scp traffic-forecast-update.tar.gz \
  traffic-forecast-collector:~/ \
  --zone=asia-southeast1-a

# Extract and restart
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    tar -xzf ~/traffic-forecast-update.tar.gz
    rm ~/traffic-forecast-update.tar.gz
    sudo systemctl restart traffic-collection.service
  "
```

## 📊 Data Analysis (Local)

```bash
# After downloading data, use control panel
bash scripts/control_panel.sh

# Options:
#   5) View Collections - Summary of all runs
#   6) Merge Collections - Combine into single file
#   9) Visualize Latest Run - Generate maps
#   8) Export Latest Data - Create ZIP archive
```

## 🛡️ Security

### Rotate API Key

```bash
# 1. Generate new key in Google Cloud Console
# 2. Update .env on VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# Edit .env
nano ~/traffic-forecast/.env
# Update GOOGLE_MAPS_API_KEY

# 3. Restart service
sudo systemctl restart traffic-collection.service
```

### SSH Key Management

```bash
# List SSH keys
gcloud compute os-login ssh-keys list

# Remove SSH key
gcloud compute os-login ssh-keys remove --key=KEY_ID
```

## 📝 Maintenance Checklist

### Daily
- [ ] Check service status
- [ ] View recent logs
- [ ] Verify collections are running
- [ ] Monitor costs

### Weekly
- [ ] Download collected data
- [ ] Review error logs
- [ ] Check disk usage
- [ ] Clean old runs (>14 days)

### End of Collection Period
- [ ] Download all data
- [ ] Verify data integrity
- [ ] Stop or delete VM
- [ ] Review total costs
- [ ] Archive results

---

**Traffic Forecast v5.1** - Operations Guide
