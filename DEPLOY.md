# Deployment Guide - Traffic Forecast System

Complete guide for deploying the Traffic Forecast system on Google Cloud VM or any Linux server.

**Version**: Academic v4.0  
**Last Updated**: October 25, 2025  
**Estimated Time**: 30-45 minutes

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Deployment](#quick-deployment)
3. [Manual Deployment](#manual-deployment)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Team Access Setup](#team-access-setup)

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ or Debian 11+
- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB minimum, 50GB recommended
- **Network**: Stable internet connection

### Required Accounts

- Google Cloud account (for GCP deployment)
- Google Maps API key (optional - can use mock API)
- Git access to this repository

### Local Requirements (for deployment)

```bash
# SSH client
ssh -V

# Git
git --version
```

---

## Quick Deployment

### One-Command Setup (Recommended)

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/thatlq1812/dsp391m_project/master/scripts/gcp_setup.sh | bash
```

This will:

1. Install all dependencies
2. Setup conda environment
3. Clone repository
4. Configure system
5. Start data collection

**Then proceed to** [Team Access Setup](#team-access-setup)

---

## Manual Deployment

### Step 1: Create GCP VM

#### Using Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Compute Engine** > **VM Instances**
3. Click **CREATE INSTANCE**

**Configuration**:

```
Name: traffic-forecast-vm
Region: asia-southeast1 (Singapore) or asia-east1 (Taiwan)
Zone: any available zone
Machine type: e2-medium (2 vCPU, 4GB RAM)
Boot disk: Ubuntu 22.04 LTS, 50GB SSD
Firewall: Allow HTTP/HTTPS traffic
```

4. Click **CREATE**

#### Using gcloud CLI

```bash
gcloud compute instances create traffic-forecast-vm \
  --zone=asia-southeast1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --tags=http-server,https-server
```

### Step 2: Connect to VM

```bash
# Via gcloud
gcloud compute ssh traffic-forecast-vm --zone=asia-southeast1-a

# Via SSH (if you have external IP)
ssh username@EXTERNAL_IP
```

### Step 3: System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
  git \
  wget \
  curl \
  vim \
  htop \
  tmux \
  build-essential

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Step 4: Clone Repository

```bash
# Clone project
cd ~
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project

# Verify files
ls -la
```

### Step 5: Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate dsp

# Verify installation
python --version
python -c "import yaml; import pandas; print('Dependencies OK')"
```

### Step 6: Configuration

```bash
# Copy environment template
cp .env_template .env

# Edit configuration (optional - can use mock API)
nano .env
```

**For Production** (with Google API):

```bash
# .env
GOOGLE_MAPS_API_KEY=your_api_key_here
```

**For Development** (FREE - Mock API):

```bash
# No API key needed - uses mock API by default
# Check configs/project_config.yaml:
#   google_directions:
#     use_mock_api: true
```

### Step 7: Verify Setup

```bash
# Test collection
python scripts/collect_and_render.py --once --no-visualize

# Check schedule
python scripts/collect_and_render.py --print-schedule

# Expected output:
# Peak Hours: 4 ranges
# Collections/day: 25
# Monthly cost: $720 (or $0 if using mock API)
```

---

## Configuration

### Basic Configuration

Edit `configs/project_config.yaml`:

```yaml
# Project info
project:
  name: traffic-forecast-academic
  version: v4.0
  timezone: Asia/Ho_Chi_Minh

# Scheduler (adaptive mode)
scheduler:
  enabled: true
  mode: adaptive # or 'fixed'

# Node selection
node_selection:
  max_nodes: 64
  min_degree: 6
  min_importance_score: 40.0

# Google API
google_directions:
  use_mock_api: true # Set false for production
  limit_nodes: 64
  k_neighbors: 3
```

### Advanced Configuration

**Change collection area**:

```yaml
globals:
  area:
    center: [106.697794, 10.772465] # [lon, lat]
    radius_m: 1024 # meters
```

**Adjust peak hours**:

```yaml
scheduler:
  adaptive:
    peak_hours:
      time_ranges:
        - start: "06:30"
          end: "07:30"
        # Add more ranges as needed
```

**Storage settings**:

```yaml
data:
  parquet_dir: ./data/parquet
  json_dir: ./data
  cache_dir: ./cache
```

---

## Running the System

### Option 1: Adaptive Scheduling (Recommended)

```bash
# Start in tmux session (persists after disconnect)
tmux new -s traffic-collect

# Run adaptive scheduler
python scripts/collect_and_render.py --adaptive

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t traffic-collect
```

### Option 2: Systemd Service (Production)

Create service file:

```bash
sudo nano /etc/systemd/system/traffic-forecast.service
```

Content:

```ini
[Unit]
Description=Traffic Forecast Data Collection
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/dsp391m_project
Environment="PATH=/home/YOUR_USERNAME/miniconda3/envs/dsp/bin:/usr/bin"
ExecStart=/home/YOUR_USERNAME/miniconda3/envs/dsp/bin/python scripts/collect_and_render.py --adaptive
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable traffic-forecast.service
sudo systemctl start traffic-forecast.service

# Check status
sudo systemctl status traffic-forecast.service

# View logs
sudo journalctl -u traffic-forecast.service -f
```

### Option 3: Cron Job (Simple)

```bash
# Edit crontab
crontab -e

# Add entry for hourly collection
0 * * * * cd /home/YOUR_USERNAME/dsp391m_project && /home/YOUR_USERNAME/miniconda3/envs/dsp/bin/python scripts/collect_and_render.py --once >> /tmp/traffic-collect.log 2>&1
```

### Option 4: With History Storage

```bash
# Use history-enabled script for lag features
python scripts/collect_with_history.py --adaptive
```

---

## Monitoring

### System Monitoring

```bash
# CPU/Memory usage
htop

# Disk usage
df -h

# Check running processes
ps aux | grep python

# Network activity
netstat -tuln
```

### Application Monitoring

```bash
# View logs
tail -f logs/collector.log

# Check database size
du -sh traffic_history.db

# Check data directory size
du -sh data/

# View recent collections
ls -lt data/node/ | head -10

# Check schedule status
python scripts/collect_and_render.py --print-schedule
```

### Cost Monitoring

```bash
# Estimate current configuration cost
python -c "
from traffic_forecast.scheduler import AdaptiveScheduler
import yaml

with open('configs/project_config.yaml') as f:
    config = yaml.safe_load(f)

scheduler = AdaptiveScheduler(config['scheduler'])
cost = scheduler.get_cost_estimate(nodes=64, k_neighbors=3, days=30)

print(f\"Collections/day: {cost['collections_per_day']}\")
print(f\"Monthly cost: ${cost['total_cost_usd']:.2f}\")
"
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Problem: ModuleNotFoundError
# Solution: Verify conda environment
conda activate dsp
conda list | grep pydantic

# Reinstall if needed
pip install -r requirements.txt
```

#### 2. Permission Errors

```bash
# Problem: Permission denied
# Solution: Fix file permissions
chmod +x scripts/*.sh
chmod 755 scripts/*.py
```

#### 3. API Errors

```bash
# Problem: Google API errors
# Solution: Check API key or use mock API

# Edit config
nano configs/project_config.yaml
# Set: use_mock_api: true
```

#### 4. Database Locked

```bash
# Problem: SQLite database locked
# Solution: Stop other processes
pkill -f collect_and_render.py

# Or remove lock file
rm traffic_history.db-journal
```

#### 5. Memory Issues

```bash
# Check memory usage
free -h

# If low, reduce nodes
nano configs/project_config.yaml
# Set: max_nodes: 32
```

### Logs Location

```
logs/collector.log       - Collection logs
logs/scheduler.log       - Scheduler logs
/tmp/traffic-*.log       - Cron job logs
journalctl               - Systemd service logs
```

### Getting Help

```bash
# Check documentation
cat doc/reference/ACADEMIC_V4_SUMMARY.md

# Run help command
python scripts/collect_and_render.py --help

# Check system info
python -c "import sys; print(sys.version)"
conda list
```

---

## Team Access Setup

### Creating Team Accounts

#### 1. Create User Account

```bash
# On VM, create user for each team member
sudo adduser teammate1
sudo adduser teammate2

# Add to necessary groups
sudo usermod -aG sudo teammate1  # If need sudo access
```

#### 2. Setup SSH Access

**Option A: Password Authentication** (Simple but less secure)

```bash
# Enable password auth
sudo nano /etc/ssh/sshd_config

# Set:
# PasswordAuthentication yes

# Restart SSH
sudo systemctl restart sshd

# Share credentials with team:
# Username: teammate1
# Password: [their password]
# Server: EXTERNAL_IP or hostname
```

**Option B: SSH Key Authentication** (Recommended)

```bash
# Team member generates key on their machine
ssh-keygen -t rsa -b 4096 -C "teammate1@email.com"

# Team member shares public key (id_rsa.pub content)

# On VM, add to authorized_keys
sudo su - teammate1
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
# Paste public key
chmod 600 ~/.ssh/authorized_keys
exit

# Team member connects
ssh teammate1@EXTERNAL_IP
```

#### 3. Setup Conda Environment for Team

```bash
# As each user
conda env create -f /home/ORIGINAL_USER/dsp391m_project/environment.yml
conda activate dsp

# Or share the environment
sudo ln -s /home/ORIGINAL_USER/miniconda3 /opt/miniconda3
echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> ~/.bashrc
```

#### 4. Grant Project Access

```bash
# Make project readable by team
sudo chmod -R 755 /home/ORIGINAL_USER/dsp391m_project

# Or create shared project directory
sudo mkdir /opt/traffic-forecast
sudo cp -r /home/ORIGINAL_USER/dsp391m_project/* /opt/traffic-forecast/
sudo chown -R :team /opt/traffic-forecast
sudo chmod -R 775 /opt/traffic-forecast
```

### Access Control

#### Read-Only Access

```bash
# Create read-only group
sudo groupadd traffic-readonly
sudo usermod -aG traffic-readonly teammate1

# Set permissions
sudo chgrp -R traffic-readonly /opt/traffic-forecast
sudo chmod -R 755 /opt/traffic-forecast
```

#### Full Access

```bash
# Create full-access group
sudo groupadd traffic-admin
sudo usermod -aG traffic-admin teammate2

# Set permissions
sudo chgrp -R traffic-admin /opt/traffic-forecast
sudo chmod -R 775 /opt/traffic-forecast
```

### Team Workflow

#### Connection Instructions for Team

Share this with team members:

```bash
# Connect to VM
ssh YOUR_USERNAME@EXTERNAL_IP

# Or via gcloud
gcloud compute ssh traffic-forecast-vm --zone=asia-southeast1-a

# Activate environment
conda activate dsp

# Navigate to project
cd /opt/traffic-forecast  # or ~/dsp391m_project

# Check status
python scripts/collect_and_render.py --print-schedule

# View logs
tail -f logs/collector.log

# Run collection
python scripts/collect_and_render.py --once
```

#### Shared tmux Sessions

```bash
# Create shared session
tmux -S /tmp/traffic new -s shared

# Set permissions
chmod 777 /tmp/traffic

# Other users join
tmux -S /tmp/traffic attach -t shared
```

---

## Production Checklist

Before going live:

- [ ] VM created and accessible
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] Test collection successful
- [ ] Systemd service configured
- [ ] Monitoring setup
- [ ] Team access configured
- [ ] Backup strategy defined
- [ ] Documentation reviewed
- [ ] Cost monitoring enabled

---

## Maintenance

### Daily Tasks

```bash
# Check system status
systemctl status traffic-forecast.service

# Monitor disk usage
df -h

# Check logs for errors
grep -i error logs/collector.log
```

### Weekly Tasks

```bash
# Review collection stats
python scripts/collect_with_history.py --stats

# Clean old data
bash scripts/cleanup.sh

# Update dependencies
conda update --all
```

### Monthly Tasks

```bash
# Review costs
# Check Google Cloud billing

# Update system
sudo apt update && sudo apt upgrade -y

# Backup database
cp traffic_history.db backups/traffic_history_$(date +%Y%m%d).db
```

---

## Security Best Practices

1. **Firewall**: Only open necessary ports
2. **SSH**: Use key-based authentication
3. **Updates**: Keep system updated
4. **Passwords**: Use strong passwords
5. **API Keys**: Store in environment variables, not code
6. **Backups**: Regular automated backups
7. **Monitoring**: Enable logging and alerts

---

## Support

- **Documentation**: [doc/](doc/)
- **Issues**: GitHub Issues
- **Contact**: SE183256

---

**Version**: Academic v4.0  
**Last Updated**: October 25, 2025
