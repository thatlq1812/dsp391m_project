# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Dashboard V4 - Quick Start Guide

Get up and running with the complete control hub in 5 minutes

## Super Quick Start

```bash
# 1. Activate environment
conda activate dsp

# 2. Launch dashboard with Phase 1 schema validators
streamlit run dashboard/Dashboard.py

# 3. Open browser
# http://localhost:8501
```

---

## First-Time Setup (5 minutes)

### Step 1: Configure VM (Page 2)

```bash
# Login to gcloud
gcloud auth login

# Set project
gcloud config set project sonorous-nomad-476606-g3

# Test connection
gcloud compute instances list
```

### Step 2: Verify Data (Page 6)

- Confirm processed parquet path matches registry entry (`configs/project_config.yaml`)
- Review validation status card (must show "Validated" with zero missing columns)
- Inspect augmentation coverage charts (basic vs extreme vs synthetic)

### Step 3: Test Prediction (Page 11)

- Select best model from dropdown (reads from validated registry)
- Choose target edge/time window
- Click "Generate Predictions"
- Confirm 12-step forecast with uncertainty ribbon renders

---

## Common Tasks (1-Click)

### Task 1: Collect Fresh Data

```
Page 5 -> "Single Collection" tab -> Copy command -> Run in terminal
Takes approximately 5 minutes for 62 nodes
```

### Task 2: Train New Model

```
Page 9 -> "Start Training" tab -> Configure -> Copy CLI (with conda run) -> Execute manually
Takes 30-60 minutes depending on config
```

### Task 3: Deploy to VM

```
Page 3 -> "Deploy" tab -> Click "Deploy to VM"
Takes 2-3 minutes
```

### Task 4: Start API Server

```
Page 12 -> "Server Control" tab -> Copy `conda run` command -> Execute in terminal
Access at http://localhost:8000/docs
```

---

## Typical Workflows

### Workflow 1: New Model Development (Complete)

```
1. Page 2: Start VM                    [2 min]
2. Page 5: Collect data (1 run)        [5 min]
3. Page 6: Check data quality          [1 min]
4. Page 7: Run augmentation (48.4x)    [10 min]
5. Page 9: Train model                 [45 min]
6. Page 10: Review & tag as 'staging'  [2 min]
7. Page 11: Test predictions           [1 min]
8. Page 10: Tag as 'production'        [1 min]
───────────────────────────────────────────────
Total: approximately 67 minutes (mostly automated)
```

### Workflow 2: Production Deployment

```
1. Page 3: Commit & push code          [1 min]
2. Page 3: Deploy to VM                [3 min]
3. Page 4: Check health                [1 min]
4. Page 5: Schedule collection         [1 min]
5. Page 12: Start API server           [1 min]
6. Page 4: Monitor logs                [Continuous]
───────────────────────────────────────────────
Total: approximately 7 minutes
```

### Workflow 3: Daily Monitoring

```
1. Page 1: Check overview dashboard    [1 min]
2. Page 4: Review health checks        [2 min]
3. Page 6: Verify latest collection    [1 min]
4. Page 11: Spot-check predictions     [2 min]
5. Page 4: Check logs for errors       [2 min]
───────────────────────────────────────────────
Total: approximately 8 minutes/day
```

### Workflow 4: Emergency Rollback

```
1. Page 4: Identify error in logs      [2 min]
2. Page 3: Select previous version     [1 min]
3. Page 3: Deploy rollback             [3 min]
4. Page 12: Restart API server         [1 min]
5. Page 4: Verify health restored      [1 min]
───────────────────────────────────────────────
Total: approximately 8 minutes (crisis resolved)
```

---

## Dashboard Navigation Cheat Sheet

### Infrastructure Group (Pages 1-4)

| Page | Purpose         | When to Use                    |
| ---- | --------------- | ------------------------------ |
| 1    | System Overview | Daily health check             |
| 2    | VM Management   | Start/stop VM, check resources |
| 3    | Deployment      | Deploy code, manage Git        |
| 4    | Monitoring      | View logs, check alerts        |

### Data Group (Pages 5-7)

| Page | Purpose            | When to Use                         |
| ---- | ------------------ | ----------------------------------- |
| 5    | Collection Control | Collect new data, schedule          |
| 6    | Data Overview      | Inspect validation status, coverage |
| 7    | Data Augmentation  | Launch augmentation via CLI helpers |

### ML Group (Pages 8-10)

| Page | Purpose        | When to Use                 |
| ---- | -------------- | --------------------------- |
| 8    | Visualization  | Analyze patterns            |
| 9    | Training       | Train new models            |
| 10   | Model Registry | Manage versions, tag models |

### Production Group (Pages 11-12)

| Page | Purpose         | When to Use          |
| ---- | --------------- | -------------------- |
| 11   | Predictions     | Generate forecasts   |
| 12   | API Integration | Manage API, webhooks |

---

## Power User Tips

### Tip 1: Keyboard Shortcuts

```
F5          -> Refresh dashboard
Ctrl+K      -> Search sidebar (VS Code)
Ctrl+Shift+P -> Streamlit command palette
```

### Tip 2: Multi-Tab Workflow

```
Tab 1: Page 1 (Overview) - Keep open for status
Tab 2: Page 4 (Logs) - Monitor in real-time
Tab 3: Current task page - Switch as needed
```

### Tip 3: Quick Commands (Terminal)

```bash
# Quick collect (one cycle, no visualization)
conda run -n dsp --no-capture-output \
  python scripts/collect_and_render.py --once --no-visualize

# Quick train (registry-backed config)
conda run -n dsp --no-capture-output \
  python scripts/training/train_stmgt.py --config configs/training_config.json

# Quick deploy
bash scripts/deployment/deploy_git.sh
```

### Tip 4: Scheduled Automation

```bash
# Add to crontab (Page 5 does this for you)
0 */6 * * * cd /path/to/project && conda run -n dsp python scripts/collect_and_render.py --once
```

### Tip 5: API Quick Test

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predict (with API key)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"edge_id": 0, "timestamp": "2025-01-15T10:00:00"}'
```

---

## Instant Actions

### Need predictions RIGHT NOW?

```
Page 11 -> Load model -> Select edge -> Generate -> Done
30 seconds
```

### Need to collect data NOW?

```
Page 5 -> Copy "Single Collection" command -> Run in terminal -> Wait 5 min -> Done
5 minutes
```

### Need to deploy NOW?

```
Page 3 -> Click "Quick Deploy" -> Wait 3 min -> Done
3 minutes
```

### Need to check if VM is alive?

```
Page 2 -> See status indicator (green/red) -> Instant
0 seconds
```

---

## Quick Fixes

### Dashboard will not start

```bash
pip install --upgrade streamlit
streamlit run dashboard/Dashboard.py
```

### VM won't connect

```bash
gcloud auth login
gcloud compute instances list
```

### Model not found

```bash
ls -la outputs/stmgt_*
```

### API port busy

```bash
# Windows
netstat -ano | findstr :8000
# Kill process

# Linux/Mac
kill $(lsof -t -i:8000)
```

---

## Expected Metrics

| Metric          | Expected  | Critical  |
| --------------- | --------- | --------- |
| MAE             | 3-5 km/h  | > 10 km/h |
| R²              | 0.70-0.80 | < 0.50    |
| VM CPU          | < 80%     | > 95%     |
| VM RAM          | < 80%     | > 90%     |
| Disk            | < 80%     | > 95%     |
| Collection time | 5 min     | > 15 min  |
| Training time   | 30-60 min | > 2 hours |

---

## Success Metrics

After setup, you should see:

- Page 1: All systems green
- Page 2: VM running, resources < 80%
- Page 6: 1,839 runs (48.4x augmented)
- Page 9: Training completes without errors
- Page 10: At least 1 model tagged 'production'
- Page 11: Predictions MAE < 5.0 km/h
- Page 12: API health returns 200 OK

---

## Learning Path

### Day 1: Basics

- Launch dashboard
- Navigate all 12 pages
- Run one collection (Page 5)
- Generate one prediction (Page 11)

### Day 2: Data Pipeline

- Understand augmentation (Page 7)
- Visualize patterns (Page 8)
- Schedule automated collection (Page 5)

### Day 3: ML Workflow

- Train first model (Page 9)
- Compare versions (Page 10)
- Test predictions thoroughly (Page 11)

### Day 4: Infrastructure

- Control VM (Page 2)
- Deploy to production (Page 3)
- Monitor and logs (Page 4)

### Day 5: Production

- Setup API (Page 12)
- Configure webhooks
- End-to-end workflow

---

## Pro Workflow

The 5-Minute Daily Check:

```
1. Page 1: Glance at overview (green = good)
2. Page 4: Skim logs for errors (red text = investigate)
3. Page 11: Spot-check 1-2 predictions (reasonable?)
4. Done
```

The Weekly Deep Dive:

```
1. Page 6: Data quality review (distributions OK?)
2. Page 8: Pattern analysis (anomalies?)
3. Page 10: Model performance trends (improving?)
4. Page 4: Alert history (recurring issues?)
5. Document findings
```

---

## Emergency Contacts

Quick Links:

- Project Overview: README.md
- Architecture: docs/STMGT_ARCHITECTURE.md
- Workflows: docs/WORKFLOW.md
- Dashboard Details: dashboard/README.md

Terminal Commands:

```bash
# Dashboard help
streamlit run dashboard/Dashboard.py --help

# Script help
python scripts/collect_and_render.py --help
python scripts/training/train_stmgt.py --help
```

---

Dashboard V4 | From Zero to Hero in 5 Minutes

Last Updated: November 2, 2025
