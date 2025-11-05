# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Dashboard V4 - Quick Reference Card

One-page cheat sheet for the complete STMGT control hub

---

## Launch Command

```bash
conda activate dsp && streamlit run dashboard/Dashboard.py
```

URL: http://localhost:8501

---

## 12 Pages Overview

| #   | Page               | Purpose          | Key Action          |
| --- | ------------------ | ---------------- | ------------------- |
| 1   | System Overview    | Health check     | Glance at status    |
| 2   | VM Management      | Control GCP VM   | Start/Stop VM       |
| 3   | Deployment         | Deploy code      | Push to production  |
| 4   | Monitoring         | Watch logs       | Check for errors    |
| 5   | Data Collection    | Collect data     | Schedule collection |
| 6   | Data Overview      | Review datasets  | Verify quality      |
| 7   | Data Augmentation  | Create synthetic | Run 48.4x augment   |
| 8   | Data Visualization | Analyze patterns | View distributions  |
| 9   | Training Control   | Train models     | Start training      |
| 10  | Model Registry     | Manage versions  | Tag production      |
| 11  | Predictions        | Forecast traffic | Generate forecast   |
| 12  | API Integration    | Manage API       | Start API server    |

---

## 10-Second Actions

| I need to...               | Go to   | Action                                    |
| -------------------------- | ------- | ----------------------------------------- |
| Check if system is healthy | Page 1  | Review status cards                       |
| Start the VM               | Page 2  | Copy command or use Start Instance button |
| Deploy latest code         | Page 3  | Copy deploy command                       |
| Collect data now           | Page 5  | Copy collection command                   |
| Run augmentation           | Page 7  | Copy augmentation command                 |
| Start training             | Page 9  | Copy training command                     |
| Get predictions            | Page 11 | Generate forecast                         |
| Start API                  | Page 12 | Copy server command                       |

---

## Common Workflows

### Morning Check (2 min)

```
Page 1 -> Page 4 -> Page 6 -> Done
```

### New Model (1 hour)

```
Page 5 (collect) -> Page 7 (augment) -> Page 9 (train) -> Page 10 (tag)
```

### Deploy to Production (5 min)

```
Page 3 (deploy) -> Page 4 (verify) -> Page 12 (start API)
```

### Emergency Rollback (8 min)

```
Page 4 (find error) -> Page 3 (rollback) -> Page 4 (verify)
```

---

## Page Groups

```
INFRASTRUCTURE          DATA PIPELINE
   1. Overview             5. Collection
   2. VM Management        6. Data Overview
   3. Deployment           7. Augmentation
   4. Monitoring

ML WORKFLOW             PRODUCTION
   8. Visualization        11. Predictions
   9. Training             12. API
   10. Model Registry
```

---

## Key Shortcuts

```bash
# Terminal quick commands
conda run -n dsp --no-capture-output \
   python scripts/collect_and_render.py --once --no-visualize    # Collect
conda run -n dsp --no-capture-output \
   python scripts/training/train_stmgt.py --config configs/training_config.json  # Train
bash scripts/deployment/deploy_git.sh                            # Deploy

# Dashboard navigation
Ctrl+R         # Refresh page
Ctrl+Shift+R   # Hard refresh
F5             # Reload
```

---

## Status Indicators

| Symbol  | Meaning            |
| ------- | ------------------ |
| GREEN   | Running / Healthy  |
| YELLOW  | Warning / Degraded |
| RED     | Stopped / Error    |
| PAUSED  | Paused             |
| WORKING | In Progress        |
| CHECK   | Success            |
| CROSS   | Failed             |

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

## Quick Links

| Resource              | Location                        |
| --------------------- | ------------------------------- |
| Project Overview      | README.md                       |
| Dashboard Quick Start | docs/DASHBOARD_V4_QUICKSTART.md |
| Dashboard Reference   | docs/DASHBOARD_V4_REFERENCE.md  |
| Architecture          | docs/STMGT_ARCHITECTURE.md      |
| Workflows             | docs/WORKFLOW.md                |
| API Docs              | http://localhost:8000/docs      |

---

## Emergency Commands

```bash
# Stop everything
pkill -f streamlit
pkill -f uvicorn
pkill -f collect_and_render

# Restart dashboard
streamlit run dashboard/Dashboard.py

# Check Python environment
conda info --envs
conda activate dsp
python --version
```

---

## Pro Tips

1. Keep Page 1 open in one tab for status monitoring
2. Use Page 4 for continuous log streaming
3. Tag models in Page 10 before deploying
4. Schedule collection in Page 5 instead of manual runs
5. Export predictions in Page 11 for external use

---

## Success Checklist

After setup, verify:

- [ ] Page 1: All systems green
- [ ] Page 2: VM running
- [ ] Page 6: 1,839+ runs
- [ ] Page 9: Training completes
- [ ] Page 10: Production model tagged
- [ ] Page 11: MAE < 5.0 km/h
- [ ] Page 12: API returns 200 OK

---

## Learning Order

```
Day 1: Pages 1, 6, 11       (Basics)
Day 2: Pages 5, 7, 8        (Data)
Day 3: Pages 9, 10          (ML)
Day 4: Pages 2, 3, 4        (DevOps)
Day 5: Page 12              (API)
```

---

Dashboard V4 | Keep this card handy

Print this page for quick desk reference
