# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Production Deployment Guide

**Version:** 3.0  
**Model:** STMGT V3  
**Date:** November 10, 2025  
**Status:** Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Model Deployment](#model-deployment)
5. [API Server](#api-server)
6. [Dashboard](#dashboard)
7. [Testing and Validation](#testing-and-validation)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)

---

## Quick Start

**For experienced users - full deployment in 5 minutes:**

```bash
# 1. Clone and setup environment
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project
conda env create -f environment.yml
conda activate dsp

# 2. Install package
pip install -e .

# 3. Start API server (auto-detects V3 model)
./stmgt.sh api start
# API available at http://localhost:8080

# 4. Start dashboard (optional)
./stmgt.sh dashboard start
# Dashboard at http://localhost:8501

# 5. Test API
curl http://localhost:8080/health
```

**Expected output:**

```json
{
  "status": "healthy",
  "model": "STMGT_V3",
  "version": "3.0",
  "test_mae": 3.0468
}
```

---

## Prerequisites

### System Requirements

**Minimum (CPU-only inference):**

- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free
- OS: Windows 10/11, Linux, macOS

**Recommended (GPU inference):**

- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)
- CUDA: 11.8 or 12.1
- Storage: 20GB free

### Software Dependencies

**Required:**

- Python 3.10 or 3.11
- Conda/Miniconda
- Git

**Optional:**

- Docker (for containerized deployment)
- NVIDIA CUDA Toolkit (for GPU acceleration)

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project
```

### 2. Create Conda Environment

**From environment.yml (recommended):**

```bash
conda env create -f environment.yml
conda activate dsp
```

**Manual installation:**

```bash
conda create -n dsp python=3.10
conda activate dsp
pip install -r requirements.txt
```

### 3. Install Package

```bash
# Development mode (editable)
pip install -e .

# Production mode
pip install .
```

### 4. Verify Installation

```bash
python -c "import traffic_forecast; print('Package installed successfully')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**

```
Package installed successfully
PyTorch: 2.1.0+cu121, CUDA: True  # or False for CPU-only
```

---

## Model Deployment

### V3 Model Artifacts

**Location:** `outputs/stmgt_v2_20251110_123931/`

**Files:**

- `best_model.pt` - Model checkpoint (680K params)
- `config.json` - Training configuration
- `training_history.csv` - Training metrics
- `final_metrics.json` - Test performance
- `test_predictions.csv` - Test set predictions

### Auto-Detection (Default)

The API automatically detects the latest model:

```python
# In traffic_api/config.py
outputs_dir = config.project_root / "outputs"
model_dirs = sorted([d for d in outputs_dir.iterdir() if d.name.startswith("stmgt")],
                    key=lambda p: p.stat().st_mtime, reverse=True)
config.model_checkpoint = model_dirs[0] / "best_model.pt"
```

**No manual configuration needed** - V3 will be used automatically.

### Manual Model Selection

To use a specific model checkpoint:

```python
# Edit traffic_api/config.py
config.model_checkpoint = Path("outputs/stmgt_v2_20251110_123931/best_model.pt")
```

Or set environment variable:

```bash
export STMGT_MODEL_PATH="/path/to/model/best_model.pt"
```

### Model Validation

```bash
# Verify model loads correctly
python -c "
from traffic_api.predictor import STMGTPredictor
from pathlib import Path

checkpoint = Path('outputs/stmgt_v2_20251110_123931/best_model.pt')
predictor = STMGTPredictor(checkpoint)
print(f'Model loaded: {predictor.config[\"model_name\"]}')
print(f'Test MAE: {predictor.config[\"test_mae\"]}')
"
```

**Expected output:**

```
Model loaded: STMGT_V3
Test MAE: 3.0468
```

---

## API Server

### Starting the API

**Option 1: CLI wrapper (recommended):**

```bash
./stmgt.sh api start
```

**Option 2: Direct uvicorn:**

```bash
conda run -n dsp uvicorn traffic_api.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload
```

**Option 3: Background service:**

```bash
# Linux/macOS
nohup ./stmgt.sh api start > api.log 2>&1 &

# Windows (PowerShell)
Start-Process -NoNewWindow -FilePath "./stmgt.sh" -ArgumentList "api start"
```

### API Configuration

**File:** `traffic_api/config.py`

**Key settings:**

```python
class APIConfig:
    host: str = "0.0.0.0"          # Bind to all interfaces
    port: int = 8080                # API port
    device: str = "cuda"            # "cuda" or "cpu"
    batch_size: int = 16            # Inference batch size
    num_workers: int = 4            # Data loading workers
    prediction_cache_ttl: int = 900 # 15 min cache
```

### API Endpoints

**Health Check:**

```bash
curl http://localhost:8080/health
```

**Single Prediction:**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "route": [1, 5, 12, 18],
    "weather": {
      "temperature": 28,
      "wind_speed": 5,
      "precipitation": 0
    },
    "time_features": {
      "hour": 8,
      "day_of_week": 1,
      "is_weekend": false
    }
  }'
```

**Expected response:**

```json
{
  "predictions": [
    {
      "edge": "1->5",
      "mean_speed": 18.3,
      "std_speed": 4.8,
      "confidence_80": [13.5, 23.1],
      "mixture_components": [
        { "mean": 15.2, "std": 2.8, "weight": 0.58 },
        { "mean": 22.4, "std": 3.1, "weight": 0.32 },
        { "mean": 28.7, "std": 2.2, "weight": 0.1 }
      ]
    }
  ],
  "total_time_minutes": 12.5,
  "avg_speed_kmh": 18.3
}
```

**Route Planning:**

```bash
curl "http://localhost:8080/route?start=1&end=30&algorithm=fastest"
```

**Network Topology:**

```bash
curl http://localhost:8080/nodes
```

### API Performance Benchmarks

**Hardware:** NVIDIA RTX 3060 Laptop (6GB VRAM)

| Metric                | Value     | Target    |
| --------------------- | --------- | --------- |
| Single prediction     | ~50ms     | <100ms    |
| Batch prediction (16) | ~200ms    | <500ms    |
| Throughput            | ~20 req/s | >10 req/s |
| Memory usage          | ~2GB VRAM | <4GB      |
| Cold start            | ~3s       | <5s       |

---

## Dashboard

### Starting the Dashboard

**Option 1: CLI wrapper:**

```bash
./stmgt.sh dashboard start
```

**Option 2: Direct streamlit:**

```bash
conda run -n dsp streamlit run dashboard/Dashboard.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

**Access:** http://localhost:8501

### Dashboard Features

**Pages:**

1. **Overview** - System status, model metrics
2. **Model Comparison** - V1 vs V3 performance
3. **Predictions** - Real-time traffic visualization
4. **Calibration** - Reliability diagrams, coverage analysis
5. **Training** - Training history, loss curves

**V3 Metrics Display:**

- Test MAE: 3.0468 km/h
- Test RMSE: 4.5198 km/h
- R² Score: 0.8161
- Coverage@80: 86.0%
- Model size: 680K params
- Best epoch: 9

### Dashboard Configuration

**File:** `dashboard/Dashboard.py`

**Port configuration:**

```python
# Default: 8501
# Change via command line:
streamlit run dashboard/Dashboard.py --server.port 8502
```

---

## Testing and Validation

### Automated Tests

**Run all tests:**

```bash
pytest tests/ -v
```

**Test API:**

```bash
pytest tests/test_api.py -v
```

**Test predictor:**

```bash
pytest tests/test_predictor_direct.py -v
```

**Test integration:**

```bash
pytest tests/test_integration.py -v
```

### Manual Validation

**1. Model Performance:**

```bash
python scripts/analysis/validate_model.py \
  --checkpoint outputs/stmgt_v2_20251110_123931/best_model.pt \
  --data data/processed/all_runs_extreme_augmented.parquet
```

**Expected metrics:**

- Test MAE: 3.0468 ± 0.01
- Test RMSE: 4.5198 ± 0.02
- Coverage@80: 86.0% ± 1%

**2. API Response Time:**

```bash
# Install Apache Bench
apt-get install apache2-utils  # Linux
brew install httpd  # macOS

# Benchmark API
ab -n 100 -c 10 http://localhost:8080/health
```

**Target:** 95th percentile < 100ms

**3. End-to-End Test:**

```bash
# Start API
./stmgt.sh api start &
sleep 5  # Wait for startup

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_request.json

# Stop API
pkill -f "uvicorn traffic_api"
```

---

## Monitoring

### Metrics to Monitor

**Application Metrics:**

- API response time (p50, p95, p99)
- Request rate (req/s)
- Error rate (%)
- Model inference time (ms)
- Cache hit rate (%)

**Model Metrics:**

- Prediction MAE (should match test: 3.0468)
- Coverage@80 (should be ~86%)
- Mixture component weights (tri-modal distribution)

**System Metrics:**

- CPU usage (%)
- Memory usage (GB)
- GPU usage (%) - if available
- GPU memory (GB)
- Disk I/O

### Logging

**API logs:**

```bash
# Location: api.log (if using nohup)
tail -f api.log

# Or check uvicorn logs
# Location: stdout/stderr
```

**Training logs:**

```bash
# Location: outputs/stmgt_v2_20251110_123931/training.log
tail -f outputs/stmgt_v2_20251110_123931/training.log
```

**Dashboard logs:**

```bash
# Streamlit logs to stdout
# Redirect to file:
streamlit run dashboard/Dashboard.py > dashboard.log 2>&1
```

### Health Checks

**API health endpoint:**

```bash
# Should return HTTP 200
curl -f http://localhost:8080/health || echo "API DOWN"
```

**Automated monitoring script:**

```bash
#!/bin/bash
# scripts/monitoring/check_health.sh

while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
  if [ "$STATUS" -ne 200 ]; then
    echo "$(date): API unhealthy (HTTP $STATUS)" >> health.log
    # Send alert (email, Slack, etc.)
  fi
  sleep 60  # Check every minute
done
```

### Alerting

**Simple email alert (Linux):**

```bash
# Install mailutils
apt-get install mailutils

# Add to cron
*/5 * * * * /path/to/check_health.sh
```

**Slack webhook:**

```bash
# In check_health.sh
if [ "$STATUS" -ne 200 ]; then
  curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
    -d '{"text":"STMGT API is down!"}'
fi
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Not Found

**Symptom:**

```
FileNotFoundError: Model checkpoint not found
```

**Solution:**

```bash
# Check model exists
ls outputs/stmgt_v2_20251110_123931/best_model.pt

# If missing, retrain or download backup
# For backup, see Rollback Procedures section
```

#### Issue 2: CUDA Out of Memory

**Symptom:**

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# Option 1: Reduce batch size
# In traffic_api/config.py
config.batch_size = 8  # Default: 16

# Option 2: Use CPU
config.device = "cpu"

# Option 3: Clear cache
import torch
torch.cuda.empty_cache()
```

#### Issue 3: Slow Inference

**Symptom:**
Predictions take >200ms per request

**Solution:**

```python
# Enable GPU if available
config.device = "cuda"

# Enable TF32 (RTX 30xx+)
torch.backends.cuda.matmul.allow_tf32 = True

# Increase batch size (if memory allows)
config.batch_size = 32

# Enable model compilation (PyTorch 2.0+)
model = torch.compile(model)
```

#### Issue 4: API Won't Start

**Symptom:**

```
Address already in use: 0.0.0.0:8080
```

**Solution:**

```bash
# Find process using port 8080
lsof -i :8080  # Linux/macOS
netstat -ano | findstr :8080  # Windows

# Kill the process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn traffic_api.main:app --port 8081
```

#### Issue 5: High Error Rate

**Symptom:**
Many API requests fail with validation errors

**Diagnostic:**

```bash
# Check API logs
tail -f api.log | grep ERROR

# Test with known-good request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_request.json
```

**Solution:**

- Validate input data format
- Check node IDs exist in network
- Ensure weather values in valid range
- Check model checkpoint integrity

#### Issue 6: Dashboard Not Loading

**Symptom:**
Dashboard shows blank page or loading spinner

**Solution:**

```bash
# Check streamlit is running
ps aux | grep streamlit

# Check port is accessible
curl http://localhost:8501

# Clear streamlit cache
rm -rf ~/.streamlit/cache

# Restart dashboard
./stmgt.sh dashboard stop
./stmgt.sh dashboard start
```

### Performance Tuning

**Inference Optimization:**

```python
# Enable mixed precision (AMP)
torch.set_float32_matmul_precision('high')

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Disable gradient computation
torch.set_grad_enabled(False)

# Use torch.inference_mode() context
with torch.inference_mode():
    predictions = model(input_data)
```

**Memory Optimization:**

```python
# Reduce batch size
config.batch_size = 8

# Clear unused variables
import gc
gc.collect()
torch.cuda.empty_cache()

# Use float16 for inference
model = model.half()  # Convert to FP16
```

---

## Rollback Procedures

### Scenario 1: V3 Performance Degradation

**If V3 shows unexpected poor performance in production:**

**Step 1: Verify issue is model-related**

```bash
# Check prediction MAE
python scripts/analysis/validate_model.py \
  --checkpoint outputs/stmgt_v2_20251110_123931/best_model.pt

# If MAE > 3.2, proceed to rollback
```

**Step 2: Rollback to V1**

```python
# Edit traffic_api/config.py
config.model_checkpoint = Path("outputs/stmgt_v2_20251101_012257/best_model.pt")
```

**Step 3: Restart API**

```bash
./stmgt.sh api stop
./stmgt.sh api start
```

**Step 4: Verify V1 performance**

```bash
curl http://localhost:8080/health
# Should show model: STMGT_V1, test_mae: 3.08
```

### Scenario 2: API Service Failure

**If API crashes or becomes unresponsive:**

**Step 1: Check logs**

```bash
tail -100 api.log
# Look for error messages
```

**Step 2: Restart service**

```bash
# Kill existing process
pkill -f "uvicorn traffic_api"

# Restart
./stmgt.sh api start
```

**Step 3: If restart fails, use backup config**

```bash
# Revert to known-good configuration
git checkout traffic_api/config.py
./stmgt.sh api start
```

### Scenario 3: Data Corruption

**If training data becomes corrupted:**

**Step 1: Verify data integrity**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/all_runs_extreme_augmented.parquet')
print(f'Rows: {len(df)}, Nulls: {df.isnull().sum().sum()}')
"
```

**Step 2: Restore from backup**

```bash
# If you have backups
cp data/backups/all_runs_extreme_augmented_backup.parquet \
   data/processed/all_runs_extreme_augmented.parquet
```

**Step 3: Revalidate**

```bash
python scripts/data/validate_dataset.py \
  --data data/processed/all_runs_extreme_augmented.parquet
```

### Backup Checklist

**Before any major changes:**

1. **Backup current model:**

```bash
cp outputs/stmgt_v2_20251110_123931/best_model.pt \
   backups/v3_$(date +%Y%m%d).pt
```

2. **Backup configuration:**

```bash
cp traffic_api/config.py backups/config_$(date +%Y%m%d).py
cp configs/train_normalized_v3.json backups/
```

3. **Backup data:**

```bash
cp data/processed/all_runs_extreme_augmented.parquet \
   backups/data_$(date +%Y%m%d).parquet
```

4. **Document backup location:**

```bash
echo "Backup created: $(date)" >> backups/BACKUP_LOG.txt
```

---

## Production Checklist

**Before deploying V3 to production:**

- [ ] Environment setup complete
- [ ] Model checkpoint verified (MAE 3.0468)
- [ ] API starts successfully
- [ ] Dashboard displays V3 metrics
- [ ] All tests passing (pytest)
- [ ] Performance benchmarks met (<100ms p95)
- [ ] Monitoring configured
- [ ] Logs rotating properly
- [ ] Backup procedures documented
- [ ] Rollback tested successfully
- [ ] Team trained on new features
- [ ] Documentation updated

---

## Support and Resources

**Documentation:**

- Main README: `README.md`
- V3 Design: `docs/V3_DESIGN_RATIONALE.md`
- API Guide: `traffic_api/README.md`
- Dashboard Guide: `dashboard/README.md`
- Full Report: `docs/report/V3_FINAL_SUMMARY.md`

**Code Repository:**

- GitHub: https://github.com/thatlq1812/dsp391m_project

**Maintainer:**

- Name: THAT Le Quang
- Role: AI & DS Major Student
- GitHub: [@thatlq1812]

**Reporting Issues:**

```bash
# GitHub Issues
https://github.com/thatlq1812/dsp391m_project/issues

# Include in report:
# - Error message and stack trace
# - Steps to reproduce
# - Environment (OS, Python version, CUDA version)
# - Logs (api.log, training.log)
```

---

**End of Deployment Guide**

**Status:** Production Ready  
**Last Updated:** November 10, 2025  
**Version:** 3.0
