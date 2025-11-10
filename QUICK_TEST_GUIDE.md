# Quick API Testing Guide

## Option 1: Python Script (Easiest for Windows)

### Start API Server
```bash
# In terminal 1 (keep this running)
python scripts/deployment/start_api.py
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Test API
```bash
# In terminal 2 (new terminal)
./scripts/deployment/test_api.sh

# Or manually test
curl http://localhost:8080/health
```

---

## Option 2: Using stmgt.sh (If conda in PATH)

### Check stmgt.sh config
Edit `stmgt.sh` and update CONDA_PATH if needed:
```bash
# Current setting
CONDA_PATH="C:/ProgramData/miniconda3/Scripts/conda.exe"

# Your actual path might be:
# C:/Users/fxlqt/anaconda3/Scripts/conda.exe
# or
# C:/Users/fxlqt/miniconda3/Scripts/conda.exe
```

### Start API
```bash
./stmgt.sh api start
```

---

## Option 3: Direct uvicorn (Manual)

```bash
# Activate conda environment
source C:/ProgramData/miniconda3/Scripts/activate dsp

# Start API
uvicorn traffic_api.main:app --host 0.0.0.0 --port 8080
```

---

## Testing

### Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "STMGT_V3",
  "version": "3.0",
  "test_mae": 3.0468,
  "test_coverage": 0.86
}
```

### Full Test Suite
```bash
./scripts/deployment/test_api.sh
```

### API Documentation
Open in browser:
- API Docs: http://localhost:8080/docs
- Health: http://localhost:8080/health

---

## Troubleshooting

### Issue: "curl: command not found"

**Solution 1: Use Python**
```bash
python -c "import requests; print(requests.get('http://localhost:8080/health').json())"
```

**Solution 2: Use browser**
Open http://localhost:8080/health in browser

### Issue: "Connection refused"

Check if API is running:
```bash
# Windows
netstat -ano | findstr :8080

# If nothing, API is not running
# Start it with: python scripts/deployment/start_api.py
```

### Issue: "Model not found"

Check model exists:
```bash
ls -la outputs/stmgt_v2_20251110_123931/best_model.pt
```

If missing, the model file was not saved. You may need to use a different output directory.

---

## Recommended: Simple Python Start

**Most reliable for Windows Git Bash:**

```bash
# Terminal 1: Start API
python scripts/deployment/start_api.py

# Terminal 2: Test (after API starts)
curl http://localhost:8080/health

# Or if no curl:
python -c "import requests; r=requests.get('http://localhost:8080/health'); print(r.json())"
```

---

## Dashboard Testing (Optional)

```bash
# After API is running
streamlit run dashboard/Dashboard.py

# Access at http://localhost:8501
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start API (easy) | `python scripts/deployment/start_api.py` |
| Test API | `curl http://localhost:8080/health` |
| Full test | `./scripts/deployment/test_api.sh` |
| Stop API | Press Ctrl+C in API terminal |
| API docs | http://localhost:8080/docs |
