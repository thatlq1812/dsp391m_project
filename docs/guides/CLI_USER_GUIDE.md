# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT CLI Tool - User Guide

Simple command-line interface for managing STMGT Traffic Forecasting system.

**Version:** 2.0.0  
**Replaces:** Streamlit Dashboard (13 pages → 1 CLI tool)

---

## Why CLI Instead of Streamlit?

**Problems with Streamlit Dashboard:**

- 13 separate pages (too complex)
- Many features don't work properly
- Heavy dependencies (streamlit, plotly, etc.)
- Slow startup time
- Difficult to use in headless environments
- Over-engineered for simple tasks

**Benefits of CLI:**

- Simple, fast, lightweight
- Works in any terminal (local, SSH, Docker)
- Easy to script and automate
- Better for DevOps workflows
- Professional tool for production use
- Can be used in CI/CD pipelines

---

## Installation

```bash
# Install CLI tool
cd d:/UNI/DSP391m/project
pip install -e . -f setup_cli.py

# Or install dependencies manually
pip install click rich requests pyyaml
```

After installation, the `stmgt` command will be available globally.

---

## Quick Start

```bash
# Show help
stmgt --help

# Show system info
stmgt info

# List all models
stmgt model list

# Check API status
stmgt api status

# Start API server
stmgt api start
```

---

## Command Reference

### Global Options

```bash
stmgt --version           # Show CLI version
stmgt --help              # Show help message
```

---

### Model Management

#### List All Models

```bash
stmgt model list                    # Table format (default)
stmgt model list --format=json      # JSON format
```

**Output:**

```
┌──────────────────────────┬────────┬────────┬────────┬────────────┐
│ Name                     │ Type   │ MAE    │ Status │ Date       │
├──────────────────────────┼────────┼────────┼────────┼────────────┤
│ stmgt_v2_production      │ STMGT  │ 3.69   │ active │ 2025-11-02 │
│ lstm_baseline            │ LSTM   │ 3.94   │ tested │ 2025-11-09 │
└──────────────────────────┴────────┴────────┴────────┴────────────┘
```

#### Show Model Details

```bash
stmgt model info stmgt_v2_production
```

**Output:**

```
╭─────────────────── Model: stmgt_v2_production ───────────────────╮
│                                                                   │
│ Model: stmgt_v2_production                                       │
│ Type: STMGT                                                       │
│ Status: active                                                    │
│                                                                   │
│ Performance:                                                      │
│   MAE:  3.6900 km/h                                              │
│   RMSE: 4.8500 km/h                                              │
│   R²:   0.6600                                                   │
│                                                                   │
│ Configuration:                                                    │
│   Parameters: 4,000,000                                          │
│   Batch Size: 16                                                 │
│   Epochs:     100                                                │
│                                                                   │
│ Files:                                                            │
│   Checkpoint: outputs/stmgt_v2_20251102_200308/best_model.pt    │
│   Config:     configs/training_config.json                       │
│                                                                   │
│ Created: 2025-11-02 20:03:08                                     │
╰───────────────────────────────────────────────────────────────────╯
```

#### Compare Models

```bash
stmgt model compare stmgt_v2_production lstm_baseline
```

**Output:**

```
┌────────────────┬────────────────────────┬────────────────┐
│ Metric         │ stmgt_v2_production    │ lstm_baseline  │
├────────────────┼────────────────────────┼────────────────┤
│ MAE (km/h)     │ 3.6900                 │ 3.9400         │
│ RMSE (km/h)    │ 4.8500                 │ 5.1200         │
│ R²             │ 0.6600                 │ 0.6200         │
│ Parameters     │ 4,000,000              │ 100,000        │
│ Epochs         │ 100                    │ 100            │
└────────────────┴────────────────────────┴────────────────┘

Best model (lowest MAE): stmgt_v2_production (3.6900 km/h)
```

---

### API Server Management

#### Start API Server

```bash
# Default: http://0.0.0.0:8000
stmgt api start

# Custom host and port
stmgt api start --host=127.0.0.1 --port=5000

# Enable auto-reload (development)
stmgt api start --reload
```

**Output:**

```
Starting API server on 0.0.0.0:8000...
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Press `Ctrl+C` to stop the server.

#### Check API Status

```bash
stmgt api status
```

**Output (if online):**

```
╭─────────────── API Status ───────────────╮
│                                           │
│ API Server: Online                        │
│                                           │
│ Status: healthy                           │
│ Model Loaded: True                        │
│ Device: cuda                              │
│ Timestamp: 2025-11-09T16:30:00           │
╰───────────────────────────────────────────╯
```

**Output (if offline):**

```
API Server: Offline
Start with: stmgt api start
```

#### Test API Endpoint

```bash
# Test health endpoint (default)
stmgt api test

# Test specific endpoint
stmgt api test --endpoint=/api/traffic/current
stmgt api test --endpoint=/nodes
```

**Output:**

```
Testing: http://localhost:8000/health

Status Code: 200
Response Time: 0.123s

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-11-09T16:30:00"
}
```

---

### Training Management

#### Show Training Status

```bash
stmgt train status
```

**Output:**

```
╭─────────────── Training Status ───────────────╮
│                                                │
│ Training Run: stmgt_v2_20251109_160000        │
│ Status: running                                │
│                                                │
│ Progress:                                      │
│   Epoch: 45/100                               │
│   Progress: 45.0%                             │
│                                                │
│ Current Metrics:                               │
│   Train Loss: 12.3456                         │
│   Val Loss:   15.2345                         │
│   Val MAE:    3.7500 km/h                     │
│                                                │
│ Best Metrics:                                  │
│   Best Val Loss: 14.8900                      │
│   Best Epoch:    42                           │
│                                                │
│ Time:                                          │
│   Elapsed: 1h 23m                             │
│   Estimated Remaining: 1h 38m                 │
╰────────────────────────────────────────────────╯
```

#### Show Training Logs

```bash
# Show last 20 lines (default)
stmgt train logs

# Show last 50 lines
stmgt train logs --lines=50

# Follow logs in real-time (like tail -f)
stmgt train logs --follow
stmgt train logs -f
```

---

### Data Management

#### Show Dataset Information

```bash
stmgt data info
```

**Output:**

```
╭─────────────── Dataset Information ───────────────╮
│                                                    │
│ Dataset: all_runs_extreme_augmented.parquet       │
│                                                    │
│ Shape:                                             │
│   Rows:    205,920                                │
│   Columns: 18                                     │
│                                                    │
│ Columns:                                           │
│   run_id, timestamp, node_a_id, node_b_id, ...   │
│                                                    │
│ Date Range:                                        │
│   Start: 2025-10-28 00:00:00                      │
│   End:   2025-11-01 12:00:00                      │
│                                                    │
│ Statistics:                                        │
│   Unique Edges: 144                               │
│   Avg Speed:    30.50 km/h                        │
│   Min Speed:    5.20 km/h                         │
│   Max Speed:    65.80 km/h                        │
│                                                    │
│ File:                                              │
│   Size: 45.23 MB                                  │
│   Path: data/processed/all_runs_...parquet        │
╰────────────────────────────────────────────────────╯
```

---

### System Information

```bash
stmgt info
```

**Output:**

```
╭─────────────── System Info ───────────────╮
│                                            │
│ System Information                         │
│                                            │
│ Hardware:                                  │
│   CPU Usage:    25%                       │
│   Memory:       35% (5.6GB / 16.0GB)     │
│   Disk:         45% (135.2GB / 300.0GB)  │
│                                            │
│ Software:                                  │
│   OS:           Windows 10                │
│   Python:       3.10.18                   │
│   PyTorch:      2.5.1+cu121              │
│   CUDA:         True                      │
│   Device:       NVIDIA GeForce RTX 3060  │
│                                            │
│ Project:                                   │
│   Root:         D:\UNI\DSP391m\project    │
│   API:          http://localhost:8000     │
│   Docs:         http://localhost:8000/docs│
╰────────────────────────────────────────────╯
```

---

## Common Workflows

### 1. Check System and Start API

```bash
# Check system resources
stmgt info

# Check API status
stmgt api status

# Start API if not running
stmgt api start
```

### 2. Monitor Training

```bash
# Check training status
stmgt train status

# Follow logs in real-time
stmgt train logs -f
```

### 3. Compare Models

```bash
# List all models
stmgt model list

# Compare specific models
stmgt model compare model1 model2

# View detailed info
stmgt model info best_model
```

### 4. Quick API Test

```bash
# Start API in background
stmgt api start &

# Wait a moment for startup
sleep 5

# Test endpoints
stmgt api test
stmgt api test --endpoint=/nodes
stmgt api test --endpoint=/api/traffic/current
```

---

## Scripting and Automation

The CLI is designed to be scriptable:

```bash
#!/bin/bash
# Example: Deploy and test script

echo "Checking system..."
stmgt info

echo "Starting API..."
stmgt api start &
API_PID=$!

echo "Waiting for startup..."
sleep 10

echo "Testing API..."
stmgt api status
stmgt api test

echo "Done! API running on PID $API_PID"
```

---

## Output Formats

### JSON Output

Most commands support JSON output for scripting:

```bash
stmgt model list --format=json | jq '.[] | select(.mae < 4.0)'
```

### Rich Terminal Output

The CLI uses the `rich` library for beautiful terminal output:

- Colored text
- Tables with borders
- Panels with titles
- Progress bars
- Syntax highlighting for JSON

---

## Troubleshooting

### Command Not Found

```bash
# Error: stmgt: command not found

# Solution 1: Install CLI
pip install -e . -f setup_cli.py

# Solution 2: Run directly
python traffic_forecast/cli.py --help

# Solution 3: Add alias
alias stmgt="python d:/UNI/DSP391m/project/traffic_forecast/cli.py"
```

### API Server Won't Start

```bash
# Check if port is already in use
netstat -ano | findstr :8000

# Kill existing process
taskkill /PID <PID> /F

# Try different port
stmgt api start --port=8001
```

### Model Not Found

```bash
# List available models
stmgt model list

# Check model registry file
cat configs/model_registry.json
```

---

## Future Enhancements

**Planned features:**

- `stmgt deploy` - Deploy to cloud VM
- `stmgt backup` - Backup models and data
- `stmgt predict` - Make predictions from CLI
- `stmgt experiment` - Run experiments
- Interactive mode with prompts

---

## Comparison: Dashboard vs CLI

| Feature              | Streamlit Dashboard       | CLI Tool            |
| -------------------- | ------------------------- | ------------------- |
| **Pages**            | 13 pages                  | 1 tool              |
| **Lines of code**    | ~2000+                    | ~500                |
| **Startup time**     | 5-10 seconds              | Instant             |
| **Dependencies**     | streamlit, plotly, altair | click, rich         |
| **Works over SSH**   | No                        | Yes                 |
| **Scriptable**       | No                        | Yes                 |
| **Resource usage**   | High (browser + server)   | Low (terminal only) |
| **Production ready** | No                        | Yes                 |

---

## Migration from Dashboard

**Old workflow (Streamlit):**

```bash
streamlit run dashboard/Dashboard.py
# Navigate to http://localhost:8501
# Click through 13 pages
# Wait for page loads
```

**New workflow (CLI):**

```bash
stmgt model list          # Instant
stmgt api start          # One command
stmgt train status       # Real-time
```

**Result:** 10x faster, simpler, more reliable!

---

## Web Interface (Future)

For visualization and route planning, we'll build a **separate web interface** later:

- Lightweight HTML/CSS/JS (no framework bloat)
- Leaflet.js for maps (already done in route_planner.html)
- Real-time traffic visualization
- Route planning interface

**Separation of concerns:**

- CLI = Management and operations
- Web = Visualization and user interaction

---

## Support

**For issues:**

- Check logs: `stmgt train logs`
- Check system: `stmgt info`
- Check API: `stmgt api status`

**Documentation:**

- API docs: http://localhost:8000/docs
- Project docs: `docs/INDEX.md`
- This guide: `docs/guides/CLI_USER_GUIDE.md`

---

**Author:** THAT Le Quang  
**GitHub:** thatlq1812  
**Version:** 2.0.0  
**Date:** November 9, 2025
