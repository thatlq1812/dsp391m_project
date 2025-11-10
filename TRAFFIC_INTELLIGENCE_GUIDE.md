# STMGT Traffic Intelligence - Quick Start Guide

## How to Run the Application

### Option 1: VS Code Python Terminal (RECOMMENDED)

1. **Open VS Code**
2. **Press `` Ctrl+` `` to open terminal**
3. **Select Python environment**: Click on Python version in bottom-right corner, choose `dsp` environment
4. **Run the command**:
   ```bash
   python start_api_simple.py
   ```
5. **Open browser**: http://localhost:8080

### Option 2: Anaconda Prompt

1. **Open Anaconda Prompt** (from Start menu)
2. **Navigate to project**:
   ```bash
   cd D:\UNI\DSP391m\project
   ```
3. **Activate environment**:
   ```bash
   conda activate dsp
   ```
4. **Run API**:
   ```bash
   python start_api_simple.py
   ```
5. **Open browser**: http://localhost:8080

### Option 3: Direct Script Execution

1. **Double-click**: `start_api_simple.py` in File Explorer
2. **Open browser**: http://localhost:8080

---

## What You'll See

### 🌐 Traffic Intelligence Dashboard

**Two Modes:**

#### 1. Traffic View Mode (Default)

- Real-time traffic visualization
- Color-coded road segments by speed
- Click nodes to view traffic details
- Auto-refresh every 5 minutes

#### 2. Route Planning Mode

- Click to select start point (green marker)
- Click to select end point (red marker)
- Click "Find Best Route" button
- View optimal path with metrics:
  - Distance (km)
  - Estimated time (minutes)
  - Average speed (km/h)
  - Number of segments

---

## Features

### Traffic Visualization

- **6 Speed Categories**:
  - 🔵 Very Smooth (50+ km/h)
  - 🟢 Smooth (40-50 km/h)
  - 🟡 Normal (30-40 km/h)
  - 🟠 Slow (20-30 km/h)
  - 🟠 Congested (10-20 km/h)
  - 🔴 Heavy Traffic (<10 km/h)

### Route Planning

- **A\* Pathfinding Algorithm**
- Traffic-aware routing
- Real-time speed data
- Optimal path calculation
- Interactive map selection

### Network Statistics

- Total traffic nodes
- Total road segments
- Average network speed
- Congestion level indicator

---

## Troubleshooting

### API Won't Start

- Check Python environment is activated
- Check port 8080 is not in use
- Try: `netstat -ano | findstr :8080`
- Kill process: `taskkill /PID <PID> /F`

### Page Won't Load

- Ensure API is running (check terminal)
- Try: http://localhost:8080/health
- Check firewall settings

### No Traffic Data

- Check `/api/traffic/current` endpoint
- Verify model checkpoint loaded
- Check data files exist in `data/` folder

---

## API Endpoints

- **Home**: http://localhost:8080
- **Health**: http://localhost:8080/health
- **Traffic Data**: http://localhost:8080/api/traffic/current
- **Prediction**: http://localhost:8080/predict (POST)
- **Nodes**: http://localhost:8080/nodes

---

## Technical Details

### Frontend

- Leaflet.js for mapping
- Bootstrap 5 for UI
- Font Awesome icons
- Vanilla JavaScript (no frameworks)

### Backend

- FastAPI (Python)
- STMGT V3 Model (680K params)
- PyTorch for inference
- Real-time traffic prediction

### Algorithm

- **A\* Search** for pathfinding
- Euclidean distance heuristic
- Traffic speed-based edge weights
- Bidirectional graph construction

---

## Next Steps

1. ✅ Start API server
2. ✅ Open dashboard in browser
3. ✅ Explore traffic view
4. ✅ Try route planning
5. ✅ Click nodes for details
6. ✅ View traffic statistics

Enjoy your Traffic Intelligence Dashboard! 🚦🗺️
