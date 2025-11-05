# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Phase 1: Web MVP Completion

**Duration:** 3-4 days  
**Priority:** 🔴 HIGH - Critical for Report 3 presentation  
**Dependencies:** None (backend ready)

---

## Objectives

Build production-ready web interface for traffic forecasting:

- Google Maps visualization of 78 traffic nodes
- Real-time predictions with 3-hour forecast horizon
- Interactive UI with color-coded congestion levels
- Integration with FastAPI backend

---

## Task Breakdown

### Task 1.1: Quick Fixes (30 mins) ✅ DO FIRST

**Priority:** IMMEDIATE

Fix issues identified in project review:

```bash
# 1. Fix duplicate header in research doc
# Edit docs/STMGT_RESEARCH_CONSOLIDATED.md lines 1-10

# 2. Create .env file
echo "CONDA_ENV_NAME=dsp" > .env
echo "PYTHON_VERSION=3.10" >> .env

# 3. Verify requirements.txt is tracked
git add requirements.txt
```

**Acceptance Criteria:**

- [ ] No duplicate headers in any docs
- [ ] `.env` file exists with conda env name
- [ ] `requirements.txt` in git

---

### Task 1.2: Frontend Structure (2 hours)

Create basic HTML/JS structure for web interface.

**Location:** Create `traffic_api/static/` directory

**Files to create:**

```
traffic_api/
├── static/
│   ├── index.html           # Main page
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       ├── map.js           # Google Maps initialization
│       ├── api.js           # API client
│       └── charts.js        # Forecast charts (Chart.js)
```

**Template: index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>STMGT Traffic Forecast - HCMC</title>

    <!-- Bootstrap CSS -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet"
    />

    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/css/style.css" />
  </head>
  <body>
    <nav class="navbar navbar-dark bg-dark">
      <div class="container-fluid">
        <span class="navbar-brand mb-0 h1">🚦 STMGT Traffic Forecast</span>
        <span class="text-white" id="lastUpdate">Last update: --:--</span>
      </div>
    </nav>

    <div class="container-fluid">
      <div class="row">
        <!-- Map Panel -->
        <div class="col-md-8">
          <div id="map" style="height: 90vh;"></div>
        </div>

        <!-- Forecast Panel -->
        <div class="col-md-4">
          <div class="card mt-3">
            <div class="card-header">
              <h5 id="nodeTitle">Select a node on map</h5>
            </div>
            <div class="card-body">
              <canvas id="forecastChart"></canvas>
              <div id="nodeDetails" class="mt-3"></div>
            </div>
          </div>

          <!-- Stats Panel -->
          <div class="card mt-3">
            <div class="card-header">System Stats</div>
            <div class="card-body">
              <p>
                <strong>Total Nodes:</strong> <span id="totalNodes">--</span>
              </p>
              <p><strong>Model:</strong> STMGT v2</p>
              <p>
                <strong>Inference Time:</strong>
                <span id="inferenceTime">--</span>ms
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Google Maps API -->
    <script
      src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=initMap"
      async
      defer
    ></script>

    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Custom JS -->
    <script src="/static/js/api.js"></script>
    <script src="/static/js/charts.js"></script>
    <script src="/static/js/map.js"></script>
  </body>
</html>
```

**Acceptance Criteria:**

- [ ] HTML structure created with responsive layout
- [ ] Bootstrap integrated for UI components
- [ ] Google Maps API placeholder added
- [ ] Chart.js ready for forecast visualization

---

### Task 1.3: Google Maps Integration (3 hours)

Implement interactive map with traffic nodes.

**File:** `traffic_api/static/js/map.js`

```javascript
// Global variables
let map;
let markers = [];
let nodes = [];
let selectedNode = null;

// Color mapping based on speed
function getMarkerColor(speed) {
  if (speed > 40) return "#28a745"; // Green - fast
  if (speed > 20) return "#ffc107"; // Yellow - moderate
  return "#dc3545"; // Red - congested
}

// Initialize map
async function initMap() {
  // Center on HCMC
  const hcmcCenter = { lat: 10.772465, lng: 106.697794 };

  map = new google.maps.Map(document.getElementById("map"), {
    center: hcmcCenter,
    zoom: 13,
    styles: [
      {
        featureType: "poi",
        stylers: [{ visibility: "off" }], // Hide POIs for cleaner view
      },
    ],
  });

  // Load nodes from API
  await loadNodes();

  // Initial prediction
  await updatePredictions();

  // Auto-refresh every 15 minutes
  setInterval(updatePredictions, 15 * 60 * 1000);
}

// Load traffic nodes from API
async function loadNodes() {
  try {
    const data = await apiGetNodes();
    nodes = data.nodes;

    document.getElementById("totalNodes").textContent = nodes.length;

    // Create markers
    nodes.forEach((node) => {
      const marker = new google.maps.Marker({
        position: { lat: node.lat, lng: node.lon },
        map: map,
        title: `Node ${node.id}`,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: "#3b82f6",
          fillOpacity: 0.8,
          strokeColor: "#ffffff",
          strokeWeight: 2,
        },
      });

      // Click handler
      marker.addListener("click", () => selectNode(node, marker));

      markers.push({ node: node, marker: marker });
    });
  } catch (error) {
    console.error("Failed to load nodes:", error);
    alert("Failed to load traffic nodes. Check API connection.");
  }
}

// Update predictions for all nodes
async function updatePredictions() {
  try {
    const predictions = await apiGetPredictions();

    // Update marker colors based on current speed
    markers.forEach(({ node, marker }) => {
      const pred = predictions.find((p) => p.node_id === node.id);
      if (pred) {
        const currentSpeed = pred.predictions[0].mean; // First timestep
        marker.setIcon({
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: getMarkerColor(currentSpeed),
          fillOpacity: 0.8,
          strokeColor: "#ffffff",
          strokeWeight: 2,
        });
      }
    });

    document.getElementById(
      "lastUpdate"
    ).textContent = `Last update: ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    console.error("Failed to update predictions:", error);
  }
}

// Select a node and show forecast
async function selectNode(node, marker) {
  selectedNode = node;

  // Highlight selected marker
  markers.forEach((m) => {
    m.marker.setOptions({ scale: 8 });
  });
  marker.setOptions({ scale: 12 });

  // Update panel
  document.getElementById("nodeTitle").textContent = `Node ${node.id}`;
  document.getElementById("nodeDetails").innerHTML = `
        <p><strong>Location:</strong> ${node.lat.toFixed(
          6
        )}, ${node.lon.toFixed(6)}</p>
        <p><strong>Streets:</strong> ${
          node.street_names?.join(", ") || "N/A"
        }</p>
    `;

  // Fetch and display forecast
  try {
    const prediction = await apiGetNodePrediction(node.id);
    updateForecastChart(prediction);

    document.getElementById("inferenceTime").textContent =
      prediction.inference_time_ms.toFixed(1);
  } catch (error) {
    console.error("Failed to get node prediction:", error);
  }
}
```

**Acceptance Criteria:**

- [ ] Map centered on HCMC with 78 nodes
- [ ] Markers color-coded by current speed
- [ ] Click marker → load forecast
- [ ] Auto-refresh every 15 minutes
- [ ] Error handling for API failures

---

### Task 1.4: API Client (1 hour)

Create JavaScript API client for backend communication.

**File:** `traffic_api/static/js/api.js`

```javascript
// API Configuration
const API_BASE_URL = window.location.origin; // Same origin

// Helper function for API calls
async function apiCall(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(
      `API call failed: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

// Get all nodes
async function apiGetNodes() {
  return apiCall("/nodes");
}

// Get predictions for all nodes
async function apiGetPredictions(horizons = [1, 4, 12]) {
  return apiCall("/predict", {
    method: "POST",
    body: JSON.stringify({ horizons }),
  });
}

// Get prediction for specific node
async function apiGetNodePrediction(nodeId, horizons = [1, 2, 3, 4, 6, 8, 12]) {
  return apiCall(`/nodes/${nodeId}/predict`, {
    method: "POST",
    body: JSON.stringify({ horizons }),
  });
}

// Health check
async function apiHealthCheck() {
  return apiCall("/health");
}
```

**Acceptance Criteria:**

- [ ] All API endpoints wrapped
- [ ] Error handling included
- [ ] JSON parsing automated
- [ ] Works with FastAPI backend

---

### Task 1.5: Forecast Charts (2 hours)

Implement Chart.js visualization for 3-hour forecasts.

**File:** `traffic_api/static/js/charts.js`

```javascript
let forecastChart = null;

// Initialize or update forecast chart
function updateForecastChart(prediction) {
  const ctx = document.getElementById("forecastChart").getContext("2d");

  // Prepare data
  const labels = prediction.predictions.map((_, idx) => {
    const minutes = idx * 15; // 15-min intervals
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `+${hours}h${mins > 0 ? mins : ""}`;
  });

  const means = prediction.predictions.map((p) => p.mean);
  const upperBounds = prediction.predictions.map((p) => p.mean + p.std);
  const lowerBounds = prediction.predictions.map((p) => p.mean - p.std);

  // Destroy existing chart
  if (forecastChart) {
    forecastChart.destroy();
  }

  // Create new chart
  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Predicted Speed",
          data: means,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          tension: 0.3,
        },
        {
          label: "Upper Bound (80% CI)",
          data: upperBounds,
          borderColor: "#94a3b8",
          borderWidth: 1,
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0,
        },
        {
          label: "Lower Bound (80% CI)",
          data: lowerBounds,
          borderColor: "#94a3b8",
          borderWidth: 1,
          borderDash: [5, 5],
          fill: "-1",
          backgroundColor: "rgba(148, 163, 184, 0.1)",
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        title: {
          display: true,
          text: "3-Hour Traffic Speed Forecast",
        },
        legend: {
          display: true,
          position: "bottom",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Speed (km/h)",
          },
        },
        x: {
          title: {
            display: true,
            text: "Time Ahead",
          },
        },
      },
    },
  });
}
```

**Acceptance Criteria:**

- [ ] Chart shows mean prediction line
- [ ] Confidence intervals (80% CI) displayed
- [ ] Responsive design
- [ ] Labels show time ahead (+0h, +1h, +2h, +3h)
- [ ] Chart updates when selecting different nodes

---

### Task 1.6: Backend Enhancements (2 hours)

Update FastAPI to serve static files and add missing endpoints.

**File:** `traffic_api/main.py`

Add static file serving:

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add root redirect to index.html
@app.get("/")
async def root():
    """Redirect to web interface."""
    from fastapi.responses import FileResponse
    return FileResponse(static_dir / "index.html")

# Add node-specific prediction endpoint
@app.post("/nodes/{node_id}/predict")
async def predict_node(node_id: int, request: PredictionRequest):
    """Get prediction for specific node."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Filter prediction for this node only
        predictions = predictor.predict(horizons=request.horizons)
        node_pred = next((p for p in predictions if p["node_id"] == node_id), None)

        if node_pred is None:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        return node_pred

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Acceptance Criteria:**

- [ ] Static files served at `/static/`
- [ ] Root path `/` serves `index.html`
- [ ] `/nodes/{node_id}/predict` endpoint added
- [ ] CORS allows frontend requests
- [ ] No breaking changes to existing API

---

### Task 1.7: Styling (1 hour)

Create clean, professional CSS.

**File:** `traffic_api/static/css/style.css`

```css
:root {
  --primary: #3b82f6;
  --success: #28a745;
  --warning: #ffc107;
  --danger: #dc3545;
  --dark: #212529;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
    Ubuntu, Cantarell, sans-serif;
  margin: 0;
  padding: 0;
}

#map {
  border-radius: 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: none;
}

.card-header {
  background-color: var(--dark);
  color: white;
  font-weight: 600;
}

#forecastChart {
  max-height: 300px;
}

.navbar-brand {
  font-weight: 700;
  font-size: 1.5rem;
}

/* Loading spinner */
.spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid var(--primary);
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  #map {
    height: 50vh !important;
  }

  .col-md-4 {
    margin-top: 1rem;
  }
}
```

**Acceptance Criteria:**

- [ ] Professional, clean design
- [ ] Responsive for mobile/desktop
- [ ] Loading states styled
- [ ] Color scheme consistent with branding

---

### Task 1.8: Testing & Debugging (2 hours)

Comprehensive testing of web interface.

**Test Checklist:**

```markdown
## Functionality Tests

- [ ] Map loads centered on HCMC
- [ ] All 78 nodes visible as markers
- [ ] Click node → panel updates
- [ ] Forecast chart displays correctly
- [ ] Confidence intervals visible
- [ ] Auto-refresh works (check after 15min)
- [ ] System stats update
- [ ] Error messages for API failures

## Performance Tests

- [ ] Initial load <2 seconds
- [ ] Marker click response <200ms
- [ ] API calls <100ms (backend already tested)
- [ ] No memory leaks (open DevTools → Performance)

## Browser Compatibility

- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Edge (latest)
- [ ] Mobile Safari (iOS)
- [ ] Mobile Chrome (Android)

## Edge Cases

- [ ] No internet connection → error message
- [ ] API down → graceful degradation
- [ ] Invalid node ID → 404 handled
- [ ] Very slow connection → loading indicator
```

**Debugging Tools:**

```bash
# Run backend in debug mode
conda run -n dsp uvicorn traffic_api.main:app --reload --host 0.0.0.0 --port 8000

# Check browser console for JS errors
# Open DevTools → Console

# Test API endpoints directly
curl http://localhost:8000/health
curl http://localhost:8000/nodes
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizons": [1, 4, 12]}'
```

**Acceptance Criteria:**

- [ ] All functionality tests pass
- [ ] No console errors
- [ ] Works on 3+ browsers
- [ ] Mobile responsive
- [ ] Performance acceptable

---

### Task 1.9: Documentation (1 hour)

Document the web interface for users and developers.

**Create:** `traffic_api/README.md`

Update with web interface section:

````markdown
## Web Interface

### Quick Start

1. Start backend:
   ```bash
   conda run -n dsp uvicorn traffic_api.main:app --host 0.0.0.0 --port 8000
   ```
````

2. Open browser: http://localhost:8000

3. Interact with map:
   - View 78 traffic nodes across HCMC
   - Click any marker to see 3-hour forecast
   - Color coding:
     - 🟢 Green: >40 km/h (fast)
     - 🟡 Yellow: 20-40 km/h (moderate)
     - 🔴 Red: <20 km/h (congested)

### Features

- Real-time predictions updated every 15 minutes
- Interactive Google Maps integration
- Forecast charts with 80% confidence intervals
- Responsive design (desktop + mobile)
- <100ms inference latency

### Architecture

```
User Browser
    ↓ (HTTP)
FastAPI Server (:8000)
    ↓ (static files: /, /static/*)
    ↓ (API: /nodes, /predict)
STMGT Model (PyTorch)
    ↓ (inference)
Predictions (JSON)
```

### Customization

Edit `traffic_api/static/js/map.js` to:

- Change map center/zoom
- Adjust color thresholds
- Modify auto-refresh interval

Edit `traffic_api/static/css/style.css` for styling.

````

**Acceptance Criteria:**
- [ ] README updated with web interface docs
- [ ] Screenshots included (optional but nice)
- [ ] Architecture diagram clear
- [ ] Customization guide provided

---

### Task 1.10: Demo Preparation (1 hour)

Prepare demo for Report 3 presentation.

**Demo Script:**

```markdown
## 5-Minute Demo Script

### Setup (before presentation)
1. Start backend: `conda run -n dsp uvicorn traffic_api.main:app`
2. Open browser to http://localhost:8000
3. Prepare 2-3 interesting nodes (high traffic areas)

### Live Demo Flow

**[0:00-0:30] Introduction**
- "This is STMGT - Spatial-Temporal Multi-Modal Graph Transformer"
- "Predicts traffic speeds 3 hours ahead for 78 locations in HCMC"

**[0:30-1:30] Model Architecture Overview**
- Show architecture slide from docs/STMGT_ARCHITECTURE.md
- "267K parameters, parallel spatial-temporal processing"
- "Weather cross-attention for multi-modal fusion"
- "Gaussian mixture outputs for uncertainty quantification"

**[1:30-3:00] Web Interface Demo**
- Show map: "78 traffic nodes across HCMC"
- Color coding: "Green=fast, Yellow=moderate, Red=congested"
- Click busy intersection: "Real-time prediction with confidence intervals"
- Show forecast chart: "3-hour horizon, 15-minute intervals"
- "Inference latency <100ms per request"

**[3:00-4:00] Technical Highlights**
- "Current performance: R²=0.79, MAE=2.78 km/h on test set"
- "Beats ASTGCN baseline significantly"
- "Production-ready FastAPI backend"
- "Auto-refresh every 15 minutes"

**[4:00-5:00] Q&A**
- Prepared answers for:
  - "How does it compare to Google Traffic?" → Different approach, explains uncertainty
  - "Can it scale to more cities?" → Yes, just need data collection
  - "Deployment plan?" → Phase 3 - Docker + Google Cloud Run
````

**Backup Plans:**

```markdown
## If Demo Fails

**Backend won't start:**

- Have screenshots ready
- Explain: "In production, this runs in Docker container"

**Slow internet:**

- Use localhost (already planned)
- Pre-load nodes data

**Model fails to load:**

- Show dashboard instead (Streamlit backup)
- Explain: "This is the alternative monitoring interface"
```

**Acceptance Criteria:**

- [ ] Demo script written and rehearsed
- [ ] Timing under 5 minutes
- [ ] Backup screenshots prepared
- [ ] Q&A answers ready
- [ ] Confident presentation

---

## Phase 1 Success Criteria

✅ **All tasks completed**  
✅ **Web interface functional**  
✅ **Demo presentation ready**  
✅ **Documentation updated**  
✅ **No blocking bugs**

---

## Next Steps

After Phase 1 completion:

1. **Update CHANGELOG.md** with Phase 1 achievements
2. **Present Report 3** with live demo
3. **Gather feedback** from presentation
4. **Begin Phase 2** - Model Quality Assurance

---

## Time Tracking

| Task                   | Estimated | Actual | Notes |
| ---------------------- | --------- | ------ | ----- |
| 1.1 Quick Fixes        | 30m       |        |       |
| 1.2 Frontend Structure | 2h        |        |       |
| 1.3 Google Maps        | 3h        |        |       |
| 1.4 API Client         | 1h        |        |       |
| 1.5 Charts             | 2h        |        |       |
| 1.6 Backend            | 2h        |        |       |
| 1.7 Styling            | 1h        |        |       |
| 1.8 Testing            | 2h        |        |       |
| 1.9 Documentation      | 1h        |        |       |
| 1.10 Demo Prep         | 1h        |        |       |
| **Total**              | **15.5h** |        |       |

Spread across 3-4 days = 4-5 hours per day (manageable).

---

**Ready to start? Begin with Task 1.1 Quick Fixes! 🚀**
