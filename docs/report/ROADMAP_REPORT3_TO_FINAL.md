# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Roadmap: Report 3 → Final Delivery

**Current Status:** Report 3/4 completed  
**Timeline:** Report 3 → Model Improvements → Report 4 + Final Delivery  
**Goal:** Production-ready traffic forecasting system with web interface

---

## PHASE 1: Report 3 Completion (Current Sprint)

### Model Architecture Analysis

**Current STMGT Architecture:**

```python
Input:
  - Traffic: (batch, 62 nodes, 12 timesteps, 1 feature)
  - Weather: (batch, 62 nodes, 12 timesteps, 3 features)
  - Temporal: hour, dow, is_weekend

Model Components:
  1. Traffic Encoder: Linear(1 → 96)
  2. Weather Encoder: Linear(3 → 96)
  3. Temporal Encoder: Embedding + Cyclical (hour sin/cos)

  4. ParallelSTBlock × 3:
     - Spatial: GATv2Conv (4 heads, edge dropout 0.05)
     - Temporal: MultiheadAttention (4 heads)
     - Fusion: Gated combination

  5. Weather Cross-Attention: Traffic attends to weather

  6. Gaussian Mixture Head:
     - Components: 3
     - Output: mean, std, logits per node per horizon

Output: (batch, 62 nodes, 12 horizons, 3 components)
```

**Current Performance (Best Run: stmgt_v2_20251101_215205):**

- Validation MAE: 5.00 km/h
- Test MAE: 2.78 km/h
- Test R²: 0.79
- Test MAPE: ~18%
- Model Size: 2.7 MB (2,728,162 bytes)

**Key Hyperparameters:**

- Hidden Dim: 96
- Sequence Length: 12 (3 hours)
- Batch Size: 48
- Learning Rate: 0.0006
- Mixture Loss + MSE (weight 0.2)

### Inference Web Build

**Goal:** Simple Google Maps interface showing real-time predictions

**Tech Stack:**

```
Backend:
  - FastAPI (lightweight, async)
  - STMGT model loading from best checkpoint
  - In-memory caching for fast inference

Frontend:
  - HTML + Google Maps JavaScript API
  - Vanilla JS (no React/Vue initially - keep it simple)
  - Bootstrap for UI components
```

**Minimum Viable Product (MVP):**

1. **Backend API:**

   - `GET /health` - Health check
   - `GET /nodes` - List all 78 nodes with coordinates
   - `POST /predict` - Predict traffic for next 3 hours
   - `GET /predict/{node_id}` - Get prediction for specific node

2. **Frontend Map:**

   - Display 78 nodes as markers on HCMC map
   - Color-coded by current speed prediction:
     - 🟢 Green: >40 km/h (fast)
     - 🟡 Yellow: 20-40 km/h (moderate)
     - 🔴 Red: <20 km/h (congested)
   - Click node → show forecast chart (3h ahead)

3. **Data Flow:**
   ```
   User opens web → Frontend loads
   → Calls /nodes → Renders map markers
   → Calls /predict → Gets predictions for all nodes
   → Updates marker colors
   → User clicks marker → Shows forecast popup
   ```

---

## PHASE 2: Model Improvements (Post Report 3)

### 🔬 Improvement Areas Identified

#### A. Architecture Enhancements

**1. Attention Mechanism:**

```python
Current: 4 heads, standard MultiheadAttention
Potential:
  - Try 8 heads (more expressive)
  - Add relative positional encoding
  - Experiment with Performer/Linformer (efficiency)
```

**2. Graph Structure:**

```python
Current: Static edge_index (144 edges)
Potential:
  - Dynamic graph learning (learn edge weights)
  - Add spatial features (distance, road type)
  - Multi-hop message passing
```

**3. Weather Integration:**

```python
Current: Cross-attention after ST blocks
Potential:
  - Early fusion (inject weather in ST blocks)
  - Weather-conditioned edge weights
  - Separate weather encoder (Conv1D)
```

**4. Probabilistic Output:**

```python
Current: 3-component Gaussian mixture
Potential:
  - Increase to 5 components (more flexibility)
  - Add quantile regression head
  - Calibration post-processing
```

#### B. Training Improvements

**1. Data Augmentation:**

```python
Current: all_runs_extreme_augmented.parquet
Ideas:
  - Temporal jittering (shift windows by ±1-2 steps)
  - Spatial dropout (randomly mask nodes)
  - Weather perturbation (add noise to forecasts)
  - MixUp between different time periods
```

**2. Loss Function:**

```python
Current: Mixture NLL + 0.2 * MSE
Experiments:
  - Add MAE term for robustness
  - Weighted loss by traffic levels (penalize congestion errors)
  - Focal loss for extreme values
  - Multi-task: predict speed + uncertainty separately
```

**3. Regularization:**

```python
Current: Dropout 0.2, Edge dropout 0.05
Try:
  - Label smoothing
  - Stochastic depth (drop blocks randomly)
  - Spectral normalization on attention
  - Gradient penalty on predictions
```

**4. Optimization:**

```python
Current: AdamW, ReduceLROnPlateau
Experiments:
  - CosineAnnealingWarmRestarts (cyclic LR)
  - Lookahead optimizer wrapper
  - Gradient clipping adjustment
  - Warm-up schedule tuning
```

#### C. Evaluation Enhancements

**1. Ablation Studies:**

```
Test impact of each component:
  - STMGT vs STMGT-no-weather
  - STMGT vs STMGT-spatial-only
  - STMGT vs STMGT-temporal-only
  - Different mixture components (1, 3, 5)
```

**2. Error Analysis:**

```python
Analyze predictions by:
  - Time of day (peak vs off-peak)
  - Weather conditions (rain vs clear)
  - Road types (trunk vs primary)
  - Node degree (high vs low connectivity)
  - Traffic levels (free-flow vs congested)
```

**3. Baseline Comparisons:**

```
Compare against:
  - Historical average (naive baseline)
  - LSTM (temporal only)
  - GCN (spatial only)
  - ASTGCN (teammates' models)
  - Prophet (Facebook forecasting)
```

---

## PHASE 3: Production API (Report 4 Prep)

###API Development

**FastAPI Structure:**

```python
traffic_api/
├── __init__.py
├── main.py              # FastAPI app
├── models/
│   ├── predictor.py     # STMGT inference wrapper
│   └── schemas.py       # Pydantic models
├── routers/
│   ├── health.py        # Health checks
│   ├── nodes.py         # Node metadata
│   └── predictions.py   # Prediction endpoints
├── core/
│   ├── config.py        # API config
│   ├── cache.py         # Redis/in-memory cache
│   └── auth.py          # API key validation
└── utils/
    ├── logger.py        # Logging
    └── metrics.py       # Prometheus metrics
```

**Key Features:**

1. **Model Loading:**

   - Load best checkpoint on startup
   - Keep model in GPU memory
   - Batch multiple requests together

2. **Caching:**

   - Cache predictions for 15 minutes
   - Cache node metadata permanently
   - LRU cache for frequent queries

3. **Rate Limiting:**

   - 100 requests/minute per API key
   - Burst allowance for dashboard

4. **Monitoring:**
   - Request latency tracking
   - Error rate monitoring
   - GPU utilization metrics

### 🌐 Web Frontend

**Google Maps Integration:**

```html
<!DOCTYPE html>
<html>
  <head>
    <title>HCMC Traffic Forecast</title>
    <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_KEY"></script>
    <style>
      #map {
        height: 100vh;
      }
      .node-marker {
        cursor: pointer;
      }
      .forecast-popup {
        /* styling */
      }
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script>
      // Initialize map centered on HCMC
      // Load nodes from /nodes endpoint
      // Add markers with color coding
      // Poll /predict every 5 minutes
      // Update marker colors
      // Show forecast chart on click
    </script>
  </body>
</html>
```

**Features:**

- Real-time updates (5-min polling)
- Node clustering (when zoomed out)
- Heatmap overlay option
- Time slider (show predictions at different hours)
- Route ETA calculator (stretch goal)

### Deployment

**Docker Setup:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install CUDA if needed (for GPU inference)
# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY traffic_api/ ./traffic_api/
COPY traffic_forecast/ ./traffic_forecast/
COPY outputs/stmgt_v2_20251101_215205/ ./models/

# Run API
CMD ["uvicorn", "traffic_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Deployment Options:**

1. **Google Cloud Run:**

   - Serverless, auto-scaling
   - Pay per request
   - GPU support (if needed)

2. **Google Compute Engine VM:**

   - More control, cheaper for consistent load
   - Can use GPU instance

3. **Local Development:**
   - Run on laptop for demos
   - Use ngrok for public access

---

## PHASE 4: Final Delivery Checklist

### 📊 Report 4 Requirements

**1. Complete Documentation:**

- [x] Architecture documentation (STMGT_ARCHITECTURE.md)
- [x] Data pipeline docs (STMGT_DATA_IO.md)
- [x] Training workflow (WORKFLOW.md)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Deployment guide
- [ ] User manual for web interface

**2. Evaluation Results:**

- [x] Current performance metrics
- [ ] Ablation study results
- [ ] Baseline comparisons
- [ ] Error analysis by category
- [ ] Prediction visualizations

**3. Demo Materials:**

- [x] Streamlit dashboard (internal use)
- [ ] Google Maps web demo (public-facing)
- [ ] Video walkthrough (5-10 min)
- [ ] Presentation slides

**4. Code Quality:**

- [x] Modular architecture
- [x] Configuration management
- [ ] Comprehensive tests (>70% coverage)
- [ ] CI/CD pipeline
- [ ] Code documentation (docstrings)

**5. Reproducibility:**

- [x] Environment setup (environment.yml)
- [x] Training configs (JSON files)
- [x] Model checkpoints
- [ ] Sample inference script
- [ ] Data collection guide (if sharing)

---

## Implementation Timeline

### Week 1 (Report 3 Focus):

**Days 1-2:** Build Inference Web MVP

- [ ] Create FastAPI skeleton with /health, /nodes, /predict
- [ ] Implement STMGT inference wrapper
- [ ] Test inference speed (aim for <100ms per prediction)

**Days 3-4:** Google Maps Frontend

- [ ] Basic HTML + Maps API integration
- [ ] Display 78 nodes as markers
- [ ] Connect to FastAPI backend
- [ ] Implement color-coding logic

**Days 5-7:** Polish & Demo

- [ ] Add forecast popup charts
- [ ] Improve UI/UX
- [ ] Deploy to Google Cloud Run (or local + ngrok)
- [ ] Record demo video for Report 3

### Week 2-3 (Model Improvements):

**Week 2:** Architecture Experiments

- [ ] Run ablation studies (no weather, spatial-only, etc.)
- [ ] Try 8-head attention
- [ ] Experiment with 5-component mixture
- [ ] Test dynamic graph learning

**Week 3:** Training Enhancements

- [ ] Implement data augmentation pipeline
- [ ] Try different loss functions
- [ ] Hyperparameter search (hidden_dim, num_blocks)
- [ ] Train best configuration for 200 epochs

### Week 4 (Report 4 & Final):

**Days 1-3:** Results Analysis

- [ ] Compile all experiment results
- [ ] Create comparison tables
- [ ] Generate error analysis plots
- [ ] Write findings summary

**Days 4-5:** Documentation

- [ ] API documentation (Swagger)
- [ ] Deployment guide
- [ ] Update CHANGELOG with all improvements

**Days 6-7:** Presentation

- [ ] Create slides
- [ ] Practice demo flow
- [ ] Prepare Q&A answers
- [ ] Final testing

---

## Success Metrics

### Report 3 Goals:

- Working inference API (latency <100ms)
- Google Maps web demo (78 nodes displayed)
- Color-coded predictions (real-time)
- Forecast visualization (3-hour chart)

### Report 4 Goals:

- Test MAE <2.5 km/h (current: 2.78)
- Test R² >0.80 (current: 0.79)
- Model size <5 MB (current: 2.7 MB ✓)
- Inference latency <50ms
- > 5 ablation experiments documented
- Web demo deployed publicly

### Final Delivery:

- 🏆 Production-ready API (deployed)
- 🏆 Public web interface (accessible URL)
- 🏆 Complete documentation (all guides)
- 🏆 >80% test coverage
- 🏆 Presentation ready

---

## Notes

### Current Strengths:

- Solid architecture with proven performance (R²=0.79)
- Modular codebase ready for extensions
- Comprehensive documentation already exists
- Working dashboard for internal monitoring
- Clear data pipeline with caching

### Areas to Focus:

1. **Quick Win:** Get inference web running ASAP (Report 3)
2. **Impact:** Model improvements show research depth
3. **Polish:** API + web demo = strong final impression
4. **Safety:** Keep current best model as fallback

### Risk Mitigation:

- Don't break current working model during experiments
- Save all experiment configs and checkpoints
- Keep Report 3 demo simple (MVP only)
- Reserve time for bugs and unexpected issues

---

**Last Updated:** November 2, 2025  
**Next Milestone:** Inference Web MVP (3 days)
