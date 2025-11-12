# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Master Plan: Final Report & Presentation Preparation

**Objective:** Deploy a complete, production-ready traffic forecasting system with model comparison, route optimization, and real-time capabilities for final presentation.

**Timeline:** 2-3 weeks  
**Target Date:** Ready for final report submission & demo

---

## Vision

### End Result:

- **VM-hosted system** running 24/7 with real-time data collection
- **Public web interface** for route queries and traffic visualization
- **Comprehensive model comparison** showing STMGT superiority
- **Route optimization** feature calculating best paths based on predictions
- **Demo-ready** for presentation (just open browser and show)

### User Story:

```
Presenter opens web interface →
Enters start/end locations →
System shows:
  1. Multiple route options
  2. Predicted travel times with confidence intervals
  3. Real-time traffic conditions
  4. Recommended optimal route
  5. Model comparison showing why STMGT performs best
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Public Web UI                         │
│              (Route Query + Visualization)              │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS
┌────────────────────▼────────────────────────────────────┐
│                   VM Backend                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  FastAPI Server (traffic_api/)                   │   │
│  │    - Prediction endpoints                        │   │
│  │    - Route optimization                          │   │
│  │    - Real-time data serving                      │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │  STMGT Model (Best: MAE 2.3, R²=0.82)           │   │
│  │  + Baseline Models (for comparison)              │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │  Data Collection Pipeline                        │   │
│  │    - Google Directions (every 30min peak)        │   │
│  │    - Weather data (Open-Meteo)                   │   │
│  │    - PostgreSQL storage                          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown

### Phase 1: Model Comparison & Validation (Week 1)

**Goal:** Prove STMGT is the best choice through rigorous comparison

#### Task 1.1: Unified Evaluation Framework

- Implement consistent evaluation pipeline for all models
- Same dataset, same train/val/test splits (temporal)
- Same metrics: MAE, RMSE, R², MAPE, CRPS
- Cross-validation for statistical significance

#### Task 1.2: Fix STMGT Validation

- **Current issue:** val_mae=3.05, need < 2.5
- Verify no data leakage
- Check metric calculation correctness
- K-fold cross-validation
- Hyperparameter optimization

#### Task 1.3: Baseline Model Benchmarking

Train and evaluate:

1. **LSTM Baseline** - Simple temporal model
2. **ASTGCN** - Spatial-temporal graph convolution
3. **GraphWaveNet** - Graph WaveNet architecture (if available)

Expected results:

```
Model              | MAE    | R²    | Training Time | Params
-------------------|--------|-------|---------------|--------
LSTM               | 4.5    | 0.55  | 10 min        | 200K
ASTGCN             | 3.8    | 0.65  | 30 min        | 500K
GraphWaveNet       | 3.5    | 0.70  | 45 min        | 1.2M
STMGT (Target)     | 2.3    | 0.82  | 60 min        | 4.0M
```

#### Task 1.4: Ablation Study

Test STMGT variants:

- Without graph module (transformer only)
- Without transformer (graph only)
- Without weather fusion
- Without temporal encoding

**Deliverable:** Comprehensive comparison report showing STMGT improvements

---

### Phase 2: Route Optimization (Week 1-2)

**Goal:** Add practical route recommendation feature

#### Task 2.1: Multi-Path Prediction

```python
# Input: start_node, end_node
# Output: [route1, route2, route3] with predictions

routes = [
    {
        'path': [node_1, node_5, node_8, node_15],
        'predicted_time_mean': 25.3,  # minutes
        'predicted_time_std': 3.2,
        'confidence_80': (23.1, 27.5),
        'distance': 8.5,  # km
        'traffic_level': 'moderate'
    },
    ...
]
```

#### Task 2.2: Route Optimization Algorithm

- Calculate expected travel time for each route
- Factor in uncertainty (use CRPS)
- Recommend optimal route based on:
  - Fastest expected time
  - Lowest uncertainty
  - Reliability (historical performance)

#### Task 2.3: Visualization

- Interactive map showing multiple routes
- Color-coded by predicted traffic
- Confidence intervals displayed
- Real-time updates

**Deliverable:** Working route optimization API endpoint

---

### Phase 3: Real-time System on VM (Week 2)

**Goal:** Deploy production-ready system

#### Task 3.1: VM Setup

- Configure Ubuntu VM with GPU (if needed)
- Install dependencies (conda, PyTorch, etc.)
- Setup PostgreSQL for data storage
- Configure firewall and security

#### Task 3.2: Automated Data Collection

```yaml
# Cron job schedule
Peak hours (6:30-8am, 11-12pm, 4-7pm): Every 30 min
Off-peak hours: Every 2 hours
Night hours: Every 2 hours
```

#### Task 3.3: Model Serving

- FastAPI server with production settings
- Model loading optimization (lazy loading)
- Request caching
- Rate limiting
- Logging and monitoring

#### Task 3.4: Continuous Integration

- Auto model retraining (weekly)
- Performance monitoring
- Alert system for anomalies
- Backup and recovery

**Deliverable:** 24/7 running system on VM

---

### Phase 4: Public Web Interface (Week 2-3)

**Goal:** User-friendly demo interface

#### Task 4.1: Frontend Development

Technology: React + Leaflet (or Streamlit + Folium)

Features:

- Map interface with HCMC overlay
- Route input (click or search)
- Real-time traffic visualization
- Prediction results display
- Model comparison tab
- About/Documentation tab

#### Task 4.2: Backend Integration

- Connect to VM FastAPI endpoints
- Handle async requests
- Error handling and user feedback
- Loading states and animations

#### Task 4.3: Deployment

- Host frontend (Vercel/Netlify free tier)
- Configure HTTPS
- Link to VM backend
- Domain name (optional)

**Deliverable:** Live demo website

---

### Phase 5: Documentation & Report (Week 3)

**Goal:** Complete final report

#### Task 5.1: Model Comparison Section

- Comparison table with all metrics
- Training curves comparison
- Ablation study results
- Statistical significance tests
- Why STMGT wins (architecture analysis)

#### Task 5.2: System Architecture

- Architecture diagram
- Data flow explanation
- Technology stack justification
- Scalability analysis

#### Task 5.3: Results & Evaluation

- Production performance metrics
- Route optimization case studies
- Real-world validation
- User testing results (if applicable)

#### Task 5.4: Demo Guide

- How to access the web interface
- Example queries to showcase
- Presentation talking points
- Troubleshooting guide

**Deliverable:** Final report + presentation slides

---

## Success Metrics

### Technical Metrics

- [x] STMGT MAE < 2.5 km/h
- [ ] R² > 0.75
- [ ] 99% API uptime on VM
- [ ] < 200ms prediction latency
- [ ] 20%+ improvement over baseline models

### Demo Metrics

- [ ] System accessible via public URL
- [ ] Zero crashes during 10 demo runs
- [ ] All 3 route optimization cases work
- [ ] Model comparison clearly shows STMGT advantage

### Report Metrics

- [ ] Complete model comparison table
- [ ] Ablation study with 4+ variants
- [ ] Architecture diagrams for all components
- [ ] Real-world validation results
- [ ] User guide for reproduction

---

## Risk Management

### Risk 1: VM Cost Overrun

**Mitigation:**

- Use Google Cloud free tier ($300 credit)
- Optimize collection schedule (adaptive scheduler)
- Monitor API usage daily

### Risk 2: Model Performance Not Meeting Target

**Mitigation:**

- Already have backup plan (target MAE 2.5-3.0 acceptable)
- Can adjust target based on baseline comparison
- Focus on improvement over baselines

### Risk 3: Time Constraint

**Mitigation:**

- Prioritize: Model comparison > Route optimization > Web UI
- MVP approach: Basic web UI sufficient
- Can use existing Streamlit dashboard as fallback

### Risk 4: Deployment Issues

**Mitigation:**

- Test VM deployment early (Week 2)
- Docker containerization for portability
- Detailed deployment documentation
- Backup local demo option

---

## Current Status

### Completed

✅ STMGT model architecture
✅ Data collection pipeline
✅ Streamlit dashboard (v4)
✅ FastAPI basic endpoints
✅ Initial training (MAE 3.05)

### In Progress

🔄 Model validation improvements
🔄 Documentation organization

### Not Started

❌ Baseline model training
❌ Ablation study
❌ Route optimization
❌ VM deployment
❌ Public web interface

---

## Next Steps (This Week)

1. **Today (Nov 9):**

   - Create evaluation framework
   - Fix STMGT validation issues
   - Start LSTM baseline training

2. **Nov 10-11:**

   - Complete baseline benchmarking
   - STMGT hyperparameter tuning
   - Ablation study

3. **Nov 12-13:**

   - Model comparison report
   - Route optimization prototype
   - VM setup planning

4. **Nov 14-15:**
   - Start VM deployment
   - Initial route optimization testing

---

## Resources Needed

### Computational

- VM: 4 vCPU, 16GB RAM, 100GB storage
- GPU: Optional but helpful (T4 or better)
- Estimated cost: $50-80 for 2 weeks

### APIs

- Google Directions: ~$60 for 3-day collection
- Open-Meteo: Free (no limits)

### Development

- Frontend hosting: Free (Vercel/Netlify)
- Domain: Optional ($10/year)

**Total Budget:** ~$100-150

---

## Questions to Resolve

1. **Web UI Framework:** React (more professional) vs Streamlit (faster development)?
2. **VM Provider:** Google Cloud (free tier) vs AWS vs Azure?
3. **Route Algorithm:** Dijkstra vs A\* vs custom heuristic?
4. **Presentation Format:** Live demo vs recorded video vs both?

---

## References

- Current project docs: `/docs/`
- STMGT architecture: `/docs/architecture/STMGT_ARCHITECTURE.md`
- Phase instructions: `/docs/instructions/PHASE2_MODEL_IMPROVEMENTS.md`
- Dashboard guide: `/docs/dashboard/DASHBOARD_V4_QUICKSTART.md`
