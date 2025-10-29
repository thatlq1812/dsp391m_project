# Traffic Forecast v5.0 Documentation

## Documentation Index

### Quick Start

1. **[README_V5.md](README_V5.md)** - Main English documentation (550 lines)

- System overview
- Architecture
- Installation
- Usage

2. **[HUONG_DAN_V5.md](HUONG_DAN_V5.md)** - Vietnamese quick start guide (300 lines)

- Hướng dẫn cài đặt
- Sử dụng cơ bản
- Ví dụ thực tế

3. **[COLLECTION_OPTIMIZATION_V5.1.md](COLLECTION_OPTIMIZATION_V5.1.md)** - v5.1 Optimization guide (500 lines) NEW

- Adaptive scheduling strategy
- Weather grid caching (32% reduction)
- Permanent topology cache
- Cost analysis ($21/day, 25% savings)
- Testing results and validation

### Implementation

3. **[BAO_CAO_CAI_TIEN_V5.md](BAO_CAO_CAI_TIEN_V5.md)** - Implementation report (400 lines)

- Changes from v4.0 to v5.0
- Technical improvements
- Performance comparisons

### Deployment

4. **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - GCP deployment guide (500 lines)

- VM setup instructions
- Cost analysis
- Monitoring procedures
- Troubleshooting

5. **[DEPLOYMENT_READY_SUMMARY.md](DEPLOYMENT_READY_SUMMARY.md)** - Complete system overview (600 lines)

- System specifications
- Configuration details
- File structure
- Quick commands

### Final Summary

6. **[HOAN_TAT_V5.md](HOAN_TAT_V5.md)** - Project completion summary (400 lines)

- Completion checklist
- Final specifications
- Next steps
- Academic deliverables

### Notebook Documentation

7. **[DATA_DASHBOARD.md](DATA_DASHBOARD.md)** - Data dashboard documentation

- EDA procedures
- Visualization guide

---

## Key Changes in v5.0 → v5.1

### v5.0 Core Improvements

- **Real API Only** - Removed all mock API fallbacks
- **Coverage Expansion** - Radius 1024m → 2048m (16x area)
- **Distance Filtering** - 200m minimum between nodes
- **Topology Caching** - One-time Overpass collection
- **Deployment Automation** - Full GCP VM deployment scripts

### v5.1 Optimizations NEW

- **Adaptive Scheduling** - Peak/off-peak/night modes
- **Weather Grid Cache** - 1km² cells (32% API reduction)
- **Permanent Topology Cache** - Never re-fetch static data
- **25% Cost Reduction** - $28/day → $21/day

### Data Collection

- **78 nodes** (filtered from 220)
- **234 edges** per collection (78 × 3 neighbors)
- **100% API success rate** verified
- **18 collections/day** (adaptive schedule)
- **Peak hours:** 30-min intervals (dense sampling)
- **Off-peak:** 90-min intervals (adequate coverage)
- **Night:**Skip completely (stable traffic)

### Cost Optimization

- **$21.06/day** for adaptive collection
- **$147 for 7 days** (25% savings vs v5.0)
- **~30,000 measurements** in 7 days (quality-optimized)

---

## Related Files

### Notebooks (../notebooks/)

- **CONTROL_PANEL.ipynb** - Main pipeline control
- **GCP_DEPLOYMENT.ipynb** - GCP VM management
- **DATA_DASHBOARD.ipynb** - EDA and visualization
- **ML_TRAINING.ipynb** - Model training

### Configuration (../configs/)

- **project_config.yaml** - v5.0 configuration

### Cache (../cache/)

- **overpass_topology.json** - 78 nodes cached

### Scripts (../scripts/)

- **deploy_gcp_vm.sh** - Automated deployment
- **feature_importance_analysis.py** - Feature analysis
- **cross_validation.py** - 5-fold CV

---

## Quick Start

### Local Testing

```bash
# Open control panel
jupyter notebook ../notebooks/CONTROL_PANEL.ipynb

# Test collection
conda run -n dsp python traffic_forecast/collectors/google/collector.py
```

### Deploy to GCP

```bash
# Open GCP deployment notebook
jupyter notebook ../notebooks/GCP_DEPLOYMENT.ipynb

# Or use automated script
./scripts/deploy_gcp_vm.sh
```

---

**Total Documentation:** ~3,250 lines (including v5.1 optimization)
**Languages:**English + Vietnamese
**Version:** 5.1 (Optimized)
**Date:**October 29, 2025
