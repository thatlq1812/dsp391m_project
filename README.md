# STMGT Traffic Forecasting Project

**Spatio-Temporal Multi-Graph Transformer for Traffic Speed Forecasting**

[![Status](https://img.shields.io/badge/status-production-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-orange)]()
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red)]()

---

## Quick Start

### Interactive Dashboard (NEW v3.0)

```bash
conda activate dsp
streamlit run dashboard/dashboard.py
# Access at http://localhost:8501
```

**Dashboard Features:**

- **Data Overview:** Collection stats, augmentation summary, data quality
- **Data Augmentation:** Configure parameters, validate quality, run augmentation
- **Visualization:** Speed analysis, temporal patterns, correlation heatmap
- **Training Monitor:** Start/stop training, tune hyperparameters, export reports
- **Predictions:** Real-time forecast, scenarios, alerts, export results

### Training (CLI)

```bash
conda activate dsp
python scripts/training/train_stmgt.py
```

### Monitoring

```bash
python scripts/monitoring/monitor_training.py
```

---

## Current Performance

- **MAE:** 3.05 km/h -OK- (Target: <5.0)
- **R²:** 0.769 -OK- (Target: >0.45)
- **MAPE:** 22.98% -OK- (Target: <30%)

---

## Dashboard V3 Highlights

**Page 2: Data Augmentation**

- Interactive noise & interpolation controls
- Strategy comparison (Basic vs Extreme)
- KS test & correlation validation
- One-click augmentation execution

**Page 4: Training Control**

- Subprocess-based training (Start/Stop)
- Live resource monitoring (CPU/GPU/Memory)
- Hyperparameter grid/random search
- HTML report export to `docs/report/`

**Page 5: Advanced Predictions**

- 12-step forecast with uncertainty (GMM)
- Weather scenario simulation
- Congestion alert system
- Multi-format export (CSV/Parquet/JSON)

---

## Documentation

**Complete documentation is in the `docs/` directory:**

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Standard workflows
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current status
- **[docs/STMGT_ARCHITECTURE.md](docs/STMGT_ARCHITECTURE.md)** - Model architecture
- **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Development guide
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Training guide
- **[docs/DASHBOARD_V3_COMPLETE.md](docs/DASHBOARD_V3_COMPLETE.md)** - Dashboard V3 guide (NEW)
- **[docs/DASHBOARD_V3_QUICKREF.md](docs/DASHBOARD_V3_QUICKREF.md)** - Quick reference (NEW)

---

## Project Structure

```
project/
├── dashboard/           # Streamlit Dashboard V3 (NEW)
│   ├── dashboard.py    # Main entry
│   └── pages/         # 5 interactive pages
│       ├── 1_Data_Overview.py
│       ├── 2_Data_Augmentation.py    # NEW
│       ├── 3_Visualization.py        # ENHANCED
│       ├── 4_Training_Monitor.py     # ENHANCED
│       └── 5_Predictions.py          # ENHANCED
│
├── traffic_forecast/      # Core library
│   ├── models/           # STMGT implementation
│   ├── data/            # Dataset & DataLoader
│   └── augmentation/    # Data augmentation
│
├── scripts/
│   ├── training/        # Training scripts
│   ├── data/           # Data processing
│   ├── analysis/       # Analysis tools
│   └── monitoring/     # Monitoring tools
│
├── data/
│   ├── processed/      # Processed datasets
│   └── predictions/    # Prediction exports (NEW)
│
├── docs/
│   └── report/        # Training reports (NEW)
│
├── outputs/            # Training outputs
└── archive/           # Archived files
```

---

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .
```

---

## Learn More

- **Architecture:** [docs/STMGT_ARCHITECTURE.md](docs/STMGT_ARCHITECTURE.md)
- **Workflows:** [docs/WORKFLOW.md](docs/WORKFLOW.md)
- **Development:** [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)

---

## Team

DSP391m Team - Traffic Forecasting Project

**Repository:** https://github.com/thatlq1812/dsp391m_project

---

_For detailed documentation, see the `docs/` directory._
