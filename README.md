# STMGT Traffic Forecasting Project

**Spatio-Temporal Multi-Graph Transformer for Traffic Speed Forecasting**

[![Status](https://img.shields.io/badge/status-production-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-orange)]()

---

## 🚀 Quick Start

### Training
```bash
conda activate dsp
python scripts/training/train_stmgt_v2.py
```

### Monitoring
```bash
python scripts/monitoring/monitor_training.py
```

---

## 📊 Current Performance

- **MAE:** 3.05 km/h ✅ (Target: <5.0)
- **R²:** 0.769 ✅ (Target: >0.45)
- **MAPE:** 22.98% ✅ (Target: <30%)

---

## 📚 Documentation

**Complete documentation is in the `docs/` directory:**

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Standard workflows
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current status
- **[docs/STMGT_ARCHITECTURE.md](docs/STMGT_ARCHITECTURE.md)** - Model architecture
- **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Development guide
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Training guide

---

## 📁 Project Structure

```
project/
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
│   └── processed/      # Processed datasets
│
├── outputs/            # Training outputs
├── docs/              # Documentation
└── archive/           # Archived files
```

---

## 🛠️ Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .
```

---

## 📖 Learn More

- **Architecture:** [docs/STMGT_ARCHITECTURE.md](docs/STMGT_ARCHITECTURE.md)
- **Workflows:** [docs/WORKFLOW.md](docs/WORKFLOW.md)
- **Development:** [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)

---

## 👥 Team

DSP391m Team - Traffic Forecasting Project

**Repository:** https://github.com/thatlq1812/dsp391m_project

---

*For detailed documentation, see the `docs/` directory.*
