# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic Forecasting System

**Spatio-Temporal Multi-Modal Graph Transformer** for real-time traffic speed forecasting in Ho Chi Minh City.**Spatio-Temporal Multi-Graph Transformer for traffic speed forecasting in Ho Chi Minh City.**

🎯 **Current Performance:** MAE 2.54 km/h (beats SOTA by 36-43%)

🚀 **Status:** Production-ready with REST API and CLI

📊 **Coverage:** 62 nodes, 144 road segments, district-level forecasting---

---## Quick Start

## Quick Start (5 Minutes)### 1. Environment

### 1. Installation```bash

conda env create -f environment.yml

````bashconda activate dsp

# Clone repositorypip install -e .

git clone https://github.com/thatlq1812/dsp391m_project.git```

cd dsp391m_project

# Create environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .
```

All scripts assume the Conda environment name `dsp` recorded in `.env`.

### 2. CLI Tool (Recommended)

**Simple, fast command-line interface for all operations:**

```bash
# For Git Bash on Windows - use wrapper script
cd /d/UNI/DSP391m/project
./stmgt.sh --help

# Common commands
./stmgt.sh model list              # List all models
./stmgt.sh api start               # Start API server
./stmgt.sh data info               # Show dataset info
./stmgt.sh train status            # Training status

# Add to PATH (optional)
export PATH="$PATH:/d/UNI/DSP391m/project"
stmgt --help
```

**Full documentation:** `docs/01_getting_started/CLI.md`

### 3. Web Interface

**Interactive traffic visualization and route planning:**

```bash
# Start API server first
./stmgt.sh api start

# Open in browser
# http://localhost:8000/route_planner.html
```

**Features:**

- Real-time traffic visualization with gradient colors
- Route planning with 3 algorithms (fastest/shortest/balanced)
- Interactive map with Leaflet.js

### 4. Training via CLI

```bash
# Complete training workflow
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py \
  --config configs/training_config.json

# See: docs/03_models/TRAINING_WORKFLOW.md for step-by-step guide
```

The trainer loads configuration through `traffic_forecast/core/config_loader.py` and validates datasets with `traffic_forecast/data/validators.py` before any run begins.

---

## Current Performance (Nov 15, 2025)

### STMGT V3 Production Model

**Performance Metrics:**

- **Test MAE:** 2.54 km/h (best among all baselines)
- **Test RMSE:** 3.01 km/h
- **R² Score:** 0.85 (explains 85% of variance)
- **MAPE:** 14.17%
- **CRPS:** 1.94 (probabilistic score)
- **Coverage@80:** 81.94% (well-calibrated uncertainty)
- **Model Capacity:** 680K parameters (proven optimal)

**Key Achievements:**

- 5 capacity experiments confirmed 680K params optimal (U-shaped curve)
- Training improvements (dropout 0.25, LR 0.0008, regularization) beat baseline
- Best-in-class uncertainty quantification (81.94% coverage)
- Production-ready with comprehensive testing and verification

**Artifacts:**

- Model checkpoint: `outputs/stmgt_baseline_1month_20251115_132552/best_model.pt`
- Final Report: `docs/05_final_report/DSP391m_final_report_V3.pdf` (85 pages)
- Documentation: See [Documentation Map](#documentation-map) below

### Benchmark Comparison

| Model                    | MAE (km/h) | vs STMGT    |
| ------------------------ | ---------- | ----------- |
| Naive (last value)       | 7.20       | +183% worse |
| LSTM Sequential          | 4.85       | +91% worse  |
| GCN Graph                | 3.91       | +54% worse  |
| GraphWaveNet (SOTA 2019) | 3.95       | +56% worse  |
| **STMGT V3**             | **2.54**   | **BEST** ✓  |

---

## Documentation Map

### Quick Access

**[Final Report](docs/05_final_report/DSP391m_final_report_V3.pdf)** - Complete 85-page IEEE format report
**[Project Status](PROJECT_STATUS.txt)** - Current verification status
**[Changelog](docs/CHANGELOG.md)** - Complete project history (4900+ lines)

### For Users

**Getting Started:**

- [CLI Guide](docs/01_getting_started/CLI.md) - Command-line interface
- [API Guide](docs/01_getting_started/API.md) - REST API documentation
- [Deployment Guide](docs/01_getting_started/DEPLOYMENT.md) - Production deployment

**Using the System:**

- REST API: `http://localhost:8000/docs` (Swagger UI)
- Web Interface: `http://localhost:8000/route_planner.html`
- CLI Tool: `./stmgt.sh --help`

### For Developers

**Data Pipeline:**

- [Data Overview](docs/02_data/DATA.md) - Dataset structure and specs
- [Augmentation](docs/02_data/AUGMENTATION.md) - Data augmentation strategies
- [Super Dataset](docs/02_data/super_dataset/) - 1-year simulation dataset

**Model Development:**

- [Model Overview](docs/03_models/MODEL.md) - Architecture and capabilities
- [Training Workflow](docs/03_models/TRAINING_WORKFLOW.md) - Complete training pipeline
- [Architecture Details](docs/03_models/architecture/STMGT_ARCHITECTURE.md) - Design decisions

**Evaluation:**

- [Metrics Verification](docs/04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md) - Cross-model comparison
- [GraphWaveNet Investigation](docs/04_evaluation/graphwavenet/) - Baseline analysis

### Project Information

- [Final Report (LaTeX)](docs/05_final_report/) - Source files and figures
- [Documentation Index](docs/README.md) - Complete documentation structure

---

## Project Structure

```
project/
├── docs/
│   ├── 01_getting_started/      # User guides (CLI, API, Deployment)
│   ├── 02_data/                 # Data documentation
│   ├── 03_models/               # Model and training docs
│   ├── 04_evaluation/           # Evaluation reports
│   ├── 05_final_report/         # IEEE format report (85 pages)
│   │   ├── DSP391m_final_report_V3.pdf
│   │   ├── DSP391m_final_report_V3.tex
│   │   ├── references.bib       # Bibliography (20+ citations)
│   │   └── figures/             # All 20 figures
│   └── CHANGELOG.md             # Complete project history
├── traffic_forecast/
│   ├── core/                    # Config loader, registry, logging
│   ├── data/                    # Dataset definitions and validators
│   ├── models/                  # STMGT architecture
│   └── utils/                   # Shared utilities
├── scripts/
│   ├── data/                    # Data collection and preprocessing
│   ├── training/                # Training scripts
│   ├── visualization/           # Figure generation (20 figures)
│   ├── evaluation/              # Model evaluation
│   └── demo/                    # Demo scripts
├── configs/
│   ├── training/                # Training configurations
│   ├── data/                    # Data pipeline configs
│   └── project_config.yaml      # Project-level config
├── data/
│   ├── processed/               # Preprocessed parquet files
│   └── runs/                    # Raw data collection runs
├── outputs/
│   └── stmgt_baseline_1month_20251115_132552/  # Production model
├── traffic_api/                 # FastAPI REST API
├── stmgt.sh                     # CLI wrapper script
└── README.md                    # This file
```

---

## Features

### Core Capabilities

**Traffic Forecasting:**

- 3-hour ahead prediction (15-minute intervals)
- Probabilistic forecasts with calibrated uncertainty
- 62 nodes, 144 road segments in Ho Chi Minh City

**Model Architecture:**

- Graph Attention Network (GATv2) for spatial dependencies
- Transformer for temporal patterns
- Gaussian Mixture Model (K=5) for uncertainty quantification
- 680K parameters (optimal capacity)

**Production Features:**

- REST API with Swagger documentation
- Command-line interface (CLI)
- Interactive web dashboard
- Route planning (fastest/shortest/balanced)

---

## Usage Examples

### Example 1: REST API

```python
import requests

# Get current traffic
response = requests.get("http://localhost:8000/traffic/current")
traffic = response.json()

# Get 3-hour forecast
response = requests.get(
    "http://localhost:8000/traffic/forecast",
    params={"edge_id": "123-456", "horizon_minutes": 180}
)
forecast = response.json()

# Plan route
response = requests.post(
    "http://localhost:8000/route/plan",
    json={
        "origin_node": 123,
        "dest_node": 456,
        "algorithm": "fastest"
    }
)
route = response.json()
```

### Example 2: CLI Operations

```bash
# List all models with performance
./stmgt.sh model list

# Check dataset info
./stmgt.sh data info

# Start API server
./stmgt.sh api start --port 8000

# Monitor training
./stmgt.sh train status
```

### Example 3: Training Pipeline

```bash
# Step 1: Preprocess data
python scripts/data/preprocess_runs.py

# Step 2: Train model
python scripts/training/train_stmgt.py \
  --config configs/train_normalized_v3.json

# Step 3: Evaluate
python scripts/evaluation/evaluate_model.py \
  --model-dir outputs/stmgt_baseline_1month_20251115_132552

# See: docs/03_models/TRAINING_WORKFLOW.md for complete guide
```

---

## Recent Updates

### November 15, 2025 - Final Verification

**Project Verification Complete:**

- Bibliography citations fixed (all [?] → [1], [2], [3]...)
- All 20 figures regenerated with current code
- Code cleanup (removed 12 TODO comments)
- Final report compiled (85 pages, 6.0 MB PDF)

**Quality Assurance:**

- All figures consistent with V3 performance (MAE 2.54, R² 0.85)
- No undefined references in report
- Clean codebase ready for review
- Production-ready status confirmed

**Documentation:**

- Created `PROJECT_STATUS.txt` - Detailed verification status
- Updated `docs/CHANGELOG.md` - Complete project history
- All 20 figures verified and embedded in report

### November 12, 2025 - Documentation Consolidation

**Report Approval:**

- Final report V3 approved by supervisor
- Fixed all internal contradictions
- Updated all tables to V3 values
- Bibliography section added with 20+ citations

**Code Quality:**

- Comprehensive test suite
- Production-ready API/CLI
- Clean project structure
- All baselines verified

---

## Requirements

**Python:** 3.10+
**Hardware:** NVIDIA GPU recommended (CPU supported)
**OS:** Linux, macOS, Windows (Git Bash)

**Key Dependencies:**

- PyTorch 2.0+
- PyTorch Geometric
- FastAPI
- Pandas, NumPy
- Matplotlib, Seaborn

**Full list:** See `environment.yml` and `requirements.txt`

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{stmgt2025,
  title={STMGT: Spatio-Temporal Multi-Modal Graph Transformer for Traffic Forecasting},
  author={Le Quang, THAT and Le Minh, HUNG and Nguyen Quy, TOAN},
  year={2025},
  institution={FPT University},
  note={DSP391m Capstone Project}
}
```

---

## Contributing

### Development Workflow

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test: `pytest tests/`
4. Update documentation
5. Update `docs/CHANGELOG.md`
6. Submit pull request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings and type hints
- Write unit tests for new features
- Keep code modular and clean
- Document major changes

---

## License

MIT License - See LICENSE file for details

---

## Support

**Issues:** Open GitHub issue at [thatlq1812/dsp391m_project](https://github.com/thatlq1812/dsp391m_project)
**Documentation:** See [docs/](docs/) directory
**Final Report:** [docs/05_final_report/DSP391m_final_report_V3.pdf](docs/05_final_report/DSP391m_final_report_V3.pdf)
**Contact:** THAT Le Quang - thatlq1812

---

## Acknowledgments

- **FPT University** - Academic support and supervision
- **OpenStreetMap & Google Maps API** - Road network and traffic data
- **PyTorch & PyTorch Geometric** - Deep learning frameworks
- **Research Community** - Graph neural networks and spatio-temporal modeling

---
````
