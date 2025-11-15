# Maintainer Profile# Maintainer Profile

**Name:** THAT Le Quang**Name:** THAT Le Quang

- **Role:** AI & DS Major Student- **Role:** AI & DS Major Student

- **GitHub:** [thatlq1812]- **GitHub:** [thatlq1812]

---

# STMGT Traffic Forecasting System# STMGT Traffic Forecasting Project

**Spatio-Temporal Multi-Modal Graph Transformer** for real-time traffic speed forecasting in Ho Chi Minh City.**Spatio-Temporal Multi-Graph Transformer for traffic speed forecasting in Ho Chi Minh City.**

🎯 **Current Performance:** MAE 3.08 km/h (beats SOTA by 21-28%) The repository now reflects Phase 1 completion: dataset validation is enforced across collection, augmentation, and training flows; registry configuration is schema-validated via Pydantic; and the Streamlit dashboard wiring targets the new helper utilities. See the documentation index at `docs/INDEX.md` and the consolidated research at `docs/STMGT_RESEARCH_CONSOLIDATED.md`.

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

All scripts assume the Conda environment name `dsp` recorded in `.env`.

# Create environment

conda env create -f environment.yml### 2. CLI Tool (NEW - Replaces Dashboard)

conda activate dsp

**Simple, fast command-line interface for all operations:**

# Install package

pip install -e .```bash

```# For Git Bash on Windows - use wrapper script

cd /d/UNI/DSP391m/project

### 2. First Run - API Server./stmgt.sh --help



```bash# Common commands

# Start FastAPI server./stmgt.sh model list              # List all models

python traffic_api/main.py./stmgt.sh api start               # Start API server

./stmgt.sh data info               # Show dataset info

# Access API./stmgt.sh train status            # Training status

curl http://localhost:8000/health

# Add to PATH (optional)

# Web interfaceexport PATH="$PATH:/d/UNI/DSP391m/project"

open http://localhost:8000/route_planner.htmlstmgt --help

````

### 3. First Run - CLI Tool**Full documentation:** `docs/guides/CLI_USER_GUIDE.md`

````bash### 3. Web Interface

# Show help

python tools/stmgt_cli.py --help**Interactive traffic visualization and route planning:**



# List models```bash

python tools/stmgt_cli.py model list# Start API server first

./stmgt.sh api start

# Check data

python tools/stmgt_cli.py data info# Open in browser

```# http://localhost:8000/route_planner.html

````

### 4. Train Model

Features:

````bash

# Complete training workflow- Real-time traffic visualization with gradient colors

python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json- Route planning with 3 algorithms (fastest/shortest/balanced)

- Interactive map with Leaflet.js

# See: TRAINING_WORKFLOW.md for step-by-step guide

```### 4. Training via CLI



---```bash

conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/training_config.json

## Performance Highlights```



### Latest Results (November 12, 2025)The trainer loads configuration through `traffic_forecast/core/config_loader.py` and validates datasets with `traffic_forecast/data/validators.py` before any run begins.



**Baseline (No Augmentation):**### 4. Analytical Reports

- Test MAE: 3.1068 km/h

- Test R²: 0.8157```bash

- Test MAPE: 20.11%conda run -n dsp python scripts/analysis/analyze_augmentation_strategy.py --dataset data/processed/all_runs_combined.parquet

conda run -n dsp python scripts/analysis/analyze_data_distribution.py --dataset data/processed/all_runs_combined.parquet

**With SafeTrafficAugmentor:**```

- Test MAE: 3.0774 km/h ✓ (0.95% improvement)

- Test R²: 0.8175 ✓These scripts now share CLI patterns, automatic data-root resolution, and dataset validation hooks.

- Test MAPE: 19.68% ✓

- **Result:** Augmentation works, no data leakage---



### Benchmark Comparison## Current Performance (Nov 10, 2025) - V3 Production



| Model                    | MAE (km/h) | vs STMGT       |**STMGT V3** is the current production model, achieved through systematic capacity exploration and training refinement:

| ------------------------ | ---------- | -------------- |

| Naive (last value)       | 7.20       | +134% worse    |- **Test MAE:** 3.0468 km/h (1.1% better than V1)

| LSTM Sequential          | 4.42-4.85  | +43-57% worse  |- **Test RMSE:** 4.5198 km/h

| GCN Graph                | 3.91       | +27% worse     |- **R² Score:** 0.8161

| GraphWaveNet (SOTA 2019) | 3.95       | +28% worse     |- **MAPE:** 18.89%

| **STMGT V1**             | **3.08**   | **BEST** ✓     |- **Coverage@80:** 86.0% (excellent calibration)

- **Model Capacity:** 680K parameters (proven optimal)

---

**Key Achievements:**

## Documentation

- 5 capacity experiments confirmed 680K params optimal (U-shaped curve)

### For Users- Training improvements (dropout, LR, regularization) beat baseline without architectural changes

- Best-in-class uncertainty quantification (86% coverage)

#### Getting Started- Production-ready with comprehensive testing

- 📖 **[Installation & Setup](#quick-start-5-minutes)** - 5-minute quick start

- 🛠️ **[CLI Guide](docs/CLI.md)** - Command-line interface**Model artifacts:** `outputs/stmgt_v2_20251110_123931/`

- 🌐 **[API Reference](docs/API.md)** - REST API documentation**Documentation:** `docs/V3_DESIGN_RATIONALE.md`, `docs/CHANGELOG.md`

- 🚀 **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment

---

#### Using the System

- **REST API:** `http://localhost:8000/docs` (interactive Swagger UI)## Project Structure

- **Web Interface:** `http://localhost:8000/route_planner.html` (route planning)

- **CLI Tool:** `python tools/stmgt_cli.py --help` (operations)```

project/

### For Developers├── dashboard/

│   ├── Dashboard.py

#### Training & Development│   └── pages/            # Streamlit V4 page suite (registry, training, inference)

- 📊 **[Training Workflow](TRAINING_WORKFLOW.md)** - Complete training pipeline├── traffic_forecast/

- 🎓 **[Training Guide](docs/TRAINING.md)** - Advanced training options│   ├── core/             # Config loader, registry validation, logging helpers

- 📈 **[Augmentation Guide](docs/AUGMENTATION.md)** - Data augmentation│   ├── data/             # Dataset definitions and validators

- 🗃️ **[Data Guide](docs/DATA.md)** - Data concepts and schemas│   ├── models/           # STMGT architecture and components

│   └── utils/            # Shared utilities

#### Architecture & Research├── scripts/

- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** - System design│   ├── analysis/         # CLI-first analytics (augmentation, distribution reports)

- 🧠 **[Model Overview](docs/MODEL.md)** - Model capabilities and limits│   ├── data/             # Collection, augmentation, and maintenance tasks

- 🔬 **[Research Summary](docs/RESEARCH.md)** - Academic contributions│   ├── monitoring/       # Dashboard and training telemetry

- 🐛 **[Critical Fixes](docs/FIXES.md)** - Data leakage fix│   └── training/         # Training entrypoints

├── configs/              # Environment, registry, and training configuration

### Project Information├── data/                 # Raw and processed datasets (validated at runtime)

- 📝 **[Changelog](docs/CHANGELOG.md)** - Version history├── docs/                 # Project documentation (see below)

- 📄 **[Final Report](docs/final_report/)** - Complete project report├── outputs/              # Training artifacts and evaluation plots

└── models/training_runs/ # Legacy training archives

---```



## Project Structure---



```## Documentation Map

project/

├── README.md                       # This fileDocumentation lives in `docs/` (all files include Maintainer metadata):

├── TRAINING_WORKFLOW.md            # Complete training guide

├── traffic_forecast/               # Main package**Core Documentation:**

│   ├── core/                       # Config, artifacts, reporting

│   ├── data/                       # Datasets and augmentation- `docs/CHANGELOG.md` – Project-level change log with full history

│   │   ├── stmgt_dataset.py        # PyTorch dataset- `docs/INDEX.md` – Canonical index of all documentation (reorganized Nov 2025)

│   │   └── augmentation_safe.py    # Leak-free augmentation- `docs/STMGT_ARCHITECTURE.md` – Model architecture and design decisions

│   ├── models/stmgt/               # STMGT model- `docs/STMGT_DATA_IO.md` – Data schemas and I/O specifications

│   │   ├── model.py                # Architecture- `docs/STMGT_RESEARCH_CONSOLIDATED.md` – Merged research findings (Claude, Gemini, OpenAI)

│   │   ├── train.py                # Training loop

│   │   └── evaluate.py             # Evaluation**Guides (`docs/guides/`):**

│   └── utils/                      # Utilities

├── traffic_api/                    # FastAPI server- `docs/guides/README_SETUP.md` – Environment setup and installation

│   ├── main.py                     # API entry point- `docs/guides/WORKFLOW.md` – Development workflow and operations

│   └── route_planner.html          # Web interface- `docs/guides/PROCESSED_DATA_PIPELINE.md` – Data collection and processing pipeline

├── tools/

│   └── stmgt_cli.py                # CLI tool**Quality Audits (`docs/audits/`):**

├── scripts/

│   ├── data/                       # Data collection/preprocessing- `docs/audits/PROJECT_TRANSPARENCY_AUDIT.md` – Comprehensive project evaluation (8.7/10)

│   │   ├── preprocess_runs.py      # JSON → Parquet- `docs/audits/GRAPHWAVENET_TRANSPARENCY_AUDIT.md` – Baseline model analysis

│   │   └── augment_safe.py         # Safe augmentation

│   ├── training/                   # Training scripts**Dashboard:**

│   │   └── train_stmgt.py          # Main training script

│   └── analysis/                   # Analytics scripts- `docs/DASHBOARD_V4_QUICKSTART.md` – Dashboard quick start guide

├── configs/                        # Configuration files- `docs/DASHBOARD_V4_REFERENCE.md` – Complete dashboard reference

│   ├── train_normalized_v3.json    # Training config

│   ├── augmentation_config.json    # Augmentation config**Phase Instructions (`docs/instructions/`):**

│   └── model_registry.json         # Model registry

├── data/- Phase 1-4 implementation guides and task lists

│   ├── runs/                       # Raw data (JSON)

│   └── processed/                  # Processed data (Parquet)**Archived Content:**

├── docs/                           # Documentation

│   ├── CLI.md                      # CLI reference- `archive/README.md` – Experimental code, old training runs, retention policy

│   ├── API.md                      # API reference

│   ├── TRAINING.md                 # Training guide---

│   ├── DATA.md                     # Data guide

│   ├── MODEL.md                    # Model overview## Contributing and Testing

│   ├── ARCHITECTURE.md             # Architecture

│   ├── DEPLOYMENT.md               # Deployment- Format and lint using the standard tools specified in `pyproject.toml`.

│   ├── AUGMENTATION.md             # Augmentation- Run targeted tests with `conda run -n dsp pytest tests/test_stmgt_utils.py` after modifying core utilities.

│   ├── RESEARCH.md                 # Research- Record notable changes in `docs/CHANGELOG.md` and update relevant workflow or quickstart guides.

│   ├── FIXES.md                    # Critical fixes

│   ├── CHANGELOG.md                # Project history---

│   ├── final_report/               # Final report

│   └── archive/                    # Historical docs## Support

└── outputs/                        # Training outputs

    └── stmgt_v2_TIMESTAMP/         # Model checkpointsFor issues or feature requests please open a GitHub issue on [thatlq1812/dsp391m_project](https://github.com/thatlq1812/dsp391m_project).

````

---

---

_Detailed documentation and run logs are maintained in the `docs/` and `outputs/` directories._

## Features

### 🎯 Core Features

**Traffic Forecasting:**

- 3-hour ahead prediction (15-minute intervals)
- Probabilistic forecasts with uncertainty quantification
- 62 nodes, 144 road segments coverage

**Model Architecture:**

- Graph Neural Network (3-hop spatial propagation)
- Transformer attention (temporal patterns)
- Gaussian Mixture Model (uncertainty)
- 680K parameters (optimal capacity)

**Data Pipeline:**

- JSON collection → Parquet preprocessing
- Leak-free data augmentation
- Temporal split validation
- Automated data validation

### 🚀 Production Features

**REST API:**

- `/health` - Health check
- `/traffic/current` - Current traffic state
- `/traffic/forecast` - 3-hour forecast
- `/route/plan` - Route optimization

**CLI Tool:**

- Model management (list, info, compare)
- API control (start, stop, status)
- Data operations (info, validate)
- Training monitoring

**Web Interface:**

- Interactive traffic map
- Route planning (fastest/shortest/balanced)
- Real-time visualization
- Leaflet.js integration

---

## Usage Examples

### Example 1: Train Model from Scratch

```bash
# Step 1: Preprocess data
python scripts/data/preprocess_runs.py

# Step 2: Train baseline
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json

# Step 3: Augment data (optional)
python scripts/data/augment_safe.py --preset moderate

# Step 4: Train with augmentation
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json

# See: TRAINING_WORKFLOW.md for detailed steps
```

### Example 2: API Usage

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

### Example 3: CLI Operations

```bash
# List all models with performance
python tools/stmgt_cli.py model list

# Compare two model runs
python tools/stmgt_cli.py model compare \
    outputs/stmgt_v2_20251112_085612 \
    outputs/stmgt_v2_20251112_091929

# Check dataset info
python tools/stmgt_cli.py data info

# Start API server
python tools/stmgt_cli.py api start --port 8000
```

---

## Recent Updates

### November 12, 2025

**Augmentation Experiment Success:**

- Baseline: MAE 3.1068 km/h
- Augmented: MAE 3.0774 km/h (0.95% improvement)
- Validated no data leakage

**Documentation Consolidation:**

- Reduced from 50+ docs to 14 essential docs
- Created master README (this file)
- Archived historical documentation
- Clear navigation structure

**Training Workflow:**

- Created comprehensive TRAINING_WORKFLOW.md
- Step-by-step pipeline from preprocessing to evaluation
- SafeTrafficAugmentor integration guide

### November 10, 2025

**STMGT V3 Production Release:**

- MAE 3.0468 km/h (best to date)
- 5 capacity experiments (350K-1.15M params)
- Confirmed 680K parameters optimal
- Excellent uncertainty calibration (86%)

---

## Requirements

**Python:** 3.9+
**Hardware:** NVIDIA GPU recommended (CPU supported)
**OS:** Linux, macOS, Windows (Git Bash)

**Key Dependencies:**

- PyTorch 2.0+
- PyTorch Geometric
- FastAPI
- Pandas
- NumPy

**Full list:** See `environment.yml` and `requirements.txt`

---

## Contributing

### Development Workflow

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes
4. Run tests: `pytest tests/`
5. Update CHANGELOG.md
6. Submit pull request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write unit tests
- Update documentation

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{stmgt2025,
  title={STMGT: Spatio-Temporal Multi-Modal Graph Transformer for Traffic Forecasting},
  author={Le Quang, THAT},
  year={2025},
  institution={FPT University},
  note={DSP391m Project}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support

**Issues:** Open GitHub issue at [thatlq1812/dsp391m_project](https://github.com/thatlq1812/dsp391m_project)
**Documentation:** See [docs/](docs/) directory
**Contact:** THAT Le Quang - thatlq1812

---

## Acknowledgments

- **FPT University** - Academic support
- **OpenStreetMap** - Road network data
- **PyTorch Team** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library

---

## Data Pipeline

- Baseline dataset: `python scripts/data/run_data_pipeline.py baseline`
- Augmented dataset: `python scripts/data/run_data_pipeline.py augmented`
- Quick validate both: `python scripts/data/04_analysis/quick_validate_datasets.py --preset both`

## Training (Defaults Wired)

- `python scripts/training/train_stmgt.py` (defaults to `configs/training/stmgt_baseline_1month.json` and `data/processed/baseline_1month.parquet`)
- `python scripts/training/train_stmgt.py --config configs/training/stmgt_augmented_1year.json`
- `python scripts/training/train_lstm_baseline.py --dataset-preset baseline|augmented`
- `python scripts/training/train_graphwavenet_baseline.py --dataset-preset baseline|augmented`

---

**Last Updated:** November 12, 2025  
**Version:** 1.0.0  
**Status:** Production Ready
