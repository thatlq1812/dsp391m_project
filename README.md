# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic Forecasting Project

**Spatio-Temporal Multi-Graph Transformer for traffic speed forecasting in Ho Chi Minh City.**

The repository now reflects Phase 1 completion: dataset validation is enforced across collection, augmentation, and training flows; registry configuration is schema-validated via Pydantic; and the Streamlit dashboard wiring targets the new helper utilities. See the documentation index at `docs/INDEX.md` and the consolidated research at `docs/STMGT_RESEARCH_CONSOLIDATED.md`.

---

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate dsp
pip install -e .
```

All scripts assume the Conda environment name `dsp` recorded in `.env`.

### 2. Interactive Dashboard (Streamlit V4)

```bash
conda activate dsp
streamlit run dashboard/Dashboard.py
# Visit http://localhost:8501
```

Key pages:

- `pages/2_VM_Management.py`: VM orchestration shortcuts and health checks.
- `pages/5_Data_Collection.py`: Collection schedule, validation status, and Overpass cache previews.
- `pages/6_Data_Overview.py`: Dataset profiler pulling from the validated parquet registry.
- `pages/9_Training_Control.py`: Launch STMGT training via the schema-checked registry entries.
- `pages/11_Predictions.py`: Batch inference using the currently selected checkpoint.

### 3. Training via CLI

```bash
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/training_config.json
```

The trainer loads configuration through `traffic_forecast/core/config_loader.py` and validates datasets with `traffic_forecast/data/validators.py` before any run begins.

### 4. Analytical Reports

```bash
conda run -n dsp python scripts/analysis/analyze_augmentation_strategy.py --dataset data/processed/all_runs_combined.parquet
conda run -n dsp python scripts/analysis/analyze_data_distribution.py --dataset data/processed/all_runs_combined.parquet
```

These scripts now share CLI patterns, automatic data-root resolution, and dataset validation hooks.

---

## Current Benchmarks (Oct 31, 2025)

- **Validation MAE:** 3.05 km/h
- **Validation R²:** 0.77
- **Validation MAPE:** 22.9%

Benchmark artifacts live under `outputs/stmgt_v2_20251101_012257/` and are surfaced in the dashboard registry.

---

## Project Structure

```
project/
├── dashboard/
│   ├── Dashboard.py
│   └── pages/            # Streamlit V4 page suite (registry, training, inference)
├── traffic_forecast/
│   ├── core/             # Config loader, registry validation, logging helpers
│   ├── data/             # Dataset definitions and validators
│   ├── models/           # STMGT architecture and components
│   └── utils/            # Shared utilities
├── scripts/
│   ├── analysis/         # CLI-first analytics (augmentation, distribution reports)
│   ├── data/             # Collection, augmentation, and maintenance tasks
│   ├── monitoring/       # Dashboard and training telemetry
│   └── training/         # Training entrypoints
├── configs/              # Environment, registry, and training configuration
├── data/                 # Raw and processed datasets (validated at runtime)
├── docs/                 # Project documentation (see below)
├── outputs/              # Training artifacts and evaluation plots
└── models/training_runs/ # Legacy training archives
```

---

## Documentation Map

Documentation lives in `docs/` (all files include Maintainer metadata):

- `docs/CHANGELOG.md` – Project-level change log updated for Phase 1 completion.
- `docs/INDEX.md` – Canonical index of all documentation.
- `docs/WORKFLOW.md` – Step-by-step operations workflow, now referencing validation checkpoints.
- `docs/DASHBOARD_V4_QUICKSTART.md` – Task-oriented dashboard onboarding.
- `docs/DASHBOARD_V4_REFERENCE.md` – Detailed dashboard feature catalogue.
- `docs/PROCESSED_DATA_PIPELINE.md` – Collection-to-parquet pipeline with validator requirements.
- `docs/VM_CONFIG_INTEGRATION.md` – VM provisioning and sync notes.
- `docs/README_SETUP.md` – Environment bootstrap and task runner guidance.
- `docs/README_FULLREPORT.md` – Full technical report and experiment log.
- `docs/STMGT_RESEARCH_CONSOLIDATED.md` – Merged research findings (Claude, Gemini, OpenAI).
- `docs/TaskofNov02.md` – Phase plan and execution checklist (latest status recorded).

---

## Contributing and Testing

- Format and lint using the standard tools specified in `pyproject.toml`.
- Run targeted tests with `conda run -n dsp pytest tests/test_stmgt_utils.py` after modifying core utilities.
- Record notable changes in `docs/CHANGELOG.md` and update relevant workflow or quickstart guides.

---

## Support

For issues or feature requests please open a GitHub issue on [thatlq1812/dsp391m_project](https://github.com/thatlq1812/dsp391m_project).

---

_Detailed documentation and run logs are maintained in the `docs/` and `outputs/` directories._
