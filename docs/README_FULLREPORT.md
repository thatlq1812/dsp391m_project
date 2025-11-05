# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic Forecasting Full Report

## Project Overview

This project delivers a production-grade traffic speed forecasting system for Ho Chi Minh City built around the Spatio-Temporal Multi-Graph Transformer (STMGT) architecture. It combines multi-source data collection, graph-based preprocessing, a mixture-density transformer model, and a Streamlit dashboard for analysis and monitoring. The current codebase targets repeatable experimentation and operational deployment via Conda-managed environments and reproducible configuration files.

## Data Acquisition and Augmentation Pipeline

- **Collection stack** (`scripts/data/`): orchestrates Overpass topology pulls, weather ingestion, and Google Directions crawling under an adaptive scheduler defined in `configs/project_config.yaml`.
- **Storage layout**: raw artifacts accumulate under `data/runs/`, consolidated parquet datasets in `data/processed/`, and cached graph metadata in `cache/`.
- **Augmentation variants**:
  - `all_runs_augmented.parquet`: baseline augmented corpus covering mixed traffic conditions.
  - `all_runs_extreme_augmented.parquet`: emphasizes congested and extreme weather snapshots to stress-test the model.
- **Dataset loader** (`traffic_forecast/data/stmgt_dataset.py`):
  - Generates sliding windows indexed by run IDs so each timestep represents a dense graph snapshot (62 nodes × 144 directed edges).
  - Emits graph-aware batches comprising traffic histories, weather sequences, temporal embeddings, and multi-horizon targets.
  - Supports configurable worker pools, pinned memory, persistent workers, and prefetch factors for high-throughput GPU training on Windows.

## Model Architecture

- **Core model** (`traffic_forecast/models/stmgt.py`): multi-head attention over temporal windows augmented with graph message passing.
- **Output head**: Gaussian mixture density estimator producing logits, means, and standard deviations per node and horizon, enabling calibrated predictive distributions.
- **Loss functions**:
  - Primary: mixture negative log-likelihood (`mixture_nll_loss`).
  - Auxiliary (new): optional mean-squared error term weighted by `mse_loss_weight` to stabilize R² and mean accuracy.

## Training Pipeline Enhancements

- **Entrypoint**: `scripts/training/train_stmgt.py` consolidates configuration parsing, device preparation, loader instantiation, and dashboard-compatible artifact generation.
- **Automatic resource tuning**:
  - Dataloader workers resolved from available CPU cores (clamped to ≤16) with CUDA-aware pinning and persistent worker control.
  - Float32 matmul precision set via `torch.set_float32_matmul_precision`, enabling TF32 when appropriate while supporting non-CUDA fallbacks.
  - Metadata about GPU, worker count, and matmul settings stored in each run’s `config.json`.
- **Training loop features**:
  - AMP via `GradScaler` with graceful fallback for older Torch builds.
  - Gradient clipping, accumulation steps, early stopping, ReduceLROnPlateau scheduler, and per-epoch CSV history logging.
  - Comprehensive validation metrics (loss, MAE, RMSE, R², MAPE, CRPS, coverage@80).
- **Output structure**: every run writes to `outputs/stmgt_v2_*/` with `config.json`, `training_history.csv`, `history.json`, `best_model.pt`, `final_model.pt`, and optional `test_results.json`.

## Experiment Log

| Run Label | Config File | Dataset | Epochs (best) | Batch Size | MAE (Val/Test) | R² (Val/Test) | Notes |
|-----------|-------------|---------|---------------|------------|----------------|---------------|-------|
| `smoke_test_epoch1` | `configs/train_smoke_test.json` | `all_runs_extreme_augmented.parquet` | 1 | 16 | 10.25 / – | – / – | CPU-only AMP off sanity check; validates pipeline wiring. |
| `augmented_normal_e10_tuned` | `configs/train_augmented_normal_10epoch.json` | `all_runs_augmented.parquet` | 10 | 48 | 5.48 / – | 0.26 / – | Demonstrated improved convergence versus earlier baseline; dashboard-ready artifacts produced. |
| `augmented_extreme_e100` | `configs/train_extreme_augmented_100epoch.json` | `all_runs_extreme_augmented.parquet` | 48 (early stop) | 48 | 5.00 / 2.78 | 0.57 / 0.79 | Current flagship model; high R² and low MAPE on extreme-focused test split. |

_All runs executed inside Conda environment `dsp` via `conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config <config-path>`._

## Performance Analysis

- **R² stabilization**: blending the auxiliary MSE term (`mse_loss_weight = 0.2`) tightened predictive means, elevating validation R² from ~0.1 (baseline) to ~0.55 on extreme validation; test performance reached 0.79.
- **MAE trend**: progressive plateau near 5 km/h on validation for extreme data; test MAE under 3 km/h indicates strong generalization to unseen extreme scenarios.
- **MAPE and CRPS**: MAPE settled near 18% on test, and CRPS 2.10 with coverage@80 ≈ 0.87 confirms calibrated predictive intervals.
- **Early stopping behavior**: patience window (25 epochs) triggered stop at epoch 48, balancing fit quality and training duration without overfitting.

## Resource Utilization Profile

- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, CC 8.6) with TF32 acceleration enabled.
- **CPU footprint**: 15 dataloader workers saturate available cores during parquet loading, amortizing GPU wait time. Batch size 48 ensures GPU receives sufficiently large tensors to maintain utilization.
- **Runtime**: full extreme run (48 epochs) completed in ≈1 hour 50 minutes, dominated by I/O and evaluation passes on large validation/test sets.

## Dashboard and Monitoring Integration

- Streamlit pages `dashboard/pages/9_Training_Control.py` and `dashboard/pages/10_Model_Registry.py` now ingest `training_history.csv`, `config.json`, and `test_results.json` generated by the new trainer.
- `dashboard/realtime_stats.py` aggregates best runs from both legacy `models/training_runs/` and new `outputs/` directories, ensuring the dashboard surfaces current best MAE values automatically.
- `visualize.py` and `scripts/live_dashboard.py` can be launched via predefined VS Code tasks listed in workspace metadata for rapid inspection during experiments.

## Reproducibility Checklist

1. **Environment**: Create/activate Conda env via `environment.yml`, then load `.env` for secrets (Google Maps API key, etc.).
2. **Data readiness**: Ensure desired parquet file (`data/processed/<name>.parquet`) and cache artifacts exist; run collection scripts if necessary.
3. **Select config**: Choose between `configs/train_augmented_normal_10epoch.json` or `configs/train_extreme_augmented_100epoch.json`, or craft a new JSON following the same schema.
4. **Execute training**: `conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config <config>`.
5. **Inspect artifacts**: Review `outputs/<run>/training_history.csv`, `history.json`, and `test_results.json`; optionally feed them into the dashboard for visualization.
6. **Testing**: Run `pytest` with focus on `tests/test_integration.py` and `tests/test_model_with_data.py` to validate model-dataloader compatibility.

## Future Directions

- Expand hyperparameter sweeps (hidden dimensions, attention heads, accumulation steps) using the existing configuration-driven workflow.
- Introduce k-fold temporal cross-validation to better quantify variance and establish confidence intervals for MAE/R².
- Explore deployment packaging (FastAPI serving via `dashboard/pages/12_API_Integration.py`) using the best checkpoint from `outputs/stmgt_v2_20251101_215205`.
- Document failure cases by mining `training_history.csv` gradients and comparing node-wise residuals to guide targeted data augmentation.

## Key Artifacts

- Trainer: `scripts/training/train_stmgt.py`
- Dataloader: `traffic_forecast/data/stmgt_dataset.py`
- Primary configs: `configs/train_augmented_normal_10epoch.json`, `configs/train_extreme_augmented_100epoch.json`
- Best run outputs: `outputs/stmgt_v2_20251101_215205/`
- Dashboard pages: `dashboard/pages/9_Training_Control.py`, `dashboard/pages/10_Model_Registry.py`

This full report consolidates the project’s end-to-end workflow, experimental evidence, and reproducibility guidance for coursework evaluation and potential research follow-up.
