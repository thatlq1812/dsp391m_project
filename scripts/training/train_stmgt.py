"""Optimized STMGT training entrypoint with dashboard integration."""

from __future__ import annotations

# Suppress TensorFlow warnings BEFORE any other imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.core.artifacts import (
    prepare_output_dir,
    save_json_artifact,
    save_run_config,
    write_training_history,
)
from traffic_forecast.core.config_loader import (
    RunConfig,
    TrainingConfig,
    get_data_root,
    load_run_config,
)
from traffic_forecast.core.reporting import (
    format_metric_line,
    log_dataloader_settings,
    log_device_summary,
    print_section,
)
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders
from traffic_forecast.data.dataset_validation import (
    DEFAULT_REQUIRED_COLUMNS,
    validate_processed_dataset,
)
from traffic_forecast.models.stmgt.evaluate import evaluate_model
from traffic_forecast.models.stmgt.model import STMGT
from traffic_forecast.models.stmgt.train import EarlyStopping, train_epoch

MODEL_KEY = "stmgt_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STMGT v2 model")
    parser.add_argument("--config", type=Path, default=None, help="Path to dashboard-generated config JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = load_run_config(args.config)
    training_cfg = run_cfg.training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training_cfg.use_amp and device.type != "cuda":
        print("WARNING: AMP requested but CUDA unavailable. Disabling AMP.")
        training_cfg.use_amp = False

    try:
        torch.set_float32_matmul_precision(training_cfg.matmul_precision)
    except (TypeError, ValueError):
        print(
            f"WARNING: Unsupported matmul precision '{training_cfg.matmul_precision}'. Falling back to 'high'."
        )
        torch.set_float32_matmul_precision("high")
        training_cfg.matmul_precision = "high"

    resolved_num_workers = training_cfg.num_workers
    if resolved_num_workers < 0:
        cpu_count = os.cpu_count() or 1
        target_workers = max(cpu_count - 1, 0)
        if device.type == "cuda":
            target_workers = max(target_workers, 2)
        resolved_num_workers = min(target_workers, 16)

    resolved_num_workers = max(resolved_num_workers, 0)

    resolved_pin_memory = training_cfg.pin_memory and device.type == "cuda"
    resolved_persistent_workers = training_cfg.persistent_workers and resolved_num_workers > 0
    resolved_prefetch = training_cfg.prefetch_factor if resolved_num_workers > 0 else None
    if isinstance(resolved_prefetch, int) and resolved_prefetch <= 0:
        resolved_prefetch = None

    training_cfg.num_workers = resolved_num_workers
    training_cfg.pin_memory = resolved_pin_memory
    training_cfg.persistent_workers = resolved_persistent_workers
    training_cfg.prefetch_factor = resolved_prefetch

    gpu_name = log_device_summary(device, training_cfg)
    log_dataloader_settings(training_cfg)

    candidate_path = training_cfg.resolve_data_path(strict=False)
    if candidate_path.exists():
        data_path = candidate_path
    else:
        fallback = get_data_root() / "all_runs_augmented.parquet"
        if fallback.exists() and fallback != candidate_path:
            print(f"WARNING: {candidate_path.name} missing, falling back to {fallback.name}")
            data_path = fallback
        else:
            raise FileNotFoundError(
                f"No processed data parquet available. Expected {candidate_path} or {fallback}."
            )

    print_section("Validating Dataset")
    validation = validate_processed_dataset(data_path, list(DEFAULT_REQUIRED_COLUMNS))
    print(
        f"Path: {validation.path}\nRows: {validation.rows}\nMissing Columns: {validation.missing_columns}"
    )
    if validation.errors:
        for error in validation.errors:
            print(f"ERROR: {error}")
    if not validation.is_valid:
        raise RuntimeError("Processed dataset validation failed; review the dataset before training.")

    output_dir = prepare_output_dir(MODEL_KEY, args.output_dir)

    print_section("Creating Dataloaders")
    train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
        data_path=str(data_path),
        batch_size=training_cfg.batch_size,
        num_workers=training_cfg.num_workers,
        seq_len=run_cfg.model.seq_len,
        pred_len=run_cfg.model.pred_len,
        pin_memory=training_cfg.pin_memory,
        persistent_workers=training_cfg.persistent_workers,
        prefetch_factor=training_cfg.prefetch_factor,
    )

    run_cfg.model.num_nodes = num_nodes
    run_cfg.metadata.update(
        {
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "device": device.type,
            "num_workers": training_cfg.num_workers,
            "pin_memory": training_cfg.pin_memory,
            "persistent_workers": training_cfg.persistent_workers,
            "prefetch_factor": training_cfg.prefetch_factor,
            "matmul_precision": training_cfg.matmul_precision,
            "mse_loss_weight": training_cfg.mse_loss_weight,
            "data_rows": validation.rows,
            "data_missing_columns": ",".join(validation.missing_columns),
        }
    )
    if gpu_name:
        run_cfg.metadata["gpu_name"] = gpu_name

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Nodes:         {num_nodes}")
    print(f"Edges:         {edge_index.size(1)}")

    print_section("Creating Model")
    model = STMGT(
        num_nodes=num_nodes,
        mixture_components=run_cfg.model.mixture_components,
        hidden_dim=run_cfg.model.hidden_dim,
        num_heads=run_cfg.model.num_heads,
        num_blocks=run_cfg.model.num_blocks,
        seq_len=run_cfg.model.seq_len,
        pred_len=run_cfg.model.pred_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    scaler: Optional[GradScaler] = None
    if training_cfg.use_amp:
        try:
            scaler = GradScaler(device_type=device.type)
        except TypeError:
            scaler = GradScaler()
    early_stopping = EarlyStopping(patience=training_cfg.patience, mode="min")

    history_rows: List[Dict[str, float]] = []
    best_val_mae = float("inf")

    print_section("Starting Training")
    print(
        f"Config: batch_size={training_cfg.batch_size}, num_workers={training_cfg.num_workers}, "
        f"pin_memory={training_cfg.pin_memory}, AMP={training_cfg.use_amp}, "
        f"drop_edge_p={training_cfg.drop_edge_p}, mse_loss_weight={training_cfg.mse_loss_weight}"
    )

    for epoch in range(1, training_cfg.max_epochs + 1):
        print(f"\nEpoch {epoch}/{training_cfg.max_epochs}")

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            training_cfg.drop_edge_p,
            training_cfg.accumulation_steps,
            training_cfg.mse_loss_weight,
        )
        print(f"Train Loss: {train_loss:.4f}")
        train_metric_line = format_metric_line({k: v for k, v in train_metrics.items() if k != "loss"})
        if train_metric_line:
            print(f"Train Metrics -> {train_metric_line}")

        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_metrics.items() if k != "loss"})

        if val_loader and len(val_loader) > 0:
            val_metrics = evaluate_model(model, val_loader, device)
            print(f"Val Metrics -> {format_metric_line(val_metrics)}")
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            scheduler.step(val_metrics["loss"])
            early_stopping(val_metrics["mae"], model)

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"Saved best model checkpoint (MAE: {best_val_mae:.4f})")

            if early_stopping.early_stop:
                print(f"\nEARLY STOP triggered at epoch {epoch}")
                break

        history_rows.append(row)
        write_training_history(history_rows, output_dir / "training_history.csv")

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    run_cfg.metadata.update(
        {
            "best_val_mae": best_val_mae,
            "completed_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }
    )

    save_json_artifact(history_rows, output_dir / "history.json")
    save_run_config(run_cfg, output_dir / "config.json")

    print_section("Training Complete")
    print(f"Results directory: {output_dir}")
    print(f"Best validation MAE: {best_val_mae:.4f}")

    if test_loader and len(test_loader) > 0:
        print_section("Test Evaluation")
        test_metrics = evaluate_model(model, test_loader, device)
        for key, value in test_metrics.items():
            label = key.upper() if key != "coverage_80" else "COVERAGE@80"
            if key == "mape":
                print(f"  {label}: {value:.2f}%")
            else:
                print(f"  {label}: {value:.4f}")
        save_json_artifact(test_metrics, output_dir / "test_results.json")

    print_section("All Done")


if __name__ == "__main__":
    main()
