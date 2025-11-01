"""Optimized STMGT training entrypoint with dashboard integration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders
from traffic_forecast.models.stmgt import STMGT, mixture_nll_loss


@dataclass
class ModelConfig:
    seq_len: int = 12
    pred_len: int = 12
    hidden_dim: int = 64
    num_heads: int = 4
    num_blocks: int = 2
    mixture_components: int = 3
    num_nodes: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 20
    drop_edge_p: float = 0.2
    num_workers: int = -1
    use_amp: bool = True
    accumulation_steps: int = 1
    data_source: str = "all_runs_extreme_augmented.parquet"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 2
    matmul_precision: str = "medium"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model_cfg = ModelConfig(**payload.get("model", {}))
        training_cfg = TrainingConfig(**payload.get("training", {}))
        metadata = payload.get("metadata", {})
        metadata["source_config"] = str(path)
        return cls(model=model_cfg, training=training_cfg, metadata=metadata)

    def to_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            "model": {k: getattr(self.model, k) for k in vars(self.model)},
            "training": {k: getattr(self.training, k) for k in vars(self.training)},
            "metadata": self.metadata,
        }


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, Tensor]] = None
        self.early_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def __call__(self, score: float, model: torch.nn.Module) -> None:
        if self._is_improvement(score):
            self.best_score = score
            self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class MetricsCalculator:
    @staticmethod
    def mae(pred: Tensor, target: Tensor) -> float:
        return torch.mean(torch.abs(pred - target)).item()

    @staticmethod
    def rmse(pred: Tensor, target: Tensor) -> float:
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()

    @staticmethod
    def r2(pred: Tensor, target: Tensor) -> float:
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        return (1 - ss_res / (ss_tot + 1e-8)).item()

    @staticmethod
    def mape(pred: Tensor, target: Tensor, epsilon: float = 1e-3) -> float:
        mask = target.abs() > epsilon
        if mask.sum() == 0:
            return 0.0
        return torch.mean(torch.abs((target[mask] - pred[mask]) / target[mask])).item() * 100

    @staticmethod
    def crps_gaussian(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> float:
        z = (target - pred_mean) / (pred_std + 1e-6)
        normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        phi_z = normal.cdf(z)
        pdf_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        crps = pred_std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        return torch.mean(crps).item()

    @staticmethod
    def coverage_80(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> float:
        lower = pred_mean - 1.28 * pred_std
        upper = pred_mean + 1.28 * pred_std
        return ((target >= lower) & (target <= upper)).float().mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STMGT v2 model")
    parser.add_argument("--config", type=Path, default=None, help="Path to dashboard-generated config JSON")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for output directory")
    return parser.parse_args()


def load_run_config(args: argparse.Namespace) -> RunConfig:
    if args.config is not None and args.config.exists():
        cfg = RunConfig.from_json(args.config)
    else:
        cfg = RunConfig()
        if args.config is not None:
            print(f"WARNING: Config not found at {args.config}, using defaults")
    cfg.metadata.setdefault("launched_at", datetime.utcnow().isoformat())
    return cfg


def resolve_data_path(cfg: TrainingConfig) -> Path:
    candidate = PROJECT_ROOT / "data" / "processed" / cfg.data_source
    if candidate.exists():
        return candidate
    fallback = PROJECT_ROOT / "data" / "processed" / "all_runs_augmented.parquet"
    if fallback.exists():
        print(f"WARNING: {candidate.name} missing, falling back to {fallback.name}")
        return fallback
    raise FileNotFoundError("No processed data parquet available for training")


def mixture_to_moments(pred_params: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
    means = pred_params["means"]
    stds = pred_params["stds"]
    weights = torch.softmax(pred_params["logits"], dim=-1)
    pred_mean = torch.sum(means * weights, dim=-1)
    second_moment = torch.sum((stds ** 2 + means ** 2) * weights, dim=-1)
    pred_var = torch.clamp(second_moment - pred_mean ** 2, min=1e-6)
    pred_std = torch.sqrt(pred_var)
    return pred_mean, pred_std


def train_epoch(
    model: STMGT,
    loader,
    optimizer: AdamW,
    device: torch.device,
    scaler: Optional[GradScaler],
    drop_edge_p: float,
    accumulation_steps: int,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    preds: List[Tensor] = []
    targets: List[Tensor] = []
    stds: List[Tensor] = []

    optimizer.zero_grad()
    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(loop):
        x_traffic = batch["x_traffic"].to(device)
        x_weather = batch["x_weather"].to(device)
        edge_index = batch["edge_index"].to(device)
        temporal_features = {k: v.to(device) for k, v in batch["temporal_features"].items()}
        y_target = batch["y_target"].to(device)

        if drop_edge_p > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_edge_p
            edge_index = edge_index[:, mask]

        with autocast(device_type=device.type, enabled=scaler is not None):
            pred_params = model(x_traffic, edge_index, x_weather, temporal_features)
            loss = mixture_nll_loss(pred_params, y_target) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size

        with torch.no_grad():
            pred_mean, pred_std = mixture_to_moments(pred_params)
            preds.append(pred_mean.detach().cpu())
            stds.append(pred_std.detach().cpu())
            targets.append(y_target.detach().cpu())

        loop.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_loss = total_loss / max(total_samples, 1)

    metrics: Dict[str, float] = {"loss": avg_loss}
    if preds:
        pred_tensor = torch.cat(preds)
        target_tensor = torch.cat(targets)
        std_tensor = torch.cat(stds)
        metrics.update(
            {
                "mae": MetricsCalculator.mae(pred_tensor, target_tensor),
                "rmse": MetricsCalculator.rmse(pred_tensor, target_tensor),
                "r2": MetricsCalculator.r2(pred_tensor, target_tensor),
                "mape": MetricsCalculator.mape(pred_tensor, target_tensor),
                "crps": MetricsCalculator.crps_gaussian(pred_tensor, std_tensor, target_tensor),
                "coverage_80": MetricsCalculator.coverage_80(pred_tensor, std_tensor, target_tensor),
            }
        )
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model: STMGT, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[Tensor] = []
    targets: List[Tensor] = []
    stds: List[Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        x_traffic = batch["x_traffic"].to(device)
        x_weather = batch["x_weather"].to(device)
        edge_index = batch["edge_index"].to(device)
        temporal_features = {k: v.to(device) for k, v in batch["temporal_features"].items()}
        y_target = batch["y_target"].to(device)

        pred_params = model(x_traffic, edge_index, x_weather, temporal_features)
        loss = mixture_nll_loss(pred_params, y_target)

        pred_mean, pred_std = mixture_to_moments(pred_params)
        preds.append(pred_mean.cpu())
        stds.append(pred_std.cpu())
        targets.append(y_target.cpu())

        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    pred_tensor = torch.cat(preds)
    target_tensor = torch.cat(targets)
    std_tensor = torch.cat(stds)

    return {
        "loss": total_loss / max(total_samples, 1),
        "mae": MetricsCalculator.mae(pred_tensor, target_tensor),
        "rmse": MetricsCalculator.rmse(pred_tensor, target_tensor),
        "r2": MetricsCalculator.r2(pred_tensor, target_tensor),
        "mape": MetricsCalculator.mape(pred_tensor, target_tensor),
        "crps": MetricsCalculator.crps_gaussian(pred_tensor, std_tensor, target_tensor),
        "coverage_80": MetricsCalculator.coverage_80(pred_tensor, std_tensor, target_tensor),
    }


def write_training_history(rows: List[Dict[str, float]], destination: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(destination, index=False)


def main() -> None:
    args = parse_args()
    run_cfg = load_run_config(args)
    training_cfg = run_cfg.training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training_cfg.use_amp and device.type != "cuda":
        print("WARNING: AMP requested but CUDA unavailable. Disabling AMP.")
        training_cfg.use_amp = False

    gpu_name: Optional[str] = None

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

    if device.type == "cuda":
        cuda_index = device.index or 0
        gpu_name = torch.cuda.get_device_name(cuda_index)
        gpu_capability = torch.cuda.get_device_capability(cuda_index)
        total_mem_gb = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
        allow_tf32 = training_cfg.matmul_precision.lower() != "high"
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {gpu_name} ({total_mem_gb:.1f} GB, CC {gpu_capability[0]}.{gpu_capability[1]})")
        print(f"TF32 enabled: {allow_tf32}")
    else:
        print(f"Using CPU device: {platform.processor() or 'generic CPU'}")

    print(
        "Resolved dataloader settings -> "
        f"workers={training_cfg.num_workers}, pin_memory={training_cfg.pin_memory}, "
        f"persistent_workers={training_cfg.persistent_workers}, prefetch_factor={training_cfg.prefetch_factor}"
    )

    data_path = resolve_data_path(training_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = PROJECT_ROOT / "outputs" / f"stmgt_v2_{timestamp}"
    output_dir = args.output_dir or default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
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
        }
    )
    if gpu_name:
        run_cfg.metadata["gpu_name"] = gpu_name

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"Nodes:         {num_nodes}")
    print(f"Edges:         {edge_index.size(1)}")

    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(
        f"Config: batch_size={training_cfg.batch_size}, num_workers={training_cfg.num_workers}, "
        f"pin_memory={training_cfg.pin_memory}, AMP={training_cfg.use_amp}, "
        f"drop_edge_p={training_cfg.drop_edge_p}"
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
        )
        print(f"Train Loss: {train_loss:.4f}")

        row: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_metrics.items() if k != "loss"})

        if val_loader and len(val_loader) > 0:
            val_metrics = evaluate(model, val_loader, device)
            print(
                "Val Metrics -> "
                f"Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
                f"R2: {val_metrics['r2']:.4f}, MAPE: {val_metrics['mape']:.2f}%"
            )
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

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history_rows, handle, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(run_cfg.to_dict(), handle, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results directory: {output_dir}")
    print(f"Best validation MAE: {best_val_mae:.4f}")

    if test_loader and len(test_loader) > 0:
        print("\n" + "=" * 70)
        print("TEST EVALUATION")
        print("=" * 70)
        test_metrics = evaluate(model, test_loader, device)
        for key, value in test_metrics.items():
            label = key.upper() if key != "coverage_80" else "COVERAGE@80"
            if key == "mape":
                print(f"  {label}: {value:.2f}%")
            else:
                print(f"  {label}: {value:.4f}")
        with open(output_dir / "test_results.json", "w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
