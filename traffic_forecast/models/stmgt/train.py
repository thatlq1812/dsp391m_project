"""Training utilities for the STMGT model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm.auto import tqdm

from .inference import mixture_to_moments
from .losses import mixture_nll_loss
from .model import STMGT


class EarlyStopping:
    """Basic early stopping tracker for minimisation or maximisation targets."""

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
    """Collection of evaluation metric helpers for tensors."""

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
        pdf_z = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        crps = pred_std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        return torch.mean(crps).item()

    @staticmethod
    def coverage_80(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> float:
        lower = pred_mean - 1.28 * pred_std
        upper = pred_mean + 1.28 * pred_std
        return ((target >= lower) & (target <= upper)).float().mean().item()


def train_epoch(
    model: STMGT,
    loader,
    optimizer: AdamW,
    device: torch.device,
    scaler: Optional[GradScaler],
    drop_edge_p: float,
    accumulation_steps: int,
    mse_loss_weight: float,
) -> Tuple[float, Dict[str, float]]:
    """Run a single training epoch and return loss plus metrics."""

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
            pred_mean, pred_std = mixture_to_moments(pred_params)
            
            # Normalize target to match model output space
            y_target_norm = model.speed_normalizer(y_target.unsqueeze(-1)).squeeze(-1)
            
            base_loss = mixture_nll_loss(pred_params, y_target_norm)
            mse_term = 0.0
            if mse_loss_weight > 0:
                mse_term = mse_loss_weight * F.mse_loss(pred_mean, y_target_norm)
            loss = (base_loss + mse_term) / accumulation_steps

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
            # Denormalize predictions for metrics
            pred_mean_denorm = model.speed_normalizer.denormalize(pred_mean.unsqueeze(-1)).squeeze(-1)
            pred_std_denorm = pred_std * model.speed_normalizer.std
            
            preds.append(pred_mean_denorm.detach().cpu())
            stds.append(pred_std_denorm.detach().cpu())
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
