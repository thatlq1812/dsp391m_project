"""Evaluation helpers for STMGT."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .inference import mixture_to_moments
from .losses import mixture_nll_loss
from .model import STMGT
from .train import MetricsCalculator


@torch.no_grad()
def evaluate_model(model: STMGT, loader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model and return aggregate metrics."""

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

    pred_tensor = torch.cat(preds) if preds else torch.zeros(1)
    target_tensor = torch.cat(targets) if targets else torch.zeros(1)
    std_tensor = torch.cat(stds) if stds else torch.zeros(1)

    return {
        "loss": total_loss / max(total_samples, 1),
        "mae": MetricsCalculator.mae(pred_tensor, target_tensor),
        "rmse": MetricsCalculator.rmse(pred_tensor, target_tensor),
        "r2": MetricsCalculator.r2(pred_tensor, target_tensor),
        "mape": MetricsCalculator.mape(pred_tensor, target_tensor),
        "crps": MetricsCalculator.crps_gaussian(pred_tensor, std_tensor, target_tensor),
        "coverage_80": MetricsCalculator.coverage_80(pred_tensor, std_tensor, target_tensor),
    }
