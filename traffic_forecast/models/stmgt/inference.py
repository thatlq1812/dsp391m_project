"""Inference helpers for STMGT models."""

from __future__ import annotations

import torch


def mixture_to_moments(pred_params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert mixture components into predictive mean and standard deviation."""

    means = pred_params["means"]
    stds = pred_params["stds"]
    weights = torch.softmax(pred_params["logits"], dim=-1)

    pred_mean = torch.sum(means * weights, dim=-1)
    second_moment = torch.sum((stds**2 + means**2) * weights, dim=-1)
    pred_var = torch.clamp(second_moment - pred_mean**2, min=1e-6)
    pred_std = torch.sqrt(pred_var)
    return pred_mean, pred_std
