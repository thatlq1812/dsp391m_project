"""Loss functions for STMGT models."""

from __future__ import annotations

import numpy as np
import torch


def mixture_nll_loss(y_pred_params: dict[str, torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for Gaussian mixture outputs with regularisation."""

    mu = y_pred_params["means"]
    sigma = y_pred_params["stds"]
    logits = y_pred_params["logits"]

    pi = torch.softmax(logits, dim=-1)
    y_true = y_true.unsqueeze(-1)

    log_prob = -0.5 * ((y_true - mu) / sigma) ** 2
    log_prob = log_prob - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    nll = -torch.logsumexp(weighted_log_prob, dim=-1)

    diversity_loss = -torch.std(mu, dim=-1).mean()
    entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean()
    entropy_reg = -entropy

    return nll.mean() + 0.01 * diversity_loss + 0.001 * entropy_reg
