"""Unit tests for STMGT helper utilities."""

from __future__ import annotations

import torch

from traffic_forecast.models.stmgt.inference import mixture_to_moments
from traffic_forecast.models.stmgt.losses import mixture_nll_loss
from traffic_forecast.models.stmgt.train import EarlyStopping


def _build_pred_params(mean_values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create a simple, well-behaved mixture parameter bundle for testing."""

    stds = torch.ones_like(mean_values)
    logits = torch.zeros_like(mean_values)
    return {"means": mean_values, "stds": stds, "logits": logits}


def test_mixture_nll_loss_is_finite() -> None:
    mean_values = torch.tensor([[[[0.0, 2.0]]]], requires_grad=True)
    params = _build_pred_params(mean_values)
    target = torch.tensor([[[1.0]]])

    loss = mixture_nll_loss(params, target)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert loss.item() > 0
    assert mean_values.grad is not None


def test_mixture_to_moments_matches_manual_average() -> None:
    means = torch.tensor([[[[1.0, 3.0]]]])
    logits = torch.tensor([[[[0.0, 0.0]]]])
    stds = torch.ones_like(means)
    pred_mean, pred_std = mixture_to_moments({"means": means, "stds": stds, "logits": logits})

    assert torch.allclose(pred_mean, torch.tensor([[[2.0]]]))
    assert pred_std.shape == pred_mean.shape
    assert torch.isfinite(pred_std).all().item()


def test_early_stopping_triggers_after_patience() -> None:
    tracker = EarlyStopping(patience=2, mode="min")

    class _Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

    model = _Dummy()

    tracker(1.0, model)
    tracker(1.1, model)
    assert not tracker.early_stop

    tracker(1.2, model)
    assert tracker.early_stop
    assert tracker.best_state is not None
