"""Tests for realtime dashboard statistics helpers."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from dashboard import realtime_stats


def test_get_training_stats_reads_outputs(monkeypatch, tmp_path):
    """Ensure get_training_stats picks up metrics from outputs/stmgt directories."""

    project_root = tmp_path / "project"
    outputs_dir = project_root / "outputs" / "stmgt_v2_20240101_000000"
    outputs_dir.mkdir(parents=True)

    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.5, 1.1, 0.9],
            "train_mae": [5.0, 3.5, 2.8],
            "val_loss": [1.4, 1.0, 0.8],
            "val_mae": [4.5, 3.0, 2.4],
            "val_r2": [0.70, 0.78, 0.82],
        }
    )
    history.to_csv(outputs_dir / "training_history.csv", index=False)

    test_metrics = {"mae": 2.3, "r2": 0.81}
    with open(outputs_dir / "test_results.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle)

    def fake_root() -> Path:
        return project_root

    monkeypatch.setattr(realtime_stats, "_project_root", fake_root)

    stats = realtime_stats.get_training_stats()

    assert stats["total_runs"] == 1
    assert stats["best_mae"] == "2.30 km/h"
    assert stats["last_training"] in {"Just now", "0m ago"}
