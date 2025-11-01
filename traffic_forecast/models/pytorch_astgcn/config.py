"""Configuration utilities for the PyTorch ASTGCN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def default_output_dir() -> Path:
    """Default directory for training artifacts."""
    return Path("outputs")


def default_checkpoint_dir() -> Path:
    """Default directory for checkpoints."""
    return Path("outputs") / "astgcn_checkpoints"


@dataclass
class PyTorchASTGCNConfig:
    """Configuration bundle for training the PyTorch ASTGCN model."""

    data_path: Path = Path("data/processed/all_runs_combined.parquet")
    features: List[str] = field(
        default_factory=lambda: [
            "speed_kmh",
            "temperature_c",
            "wind_speed_kmh",
            "precipitation_mm",
        ]
    )
    target_feature: str = "speed_kmh"
    input_window: int = 8
    forecast_horizon: int = 2
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    chebyshev_order: int = 3
    spatial_channels: int = 64
    blocks_per_component: int = 1
    device: Optional[str] = None
    random_seed: int = 42
    output_dir: Path = field(default_factory=default_output_dir)
    checkpoint_dir: Path = field(default_factory=default_checkpoint_dir)

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
