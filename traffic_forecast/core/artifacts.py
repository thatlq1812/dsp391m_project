"""Artifact management helpers for training runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .config_loader import RunConfig


def prepare_output_dir(model_key: str, override: Optional[Path] = None) -> Path:
    """Create the run output directory and return its path."""

    if override is not None:
        output_dir = override
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"{model_key}_{timestamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_run_config(config: RunConfig, destination: Path) -> Path:
    """Persist the run configuration as JSON for reproducibility."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return destination


def write_training_history(rows: Iterable[Dict[str, float]], destination: Path) -> None:
    """Save training metrics history to CSV."""

    df = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def save_json_artifact(payload: object, destination: Path) -> None:
    """Write a JSON artifact to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
