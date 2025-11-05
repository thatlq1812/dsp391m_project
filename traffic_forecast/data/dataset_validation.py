"""Utilities for validating processed traffic datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


DEFAULT_REQUIRED_COLUMNS: Sequence[str] = (
    "run_id",
    "timestamp",
    "node_a_id",
    "node_b_id",
    "speed_kmh",
    "distance_km",
    "duration_min",
)


@dataclass
class DatasetValidationResult:
    """Container object describing the outcome of a dataset validation run."""

    path: Path
    exists: bool
    rows: int = 0
    columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    null_columns: List[str] = field(default_factory=list)
    duplicate_rows: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:  # pragma: no cover - property delegation is trivial
        return self.exists and not self.missing_columns and not self.errors


def _normalise_required_columns(required_columns: Iterable[str] | None) -> List[str]:
    if required_columns is None:
        return list(DEFAULT_REQUIRED_COLUMNS)
    return sorted({column.strip() for column in required_columns if column.strip()})


def validate_processed_dataset(
    dataset_path: Path,
    required_columns: Iterable[str] | None = None,
    timestamp_column: str = "timestamp",
) -> DatasetValidationResult:
    """Validate that the processed dataset exists and contains core columns."""

    dataset_path = Path(dataset_path)
    columns_to_check = _normalise_required_columns(required_columns)

    result = DatasetValidationResult(path=dataset_path, exists=dataset_path.exists())
    if not result.exists:
        result.errors.append(f"Dataset not found at {dataset_path}")
        return result

    try:
        frame = pd.read_parquet(dataset_path)
    except Exception as exc:  # pragma: no cover - pandas raises multiple subclasses
        result.errors.append(f"Failed to read parquet: {exc}")
        return result

    result.rows = len(frame)
    result.columns = list(frame.columns)

    result.missing_columns = [column for column in columns_to_check if column not in frame.columns]

    # Track columns that contain NULL values in required fields.
    result.null_columns = [
        column
        for column in columns_to_check
        if column in frame.columns and frame[column].isna().any()
    ]

    # Identify duplicate records for the typical edge granularity.
    subset = [col for col in ("run_id", "node_a_id", "node_b_id", timestamp_column) if col in frame.columns]
    if subset:
        result.duplicate_rows = int(frame.duplicated(subset=subset).sum())

    if timestamp_column in frame.columns:
        try:
            pd.to_datetime(frame[timestamp_column], errors="raise")
        except Exception as exc:  # pragma: no cover - see comment above
            result.errors.append(f"Timestamp column conversion failed: {exc}")

    return result
