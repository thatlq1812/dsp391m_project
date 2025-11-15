import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_forecast.data.dataset_validation import validate_processed_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "data" / "processed" / "baseline_1month.parquet"
AUGMENTED = PROJECT_ROOT / "data" / "processed" / "augmented_1year.parquet"


def _assert_cadence(parquet_path: Path, expected_minutes: int) -> None:
    # Read only timestamp column for performance
    frame = pd.read_parquet(parquet_path, columns=["timestamp"])  # type: ignore[arg-type]
    assert len(frame) > 0, "Dataset appears empty"
    ts = pd.to_datetime(frame["timestamp"])  # ensure datetime
    # Use unique timestamps (datasets have many edges per timestamp)
    ts_unique = ts.drop_duplicates().sort_values()
    assert len(ts_unique) > 1, "Need at least 2 unique timestamps to compute cadence"
    diffs = ts_unique.diff().dropna().dt.total_seconds() / 60.0
    assert not diffs.empty, "Not enough timestamps to compute cadence"
    mode_gap = int(diffs.round().astype(int).value_counts().idxmax())
    assert mode_gap == expected_minutes, f"Expected cadence {expected_minutes} min, got {mode_gap} min"


def _assert_speed_bounds(parquet_path: Path, min_ok: float = 0.0, max_ok: float = 120.0) -> None:
    # Read only speed column for performance
    cols = ["speed_kmh"]
    frame = pd.read_parquet(parquet_path, columns=cols)  # type: ignore[arg-type]
    assert "speed_kmh" in frame.columns, "Missing speed_kmh column"
    s = frame["speed_kmh"].dropna()
    assert len(s) > 0, "No speed values present"
    mn, mx = float(s.min()), float(s.max())
    assert mn >= min_ok - 1e-6, f"Speed min below bound: {mn} < {min_ok}"
    assert mx <= max_ok + 1e-6, f"Speed max above bound: {mx} > {max_ok}"


def test_baseline_dataset_basic_validations():
    if not BASELINE.exists():
        pytest.skip(f"Baseline parquet not found at {BASELINE}")

    # Baseline schema includes distance_km but not run_id/duration_min
    required = ['timestamp', 'node_a_id', 'node_b_id', 'speed_kmh', 'distance_km']
    result = validate_processed_dataset(BASELINE, required)
    assert result.is_valid, f"Baseline dataset invalid: {result.errors or result.missing_columns}"
    _assert_cadence(BASELINE, expected_minutes=15)
    _assert_speed_bounds(BASELINE)


def test_augmented_dataset_basic_validations():
    if not AUGMENTED.exists():
        pytest.skip(f"Augmented parquet not found at {AUGMENTED}")

    # Augmented schema is lean: require core columns only
    required = ['timestamp', 'node_a_id', 'node_b_id', 'speed_kmh']
    result = validate_processed_dataset(AUGMENTED, required)
    assert result.is_valid, f"Augmented dataset invalid: {result.errors or result.missing_columns}"
    _assert_cadence(AUGMENTED, expected_minutes=10)
    _assert_speed_bounds(AUGMENTED)
