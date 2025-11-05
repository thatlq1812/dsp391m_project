"""Temporal feature engineering helpers used by the integration tests."""

from __future__ import annotations

import pandas as pd


def create_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` enriched with basic temporal signals."""

    if "timestamp" not in frame.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], utc=False, errors="coerce")

    if timestamps.isna().any():
        raise ValueError("Timestamp column contains invalid values")

    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    return result
