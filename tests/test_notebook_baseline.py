"""Tests for the ASTGCN runner."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from traffic_forecast.models.astgcn import NotebookBaselineConfig, NotebookBaselineRunner

matplotlib.use("Agg")



def test_astgcn_creates_artifacts(tmp_path):
    data_path = tmp_path / "sample.parquet"
    df = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:05:00",
                    "2025-01-02 00:00:00",
                ]
            ),
            "node_a_id": [101, 102, 101],
            "node_b_id": [201, 202, 201],
            "speed_kmh": [30.0, 15.0, 45.0],
            "distance_km": [1.2, 0.8, 1.5],
            "duration_min": [2.5, 3.0, 2.0],
            "lat_a": [10.0, 10.1, 10.0],
            "lon_a": [106.0, 106.1, 106.0],
            "lat_b": [10.05, 10.15, 10.05],
            "lon_b": [106.05, 106.15, 106.05],
            "temperature_c": [28.0, 29.0, 30.0],
            "wind_speed_kmh": [5.0, 4.0, 6.0],
            "precipitation_mm": [0.0, 1.5, 0.2],
            "humidity_percent": [70, 72, 68],
            "weather_description": ["clear", "rain", "clear"],
        }
    )
    df.to_parquet(data_path)

    output_root = tmp_path / "outputs"
    config = NotebookBaselineConfig(
        data_path=Path(data_path),
        output_root=output_root,
    )
    runner = NotebookBaselineRunner(config)
    outputs = runner.run()

    expected_keys = {
        "summary",
        "describe",
        "distributions",
        "scatter",
        "heatmap",
        "node_stats",
        "map",
        "congestion_plot",
        "congestion_data",
    }
    assert expected_keys.issubset(outputs.keys())
    for path in outputs.values():
        assert path.exists()
