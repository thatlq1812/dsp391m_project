"""
Model training pipeline: Train model based on config.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from traffic_forecast import PROJECT_ROOT
from traffic_forecast.models.registry import build_model, list_available_models
from traffic_forecast.pipelines.preprocess import run_pipeline

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
    return {}
    with CONFIG_PATH.open(encoding="utf-8") as fh:
    return yaml.safe_load(fh) or {}


def _load_processed_dataset(
    preprocess_cfg: dict,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, Path]]:
    output_dir = PROJECT_ROOT / preprocess_cfg.get("output_dir", "data/processed")
    metadata_path = PROJECT_ROOT / preprocess_cfg.get("metadata_path", "data/processed/metadata.json")
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    if not (train_path.exists() and val_path.exists() and metadata_path.exists()):
    return None

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    with metadata_path.open(encoding="utf-8") as fh:
    metadata = json.load(fh)

    return train_df, val_df, metadata, metadata_path


def train_model(config: dict) -> None:
    pipelines_cfg = config.get("pipelines", {})
    model_config = pipelines_cfg.get("model", {})
    preprocess_cfg = pipelines_cfg.get("preprocess", {})

    if not model_config:
    print("Model pipeline configuration missing.")
    return

    dataset = _load_processed_dataset(preprocess_cfg)

    if dataset is None:
    print("Processed dataset not found. Running preprocessing pipeline...")
    try:
    run_pipeline(config)
    dataset = _load_processed_dataset(preprocess_cfg)
    except Exception as exc:
    print(f"Preprocessing failed: {exc}")
    return

    if dataset is None:
    print("Processed dataset still unavailable after preprocessing.")
    return

    train_df, val_df, metadata, metadata_path = dataset

    feature_columns = metadata.get("feature_columns") or [
        col for col in train_df.columns if col not in metadata.get(
            "keep_columns", []) and col != metadata.get("target_column")]
    target_column = model_config.get("target_column") or metadata.get("target_column") or "avg_speed_kmh"

    if not feature_columns:
    print("No feature columns available for training.")
    return

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy(dtype=float)
    X_val = val_df[feature_columns].to_numpy(dtype=float)
    y_val = val_df[target_column].to_numpy(dtype=float)

    model_type = model_config.get("type", "linear_regression")
    model_params = model_config.get("params", {})
    try:
    model = build_model(model_type, **model_params)
    except ValueError as exc:
    available = ", ".join(list_available_models())
    raise ValueError(f"{exc}. Available models: {available}") from exc
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.4f} ({len(y_val)} samples)")

    val_predictions_path = PROJECT_ROOT / "data" / "processed" / "val_predictions.csv"
    val_report_df = val_df.copy()
    val_report_df["prediction"] = y_pred
    val_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    val_report_df.to_csv(val_predictions_path, index=False)

    # Refit model on full dataset (train + validation) for final export
    X_full = pd.concat([train_df[feature_columns], val_df[feature_columns]], axis=0).to_numpy(dtype=float)
    y_full = pd.concat([train_df[target_column], val_df[target_column]], axis=0).to_numpy(dtype=float)
    model.fit(X_full, y_full)

    output_model = PROJECT_ROOT / model_config.get("output_model", "models/linear_model.pkl")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)
    print(f"Model saved to {output_model}")

    model_metadata = {
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "metrics": {
            "mae": float(mae),
            "validation_samples": int(len(y_val)),
        },
        "preprocess_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "scaler_path": metadata.get("scaler_path"),
        "output_model": str(output_model.relative_to(PROJECT_ROOT)),
        "validation_predictions_file": str(val_predictions_path.relative_to(PROJECT_ROOT)),
        "model_params": model_params,
    }

    metadata_path_cfg = PROJECT_ROOT / model_config.get("metadata_path", "models/model_metadata.json")
    metadata_path_cfg.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path_cfg.open("w", encoding="utf-8") as fh:
    json.dump(model_metadata, fh, indent=2)
    print(f"Model metadata saved to {metadata_path_cfg}")

    training_report = {
        "summary": {
            "model_type": model_type,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "mae": float(mae),
        },
        "hyperparameters": model_params,
        "artifacts": {
            "model_path": str(output_model.relative_to(PROJECT_ROOT)),
            "metadata_path": str(metadata_path_cfg.relative_to(PROJECT_ROOT)),
            "val_predictions_path": str(val_predictions_path.relative_to(PROJECT_ROOT)),
        },
        "available_models": list(list_available_models()),
    }

    training_report_path = PROJECT_ROOT / "models" / "training_report.json"
    training_report_path.parent.mkdir(parents=True, exist_ok=True)
    training_report_path.write_text(json.dumps(training_report, indent=2), encoding="utf-8")
    print(f"Training report saved to {training_report_path}")


if __name__ == "__main__":
    config = load_config()
    train_model(config)
