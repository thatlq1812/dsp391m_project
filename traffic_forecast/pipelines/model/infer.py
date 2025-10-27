"""
Infer batch pipeline: Load model and predict based on config.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from traffic_forecast import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "features_nodes_v2.json"


def load_config() -> dict:
 if not CONFIG_PATH.exists():
 return {}
 with CONFIG_PATH.open(encoding="utf-8") as fh:
 return yaml.safe_load(fh) or {}


def load_latest_features() -> List[dict]:
 if not FEATURES_PATH.exists():
 return []
 with FEATURES_PATH.open(encoding="utf-8") as fh:
 return json.load(fh)


def _load_metadata(model_config: dict, preprocess_config: dict) -> Tuple[Dict, Dict]:
 model_meta_path = PROJECT_ROOT / model_config.get("metadata_path", "models/model_metadata.json")
 model_metadata: Dict = {}
 if model_meta_path.exists():
 with model_meta_path.open(encoding="utf-8") as fh:
 model_metadata = json.load(fh)

 preprocess_meta_path = PROJECT_ROOT / preprocess_config.get("metadata_path", "data/processed/metadata.json")
 if model_metadata.get("preprocess_metadata"):
 preprocess_meta_path = PROJECT_ROOT / model_metadata["preprocess_metadata"]

 preprocess_metadata: Dict = {}
 if preprocess_meta_path.exists():
 with preprocess_meta_path.open(encoding="utf-8") as fh:
 preprocess_metadata = json.load(fh)

 return model_metadata, preprocess_metadata


def _load_scaler(model_metadata: Dict, preprocess_metadata: Dict, preprocess_config: dict):
 scaler_path: Optional[Path] = None
 if model_metadata.get("scaler_path"):
 scaler_path = PROJECT_ROOT / model_metadata["scaler_path"]
 elif preprocess_metadata.get("scaler_path"):
 scaler_path = PROJECT_ROOT / preprocess_metadata["scaler_path"]
 else:
 scaler_path = PROJECT_ROOT / preprocess_config.get("scaler_path", "models/feature_scaler.pkl")

 if scaler_path and scaler_path.exists():
 return joblib.load(scaler_path)
 return None


def _prepare_feature_frame(
 features: List[dict],
 feature_columns: List[str],
 imputation_values: Dict[str, float],
) -> pd.DataFrame:
 rows = []
 for feature in features:
 row = {col: feature.get(col) for col in feature_columns}
 rows.append(row)
 df = pd.DataFrame(rows, columns=feature_columns)
 for col in feature_columns:
 df[col] = pd.to_numeric(df[col], errors="coerce")
 fill_value = imputation_values.get(col)
 if fill_value is None:
 fill_value = float(df[col].median(skipna=True) or 0.0)
 df[col] = df[col].fillna(fill_value)
 return df


def _predict_with_metadata(
 model,
 features: List[dict],
 feature_columns: List[str],
 imputation_values: Dict[str, float],
 scaler,
) -> List[Dict]:
 if not features:
 return []
 feature_frame = _prepare_feature_frame(features, feature_columns, imputation_values)
 matrix = feature_frame.to_numpy(dtype=float)
 if scaler is not None:
 X = scaler.transform(matrix)
 else:
 X = matrix

 preds = model.predict(X)
 results = []
 for feature, pred in zip(features, preds):
 results.append(
 {
 "node_id": feature.get("node_id"),
 "ts": feature.get("ts"),
 "predicted_speed_kmh": float(pred),
 "horizon_min": feature.get("horizon_min", 15),
 }
 )
 return results


def _predict_legacy(model, features: List[dict]) -> List[Dict]:
 predictions = []
 for feature in features:
 fv = feature["feature_vector"]
 f5_temp = feature.get("forecast_temp_t5_c", 0) or 0
 f5_rain = feature.get("forecast_rain_t5_mm", 0) or 0
 f5_wind = feature.get("forecast_wind_t5_kmh", 0) or 0
 X = np.array([fv + [f5_temp, f5_rain, f5_wind]])
 pred = model.predict(X)[0]
 predictions.append(
 {
 "node_id": feature["node_id"],
 "ts": feature["ts"],
 "predicted_speed_kmh": float(pred),
 "horizon_min": feature.get("horizon_min", 15),
 }
 )
 return predictions


def infer_batch() -> None:
 config = load_config()
 pipelines_cfg = config.get("pipelines", {})
 model_config = pipelines_cfg.get("model", {})
 preprocess_config = pipelines_cfg.get("preprocess", {})

 if not model_config:
 print("Model pipeline configuration missing.")
 return

 features = load_latest_features()
 if not features:
 print("No features to infer.")
 return

 model_path = PROJECT_ROOT / model_config.get("output_model", "")
 if not model_path or not model_path.exists():
 print("Model not trained yet.")
 return

 model = joblib.load(model_path)

 model_metadata, preprocess_metadata = _load_metadata(model_config, preprocess_config)
 feature_columns = model_metadata.get("feature_columns") or preprocess_metadata.get("feature_columns")
 imputation_values = preprocess_metadata.get("imputation_values", {})
 scaler = _load_scaler(model_metadata, preprocess_metadata, preprocess_config)

 limit = model_config.get("infer_limit", len(features))
 subset = features[:limit]

 if feature_columns:
 predictions = _predict_with_metadata(model, subset, feature_columns, imputation_values, scaler)
 else:
 print("Falling back to legacy feature processing for inference.")
 predictions = _predict_legacy(model, subset)

 predictions_path = PROJECT_ROOT / model_config.get("predictions_file", "data/predictions.json")
 predictions_path.parent.mkdir(parents=True, exist_ok=True)
 with predictions_path.open("w", encoding="utf-8") as fh:
 json.dump(predictions, fh, indent=2)

 summary = {
 "count": len(predictions),
 "model_type": model_config.get("type", "unknown"),
 "created_at": datetime.utcnow().isoformat(),
 }
 summary_path = predictions_path.with_name("predictions_summary.json")
 with summary_path.open("w", encoding="utf-8") as fh:
 json.dump(summary, fh, indent=2)

 print(f"Inferred {len(predictions)} predictions with {model_config.get('type', 'unknown')} model.")


if __name__ == "__main__":
 infer_batch()
