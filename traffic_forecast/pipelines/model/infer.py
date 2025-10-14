"""
Infer batch pipeline: Load model and predict based on config.
"""

import json
from pathlib import Path
import numpy as np
import joblib
import yaml
from traffic_forecast import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "features_nodes_v2.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_latest_features() -> list:
    if not FEATURES_PATH.exists():
        return []
    with FEATURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def infer_batch() -> None:
    config = load_config()
    model_config = config.get("pipelines", {}).get("model", {})
    if not model_config:
        print("Model pipeline configuration missing.")
        return

    features = load_latest_features()
    if not features:
        print("No features to infer.")
        return

    output_model = PROJECT_ROOT / model_config.get("output_model", "")
    if not output_model or not output_model.exists():
        print("Model not trained yet.")
        return

    model = joblib.load(output_model)

    predictions = []
    limit = model_config.get("infer_limit", len(features))
    for feature in features[:limit]:
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
                "horizon_min": 15,
            }
        )

    predictions_path = PROJECT_ROOT / model_config.get("predictions_file", "data/predictions.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2)

    print(f"Inferred {len(predictions)} predictions with {model_config.get('type', 'unknown')} model.")


if __name__ == "__main__":
    infer_batch()
