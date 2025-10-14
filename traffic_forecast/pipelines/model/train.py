"""
Model training pipeline: Train model based on config.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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


def load_features() -> list:
    if not FEATURES_PATH.exists():
        return []
    with FEATURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def prepare_data(features: list) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features and targets."""
    X, y = [], []
    for feature in features:
        if feature.get("avg_speed_kmh") is not None:
            fv = feature["feature_vector"]
            f5_temp = feature.get("forecast_temp_t5_c", 0) or 0
            f5_rain = feature.get("forecast_rain_t5_mm", 0) or 0
            f5_wind = feature.get("forecast_wind_t5_kmh", 0) or 0
            X.append(fv + [f5_temp, f5_rain, f5_wind])
            y.append(feature["avg_speed_kmh"])
    return np.array(X), np.array(y)


def train_model(config: dict) -> None:
    features = load_features()
    X, y = prepare_data(features)

    if len(X) == 0:
        print("Not enough data for training.")
        return

    model_config = config.get("pipelines", {}).get("model", {})
    if not model_config:
        print("Model pipeline configuration missing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_config.get("test_size", 0.2),
        random_state=model_config.get("random_state", 42),
    )

    if model_config.get("type") == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model type: {model_config.get('type')}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. MAE: {mae:.4f}")

    output_model = PROJECT_ROOT / model_config.get("output_model", "models/linear_model.pkl")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)


if __name__ == "__main__":
    config = load_config()
    train_model(config)
