"""
Model training pipeline: Train model based on config.
"""

import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import yaml

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_features():
    with open('data/features_nodes_v2.json', 'r') as f:
        return json.load(f)

def prepare_data(features):
    """Prepare features and targets."""
    X, y = [], []
    for f in features:
        if f.get('avg_speed_kmh') is not None:
            # Use feature_vector + forecasts
            fv = f['feature_vector']
            f5_temp = f.get('forecast_temp_t5_c', 0) or 0
            f5_rain = f.get('forecast_rain_t5_mm', 0) or 0
            f5_wind = f.get('forecast_wind_t5_kmh', 0) or 0
            X.append(fv + [f5_temp, f5_rain, f5_wind])
            y.append(f['avg_speed_kmh'])
    return np.array(X), np.array(y)

def train_model(config):
    features = load_features()
    X, y = prepare_data(features)
    
    if len(X) == 0:
        print("Not enough data for training.")
        return
    
    model_config = config['pipelines']['model']
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['test_size'], random_state=model_config['random_state'])
    
    # Model
    if model_config['type'] == 'linear_regression':
        model = LinearRegression()
    # elif model_config['type'] == 'lstm':
    #     # Implement LSTM
    #     pass
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. MAE: {mae:.4f}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_config['output_model'])

if __name__ == "__main__":
    config = load_config()
    train_model(config)