"""
Infer batch pipeline: Load model and predict based on config.
"""

import json
import os
import numpy as np
import joblib
import yaml

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_latest_features():
    with open('data/features_nodes_v2.json', 'r') as f:
        return json.load(f)

def infer_batch():
    config = load_config()
    model_config = config['pipelines']['model']
    
    features = load_latest_features()
    if not features:
        print("No features to infer.")
        return
    
    # Load model
    if not os.path.exists(model_config['output_model']):
        print("Model not trained yet.")
        return
    
    model = joblib.load(model_config['output_model'])
    
    predictions = []
    limit = model_config.get('infer_limit', len(features))
    for f in features[:limit]:
        # Use feature_vector + forecasts
        fv = f['feature_vector']
        f5_temp = f.get('forecast_temp_t5_c', 0) or 0
        f5_rain = f.get('forecast_rain_t5_mm', 0) or 0
        f5_wind = f.get('forecast_wind_t5_kmh', 0) or 0
        X = np.array([fv + [f5_temp, f5_rain, f5_wind]])
        
        pred = model.predict(X)[0]
        predictions.append({
            'node_id': f['node_id'],
            'ts': f['ts'],
            'predicted_speed_kmh': float(pred),
            'horizon_min': 15
        })
    
    with open(model_config['predictions_file'], 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Inferred {len(predictions)} predictions with {model_config['type']}.")

if __name__ == "__main__":
    infer_batch()