from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
import yaml

load_dotenv()

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
api_config = config['api']

app = FastAPI(title=api_config['title'])

class ForecastResponse(BaseModel):
    node_id: str
    horizon_min: int
    speed_kmh_pred: float
    congestion_level: int

@app.get("/")
def read_root():
    return {"message": api_config['description']}

@app.get("/v1/nodes/{node_id}/forecast")
def get_forecast(node_id: str, horizon: int = 15):
    # Load predictions
    if os.path.exists(api_config['predictions_file']):
        with open(api_config['predictions_file'], 'r') as f:
            preds = json.load(f)
        pred = next((p for p in preds if p['node_id'] == node_id), None)
        if pred:
            return ForecastResponse(
                node_id=node_id,
                horizon_min=horizon,
                speed_kmh_pred=pred['predicted_speed_kmh'],
                congestion_level=2  # Mock
            )
    
    # Fallback to mock
    return ForecastResponse(
        node_id=node_id,
        horizon_min=horizon,
        speed_kmh_pred=40.0,
        congestion_level=2
    )