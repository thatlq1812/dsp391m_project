from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
import yaml

load_dotenv()

def load_config():
    try:
        with open('configs/project_config.yaml', 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

config = load_config()
# Allow the API to start even when the `api` section is missing in the YAML
api_config = config.get('api', {})

# sensible defaults if the config doesn't provide them
api_title = api_config.get('title', 'Traffic Forecast API')
api_description = api_config.get('description', 'Traffic Forecast API')
api_predictions_file = api_config.get('predictions_file', 'data/predictions.json')

app = FastAPI(title=api_title, description=api_description)

class ForecastResponse(BaseModel):
    node_id: str
    horizon_min: int
    speed_kmh_pred: float
    congestion_level: int

@app.get("/")
def read_root():
    return {"message": api_config.get('description', api_description)}


@app.get('/health')
def health_check():
    """Basic health check: verifies predictions file presence and DB DSN env var."""
    preds_file = api_config.get('predictions_file', api_predictions_file)
    preds_ok = os.path.exists(preds_file)
    db_dsn = os.getenv('POSTGRES_DSN')
    return {
        'status': 'ok' if preds_ok else 'degraded',
        'predictions_file': preds_file,
        'predictions_file_exists': preds_ok,
        'postgres_dsn_set': bool(db_dsn)
    }

@app.get("/v1/nodes/{node_id}/forecast")
def get_forecast(node_id: str, horizon: int = 15):
    # Load predictions from configured file (or default)
    preds_file = api_config.get('predictions_file', api_predictions_file)
    if os.path.exists(preds_file):
        try:
            with open(preds_file, 'r') as f:
                preds = json.load(f)
            pred = next((p for p in preds if p.get('node_id') == node_id), None)
            if pred:
                return ForecastResponse(
                    node_id=node_id,
                    horizon_min=horizon,
                    speed_kmh_pred=pred.get('predicted_speed_kmh', 40.0),
                    congestion_level=pred.get('congestion_level', 2)
                )
        except Exception:
            # If predictions file is corrupt or unreadable, fall back to mock below
            pass
    
    # Fallback to mock
    return ForecastResponse(
        node_id=node_id,
        horizon_min=horizon,
        speed_kmh_pred=40.0,
        congestion_level=2
    )