from fastapi import FastAPI
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import yaml
from traffic_forecast import PROJECT_ROOT

load_dotenv()

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "predictions.json"


def load_config() -> dict:
 if not CONFIG_PATH.exists():
 return {}
 with CONFIG_PATH.open(encoding="utf-8") as fh:
 return yaml.safe_load(fh) or {}

config = load_config()
# Allow the API to start even when the `api` section is missing in the YAML
api_config = config.get('api', {})

# sensible defaults if the config doesn't provide them
api_title = api_config.get('title', 'Traffic Forecast API')
api_description = api_config.get('description', 'Traffic Forecast API')
api_predictions_file = Path(api_config.get('predictions_file', DEFAULT_PREDICTIONS)).resolve()

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
 preds_file = Path(api_config.get('predictions_file', api_predictions_file)).resolve()
 preds_ok = preds_file.exists()
 db_dsn = os.getenv('POSTGRES_DSN')
 return {
 'status': 'ok' if preds_ok else 'degraded',
 'predictions_file': str(preds_file),
 'predictions_file_exists': preds_ok,
 'postgres_dsn_set': bool(db_dsn)
 }

@app.get("/v1/nodes/{node_id}/forecast")
def get_forecast(node_id: str, horizon: int = 15):
 # Load predictions from configured file (or default)
 preds_file = Path(api_config.get('predictions_file', api_predictions_file)).resolve()
 if preds_file.exists():
 try:
 with preds_file.open(encoding="utf-8") as f:
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
