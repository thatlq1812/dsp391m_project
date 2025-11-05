# Traffic API README

## Quick Start

### 1. Install Dependencies

```bash
conda activate dsp
pip install fastapi uvicorn pydantic
```

### 2. Run API

```bash
# Development mode (auto-reload)
python -m uvicorn traffic_api.main:app --host 0.0.0.0 --port 8080 --reload

# Or use the script
bash run_api.sh
```

### 3. Test API

```bash
# Health check
curl http://localhost:8080/health

# Get all nodes
curl http://localhost:8080/nodes

# Get predictions
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 4. API Documentation

Open browser: http://localhost:8080/docs

## Endpoints

### GET /health

Health check

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_checkpoint": "outputs/stmgt_v2_20251101_215205/best_model.pt",
  "device": "cuda",
  "timestamp": "2025-11-02T..."
}
```

### GET /nodes

Get all traffic nodes

Response:

```json
[
  {
    "node_id": "node-10.737984-106.721606",
    "lat": 10.7379839,
    "lon": 106.7216056,
    "degree": 5,
    "importance_score": 30.0,
    "road_type": "trunk",
    "street_names": ["Nguyễn Văn Linh", "Nguyễn Thị Thập"],
    "intersection_name": "Nguyễn Văn Linh - Nguyễn Thị Thập",
    "is_major_intersection": true
  },
  ...
]
```

### GET /nodes/{node_id}

Get specific node

### POST /predict

Generate predictions

Request:

```json
{
  "timestamp": "2025-11-02T14:00:00", // optional
  "node_ids": ["node-10.737984-106.721606"], // optional, defaults to all
  "horizons": [1, 2, 3, 6, 9, 12] // optional, timesteps (15min each)
}
```

Response:

```json
{
  "timestamp": "2025-11-02T14:00:00",
  "forecast_time": "2025-11-02T14:15:00",
  "model_version": "stmgt_v2",
  "inference_time_ms": 87.5,
  "nodes": [
    {
      "node_id": "node-10.737984-106.721606",
      "lat": 10.7379839,
      "lon": 106.7216056,
      "current_speed": 45.2,
      "forecasts": [
        {
          "horizon": 1,
          "horizon_minutes": 15,
          "mean": 42.5,
          "std": 3.2,
          "lower_80": 38.4,
          "upper_80": 46.6
        },
        ...
      ]
    },
    ...
  ]
}
```

## Deployment

### Local Development

```bash
python -m uvicorn traffic_api.main:app --reload
```

### Production (Gunicorn)

```bash
gunicorn traffic_api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker

```bash
docker build -t traffic-api .
docker run -p 8080:8080 traffic-api
```

### Google Cloud Run

```bash
gcloud run deploy traffic-api \
  --source . \
  --platform managed \
  --region asia-southeast1
```

## Configuration

Edit `traffic_api/config.py`:

```python
class APIConfig(BaseModel):
    device: str = "cuda"  # "cuda" or "cpu"
    batch_size: int = 16
    host: str = "0.0.0.0"
    port: int = 8080
    prediction_cache_ttl: int = 900  # 15 minutes
```

## Architecture

```
traffic_api/
├── __init__.py
├── config.py        # Configuration
├── main.py          # FastAPI app
├── predictor.py     # STMGT inference wrapper
└── schemas.py       # Pydantic models
```

## TODO

- [ ] Implement real historical data loading (from parquet/DB)
- [ ] Add Redis caching for predictions
- [ ] Add API key authentication
- [ ] Add rate limiting
- [ ] Add Prometheus metrics
- [ ] Optimize batch inference
- [ ] Add Docker support
