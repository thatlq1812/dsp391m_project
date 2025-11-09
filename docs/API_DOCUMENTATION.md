# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic API Documentation

Complete API documentation for STMGT Traffic Forecasting System

**Base URL:** `http://localhost:8000`  
**Version:** 0.2.0  
**Model:** STMGT v2

---

## Overview

RESTful API for real-time traffic speed forecasting and route optimization in Ho Chi Minh City. Uses STMGT (Spatial-Temporal Multi-Modal Graph Transformer) for probabilistic predictions.

---

## Endpoints

### 1. Health Check

```http
GET /health
```

Check API health and model status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_checkpoint": "/path/to/checkpoint.pt",
  "device": "cuda",
  "timestamp": "2025-11-09T16:00:00"
}
```

---

### 2. Get Current Traffic

```http
GET /api/traffic/current
```

Get real-time traffic status for all edges with gradient color coding.

**Response:**

```json
{
  "edges": [
    {
      "edge_id": "node1_node2",
      "node_a_id": "node1",
      "node_b_id": "node2",
      "speed_kmh": 35.5,
      "color": "#90EE90",
      "color_category": "light_green",
      "timestamp": "2025-11-09T16:00:00",
      "lat_a": 10.8231,
      "lon_a": 106.6297,
      "lat_b": 10.8245,
      "lon_b": 106.631
    }
  ],
  "timestamp": "2025-11-09T16:00:00",
  "total_edges": 144
}
```

**Color Gradient:**

- `#0066FF` (blue): 50+ km/h - Very smooth
- `#00CC00` (green): 40-50 km/h - Smooth
- `#90EE90` (light_green): 30-40 km/h - Normal
- `#FFD700` (yellow): 20-30 km/h - Slow
- `#FF8800` (orange): 10-20 km/h - Congested
- `#FF0000` (red): <10 km/h - Heavy traffic

---

### 3. Plan Route

```http
POST /api/route/plan
```

Find optimal routes from start to end node.

**Request Body:**

```json
{
  "start_node_id": "node1",
  "end_node_id": "node10",
  "departure_time": "2025-11-09T16:00:00" // Optional, defaults to now
}
```

**Response:**

```json
{
  "start_node_id": "node1",
  "end_node_id": "node10",
  "departure_time": "2025-11-09T16:00:00",
  "routes": [
    {
      "route_type": "fastest",
      "segments": [
        {
          "edge_id": "node1_node2",
          "node_a_id": "node1",
          "node_b_id": "node2",
          "distance_km": 1.0,
          "predicted_speed_kmh": 45.0,
          "predicted_travel_time_min": 1.33,
          "uncertainty_std": 0.20
        }
      ],
      "total_distance_km": 5.5,
      "expected_travel_time_min": 8.5,
      "travel_time_uncertainty_min": 1.2,
      "confidence_level": 0.8
    },
    {
      "route_type": "shortest",
      "segments": [...],
      "total_distance_km": 4.8,
      "expected_travel_time_min": 9.2,
      "travel_time_uncertainty_min": 1.5,
      "confidence_level": 0.8
    },
    {
      "route_type": "balanced",
      "segments": [...],
      "total_distance_km": 5.2,
      "expected_travel_time_min": 8.8,
      "travel_time_uncertainty_min": 1.3,
      "confidence_level": 0.8
    }
  ],
  "timestamp": "2025-11-09T16:00:00"
}
```

**Route Types:**

- **fastest**: Minimizes travel time based on predicted speeds
- **shortest**: Minimizes distance (fewest hops)
- **balanced**: Compromise between speed and distance

---

### 4. Predict Edge Speed

```http
GET /api/predict/{edge_id}?horizon=12
```

Get speed prediction for specific edge.

**Path Parameters:**

- `edge_id` (string): Edge identifier (format: `node_a_id_node_b_id`)

**Query Parameters:**

- `horizon` (int): Forecast horizon in timesteps (default: 12)

**Response:**

```json
{
  "edge_id": "node1_node2",
  "node_a_id": "node1",
  "node_b_id": "node2",
  "horizon": 12,
  "predicted_speed_kmh": 35.0,
  "uncertainty_std": 5.0,
  "timestamp": "2025-11-09T16:00:00"
}
```

---

### 5. Get Nodes

```http
GET /nodes
```

Get information about all traffic nodes.

**Response:**

```json
[
  {
    "node_id": "node1",
    "lat": 10.8231,
    "lon": 106.6297,
    "degree": 3,
    "importance_score": 0.85,
    "road_type": "major",
    "street_names": ["Nguyen Hue", "Le Loi"],
    "intersection_name": "Ben Thanh Market",
    "is_major_intersection": true
  }
]
```

---

### 6. Get Node Details

```http
GET /nodes/{node_id}
```

Get information about specific node.

**Path Parameters:**

- `node_id` (string): Node identifier

**Response:**

```json
{
  "node_id": "node1",
  "lat": 10.8231,
  "lon": 106.6297,
  "degree": 3,
  "importance_score": 0.85,
  "road_type": "major",
  "street_names": ["Nguyen Hue", "Le Loi"],
  "intersection_name": "Ben Thanh Market",
  "is_major_intersection": true
}
```

---

### 7. Generate Predictions

```http
POST /predict
```

Generate traffic predictions for nodes.

**Request Body:**

```json
{
  "timestamp": "2025-11-09T16:00:00", // Optional
  "node_ids": ["node1", "node2"], // Optional, defaults to all
  "horizons": [1, 2, 3, 6, 9, 12] // Forecast horizons (timesteps)
}
```

**Response:**

```json
{
  "timestamp": "2025-11-09T16:00:00",
  "forecast_time": "2025-11-09T16:00:00",
  "nodes": [
    {
      "node_id": "node1",
      "lat": 10.8231,
      "lon": 106.6297,
      "forecasts": [
        {
          "horizon": 1,
          "mean": 35.5,
          "std": 4.2,
          "lower": 27.1,
          "upper": 43.9
        }
      ],
      "current_speed": 32.0
    }
  ],
  "model_version": "stmgt_v2",
  "inference_time_ms": 125.5
}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-11-09T16:00:00"
}
```

**Status Codes:**

- `200`: Success
- `404`: Not found
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

---

## Usage Examples

### Python

```python
import requests

# Get current traffic
response = requests.get('http://localhost:8000/api/traffic/current')
traffic = response.json()

# Plan route
response = requests.post(
    'http://localhost:8000/api/route/plan',
    json={
        'start_node_id': 'node1',
        'end_node_id': 'node10'
    }
)
routes = response.json()

# Get fastest route
fastest = routes['routes'][0]
print(f"Travel time: {fastest['expected_travel_time_min']:.1f} ± {fastest['travel_time_uncertainty_min']:.1f} min")
```

### JavaScript

```javascript
// Get current traffic
fetch("/api/traffic/current")
  .then((res) => res.json())
  .then((data) => {
    data.edges.forEach((edge) => {
      console.log(
        `${edge.edge_id}: ${edge.speed_kmh} km/h (${edge.color_category})`
      );
    });
  });

// Plan route
fetch("/api/route/plan", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    start_node_id: "node1",
    end_node_id: "node10",
  }),
})
  .then((res) => res.json())
  .then((data) => {
    console.log("Routes:", data.routes.length);
  });
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get current traffic
curl http://localhost:8000/api/traffic/current

# Plan route
curl -X POST http://localhost:8000/api/route/plan \
  -H "Content-Type: application/json" \
  -d '{"start_node_id":"node1","end_node_id":"node10"}'

# Predict edge
curl "http://localhost:8000/api/predict/node1_node2?horizon=12"
```

---

## Rate Limiting

No rate limiting currently implemented. Use responsibly.

---

## CORS

CORS enabled for all origins. Configure in `traffic_api/config.py`:

```python
allow_origins = ["*"]  # Or specific domains
```

---

## Model Information

**Architecture:** STMGT (Spatial-Temporal Multi-Modal Graph Transformer)

**Features:**

- Graph attention for spatial relationships
- Transformer for temporal patterns
- Weather integration (temperature, wind, precipitation)
- Probabilistic predictions (Gaussian mixture)

**Performance:**

- Val MAE: 3.69 km/h (6.3% better than LSTM baseline)
- Inference time: ~100-200ms per request
- Scales dynamically with number of nodes

---

## Deployment

### Local Development

```bash
# Start API server
python -m traffic_api.main

# Or with uvicorn
uvicorn traffic_api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# With gunicorn
gunicorn traffic_api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Web Interface

Access web interface at: `http://localhost:8000/route_planner.html`

**Features:**

- Interactive map with Leaflet.js
- Real-time traffic visualization
- Gradient color coding (6 levels)
- Route planning with 3 options
- Auto-refresh every 5 minutes

---

## Support

For issues or questions, contact: thatlq1812@github.com
