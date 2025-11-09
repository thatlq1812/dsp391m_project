# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic API - Testing Guide

Quick guide for testing the STMGT Traffic API locally.

## Quick Start

### 1. Start API Server

**Option A: Using script (recommended)**

```bash
./scripts/run_api_local.sh
```

**Option B: Direct command**

```bash
# With conda
conda run -n dsp python -m uvicorn traffic_api.main:app --reload --host 0.0.0.0 --port 8000

# Without conda
python -m uvicorn traffic_api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access URLs

Once server is running:

- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/route_planner.html
- **Health Check**: http://localhost:8000/health

## Testing Endpoints

### Using Browser

1. Open http://localhost:8000/docs for interactive API documentation
2. Click on any endpoint to expand
3. Click "Try it out" button
4. Fill in parameters (if needed)
5. Click "Execute" to test

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get all nodes
curl http://localhost:8000/nodes

# Get current traffic
curl http://localhost:8000/api/traffic/current

# Plan route
curl -X POST http://localhost:8000/api/route/plan \
  -H "Content-Type: application/json" \
  -d '{
    "start_node_id": "node-10.737984-106.721606",
    "end_node_id": "node-10.764404-106.698972"
  }'

# Predict edge speed
curl "http://localhost:8000/api/predict/node-10.737984-106.721606_node-10.764404-106.698972?horizon=12"
```

### Using Python

```python
import requests

# Base URL
base_url = "http://localhost:8000"

# 1. Health check
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# 2. Get all nodes
response = requests.get(f"{base_url}/nodes")
nodes = response.json()
print(f"Total nodes: {len(nodes)}")

# 3. Get current traffic
response = requests.get(f"{base_url}/api/traffic/current")
traffic = response.json()
print(f"Total edges: {traffic['total_edges']}")

# Show first edge with gradient color
if traffic['edges']:
    edge = traffic['edges'][0]
    print(f"Sample: {edge['edge_id']}")
    print(f"  Speed: {edge['speed_kmh']} km/h")
    print(f"  Color: {edge['color']} ({edge['color_category']})")

# 4. Plan route (use actual node IDs from nodes list)
node_a = nodes[0]['node_id']
node_b = nodes[10]['node_id']

response = requests.post(
    f"{base_url}/api/route/plan",
    json={
        "start_node_id": node_a,
        "end_node_id": node_b
    }
)

routes = response.json()
print(f"\nRoutes from {node_a} to {node_b}:")
for route in routes['routes']:
    print(f"  {route['route_type'].upper()}:")
    print(f"    Distance: {route['total_distance_km']:.1f} km")
    print(f"    Time: {route['expected_travel_time_min']:.1f} ± {route['travel_time_uncertainty_min']:.1f} min")
    print(f"    Confidence: {route['confidence_level']:.2f}")
```

## Web Interface Testing

1. Open http://localhost:8000/route_planner.html
2. You should see:
   - Interactive map centered on Ho Chi Minh City
   - Traffic edges colored by speed (gradient: blue→green→yellow→orange→red)
   - Control panel on the left with route planning form
3. Test route planning:

   - Select start node from dropdown
   - Select end node from dropdown
   - Click "Plan Routes"
   - Three route options should appear
   - Click on a route card to highlight it on the map

4. Test traffic visualization:
   - Click on any edge (line) on the map
   - Popup should show edge ID and current speed
   - Color should match speed:
     - Blue: 50+ km/h (very smooth)
     - Green: 40-50 km/h (smooth)
     - Light Green: 30-40 km/h (normal)
     - Yellow: 20-30 km/h (slow)
     - Orange: 10-20 km/h (congested)
     - Red: <10 km/h (heavy traffic)

## Automated Testing

Run the test suite:

```bash
# Using pytest
python -m pytest tests/test_api_endpoints.py -v

# Direct execution
python tests/test_api_endpoints.py
```

**Note:** Automated tests may timeout due to model loading time (normal behavior).

## Common Issues

### 1. Port Already in Use

```
Error: Address already in use
```

**Solution:** Kill existing process or use different port

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python -m uvicorn traffic_api.main:app --port 8001
```

### 2. Model Not Found

```
Error: No model checkpoint found
```

**Solution:** Check model path in `traffic_api/config.py`

```python
MODEL_CHECKPOINT = "outputs/stmgt_v2_20251102_200308/best_model.pt"
```

### 3. Module Not Found

```
ModuleNotFoundError: No module named 'uvicorn'
```

**Solution:** Install dependencies

```bash
pip install fastapi uvicorn torch networkx httpx
```

### 4. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Use CPU inference

```python
# In traffic_api/config.py
DEVICE = "cpu"  # Change from "cuda"
```

## Performance Benchmarks

Expected performance on typical hardware:

- **Health check**: <10ms
- **Get nodes**: <50ms
- **Current traffic**: 100-200ms (includes model inference)
- **Route planning**: 150-300ms (includes shortest path computation)
- **Edge prediction**: 100-200ms

## Next Steps

After successful local testing:

1. **Performance Testing**

   - Load testing with Apache Bench or Locust
   - Measure latency under concurrent requests
   - Identify bottlenecks

2. **VM Deployment**

   - Deploy to cloud VM (AWS/Azure/GCP)
   - Configure nginx reverse proxy
   - Set up SSL certificate
   - Configure systemd service for auto-restart

3. **Production Monitoring**
   - Add logging (structlog or loguru)
   - Set up Prometheus metrics
   - Configure error tracking (Sentry)
   - Create dashboard for monitoring

## Support

For issues or questions:

- Check logs in terminal where server is running
- Review API documentation: `docs/API_DOCUMENTATION.md`
- Check project documentation: `docs/INDEX.md`
- Contact: thatlq1812@github.com
