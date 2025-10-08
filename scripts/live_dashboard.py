from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
from typing import List

app = FastAPI()


def load_nodes() -> List[dict]:
    path = 'data/nodes.json'
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_traffic_snapshot() -> List[dict]:
    # Try normalized snapshot first
    paths = ['data/traffic_snapshot_normalized.json', 'data/traffic_snapshot.json', 'data/traffic_snapshot.csv']
    for p in paths:
        if os.path.exists(p):
            try:
                if p.endswith('.json'):
                    with open(p, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    # CSV fallback: load minimal fields
                    import csv
                    rows = []
                    with open(p, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            rows.append(r)
                    return rows
            except Exception:
                return []
    return []


def compute_congestion(speed_kmh):
    # Higher congestion -> higher score (1..5)
    try:
        s = float(speed_kmh)
    except Exception:
        return 3
    if s >= 50:
        return 1
    if s >= 40:
        return 2
    if s >= 30:
        return 3
    if s >= 20:
        return 4
    return 5


def color_for_congestion(c):
    # Map 1..5 to colors (green -> red)
    mapping = {1: '#2ecc71', 2: '#f1c40f', 3: '#f39c12', 4: '#e67e22', 5: '#e74c3c'}
    return mapping.get(int(c), '#95a5a6')


@app.get('/', response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Live Traffic Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>body, html { height:100%; margin:0 } #map { height:100vh }</style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([10.78, 106.7], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    const markers = {};

    function congestionColor(level) {
      const map = {1: '#2ecc71', 2: '#f1c40f', 3: '#f39c12', 4: '#e67e22', 5: '#e74c3c'};
      return map[level] || '#95a5a6';
    }

    async function update() {
      try {
        const res = await fetch('/nodes');
        const data = await res.json();
        data.nodes.forEach(n => {
          const id = n.id;
          const lat = +n.lat;
          const lon = +n.lon;
          const level = +n.congestion;
          const color = congestionColor(level);
          const radius = 50 + (6 - level) * 30; // visual radius

          if (markers[id]) {
            markers[id].setLatLng([lat, lon]);
            markers[id].setStyle({color: color, fillColor: color});
            markers[id].setRadius(radius);
          } else {
            const circle = L.circle([lat, lon], {radius: radius, color: color, fillColor: color, fillOpacity: 0.6});
            circle.addTo(map);
            markers[id] = circle;
          }
        });
      } catch (err) {
        console.error('update failed', err);
      }
    }

    // initial and periodic update
    update();
    setInterval(update, 5000);
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get('/nodes')
def nodes():
    nodes = load_nodes()
    traffic = load_traffic_snapshot()

    # build map node_id -> speed
    speed_map = {}
    for t in traffic:
        nid = t.get('node_id') or t.get('node') or t.get('id')
        if not nid:
            continue
        # try common keys
        val = t.get('avg_speed_kmh') or t.get('speed_kmh') or t.get('speed') or t.get('temperature_c')
        speed_map[nid] = val

    out = []
    for n in nodes:
        nid = n.get('node_id') or n.get('id')
        lat = n.get('lat')
        lon = n.get('lon')
        speed = speed_map.get(nid)
        congestion = compute_congestion(speed) if speed is not None else 3
        out.append({'id': nid, 'lat': lat, 'lon': lon, 'speed': speed, 'congestion': congestion, 'color': color_for_congestion(congestion)})

    return JSONResponse({'nodes': out})


if __name__ == '__main__':
    # Run with: python scripts/live_dashboard.py
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8070)
