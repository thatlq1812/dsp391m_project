from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
from typing import List

from traffic_forecast import PROJECT_ROOT

app = FastAPI()


def load_nodes() -> List[dict]:
 path = PROJECT_ROOT / 'data' / 'nodes.json'
 if not path.exists():
 return []
 with path.open('r', encoding='utf-8') as f:
 return json.load(f)


def load_traffic_snapshot() -> List[dict]:
 # Try normalized snapshot first
 candidates = [
 PROJECT_ROOT / 'data' / 'traffic_snapshot_normalized.json',
 PROJECT_ROOT / 'data' / 'traffic_snapshot.json',
 PROJECT_ROOT / 'data' / 'traffic_snapshot.csv',
 ]
 for path in candidates:
 if not path.exists():
 continue
 try:
 if path.suffix == '.json':
 with path.open('r', encoding='utf-8') as f:
 return json.load(f)
 if path.suffix == '.csv':
 import csv

 with path.open('r', encoding='utf-8') as f:
 reader = csv.DictReader(f)
 return list(reader)
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
 # Smooth color from green -> yellow -> red using HSL interpolation.
 # c expected 1..5 (1 low congestion -> green, 5 high -> red)
 try:
 level = float(c)
 except Exception:
 level = 3.0
 # normalize 0..1
 t = max(0.0, min(1.0, (level - 1.0) / 4.0))
 # hue from 120 (green) to 0 (red)
 hue = (1.0 - t) * 120
 sat = 0.75
 light = 0.5

 # convert HSL to RGB
 def hsl_to_rgb(h, s, l):
 c = (1 - abs(2 * l - 1)) * s
 hp = h / 60.0
 x = c * (1 - abs(hp % 2 - 1))
 r1 = g1 = b1 = 0
 if 0 <= hp < 1:
 r1, g1, b1 = c, x, 0
 elif 1 <= hp < 2:
 r1, g1, b1 = x, c, 0
 elif 2 <= hp < 3:
 r1, g1, b1 = 0, c, x
 elif 3 <= hp < 4:
 r1, g1, b1 = 0, x, c
 elif 4 <= hp < 5:
 r1, g1, b1 = x, 0, c
 elif 5 <= hp < 6:
 r1, g1, b1 = c, 0, x
 m = l - c / 2
 r, g, b = r1 + m, g1 + m, b1 + m
 return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))

 r, g, b = hsl_to_rgb(hue, sat, light)
 return '#%02x%02x%02x' % (r, g, b)


def speed_to_hex(speed, free_flow_kmh=50.0, jam_kmh=5.0, use_black_for_jam=True):
 try:
 import colorsys
 except Exception:
 return '#999999'
 if speed is None:
 return '#999999'
 try:
 sp = float(speed)
 except Exception:
 return '#999999'
 if sp <= jam_kmh and use_black_for_jam:
 return '#000000'
 norm = max(0.0, min(1.0, sp / float(free_flow_kmh)))
 hue_deg = 120.0 * norm
 h = hue_deg / 360.0
 s = 0.9
 v = 0.35 + 0.65 * norm
 r, g, b = colorsys.hsv_to_rgb(h, s, v)
 return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))


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
 // Smooth HSL scale from green (level=1) to red (level=5)
 const t = Math.max(0, Math.min(1, (level - 1) / 4));
 const hue = (1 - t) * 120; // 120 green -> 0 red
 const s = 75; // percent
 const l = 50; // percent
 return `hsl(${hue}, ${s}%, ${l}%)`;
 }

 async function update() {
 try {
 const res = await fetch('/nodes');
 const data = await res.json();
 data.nodes.forEach(n => {
 const id = n.id;
 const lat = +n.lat;
 const lon = +n.lon;
 // prefer server-provided color if available
 const color = n.color || congestionColor(+n.congestion);
 // Fixed small pixel-size markers: make them 1/10 of previous approx.
 const radiusPx = 2; // small marker
 if (markers[id]) {
 markers[id].setLatLng([lat, lon]);
 markers[id].setStyle({color: color, fillColor: color});
 } else {
 const marker = L.circleMarker([lat, lon], {radius: radiusPx, color: color, fillColor: color, fillOpacity: 0.9, weight: 0});
 marker.addTo(map);
 markers[id] = marker;
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
 <div style="position:absolute; right:10px; top:10px; background:white; padding:8px; border-radius:4px; box-shadow:0 0 6px rgba(0,0,0,0.2); font-size:12px;">
 <div style="font-weight:600; margin-bottom:4px">Legend</div>
 <div><span style="display:inline-block;width:14px;height:14px;background:green;margin-right:6px;"></span>Free-flow</div>
 <div><span style="display:inline-block;width:14px;height:14px;background:yellow;margin-right:6px;"></span>Moderate</div>
 <div><span style="display:inline-block;width:14px;height:14px;background:red;margin-right:6px;"></span>Slow</div>
 <div><span style="display:inline-block;width:14px;height:14px;background:black;margin-right:6px;"></span>Jam</div>
 </div>
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
 # prefer direct speed -> color mapping
 color = speed_to_hex(speed)
 congestion = compute_congestion(speed) if speed is not None else 3
 out.append({'id': nid, 'lat': lat, 'lon': lon, 'speed': speed, 'congestion': congestion, 'color': color})

 return JSONResponse({'nodes': out})


if __name__ == '__main__':
 # Run with: python scripts/live_dashboard.py
 import uvicorn
 uvicorn.run(app, host='0.0.0.0', port=8070)
