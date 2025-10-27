"""
Visualization script: Plot nodes map, traffic heatmap, and forecasts.
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import yaml

from traffic_forecast import PROJECT_ROOT

RUN_DIR = os.getenv("RUN_DIR")

load_dotenv()
try:
 from PIL import Image
 import requests
except Exception:
 Image = None
 requests = None

# Module-level globals to hold a single basemap image/extent for all plots
GLOBAL_BASEMAP_IMG = None
GLOBAL_BASEMAP_EXTENT = None

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


def resolve_path(run_dir, relative_default):
 """Return path inside run_dir/collectors if exists, else relative_default."""
 if not run_dir:
 return str(PROJECT_ROOT / relative_default)
 candidate = os.path.join(run_dir, "collectors")
 if os.path.exists(candidate):
 fname = os.path.basename(relative_default)
 for root, dirs, files in os.walk(candidate):
 if fname in files:
 return os.path.join(root, fname)
 return str(PROJECT_ROOT / relative_default)


def load_config():
 if not CONFIG_PATH.exists():
 return {}
 with CONFIG_PATH.open(encoding="utf-8") as f:
 return yaml.safe_load(f) or {}


def load_nodes(limit=None):
 # Try consolidated run_data.json first
 run_data_path = os.path.join(RUN_DIR or ".", "run_data.json")
 if os.path.exists(run_data_path):
 with open(run_data_path, "r", encoding="utf-8") as f:
 run_data = json.load(f)
 nodes = run_data.get("nodes.json", [])
 if isinstance(nodes, list):
 if limit:
 nodes = nodes[:limit]
 return nodes
 # Fallback to old method
 path = resolve_path(RUN_DIR, "data/nodes_enriched.json")
 if not os.path.exists(path):
 path = resolve_path(RUN_DIR, "data/nodes.json")
 if not os.path.exists(path):
 return []
 with open(path, "r", encoding="utf-8") as f:
 nodes = json.load(f)
 if limit:
 nodes = nodes[:limit]
 return nodes


def load_traffic():
 run_data_path = os.path.join(RUN_DIR or ".", "run_data.json")
 if os.path.exists(run_data_path):
 with open(run_data_path, "r", encoding="utf-8") as f:
 run_data = json.load(f)
 traffic = run_data.get("traffic_snapshot_normalized.json", [])
 if isinstance(traffic, list):
 return traffic
 path = resolve_path(RUN_DIR, "data/traffic_snapshot_normalized.json")
 if not os.path.exists(path):
 return []
 with open(path, "r", encoding="utf-8") as f:
 try:
 return json.load(f)
 except Exception:
 return []


def speed_to_hex(speed, free_flow_kmh=50.0, jam_kmh=5.0, use_black_for_jam=True):
 """Map speed (km/h) to a hex color."""
 try:
 import colorsys
 except Exception:
 return "#999999"
 if speed is None:
 return "#999999"
 try:
 sp = float(speed)
 except Exception:
 return "#999999"
 if sp <= jam_kmh and use_black_for_jam:
 return "#000000"
 norm = max(0.0, min(1.0, sp / float(free_flow_kmh)))
 hue_deg = 120.0 * norm
 h = hue_deg / 360.0
 s = 0.9
 v = 0.35 + 0.65 * norm
 r, g, b = colorsys.hsv_to_rgb(h, s, v)
 return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def load_events():
 run_data_path = os.path.join(RUN_DIR or ".", "run_data.json")
 if os.path.exists(run_data_path):
 with open(run_data_path, "r", encoding="utf-8") as f:
 run_data = json.load(f)
 events = run_data.get("events.json", [])
 if isinstance(events, list):
 return events
 path = resolve_path(RUN_DIR, "data/events_enriched.json")
 if os.path.exists(path):
 with open(path, "r") as f:
 return json.load(f)
 return []


def plot_nodes_map(config):
 """Plot map of all nodes."""
 vis_cfg = config.get("visualize") or {}
 nodes = load_nodes(vis_cfg.get("nodes_map", {}).get("limit_nodes"))
 if not nodes:
 print("No nodes found for plotting nodes_map; skipping")
 return
 lats = [n.get("lat") for n in nodes]
 lons = [n.get("lon") for n in nodes]
 cfg = vis_cfg.get("nodes_map", {})
 plt.figure(figsize=cfg.get("figsize", (10, 8)))
 try:
 if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
 plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)
 except NameError:
 pass
 traffic = {t.get("node_id") or t.get("node") or t.get("id"): t for t in load_traffic()}
 colors = []
 for n in nodes:
 sp = n.get("speed_kmh") or n.get("speed") or traffic.get(n.get("node_id", ""), {}).get("avg_speed_kmh")
 colors.append(speed_to_hex(sp))
 plt.scatter(lons, lats, s=cfg.get("s", 1), alpha=cfg.get("alpha", 0.5), c=colors, label=cfg.get("label", "Nodes"))
 plt.xlabel("Longitude")
 plt.ylabel("Latitude")
 plt.title(cfg.get("title", "Nodes map with speed coloring"))
 if cfg.get("legend"):
 plt.legend()
 if cfg.get("grid"):
 plt.grid(True, alpha=0.3)
 if cfg.get("save_path"):
 out_dir = os.getenv("RUN_IMAGE_DIR") or os.path.join(RUN_DIR or ".", "images")
 os.makedirs(out_dir, exist_ok=True)
 out_path = os.path.join(out_dir, os.path.basename(cfg["save_path"]))
 plt.savefig(out_path, dpi=cfg.get("dpi", 150))
 print(f"Saved nodes map to {out_path}")
 else:
 plt.show()


def plot_traffic_heatmap(config):
 """Plot traffic heatmap on basemap using scatter with colorbar."""
 vis_cfg = config.get("visualize") or {}
 heat_cfg = vis_cfg.get("traffic_heatmap", {})
 nodes = load_nodes(heat_cfg.get("limit_nodes"))
 traffic = load_traffic()
 if not nodes or not traffic:
 print("No nodes or traffic snapshots found for traffic_heatmap; skipping")
 return

 nodes_by_id = {n["node_id"]: n for n in nodes}
 speeds = []
 lats, lons, colors = [], [], []
 for t in traffic:
 node = nodes_by_id.get(t.get("node_id"))
 if not node:
 continue
 sp = t.get("avg_speed_kmh")
 if sp is None:
 continue
 lats.append(node["lat"])
 lons.append(node["lon"])
 speeds.append(sp)
 colors.append(speed_to_hex(sp))

 if not lats:
 print("Traffic data missing or no matching nodes; skipping heatmap")
 return

 plt.figure(figsize=heat_cfg.get("figsize", (12, 10)))
 if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
 plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)
 sc = plt.scatter(lons, lats, c=speeds, cmap=heat_cfg.get("cmap", "RdYlGn"), s=heat_cfg.get("s", 20), alpha=heat_cfg.get("alpha", 0.7), zorder=1)
 plt.colorbar(sc, label="Speed (km/h)")
 plt.xlabel("Longitude")
 plt.ylabel("Latitude")
 plt.title(heat_cfg.get("title", "Traffic heatmap"))
 out_dir = os.getenv("RUN_IMAGE_DIR") or os.path.join(RUN_DIR or ".", "images")
 os.makedirs(out_dir, exist_ok=True)
 out_path = os.getenv("RUN_IMAGE_FILE") or os.path.join(out_dir, os.path.basename(heat_cfg.get("save_path", "traffic_heatmap.png")))
 plt.savefig(out_path, dpi=heat_cfg.get("dpi", 150))
 print(f"Saved traffic heatmap to {out_path}")
 if heat_cfg.get("show"):
 plt.show()
 plt.close()


def plot_events(config):
 """Plot events with impact on map."""
 vis_cfg = config.get("visualize") or {}
 ev_cfg = vis_cfg.get("events", {})
 if not ev_cfg.get("enabled", True):
 return
 events = load_events()
 nodes = load_nodes(ev_cfg.get("limit_nodes"))
 if not events or not nodes:
 print("No events or nodes for plotting events; skipping")
 return

 nodes_by_id = {n["node_id"]: n for n in nodes}
 plt.figure(figsize=ev_cfg.get("figsize", (12, 10)))
 if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
 plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)

 for event in events:
 impacts = event.get("impacts") or []
 for imp in impacts[: ev_cfg.get("max_impacts_per_event", 10)]:
 node = nodes_by_id.get(imp.get("node_id"))
 if not node:
 continue
 lat, lon = node["lat"], node["lon"]
 impact_score = imp.get("impact_score", 0.2)
 plt.scatter([lon], [lat], s=impact_score * ev_cfg.get("impact_scale", 500), c=ev_cfg.get("impact_color", "#ff0000"), alpha=0.6, label=event.get("title") if ev_cfg.get("legend") else None)
 if ev_cfg.get("annotate"):
 plt.annotate(event.get("title", "event"), (lon, lat), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)

 if ev_cfg.get("legend"):
 plt.legend()
 plt.xlabel("Longitude")
 plt.ylabel("Latitude")
 plt.title(ev_cfg.get("title", "Events impact map"))
 out_dir = os.getenv("RUN_IMAGE_DIR") or os.path.join(RUN_DIR or ".", "images")
 os.makedirs(out_dir, exist_ok=True)
 out_path = os.path.join(out_dir, os.path.basename(ev_cfg.get("save_path", "events_map.png")))
 plt.savefig(out_path, dpi=ev_cfg.get("dpi", 150))
 print(f"Saved events map to {out_path}")
 if ev_cfg.get("show"):
 plt.show()
 plt.close()


def prepare_basemap(config):
 global GLOBAL_BASEMAP_IMG, GLOBAL_BASEMAP_EXTENT
 vis_cfg = config.get("visualize") or {}
 base_cfg = vis_cfg.get("basemap", {})
 if not base_cfg.get("enabled", True):
 GLOBAL_BASEMAP_IMG = None
 GLOBAL_BASEMAP_EXTENT = None
 return

 if base_cfg.get("use_cached") and GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
 return

 nodes = load_nodes()
 if not nodes:
 print("No nodes available to compute basemap extent; skipping basemap")
 return
 avg_lat = sum(n["lat"] for n in nodes) / len(nodes)
 avg_lon = sum(n["lon"] for n in nodes) / len(nodes)

 zoom = base_cfg.get("zoom", 13)
 size = base_cfg.get("size", 800)
 try:
 img, extent = fetch_static_basemap(avg_lat, avg_lon, zoom=zoom, size=size)
 GLOBAL_BASEMAP_IMG = img
 GLOBAL_BASEMAP_EXTENT = extent
 except Exception as exc:
 print(f"Failed to fetch basemap: {exc}")
 GLOBAL_BASEMAP_IMG = None
 GLOBAL_BASEMAP_EXTENT = None


def run_visualization():
 config = load_config()
 prepare_basemap(config)
 plots = os.getenv("VIS_PLOTS", "nodes,heatmap").split(",")
 plots = [p.strip() for p in plots if p.strip()]
 if "nodes" in plots:
 plot_nodes_map(config)
 if "heatmap" in plots:
 plot_traffic_heatmap(config)
 if "events" in plots:
 plot_events(config)
 if plots == ["heatmap"]:
 out_file = os.getenv("RUN_IMAGE_FILE") or os.path.join(RUN_DIR or ".", "images", "traffic_heatmap.png")
 print(f"Visualization saved to {out_file}")


def deg2num(lat_deg, lon_deg, zoom):
 """Convert lat/lon to fractional tile numbers at given zoom."""
 lat_rad = math.radians(lat_deg)
 n = 2.0 ** zoom
 xtile = (lon_deg + 180.0) / 360.0 * n
 ytile = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
 return xtile, ytile


def num2deg(xtile, ytile, zoom):
 """Convert fractional tile numbers to lat/lon of NW corner."""
 n = 2.0 ** zoom
 lon_deg = xtile / n * 360.0 - 180.0
 lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
 lat_deg = math.degrees(lat_rad)
 return lat_deg, lon_deg


def fetch_static_basemap(center_lat, center_lon, zoom=14, size=800):
 """Fetch a static map image centered at lat/lon using Google or OpenStreetMap as fallback."""
 if requests is None or Image is None:
 raise RuntimeError("PIL or requests not available")
 use_google = os.environ.get("VIS_USE_GOOGLE", "0") == "1" or os.environ.get("GOOGLE_MAPS_API_KEY")
 if use_google and os.environ.get("GOOGLE_MAPS_API_KEY"):
 key = os.environ.get("GOOGLE_MAPS_API_KEY")
 gsize = f"{min(size, 640)}x{min(size, 640)}"
 maptype = os.environ.get("VIS_GOOGLE_MAPTYPE", "satellite")
 print(f"Trying Google Static Maps (type={maptype}, size={gsize})")
 url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lon}&zoom={zoom}&size={gsize}&maptype={maptype}&key={key}&scale=1"
 try:
 r = requests.get(url, timeout=30)
 r.raise_for_status()
 img = Image.open(io.BytesIO(r.content)).convert("RGBA")
 print("Fetched basemap image from Google")
 except Exception as e:
 print(f"Google Maps failed: {e}, falling back to OpenStreetMap")
 use_google = False
 if not use_google:
 print("Using OpenStreetMap static map service")
 url = f"https://staticmap.openstreetmap.de/staticmap.php?center={center_lat},{center_lon}&zoom={zoom}&size={size}x{size}&maptype=mapnik"
 r = requests.get(url, timeout=30)
 r.raise_for_status()
 img = Image.open(io.BytesIO(r.content)).convert("RGBA")
 print("Fetched basemap image from OpenStreetMap")

 cx, cy = deg2num(center_lat, center_lon, zoom)
 tiles_per_img_x = size / 256.0
 tiles_per_img_y = size / 256.0
 left_xtile = cx - tiles_per_img_x / 2.0
 right_xtile = cx + tiles_per_img_x / 2.0
 top_ytile = cy - tiles_per_img_y / 2.0
 bottom_ytile = cy + tiles_per_img_y / 2.0

 top_lat, left_lon = num2deg(left_xtile, top_ytile, zoom)
 bottom_lat, right_lon = num2deg(right_xtile, bottom_ytile, zoom)
 extent = [left_lon, right_lon, bottom_lat, top_lat]
 return img, extent


def compute_basemap_extent(center_lat, center_lon, zoom=14, size=800):
 """Compute extent for a basemap image without performing network calls."""
 cx, cy = deg2num(center_lat, center_lon, zoom)
 tiles_per_img_x = size / 256.0
 tiles_per_img_y = size / 256.0
 left_xtile = cx - tiles_per_img_x / 2.0
 right_xtile = cx + tiles_per_img_x / 2.0
 top_ytile = cy - tiles_per_img_y / 2.0
 bottom_ytile = cy + tiles_per_img_y / 2.0
 top_lat, left_lon = num2deg(left_xtile, top_ytile, zoom)
 bottom_lat, right_lon = num2deg(right_xtile, bottom_ytile, zoom)
 extent = [left_lon, right_lon, bottom_lat, top_lat]
 return extent


def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--run-dir", help="Run directory to read data from and write images into (overrides RUN_DIR env)")
 parser.add_argument("--plots", help="Comma-separated list of plots to generate: nodes,heatmap,events")
 args = parser.parse_args()
 global RUN_DIR
 RUN_DIR = args.run_dir or os.getenv("RUN_DIR")
 if args.plots:
 os.environ["VIS_PLOTS"] = args.plots
 run_visualization()


if __name__ == "__main__":
 main()
