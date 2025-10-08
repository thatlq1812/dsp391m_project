"""
Visualization script: Plot nodes map, traffic heatmap, and forecasts.
"""

import json
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import yaml
import argparse
import math
import io

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


def resolve_path(run_dir, relative_default):
    """Return path inside run_dir/collectors if exists, else relative_default."""
    if not run_dir:
        return relative_default
    # try collectors subdir for common files
    candidate = os.path.join(run_dir, 'collectors')
    if os.path.exists(candidate):
        # map expected file names to their location under collectors
        base = candidate
        # if relative_default is 'data/nodes.json' -> collectors/*/nodes.json (best-effort)
        fname = os.path.basename(relative_default)
        # search for fname in collectors subdirs
        for root, dirs, files in os.walk(base):
            if fname in files:
                return os.path.join(root, fname)
    return relative_default

def load_config():
    try:
        with open('configs/project_config.yaml', 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def load_nodes(limit=None):
    # Try consolidated run_data.json first
    run_data_path = os.path.join(RUN_DIR or '.', 'run_data.json')
    if os.path.exists(run_data_path):
        with open(run_data_path, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        nodes = run_data.get('nodes.json', [])
        if isinstance(nodes, list):
            if limit:
                nodes = nodes[:limit]
            return nodes
    # Fallback to old method
    # prefer enriched nodes if available
    path = resolve_path(RUN_DIR, 'data/nodes_enriched.json')
    if not os.path.exists(path):
        path = resolve_path(RUN_DIR, 'data/nodes.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    if limit:
        nodes = nodes[:limit]
    return nodes

def load_traffic():
    # Try consolidated run_data.json first
    run_data_path = os.path.join(RUN_DIR or '.', 'run_data.json')
    if os.path.exists(run_data_path):
        with open(run_data_path, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        traffic = run_data.get('traffic_snapshot_normalized.json', [])
        if isinstance(traffic, list):
            return traffic
    # Fallback
    path = resolve_path(RUN_DIR, 'data/traffic_snapshot_normalized.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return []


def speed_to_hex(speed, free_flow_kmh=50.0, jam_kmh=5.0, use_black_for_jam=True):
    """Map speed (km/h) to a hex color.
    - speeds >= free_flow_kmh -> green
    - speeds near 0 -> red, and <= jam_kmh optionally black
    """
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
    # normalize 0..1 with respect to free flow
    norm = max(0.0, min(1.0, sp / float(free_flow_kmh)))
    hue_deg = 120.0 * norm  # 0..120 deg
    h = hue_deg / 360.0
    s = 0.9
    v = 0.35 + 0.65 * norm
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def load_events():
    # Try consolidated run_data.json first
    run_data_path = os.path.join(RUN_DIR or '.', 'run_data.json')
    if os.path.exists(run_data_path):
        with open(run_data_path, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        events = run_data.get('events.json', [])
        if isinstance(events, list):
            return events
    # Fallback
    path = resolve_path(RUN_DIR, 'data/events_enriched.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def plot_nodes_map(config):
    """Plot map of all nodes."""
    vis_cfg = config.get('visualize') or {}
    nodes = load_nodes(vis_cfg.get('nodes_map', {}).get('limit_nodes'))
    if not nodes:
        print('No nodes found for plotting nodes_map; skipping')
        return
    lats = [n.get('lat') for n in nodes]
    lons = [n.get('lon') for n in nodes]
    cfg = vis_cfg.get('nodes_map', {})
    plt.figure(figsize=cfg.get('figsize', (10, 8)))
    # draw basemap if provided via global variables
    try:
        if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
            plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)
    except NameError:
        pass
    # color by speed if available in nodes or from traffic snapshot
    traffic = {t.get('node_id') or t.get('node') or t.get('id'): t for t in load_traffic()}
    colors = []
    for n in nodes:
        sp = n.get('speed_kmh') or n.get('speed') or traffic.get(n.get('node_id', ''), {}).get('avg_speed_kmh')
        colors.append(speed_to_hex(sp))
    plt.scatter(lons, lats, s=cfg.get('s', 1), alpha=cfg.get('alpha', 0.5), c=colors, label=cfg.get('label', 'Nodes'))
    plt.xlabel(cfg.get('xlabel', 'Longitude'))
    plt.ylabel(cfg.get('ylabel', 'Latitude'))
    plt.title(cfg.get('title', 'Nodes Map'))
    plt.legend()
    out_dir = os.getenv('RUN_IMAGE_DIR') or os.path.join(RUN_DIR or '.', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cfg.get('save_path', 'nodes_map.png')))
    plt.savefig(out_path)
    # plt.show()  # Remove to avoid issues in terminal

def plot_traffic_heatmap(config):
    """Plot traffic speed heatmap."""
    nodes = load_nodes()
    traffic = {t.get('node_id') or t.get('node') or t.get('id'): t for t in load_traffic()}
    if not nodes or not traffic:
        print('No nodes or traffic data for heatmap; skipping')
        return

    lats, lons, speeds = [], [], []
    for node in nodes:
        nid = node.get('node_id') or node.get('id')
        if nid in traffic:
            lats.append(node.get('lat'))
            lons.append(node.get('lon'))
            speeds.append(traffic[nid].get('avg_speed_kmh'))

    vis_cfg = config.get('visualize') or {}
    cfg = vis_cfg.get('traffic_heatmap', {})
    plt.figure(figsize=cfg.get('figsize', (10, 8)))
    try:
        if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
            plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)
    except NameError:
        pass
    # map speeds -> colors using speed_to_hex for better contrast at low speeds
    colors = [speed_to_hex(s) for s in speeds]
    scatter = plt.scatter(lons, lats, s=cfg.get('s', 5), c=colors, alpha=cfg.get('alpha', 0.7))
    plt.xlabel(cfg.get('xlabel', 'Longitude'))
    plt.ylabel(cfg.get('ylabel', 'Latitude'))
    plt.title(cfg.get('title', 'Traffic Speed Heatmap'))
    out_path = os.getenv('RUN_IMAGE_FILE') or os.path.join(os.getenv('RUN_IMAGE_DIR') or os.path.join(RUN_DIR or '.', 'images'), os.path.basename(cfg.get('save_path', 'traffic_heatmap.png')))
    plt.savefig(out_path)
    # plt.show()

def plot_events(config):
    """Plot events on map."""
    nodes = load_nodes(limit=1000)  # Limit for performance
    events = load_events()
    if not nodes and not events:
        print('No nodes or events to plot; skipping')
        return

    vis_cfg = config.get('visualize') or {}
    cfg = vis_cfg.get('events_map', {})
    plt.figure(figsize=cfg.get('figsize', (10, 8)))
    try:
        if GLOBAL_BASEMAP_IMG is not None and GLOBAL_BASEMAP_EXTENT is not None:
            plt.imshow(GLOBAL_BASEMAP_IMG, extent=GLOBAL_BASEMAP_EXTENT, zorder=0)
    except NameError:
        pass
    # Plot nodes
    lats = [n.get('lat') for n in nodes]
    lons = [n.get('lon') for n in nodes]
    plt.scatter(lons, lats, s=cfg.get('node_s', 1), alpha=cfg.get('node_alpha', 0.3), c=cfg.get('node_color', 'blue'), label='Nodes')

    # Plot events
    for event in events:
        if 'venue_lat' in event and 'venue_lon' in event:
            plt.scatter(event['venue_lon'], event['venue_lat'], s=cfg.get('event_s', 100), c=cfg.get('event_color', 'red'), marker=cfg.get('event_marker', 'x'), label='Event')

    plt.xlabel(cfg.get('xlabel', 'Longitude'))
    plt.ylabel(cfg.get('ylabel', 'Latitude'))
    plt.title(cfg.get('title', 'Events and Nodes'))
    plt.legend()
    out_dir = os.getenv('RUN_IMAGE_DIR') or os.path.join(RUN_DIR or '.', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cfg.get('save_path', 'events_map.png')))
    plt.savefig(out_path)
    # plt.show()

def run_visualization():
    config = load_config()
    print("Generating visualizations...")
    # Enable basemap by default
    if not os.environ.get('VIS_USE_BASEMAP'):
        os.environ['VIS_USE_BASEMAP'] = '1'
    # optionally fetch basemap if requested via env flags
    basemap = os.environ.get('VIS_USE_BASEMAP', '0') == '1'
    basemap_zoom = int(os.environ.get('VIS_BASEMAP_ZOOM', 14))
    basemap_size = int(os.environ.get('VIS_BASEMAP_SIZE', 800))
    basemap_img = None
    basemap_extent = None
    # determine run image dir (where per-run basemap should be stored)
    run_image_dir = os.path.dirname(os.getenv('RUN_IMAGE_FILE') or os.path.join(RUN_DIR or '.', 'images', 'temp.png'))
    os.makedirs(run_image_dir, exist_ok=True)
    run_basemap_path = os.path.join(run_image_dir, 'basemap.png')

    if basemap and Image is not None and requests is not None:
        # compute center from nodes if available
        nodes = load_nodes()
        if nodes:
            lats = [n.get('lat') for n in nodes]
            lons = [n.get('lon') for n in nodes]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            try:
                # First check per-run basemap (download once per run)
                if os.path.exists(run_basemap_path):
                    try:
                        basemap_img = Image.open(run_basemap_path).convert('RGBA')
                        basemap_extent = compute_basemap_extent(center_lat, center_lon, basemap_zoom, basemap_size)
                    except Exception:
                        basemap_img = None
                        basemap_extent = None
                # If no per-run basemap, check global cache by rounded center
                if basemap_img is None:
                    ck_lat = round(center_lat, 5)
                    ck_lon = round(center_lon, 5)
                    cache_dir = os.path.join('.cache', 'basemap')
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_key = f"bm_z{basemap_zoom}_s{basemap_size}_lat{ck_lat}_lon{ck_lon}.png"
                    cache_path = os.path.join(cache_dir, cache_key)
                    if os.path.exists(cache_path):
                        try:
                            basemap_img = Image.open(cache_path).convert('RGBA')
                            basemap_extent = compute_basemap_extent(center_lat, center_lon, basemap_zoom, basemap_size)
                            # also copy into run_basemap for per-run single-download semantics
                            try:
                                basemap_img.save(run_basemap_path)
                            except Exception:
                                pass
                        except Exception:
                            basemap_img = None
                            basemap_extent = None
                # If still no basemap, fetch (prefer Google Satellite if key present)
                if basemap_img is None:
                    # set VIS_USE_GOOGLE if GOOGLE_MAPS_API_KEY exists and user didn't explicitly disable
                    if os.environ.get('VIS_USE_GOOGLE', '0') != '1' and os.environ.get('GOOGLE_MAPS_API_KEY'):
                        os.environ['VIS_USE_GOOGLE'] = '1'
                    basemap_img, basemap_extent = fetch_static_basemap(center_lat, center_lon, basemap_zoom, basemap_size)
                    try:
                        # save to both run-level and global cache
                        basemap_img.save(run_basemap_path)
                    except Exception:
                        pass
                    try:
                        # also save to global cache
                        os.makedirs(cache_dir, exist_ok=True)
                        basemap_img.save(cache_path)
                    except Exception:
                        pass
            except Exception as e:
                print('Warning: failed to fetch basemap:', e)
        else:
            print('No nodes found; skipping basemap fetch')
    # set module-level globals so plotting functions can access
    global GLOBAL_BASEMAP_IMG, GLOBAL_BASEMAP_EXTENT
    GLOBAL_BASEMAP_IMG = basemap_img
    GLOBAL_BASEMAP_EXTENT = basemap_extent
    # Only plot traffic heatmap
    plot_traffic_heatmap(config)
    out_file = os.getenv('RUN_IMAGE_FILE') or os.path.join(RUN_DIR or '.', 'images', 'traffic_heatmap.png')
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
    """Fetch a static map image centered at lat/lon using Google or OpenStreetMap as fallback.

    Returns: (PIL.Image, extent) where extent = [min_lon, max_lon, min_lat, max_lat]
    """
    if requests is None or Image is None:
        raise RuntimeError('PIL or requests not available')
    # Determine service: prefer Google Static Maps if requested and key available
    use_google = os.environ.get('VIS_USE_GOOGLE', '0') == '1' or os.environ.get('GOOGLE_MAPS_API_KEY')
    if use_google and os.environ.get('GOOGLE_MAPS_API_KEY'):
        key = os.environ.get('GOOGLE_MAPS_API_KEY')
        # Google static maps accepts size up to 640x640 for free tier unless using scale=2;
        # choose size within limits and prefer satellite imagery
        gsize = f"{min(size,640)}x{min(size,640)}"
        maptype = os.environ.get('VIS_GOOGLE_MAPTYPE', 'satellite')
        print(f'Trying Google Static Maps (type={maptype}, size={gsize})')
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lon}&zoom={zoom}&size={gsize}&maptype={maptype}&key={key}&scale=1"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert('RGBA')
            print('Fetched basemap image from Google')
        except Exception as e:
            print(f'Google Maps failed: {e}, falling back to OpenStreetMap')
            use_google = False
    if not use_google:
        # build URL (openstreetmap.de static map service)
        print('Using OpenStreetMap static map service')
        url = f"https://staticmap.openstreetmap.de/staticmap.php?center={center_lat},{center_lon}&zoom={zoom}&size={size}x{size}&maptype=mapnik"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('RGBA')
        print('Fetched basemap image from OpenStreetMap')

    # compute geographic extent from center, zoom and image size
    cx, cy = deg2num(center_lat, center_lon, zoom)
    tiles_per_img_x = size / 256.0
    tiles_per_img_y = size / 256.0
    left_xtile = cx - tiles_per_img_x / 2.0
    right_xtile = cx + tiles_per_img_x / 2.0
    top_ytile = cy - tiles_per_img_y / 2.0
    bottom_ytile = cy + tiles_per_img_y / 2.0

    # convert tile coordinates to lat/lon
    top_lat, left_lon = num2deg(left_xtile, top_ytile, zoom)
    bottom_lat, right_lon = num2deg(right_xtile, bottom_ytile, zoom)

    # extent for imshow is [xmin, xmax, ymin, ymax] in data coords (lon, lon, lat, lat)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Run directory to read data from and write images into (overrides RUN_DIR env)')
    parser.add_argument('--plots', help='Comma-separated list of plots to generate: nodes,heatmap,events')
    args = parser.parse_args()
    RUN_DIR = args.run_dir or os.getenv('RUN_DIR')
    # allow explicit image file passed with env
    RUN_IMAGE_FILE = os.getenv('RUN_IMAGE_FILE')
    if args.plots:
        os.environ['VIS_PLOTS'] = args.plots
    run_visualization()