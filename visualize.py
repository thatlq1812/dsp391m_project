"""
Visualization script: Plot nodes map, traffic heatmap, and forecasts.
"""

import json
import matplotlib.pyplot as plt
import os
import yaml
import argparse


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
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_nodes(limit=None):
    path = resolve_path(RUN_DIR, 'data/nodes.json')
    with open(path, 'r') as f:
        nodes = json.load(f)
    if limit:
        nodes = nodes[:limit]
    return nodes

def load_traffic():
    path = resolve_path(RUN_DIR, 'data/traffic_snapshot_normalized.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def load_events():
    path = resolve_path(RUN_DIR, 'data/events_enriched.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def plot_nodes_map(config):
    """Plot map of all nodes."""
    nodes = load_nodes(config['visualize']['nodes_map']['limit_nodes'])
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    
    cfg = config['visualize']['nodes_map']
    plt.figure(figsize=cfg['figsize'])
    plt.scatter(lons, lats, s=cfg['s'], alpha=cfg['alpha'], c=cfg['color'], label=cfg['label'])
    plt.xlabel(cfg['xlabel'])
    plt.ylabel(cfg['ylabel'])
    plt.title(cfg['title'])
    plt.legend()
    out_dir = os.path.join(RUN_DIR or '.', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cfg['save_path']))
    plt.savefig(out_path)
    # plt.show()  # Remove to avoid issues in terminal

def plot_traffic_heatmap(config):
    """Plot traffic speed heatmap."""
    nodes = load_nodes()
    traffic = {t['node_id']: t for t in load_traffic()}
    
    lats, lons, speeds = [], [], []
    for node in nodes:
        if node['node_id'] in traffic:
            lats.append(node['lat'])
            lons.append(node['lon'])
            speeds.append(traffic[node['node_id']]['avg_speed_kmh'])
    
    cfg = config['visualize']['traffic_heatmap']
    plt.figure(figsize=cfg['figsize'])
    scatter = plt.scatter(lons, lats, s=cfg['s'], c=speeds, cmap=cfg['cmap'], alpha=cfg['alpha'])
    plt.colorbar(scatter, label='Avg Speed (km/h)')
    plt.xlabel(cfg['xlabel'])
    plt.ylabel(cfg['ylabel'])
    plt.title(cfg['title'])
    out_dir = os.path.join(RUN_DIR or '.', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cfg['save_path']))
    plt.savefig(out_path)
    # plt.show()

def plot_events(config):
    """Plot events on map."""
    nodes = load_nodes(limit=1000)  # Limit for performance
    events = load_events()
    
    cfg = config['visualize']['events_map']
    plt.figure(figsize=cfg['figsize'])
    # Plot nodes
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    plt.scatter(lons, lats, s=cfg['node_s'], alpha=cfg['node_alpha'], c=cfg['node_color'], label='Nodes')
    
    # Plot events
    for event in events:
        if 'venue_lat' in event and 'venue_lon' in event:
            plt.scatter(event['venue_lon'], event['venue_lat'], s=cfg['event_s'], c=cfg['event_color'], marker=cfg['event_marker'], label='Event')
    
    plt.xlabel(cfg['xlabel'])
    plt.ylabel(cfg['ylabel'])
    plt.title(cfg['title'])
    plt.legend()
    out_dir = os.path.join(RUN_DIR or '.', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(cfg['save_path']))
    plt.savefig(out_path)
    # plt.show()

def run_visualization():
    config = load_config()
    print("Generating visualizations...")
    plot_nodes_map(config)
    plot_traffic_heatmap(config)
    plot_events(config)
    print(f"Visualizations saved to {os.path.join(RUN_DIR or '.', 'images')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Run directory to read data from and write images into (overrides RUN_DIR env)')
    args = parser.parse_args()
    RUN_DIR = args.run_dir or os.getenv('RUN_DIR')
    run_visualization()

if __name__ == "__main__":
    run_visualization()