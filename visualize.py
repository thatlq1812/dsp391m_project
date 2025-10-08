"""
Visualization script: Plot nodes map, traffic heatmap, and forecasts.
"""

import json
import matplotlib.pyplot as plt
import os
import yaml

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_nodes(limit=None):
    with open('data/nodes.json', 'r') as f:
        nodes = json.load(f)
    if limit:
        nodes = nodes[:limit]
    return nodes

def load_traffic():
    if os.path.exists('data/traffic_snapshot_normalized.json'):
        with open('data/traffic_snapshot_normalized.json', 'r') as f:
            return json.load(f)
    return []

def load_events():
    if os.path.exists('data/events_enriched.json'):
        with open('data/events_enriched.json', 'r') as f:
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
    plt.savefig(cfg['save_path'])
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
    plt.savefig(cfg['save_path'])
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
    plt.savefig(cfg['save_path'])
    # plt.show()

def run_visualization():
    config = load_config()
    print("Generating visualizations...")
    plot_nodes_map(config)
    plot_traffic_heatmap(config)
    plot_events(config)
    print("Visualizations saved to data/")

if __name__ == "__main__":
    run_visualization()