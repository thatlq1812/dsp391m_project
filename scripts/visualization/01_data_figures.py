"""
Generate data-related figures (Figures 1-4)

Section 5: Data Description
Section 6: Data Preprocessing
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import save_figure, load_parquet_data, FIGURE_DIR

def generate_fig1_speed_distribution():
    """Figure 1: Traffic Speed Distribution"""
    print("Generating Figure 1: Speed Distribution...")
    
    df = load_parquet_data()
    speeds = df['speed_kmh'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(speeds, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(speeds)
    x_range = np.linspace(speeds.min(), speeds.max(), 1000)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Statistics
    mean_speed = speeds.mean()
    median_speed = np.median(speeds)
    
    ax.axvline(mean_speed, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.1f} km/h')
    ax.axvline(median_speed, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_speed:.1f} km/h')
    
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Density')
    ax.set_title('Traffic Speed Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_figure(fig, 'fig01_speed_distribution')

def generate_fig2_network_topology():
    """Figure 2: Road Network Topology"""
    print("Generating Figure 2: Network Topology...")
    
    import json
    import networkx as nx
    
    # Load topology
    topology_path = Path(__file__).parents[2] / "cache" / "overpass_topology.json"
    with open(topology_path) as f:
        topology = json.load(f)
    
    # Load adjacency matrix
    adj_path = Path(__file__).parents[2] / "cache" / "adjacency_matrix.npy"
    adj_matrix = np.load(adj_path)
    
    # Create graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Extract positions (nodes is a list)
    nodes = topology['nodes']
    pos = {i: (nodes[i]['lon'], nodes[i]['lat']) for i in range(len(nodes))}
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray', ax=ax)
    
    # Draw nodes (colored by degree)
    degrees = dict(G.degree())
    node_colors = [degrees[i] for i in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, 
                          cmap='viridis', alpha=0.8, ax=ax)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('HCMC Road Network Topology (62 Nodes, 144 Edges)')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Node Degree')
    
    save_figure(fig, 'fig02_network_topology')

def generate_fig4_normalization():
    """Figure 4: Normalization Effects"""
    print("Generating Figure 4: Normalization Effects...")
    
    df = load_parquet_data()
    raw_speeds = df['speed_kmh'].values
    
    # Compute normalized speeds (assuming z-score normalization)
    mean = raw_speeds.mean()
    std = raw_speeds.std()
    normalized_speeds = (raw_speeds - mean) / std
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw distribution
    ax1.hist(raw_speeds, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Speed (km/h)')
    ax1.set_ylabel('Density')
    ax1.set_title('Before Normalization (Raw Data)')
    ax1.grid(alpha=0.3)
    ax1.axvline(mean, color='red', linestyle='--', label=f'μ={mean:.1f}')
    ax1.legend()
    
    # Normalized distribution
    ax2.hist(normalized_speeds, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Normalized Speed (z-score)')
    ax2.set_ylabel('Density')
    ax2.set_title('After Normalization (Z-score)')
    ax2.grid(alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', label='μ=0')
    ax2.legend()
    
    plt.tight_layout()
    save_figure(fig, 'fig04_normalization')

def main():
    """Generate all data figures"""
    print(f"Output directory: {FIGURE_DIR}\n")
    
    generate_fig1_speed_distribution()
    generate_fig2_network_topology()
    generate_fig4_normalization()
    
    print(f"\nAll data figures generated in: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
