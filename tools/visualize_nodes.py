"""
Visualize node locations on map with coverage area.

Shows where nodes are selected and the selection criteria.
"""

import json
import folium
from folium import plugins
import numpy as np


def load_nodes(filepath='data/nodes.json'):
    """Load nodes from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_coverage_map(
    nodes,
    center=[10.772465, 106.697794],  # HCMC center
    radius_m=512,
    output_file='data/node_coverage_map.html'
):
    """
    Create interactive map showing node coverage.
    
    Args:
        nodes: List of node dictionaries
        center: [lat, lon] of collection center
        radius_m: Collection radius in meters
        output_file: Output HTML file path
    """
    # Create base map
    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add center marker
    folium.Marker(
        location=center,
        popup=f'<b>Collection Center</b><br>Radius: {radius_m}m',
        icon=folium.Icon(color='red', icon='info-sign'),
        tooltip='Collection Center'
    ).add_to(m)
    
    # Add collection radius circle
    folium.Circle(
        location=center,
        radius=radius_m,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.1,
        popup=f'Collection Radius: {radius_m}m'
    ).add_to(m)
    
    # Add node markers
    node_group = folium.FeatureGroup(name='Traffic Nodes')
    
    for node in nodes:
        lat, lon = node['lat'], node['lon']
        node_id = node.get('node_id', 'Unknown')
        road_type = node.get('road_type', 'unknown')
        degree = node.get('degree', 'N/A')
        
        # Color based on road type
        color_map = {
            'primary': 'blue',
            'secondary': 'green',
            'tertiary': 'orange',
            'residential': 'gray'
        }
        color = color_map.get(road_type, 'gray')
        
        popup_html = f"""
        <b>Node ID:</b> {node_id}<br>
        <b>Road Type:</b> {road_type}<br>
        <b>Degree:</b> {degree}<br>
        <b>Coordinates:</b> {lat:.6f}, {lon:.6f}<br>
        <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank">View on Google Maps</a>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f'{node_id} ({road_type})'
        ).add_to(node_group)
    
    node_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    plugins.MiniMap().add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return m


def create_heatmap(
    nodes,
    output_file='data/node_density_heatmap.html'
):
    """
    Create heatmap showing node density.
    
    Args:
        nodes: List of node dictionaries
        output_file: Output HTML file path
    """
    # Extract coordinates
    coords = [[node['lat'], node['lon']] for node in nodes]
    
    # Calculate center
    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add heatmap
    plugins.HeatMap(coords).add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Heatmap saved to {output_file}")
    return m


def main():
    """Main function to generate visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize traffic nodes on map')
    parser.add_argument('--input', default='data/nodes.json', help='Input nodes JSON file')
    parser.add_argument('--center-lat', type=float, default=10.772465, help='Map center latitude')
    parser.add_argument('--center-lon', type=float, default=106.697794, help='Map center longitude')
    parser.add_argument('--radius', type=int, default=512, help='Collection radius in meters')
    parser.add_argument('--output-map', default='data/node_coverage_map.html', help='Output map file')
    parser.add_argument('--output-heatmap', default='data/node_density_heatmap.html', help='Output heatmap file')
    parser.add_argument('--heatmap-only', action='store_true', help='Generate only heatmap')
    parser.add_argument('--map-only', action='store_true', help='Generate only coverage map')
    
    args = parser.parse_args()
    
    print("Loading nodes...")
    nodes = load_nodes(args.input)
    print(f"Loaded {len(nodes)} nodes")
    
    center = [args.center_lat, args.center_lon]
    
    if not args.heatmap_only:
        print("Generating coverage map...")
        create_coverage_map(
            nodes,
            center=center,
            radius_m=args.radius,
            output_file=args.output_map
        )
    
    if not args.map_only:
        print("Generating density heatmap...")
        create_heatmap(
            nodes,
            output_file=args.output_heatmap
        )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()