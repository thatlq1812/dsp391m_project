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
        fillColor='red',
        fillOpacity=0.1,
        popup=f'Collection Radius: {radius_m}m',
        tooltip='Coverage Area'
    ).add_to(m)
    
    # Add nodes
    for i, node in enumerate(nodes, 1):
        lat, lon = node['lat'], node['lon']
        node_id = node.get('node_id', 'N/A')
        road_type = node.get('road_type', 'unknown')
        
        # Get additional info
        degree = node.get('degree', 'N/A')
        importance = node.get('importance_score', 'N/A')
        street_names = node.get('street_names', [])
        intersection_name = node.get('intersection_name', 'Unknown intersection')
        
        # Color by road type
        color_map = {
            'motorway': 'darkred',
            'trunk': 'red',
            'primary': 'orange',
            'secondary': 'blue',
            'tertiary': 'green',
            'residential': 'lightgray',
            'unknown': 'gray'
        }
        color = color_map.get(road_type, 'gray')
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0;">Node #{i}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>ID:</b></td><td>{node_id}</td></tr>
                <tr><td><b>Location:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                <tr><td><b>Intersection:</b></td><td>{intersection_name}</td></tr>
                <tr><td><b>Road Type:</b></td><td><span style="color: {color}; font-weight: bold;">{road_type}</span></td></tr>
                <tr><td><b>Degree:</b></td><td>{degree}</td></tr>
                <tr><td><b>Importance:</b></td><td>{importance}</td></tr>
            </table>
        """
        
        if street_names:
            popup_html += "<br><b>Streets:</b><ul style='margin: 5px 0; padding-left: 20px;'>"
            for street in street_names[:5]:  # Limit to 5
                popup_html += f"<li>{street}</li>"
            popup_html += "</ul>"
        
        popup_html += f"""
            <br>
            <a href='https://www.google.com/maps?q={lat},{lon}' target='_blank' 
               style='color: #1a73e8; text-decoration: none;'>
                📍 View on Google Maps
            </a>
        </div>
        """
        
        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Node #{i}: {intersection_name}"
        ).add_to(m)
    
    # Add heatmap layer (optional)
    heat_data = [[node['lat'], node['lon'], node.get('importance_score', 20)] 
                 for node in nodes if 'importance_score' in node]
    
    if heat_data:
        plugins.HeatMap(
            heat_data,
            name='Node Importance Heatmap',
            min_opacity=0.2,
            radius=25,
            blur=15,
            gradient={
                0.0: 'blue',
                0.5: 'lime',
                1.0: 'red'
            }
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: auto;
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; border-radius: 5px; padding: 10px;">
        <p style="margin: 0 0 10px 0;"><b>Node Legend</b></p>
        <p style="margin: 5px 0;"><span style="color: darkred;">●</span> Motorway</p>
        <p style="margin: 5px 0;"><span style="color: red;">●</span> Trunk</p>
        <p style="margin: 5px 0;"><span style="color: orange;">●</span> Primary</p>
        <p style="margin: 5px 0;"><span style="color: blue;">●</span> Secondary</p>
        <p style="margin: 5px 0;"><span style="color: green;">●</span> Tertiary</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0;"><b>Total Nodes:</b> {len(nodes)}</p>
        <p style="margin: 5px 0;"><b>Coverage:</b> {radius_m}m radius</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"✅ Map saved to: {output_file}")
    print(f"   Total nodes: {len(nodes)}")
    print(f"   Center: {center}")
    print(f"   Radius: {radius_m}m")
    
    return m


def analyze_node_distribution(nodes):
    """Analyze spatial distribution of nodes."""
    if not nodes:
        print("No nodes to analyze")
        return
    
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    
    print("\n📊 Node Distribution Analysis")
    print("=" * 60)
    print(f"Total nodes: {len(nodes)}")
    print(f"\nLatitude range: {min(lats):.6f} to {max(lats):.6f}")
    print(f"Longitude range: {min(lons):.6f} to {max(lons):.6f}")
    print(f"\nCenter: {np.mean(lats):.6f}, {np.mean(lons):.6f}")
    
    # Road type distribution
    road_types = {}
    for node in nodes:
        rt = node.get('road_type', 'unknown')
        road_types[rt] = road_types.get(rt, 0) + 1
    
    print("\n📍 Road Type Distribution:")
    for rt, count in sorted(road_types.items(), key=lambda x: -x[1]):
        pct = count / len(nodes) * 100
        print(f"  {rt:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Importance score distribution (if available)
    if any('importance_score' in n for n in nodes):
        scores = [n.get('importance_score', 0) for n in nodes if 'importance_score' in n]
        print(f"\n⭐ Importance Score:")
        print(f"  Min:  {min(scores):.1f}")
        print(f"  Max:  {max(scores):.1f}")
        print(f"  Mean: {np.mean(scores):.1f}")
        print(f"  Std:  {np.std(scores):.1f}")
    
    # Degree distribution (if available)
    if any('degree' in n for n in nodes):
        degrees = [n.get('degree', 0) for n in nodes if 'degree' in n]
        print(f"\n🔀 Intersection Degree:")
        print(f"  Min:  {min(degrees)}")
        print(f"  Max:  {max(degrees)}")
        print(f"  Mean: {np.mean(degrees):.1f}")


def create_comparison_map(
    nodes,
    center=[10.772465, 106.697794],
    show_all_roads=True,
    output_file='data/node_selection_comparison.html'
):
    """
    Create map showing selected nodes vs all possible roads.
    Helps visualize how selective the algorithm is.
    """
    m = folium.Map(
        location=center,
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add selected nodes
    feature_group_selected = folium.FeatureGroup(name='Selected Nodes', show=True)
    
    for node in nodes:
        lat, lon = node['lat'], node['lon']
        road_type = node.get('road_type', 'unknown')
        
        color_map = {
            'motorway': 'darkred',
            'trunk': 'red',
            'primary': 'orange',
            'secondary': 'blue',
            'tertiary': 'green'
        }
        color = color_map.get(road_type, 'gray')
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"{node.get('intersection_name', 'Node')}"
        ).add_to(feature_group_selected)
    
    feature_group_selected.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    m.save(output_file)
    print(f"✅ Comparison map saved to: {output_file}")


def main():
    """Main function to create visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize traffic nodes on map')
    parser.add_argument('--input', default='data/nodes.json', help='Input nodes JSON file')
    parser.add_argument('--output', default='data/node_coverage_map.html', 
                       help='Output HTML map file')
    parser.add_argument('--center-lat', type=float, default=10.772465,
                       help='Center latitude')
    parser.add_argument('--center-lon', type=float, default=106.697794,
                       help='Center longitude')
    parser.add_argument('--radius', type=int, default=512,
                       help='Coverage radius in meters')
    
    args = parser.parse_args()
    
    print("📍 Loading nodes...")
    nodes = load_nodes(args.input)
    
    print("\n📊 Analyzing distribution...")
    analyze_node_distribution(nodes)
    
    print("\n🗺️  Creating map...")
    center = [args.center_lat, args.center_lon]
    create_coverage_map(nodes, center, args.radius, args.output)
    
    print(f"\n✅ Done! Open {args.output} in your browser to view the map.")


if __name__ == '__main__':
    main()
