"""
Enricher pipeline: Geocode events and compute impact scores for nearby nodes.
"""

import json
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime

def load_events():
    with open('data/events.json', 'r') as f:
        return json.load(f)

def load_nodes():
    with open('data/nodes.json', 'r') as f:
        return json.load(f)

def geocode_event(event):
    """Geocode event venue using Nominatim (fallback)."""
    geolocator = Nominatim(user_agent="traffic-forecast")
    venue_name = event.get('venue_name', event.get('title', 'Ho Chi Minh City'))
    location = geolocator.geocode(venue_name + ", Ho Chi Minh City, Vietnam")
    if location:
        event['venue_lat'] = location.latitude
        event['venue_lon'] = location.longitude
    return event

def compute_impact_scores(events, nodes, radius_km=2.0):
    """Compute impact score for each event-node pair within radius."""
    impacts = []
    for event in events:
        if 'venue_lat' not in event or 'venue_lon' not in event:
            continue
        event_loc = (event['venue_lat'], event['venue_lon'])
        for node in nodes:
            node_loc = (node['lat'], node['lon'])
            dist = geodesic(event_loc, node_loc).km
            if dist <= radius_km:
                # Simple impact score: inverse distance, scaled by attendance
                attendance = event.get('expected_attendance', 100)
                impact_score = (attendance / 1000) * (1 / (1 + dist))  # Normalize
                impacts.append({
                    'event_id': event['event_id'],
                    'node_id': node['node_id'],
                    'distance_km': dist,
                    'impact_score': impact_score,
                    'timestamp': datetime.now().isoformat()
                })
    return impacts

def run_enrich():
    events = load_events()
    nodes = load_nodes()
    
    # Geocode events
    enriched_events = [geocode_event(event) for event in events]
    
    # Compute impacts
    impacts = compute_impact_scores(enriched_events, nodes)
    
    # Save enriched data
    with open('data/events_enriched.json', 'w') as f:
        json.dump(enriched_events, f, indent=2)
    
    with open('data/event_impacts.json', 'w') as f:
        json.dump(impacts, f, indent=2)
    
    print(f"Enriched {len(enriched_events)} events, computed {len(impacts)} impacts.")

if __name__ == "__main__":
    run_enrich()