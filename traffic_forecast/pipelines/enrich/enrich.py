"""
Enricher pipeline: Geocode events and compute impact scores for nearby nodes.
"""

import json
from datetime import datetime

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from traffic_forecast import PROJECT_ROOT


def load_events() -> list:
 path = PROJECT_ROOT / 'data' / 'events.json'
 if not path.exists():
 return []
 with path.open('r', encoding='utf-8') as fh:
 return json.load(fh)


def load_nodes() -> list:
 path = PROJECT_ROOT / 'data' / 'nodes.json'
 if not path.exists():
 return []
 with path.open('r', encoding='utf-8') as fh:
 return json.load(fh)


def geocode_event(event: dict) -> dict:
 """Geocode event venue using Nominatim (fallback)."""
 geolocator = Nominatim(user_agent="traffic-forecast")
 venue_name = event.get('venue_name', event.get('title', 'Ho Chi Minh City'))
 location = geolocator.geocode(f"{venue_name}, Ho Chi Minh City, Vietnam")
 if location:
 event['venue_lat'] = location.latitude
 event['venue_lon'] = location.longitude
 return event


def compute_impact_scores(events: list, nodes: list, radius_km: float = 2.0) -> list:
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
 attendance = event.get('expected_attendance', 100)
 impact_score = (attendance / 1000) * (1 / (1 + dist))
 impacts.append(
 {
 'event_id': event['event_id'],
 'node_id': node['node_id'],
 'distance_km': dist,
 'impact_score': impact_score,
 'timestamp': datetime.now().isoformat(),
 }
 )
 return impacts


def run_enrich() -> None:
 events = [geocode_event(event) for event in load_events()]
 nodes = load_nodes()
 impacts = compute_impact_scores(events, nodes)

 events_path = PROJECT_ROOT / 'data' / 'events_enriched.json'
 impacts_path = PROJECT_ROOT / 'data' / 'event_impacts.json'
 events_path.parent.mkdir(parents=True, exist_ok=True)

 with events_path.open('w', encoding='utf-8') as fh:
 json.dump(events, fh, indent=2)

 with impacts_path.open('w', encoding='utf-8') as fh:
 json.dump(impacts, fh, indent=2)

 print(f"Enriched {len(events)} events, computed {len(impacts)} impacts.")


if __name__ == "__main__":
 run_enrich()
