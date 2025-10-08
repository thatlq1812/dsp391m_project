#!/usr/bin/env python3
"""
Mock collectors runner for prototype.
"""

import csv
import json
import os
from datetime import datetime

def run_mock_collectors():
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)

    # Mock traffic snapshot
    traffic_data = [
        {'node_id': 'node-1', 'timestamp': datetime.now().isoformat(), 'avg_speed_kmh': '35.0', 'vehicle_count': '120'},
        {'node_id': 'node-2', 'timestamp': datetime.now().isoformat(), 'avg_speed_kmh': '45.0', 'vehicle_count': '80'},
    ]
    with open('data/traffic_snapshot.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['node_id', 'timestamp', 'avg_speed_kmh', 'vehicle_count'])
        writer.writeheader()
        writer.writerows(traffic_data)

    # Mock events
    events_data = [
        {
            'event_id': 'event-1',
            'title': 'Concert at Venue A',
            'start_time': datetime.now().isoformat(),
            'venue_lat': 10.75,
            'venue_lon': 106.7,
            'expected_attendance': 500
        }
    ]
    with open('data/events.json', 'w') as f:
        json.dump(events_data, f, indent=2)

    print("Mock collectors completed. Data saved to data/")

if __name__ == "__main__":
    run_mock_collectors()