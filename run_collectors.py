#!/usr/bin/env python3
"""
Mock collectors runner for prototype.
Writes outputs under a run directory: <run_dir>/collectors/<collector_name>/...
"""

import csv
import json
import os
import argparse
from datetime import datetime


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_mock_collectors(run_dir=None):
    if run_dir is None:
        run_dir = os.getenv('RUN_DIR', 'data_runs/latest')
    collectors_dir = os.path.join(run_dir, 'collectors')
    ensure_dir(collectors_dir)

    # Mock traffic snapshot
    traffic_data = [
        {'node_id': 'node-1', 'timestamp': datetime.now().isoformat(), 'avg_speed_kmh': '35.0', 'vehicle_count': '120'},
        {'node_id': 'node-2', 'timestamp': datetime.now().isoformat(), 'avg_speed_kmh': '45.0', 'vehicle_count': '80'},
    ]
    out_dir = os.path.join(collectors_dir, 'mock')
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, 'traffic_snapshot.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['node_id', 'timestamp', 'avg_speed_kmh', 'vehicle_count'])
        writer.writeheader()
        writer.writerows(traffic_data)

    # Mock events
    events_data = [
        {
            'event_id': 'event-1',
            'title': 'Concert at Venue A',
            'start_time': datetime.now().isoformat(),
            'venue_lat': 10.7724655,
            'venue_lon': 106.697794,
            'expected_attendance': 500
        }
    ]
    with open(os.path.join(out_dir, 'events.json'), 'w') as f:
        json.dump(events_data, f, indent=2)

    print(f"Mock collectors completed. Data saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Run directory to save outputs (overrides RUN_DIR env)')
    args = parser.parse_args()
    rd = args.run_dir or os.getenv('RUN_DIR')
    run_mock_collectors(run_dir=rd)