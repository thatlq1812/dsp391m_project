#!/usr/bin/env python3
"""
Combine monthly runs into a single parquet file - simplified version with proper encoding.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_run(run_dir: Path) -> list:
    """Load a single run's traffic data"""
    
    # Required files
    traffic_file = run_dir / 'traffic_edges.json'
    weather_file = run_dir / 'weather_snapshot.json'
    
    if not traffic_file.exists():
        return []
    
    # Parse timestamp from directory name
    run_name = run_dir.name
    timestamp_str = run_name.replace('run_', '')
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        return []
    
    # Load traffic data
    try:
        with open(traffic_file, 'r', encoding='utf-8') as f:
            traffic_data = json.load(f)
    except Exception as e:
        print(f"Error loading {traffic_file}: {e}")
        return []
    
    # Load weather if exists
    weather = {}
    if weather_file.exists():
        try:
            with open(weather_file, 'r', encoding='utf-8') as f:
                weather_data = json.load(f)
                # Handle both dict and list formats
                if isinstance(weather_data, dict):
                    weather = weather_data
                elif isinstance(weather_data, list) and len(weather_data) > 0:
                    weather = weather_data[0]  # Take first element
        except Exception:
            pass
    
    # Create records
    records = []
    for edge in traffic_data:
        record = {
            'timestamp': timestamp,
            'node_a_id': edge['node_a_id'],
            'node_b_id': edge['node_b_id'],
            'speed_kmh': edge['speed_kmh'],
            'distance_km': edge.get('distance_km', 0.0),
            'travel_time_minutes': edge.get('travel_time_minutes', 0.0),
            'temperature_c': weather.get('temperature_c', 30.0),
            'humidity_percent': weather.get('humidity_percent', 75.0),
            'precipitation_mm': weather.get('precipitation_mm', 0.0),
            'wind_speed_kmh': weather.get('wind_speed_kmh', 10.0),
            'weather_condition': weather.get('condition', 'clear'),
        }
        records.append(record)
    
    return records


def combine_monthly_runs(
    runs_dir: Path,
    output_file: Path,
):
    """Combine all runs into a single parquet file"""
    
    print("="*70)
    print("COMBINING MONTHLY RUNS")
    print("="*70)
    print(f"\nSource: {runs_dir}")
    print(f"Output: {output_file}")
    
    # Find all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('run_')])
    
    print(f"\nFound {len(run_dirs)} runs")
    
    # Load all runs
    all_records = []
    
    for run_dir in tqdm(run_dirs, desc="Loading runs"):
        records = load_run(run_dir)
        all_records.extend(records)
    
    # Create DataFrame
    print(f"\nCreating DataFrame from {len(all_records)} records...")
    df = pd.DataFrame(all_records)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['date'] = df['timestamp'].dt.date
    
    # Stats
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Unique edges: {df[['node_a_id', 'node_b_id']].drop_duplicates().shape[0]}")
    print(f"Unique timestamps: {df['timestamp'].nunique():,}")
    print(f"\nSpeed statistics:")
    print(f"  Mean: {df['speed_kmh'].mean():.2f} km/h")
    print(f"  Std: {df['speed_kmh'].std():.2f} km/h")
    print(f"  Min: {df['speed_kmh'].min():.2f} km/h")
    print(f"  Max: {df['speed_kmh'].max():.2f} km/h")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_file}")
    df.to_parquet(output_file, index=False, compression='snappy')
    
    # File size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("COMBINE COMPLETE")
    print("="*70)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine monthly runs into parquet file",
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=Path('data/runs'),
        help='Directory containing runs (default: data/runs)',
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('data/processed/all_runs_monthly.parquet'),
        help='Output parquet file (default: data/processed/all_runs_monthly.parquet)',
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else PROJECT_ROOT / args.runs_dir
    output_file = args.output_file if args.output_file.is_absolute() else PROJECT_ROOT / args.output_file
    
    # Combine
    combine_monthly_runs(runs_dir, output_file)


if __name__ == '__main__':
    main()
