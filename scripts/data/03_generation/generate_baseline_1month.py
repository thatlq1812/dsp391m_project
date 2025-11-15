#!/usr/bin/env python3
"""
Generate synthetic runs for 1-month dataset from existing 4-day data.

This script:
1. Analyzes existing 4-day runs (30/10 - 02/11/2025)
2. Generates runs for 1 month back from last run (03/10 - 02/11/2025)
3. Uses gap-fill + realistic noise to create 26 missing days
4. Maintains 15-minute intervals with randomized seconds
5. Creates realistic traffic patterns with temporal variations

Output: Synthetic runs in data/runs/ matching real run structure
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MonthlyRunGenerator:
    """Generate synthetic monthly runs from existing 4-day data"""
    
    def __init__(
        self,
        runs_dir: Path,
        output_dir: Path,
        reference_run: str = "run_20251102_110036",
        random_seed: int = 42,
    ):
        self.runs_dir = runs_dir
        self.output_dir = output_dir
        self.reference_run = reference_run
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Timeline: 1 month back from last run
        # Last run: 2025-11-02 11:00:36
        # Start: 2025-10-03 11:00:00 (1 month before)
        self.end_date = datetime(2025, 11, 2, 11, 0, 36)
        self.start_date = self.end_date - timedelta(days=30)
        
        print(f"Timeline: {self.start_date} -> {self.end_date}")
        print(f"Duration: 30 days")
    
    def load_reference_data(self) -> Dict:
        """Load reference run structure and data"""
        ref_path = self.runs_dir / self.reference_run
        
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference run not found: {ref_path}")
        
        print(f"\nLoading reference: {self.reference_run}")
        
        data = {}
        
        # Load all JSON files
        for json_file in ['nodes.json', 'edges.json', 'traffic_edges.json', 
                          'weather_snapshot.json', 'statistics.json']:
            file_path = ref_path / json_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[json_file] = json.load(f)
                print(f"  ✓ {json_file}: {len(data[json_file]) if isinstance(data[json_file], list) else 'loaded'}")
        
        return data
    
    def load_all_existing_runs(self) -> pd.DataFrame:
        """Load and combine all existing runs for pattern analysis"""
        print("\nLoading existing runs for pattern analysis...")
        
        all_traffic = []
        run_dirs = sorted([d for d in self.runs_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            traffic_file = run_dir / 'traffic_edges.json'
            if not traffic_file.exists():
                continue
            
            with open(traffic_file, 'r', encoding='utf-8') as f:
                traffic_data = json.load(f)
            
            # Parse timestamp from directory name
            timestamp_str = run_dir.name.replace('run_', '')
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            for edge in traffic_data:
                all_traffic.append({
                    'timestamp': timestamp,
                    'node_a_id': edge['node_a_id'],
                    'node_b_id': edge['node_b_id'],
                    'speed_kmh': edge['speed_kmh'],
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'is_weekend': timestamp.weekday() >= 5,
                })
        
        df = pd.DataFrame(all_traffic)
        print(f"  ✓ Loaded {len(run_dirs)} runs, {len(df)} traffic records")
        print(f"  ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze traffic patterns for realistic generation"""
        print("\nAnalyzing traffic patterns...")
        
        patterns = {}
        
        # Speed statistics per edge
        edge_stats = df.groupby(['node_a_id', 'node_b_id'])['speed_kmh'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).to_dict('index')
        
        patterns['edge_stats'] = edge_stats
        
        # Hourly patterns
        hourly = df.groupby('hour')['speed_kmh'].agg(['mean', 'std']).to_dict('index')
        patterns['hourly'] = hourly
        
        # Weekend vs weekday
        weekend_stats = df.groupby('is_weekend')['speed_kmh'].agg(['mean', 'std']).to_dict('index')
        patterns['weekend_stats'] = weekend_stats
        
        # Rush hour detection (low speed periods)
        patterns['morning_rush'] = (7, 9)  # 7-9 AM
        patterns['evening_rush'] = (17, 19)  # 5-7 PM
        
        print(f"  ✓ {len(edge_stats)} unique edges")
        print(f"  ✓ Hourly patterns: {len(hourly)} hours")
        print(f"  ✓ Avg speed: {df['speed_kmh'].mean():.2f} km/h")
        print(f"  ✓ Weekday avg: {weekend_stats[False]['mean']:.2f} km/h")
        print(f"  ✓ Weekend avg: {weekend_stats[True]['mean']:.2f} km/h")
        
        return patterns
    
    def generate_speed(
        self, 
        edge_key: tuple,
        timestamp: datetime,
        patterns: Dict,
        base_speed: float,
    ) -> float:
        """Generate realistic speed with temporal patterns and noise"""
        
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Base speed from patterns
        if edge_key in patterns['edge_stats']:
            stats = patterns['edge_stats'][edge_key]
            mean_speed = stats['mean']
            std_speed = stats['std']
        else:
            mean_speed = base_speed
            std_speed = base_speed * 0.15
        
        # Hourly adjustment
        if hour in patterns['hourly']:
            # Use median hour as baseline, fallback to mean of all hours
            baseline_hours = [h for h in patterns['hourly'].keys() if 10 <= h <= 14]
            if baseline_hours:
                baseline_speed = np.mean([patterns['hourly'][h]['mean'] for h in baseline_hours])
            else:
                baseline_speed = np.mean([patterns['hourly'][h]['mean'] for h in patterns['hourly'].keys()])
            
            hourly_factor = patterns['hourly'][hour]['mean'] / baseline_speed
        else:
            hourly_factor = 1.0
        
        # Rush hour reduction
        morning_rush = patterns['morning_rush']
        evening_rush = patterns['evening_rush']
        
        rush_factor = 1.0
        if morning_rush[0] <= hour <= morning_rush[1]:
            rush_factor = 0.7  # 30% slower
        elif evening_rush[0] <= hour <= evening_rush[1]:
            rush_factor = 0.65  # 35% slower
        
        # Weekend boost
        weekend_factor = 1.1 if is_weekend else 1.0
        
        # Random incidents (5% chance of significant slowdown)
        incident_factor = 1.0
        if np.random.random() < 0.05:
            incident_factor = np.random.uniform(0.4, 0.7)  # 30-60% slower
        
        # Combine factors
        adjusted_speed = mean_speed * hourly_factor * rush_factor * weekend_factor * incident_factor
        
        # Add Gaussian noise
        noise = np.random.normal(0, std_speed * 0.3)
        final_speed = adjusted_speed + noise
        
        # Clamp to realistic range
        final_speed = np.clip(final_speed, 3.0, 52.0)
        
        return round(final_speed, 2)
    
    def generate_weather(self, timestamp: datetime) -> Dict:
        """Generate realistic weather data"""
        
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Temperature: 25-35°C, varies by time of day
        base_temp = 30.0
        daily_variation = 5.0 * np.sin((hour - 6) * np.pi / 12)  # Peak at 2 PM
        seasonal_variation = 3.0 * np.sin(day_of_year * 2 * np.pi / 365)
        temp = base_temp + daily_variation + seasonal_variation + np.random.normal(0, 1.5)
        temp = np.clip(temp, 22.0, 38.0)
        
        # Humidity: 60-90%
        base_humidity = 75.0
        humidity_variation = -10.0 * np.sin((hour - 6) * np.pi / 12)  # Inverse of temp
        humidity = base_humidity + humidity_variation + np.random.normal(0, 5)
        humidity = np.clip(humidity, 50.0, 95.0)
        
        # Precipitation: 10% chance of rain
        precip = 0.0
        if np.random.random() < 0.1:
            precip = np.random.uniform(0.5, 15.0)
        
        # Wind speed: 5-25 km/h
        wind = np.random.uniform(5.0, 25.0)
        
        # Conditions
        if precip > 10:
            condition = "heavy_rain"
        elif precip > 2:
            condition = "rain"
        elif humidity > 85:
            condition = "cloudy"
        else:
            condition = "clear"
        
        return {
            'temperature_c': round(temp, 1),
            'humidity_percent': round(humidity, 1),
            'precipitation_mm': round(precip, 1),
            'wind_speed_kmh': round(wind, 1),
            'condition': condition,
            'timestamp': timestamp.isoformat(),
        }
    
    def generate_run(
        self,
        timestamp: datetime,
        reference_data: Dict,
        patterns: Dict,
    ) -> Dict:
        """Generate a single synthetic run"""
        
        run_data = {}
        
        # Copy nodes and edges (topology doesn't change)
        run_data['nodes.json'] = reference_data['nodes.json']
        run_data['edges.json'] = reference_data['edges.json']
        
        # Generate traffic with realistic patterns
        traffic_edges = []
        for edge in reference_data['traffic_edges.json']:
            edge_key = (edge['node_a_id'], edge['node_b_id'])
            base_speed = edge['speed_kmh']
            
            new_speed = self.generate_speed(edge_key, timestamp, patterns, base_speed)
            
            traffic_edges.append({
                'node_a_id': edge['node_a_id'],
                'node_b_id': edge['node_b_id'],
                'speed_kmh': new_speed,
                'distance_km': edge['distance_km'],
                'travel_time_minutes': round(edge['distance_km'] / new_speed * 60, 2),
            })
        
        run_data['traffic_edges.json'] = traffic_edges
        
        # Generate weather
        run_data['weather_snapshot.json'] = self.generate_weather(timestamp)
        
        # Generate statistics
        speeds = [e['speed_kmh'] for e in traffic_edges]
        run_data['statistics.json'] = {
            'timestamp': timestamp.isoformat(),
            'num_edges': len(traffic_edges),
            'avg_speed_kmh': round(np.mean(speeds), 2),
            'min_speed_kmh': round(np.min(speeds), 2),
            'max_speed_kmh': round(np.max(speeds), 2),
            'std_speed_kmh': round(np.std(speeds), 2),
        }
        
        return run_data
    
    def save_run(self, timestamp: datetime, run_data: Dict):
        """Save run to disk"""
        
        # Format: run_20251003_110247
        run_name = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, data in run_data.items():
            file_path = run_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return run_name
    
    def generate_monthly_dataset(self):
        """Generate complete 1-month dataset"""
        
        print("\n" + "="*70)
        print("GENERATING 1-MONTH DATASET")
        print("="*70)
        
        # Load reference
        reference_data = self.load_reference_data()
        
        # Load and analyze existing patterns
        existing_df = self.load_all_existing_runs()
        patterns = self.analyze_patterns(existing_df)
        
        # Generate timeline with 15-minute intervals
        print("\n" + "="*70)
        print("GENERATING RUNS")
        print("="*70)
        
        current = self.start_date
        generated_runs = []
        
        while current <= self.end_date:
            # Randomize seconds (0-59)
            current = current.replace(second=random.randint(0, 59))
            
            # Generate run
            run_data = self.generate_run(current, reference_data, patterns)
            run_name = self.save_run(current, run_data)
            
            generated_runs.append({
                'run_name': run_name,
                'timestamp': current,
                'avg_speed': run_data['statistics.json']['avg_speed_kmh'],
            })
            
            if len(generated_runs) % 96 == 0:  # Every day (96 × 15min = 24h)
                print(f"  ✓ {current.strftime('%Y-%m-%d')}: {len(generated_runs)} runs")
            
            # Next interval (15 minutes)
            current += timedelta(minutes=15)
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total runs generated: {len(generated_runs)}")
        print(f"Date range: {generated_runs[0]['timestamp']} -> {generated_runs[-1]['timestamp']}")
        print(f"Avg speed range: {min(r['avg_speed'] for r in generated_runs):.2f} - {max(r['avg_speed'] for r in generated_runs):.2f} km/h")
        
        # Save manifest
        manifest_path = self.output_dir / 'generated_runs_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'generation_date': datetime.now().isoformat(),
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'total_runs': len(generated_runs),
                'interval_minutes': 15,
                'random_seed': self.random_seed,
                'runs': [{'name': r['run_name'], 'timestamp': r['timestamp'].isoformat()} 
                        for r in generated_runs],
            }, f, indent=2)
        
        print(f"\nManifest saved: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic runs for 1-month dataset",
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=Path('data/runs'),
        help='Directory containing existing runs (default: data/runs)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/runs'),
        help='Output directory for generated runs (default: data/runs)',
    )
    parser.add_argument(
        '--reference-run',
        type=str,
        default='run_20251102_110036',
        help='Reference run for structure (default: run_20251102_110036)',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else PROJECT_ROOT / args.runs_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    
    # Generate
    generator = MonthlyRunGenerator(
        runs_dir=runs_dir,
        output_dir=output_dir,
        reference_run=args.reference_run,
        random_seed=args.random_seed,
    )
    
    generator.generate_monthly_dataset()


if __name__ == '__main__':
    main()
