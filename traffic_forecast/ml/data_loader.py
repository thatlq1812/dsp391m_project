"""
Data loader module for ML pipeline.
Loads traffic, weather, and node data from collected runs.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

from traffic_forecast import PROJECT_ROOT


class DataLoader:
    """Load and merge data from collection runs."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            data_dir: Root directory containing download folders.
                     Defaults to PROJECT_ROOT / 'data' / 'downloads'
        """
        self.data_dir = data_dir or PROJECT_ROOT / 'data' / 'downloads'
        self.runs = []
        self._scan_runs()

    def _scan_runs(self):
        """Scan data directory for available runs."""
        if not self.data_dir.exists():
            warnings.warn(f"Data directory {self.data_dir} does not exist")
            return

        for run_dir in self.data_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith('download_'):
                data_path = run_dir / 'data'
                if data_path.exists():
                    self.runs.append({
                        'name': run_dir.name,
                        'path': run_dir,
                        'data_path': data_path,
                        'timestamp': self._parse_run_timestamp(run_dir.name)
                    })

        self.runs.sort(key=lambda x: x['timestamp'], reverse=True)

    @staticmethod
    def _parse_run_timestamp(run_name: str) -> datetime:
        """Parse timestamp from run directory name."""
        try:
            # Format: download_YYYYMMDD_HHMMSS
            date_part = run_name.replace('download_', '')
            return datetime.strptime(date_part, '%Y%m%d_%H%M%S')
        except Exception:
            return datetime.min

    def list_runs(self) -> List[Dict]:
        """List all available runs with metadata."""
        return [{
            'name': run['name'],
            'timestamp': run['timestamp'],
            'path': str(run['path'])
        } for run in self.runs]

    def load_traffic_data(self, run_idx: int = 0) -> pd.DataFrame:
        """
        Load traffic edges data from a specific run.

        Args:
            run_idx: Index of run to load (0 = latest)

        Returns:
            DataFrame with columns: node_a_id, node_b_id, distance_km,
                                   duration_sec, speed_kmh, timestamp, api_type
        """
        if not self.runs:
            raise ValueError("No runs available")

        if run_idx >= len(self.runs):
            raise IndexError(f"Run index {run_idx} out of range (0-{len(self.runs)-1})")

        run = self.runs[run_idx]
        traffic_file = run['data_path'] / 'traffic_edges.json'

        if not traffic_file.exists():
            raise FileNotFoundError(f"Traffic file not found: {traffic_file}")

        with traffic_file.open('r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def load_weather_data(self, run_idx: int = 0) -> pd.DataFrame:
        """
        Load weather snapshot data from a specific run.

        Args:
            run_idx: Index of run to load (0 = latest)

        Returns:
            DataFrame with columns: node_id, lat, lon, timestamp,
                                   temperature_c, precipitation_mm, wind_speed_kmh
        """
        if not self.runs:
            raise ValueError("No runs available")

        if run_idx >= len(self.runs):
            raise IndexError(f"Run index {run_idx} out of range (0-{len(self.runs)-1})")

        run = self.runs[run_idx]
        weather_file = run['data_path'] / 'weather_snapshot.json'

        if not weather_file.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_file}")

        with weather_file.open('r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def load_nodes_data(self, run_idx: int = 0) -> pd.DataFrame:
        """
        Load nodes data from a specific run.

        Args:
            run_idx: Index of run to load (0 = latest)

        Returns:
            DataFrame with columns: node_id, lat, lon, name, tags
        """
        if not self.runs:
            raise ValueError("No runs available")

        if run_idx >= len(self.runs):
            raise IndexError(f"Run index {run_idx} out of range (0-{len(self.runs)-1})")

        run = self.runs[run_idx]
        nodes_file = run['data_path'] / 'nodes.json'

        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")

        with nodes_file.open('r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        return df

    def load_merged_data(self, run_idx: int = 0) -> pd.DataFrame:
        """
        Load and merge traffic, weather, and node data.

        Merges traffic edges with weather data based on node_a_id and timestamp.
        Adds node coordinates from nodes data.

        Args:
            run_idx: Index of run to load (0 = latest)

        Returns:
            DataFrame with all features merged
        """
        traffic_df = self.load_traffic_data(run_idx)
        weather_df = self.load_weather_data(run_idx)
        nodes_df = self.load_nodes_data(run_idx)

        # Merge traffic with weather data for node_a
        # Use nearest timestamp within 1 hour window
        merged = pd.merge_asof(
            traffic_df.sort_values('timestamp'),
            weather_df.sort_values('timestamp'),
            left_on='timestamp',
            right_on='timestamp',
            left_by='node_a_id',
            right_by='node_id',
            direction='nearest',
            tolerance=pd.Timedelta(hours=1),
            suffixes=('', '_weather')
        )

        # Add node coordinates
        merged = merged.merge(
            nodes_df[['node_id', 'lat', 'lon']],
            left_on='node_a_id',
            right_on='node_id',
            how='left',
            suffixes=('', '_node')
        )

        # Clean up duplicate columns
        if 'node_id' in merged.columns:
            merged = merged.drop(columns=['node_id'])
        if 'timestamp_weather' in merged.columns:
            merged = merged.drop(columns=['timestamp_weather'])

        return merged

    def load_multiple_runs(self, run_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load and concatenate data from multiple runs.

        Args:
            run_indices: List of run indices to load. If None, loads all runs.

        Returns:
            DataFrame with concatenated data from all specified runs
        """
        if run_indices is None:
            run_indices = list(range(len(self.runs)))

        dfs = []
        for idx in run_indices:
            try:
                df = self.load_merged_data(idx)
                df['run_name'] = self.runs[idx]['name']
                df['run_timestamp'] = self.runs[idx]['timestamp']
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Failed to load run {idx}: {e}")

        if not dfs:
            raise ValueError("No data loaded from any runs")

        return pd.concat(dfs, ignore_index=True)

    def get_data_summary(self) -> Dict:
        """Get summary statistics of available data."""
        if not self.runs:
            return {'total_runs': 0, 'date_range': None}

        try:
            latest_run = self.load_merged_data(0)

            return {
                'total_runs': len(self.runs),
                'date_range': {
                    'start': self.runs[-1]['timestamp'],
                    'end': self.runs[0]['timestamp']
                },
                'latest_run': {
                    'name': self.runs[0]['name'],
                    'records': len(latest_run),
                    'features': list(latest_run.columns),
                    'date_range': {
                        'start': latest_run['timestamp'].min(),
                        'end': latest_run['timestamp'].max()
                    }
                }
            }
        except Exception as e:
            return {
                'total_runs': len(self.runs),
                'error': str(e)
            }


def load_latest_data() -> pd.DataFrame:
    """
    Convenience function to load latest run data.

    Returns:
        DataFrame with merged traffic, weather, and node data
    """
    loader = DataLoader()
    return loader.load_merged_data(0)


def load_all_data() -> pd.DataFrame:
    """
    Convenience function to load all available runs.

    Returns:
        DataFrame with concatenated data from all runs
    """
    loader = DataLoader()
    return loader.load_multiple_runs()
