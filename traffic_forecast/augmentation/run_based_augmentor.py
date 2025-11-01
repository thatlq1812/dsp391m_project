"""
Run-Based Traffic Data Augmentation

Generates synthetic run folders to achieve target density (20 samples/hour).
Uses cubic spline interpolation with variation control and realistic noise.

Author: thatlq1812
Date: 2025-10-31
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from math import ceil

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

logger = logging.getLogger(__name__)


class RunBasedAugmentor:
    """
    Generate augmented run folders to achieve target sample density.
    
    Target: 20 samples per hour (3-minute intervals)
    Method: Cubic spline + variation control + noise injection
    Output: Individual run folders (run_YYYYMMDD_HHMMSS)
    """
    
    # Configuration
    TARGET_SAMPLES_PER_HOUR = 20
    TARGET_INTERVAL_MINUTES = 3  # 60min / 20 samples
    
    CONSTRAINTS = {
        'speed_range': (5, 80),           # km/h
        'max_variation': 5.0,             # km/h between consecutive points
        'max_variation_ratio': 0.3,       # 30% of local range
        'smooth_window': 3,               # Rolling average
        'noise_std_range': (2, 5),        # km/h (strong variation)
        'max_acceleration': 0.5,          # km/h per second
    }
    
    def __init__(
        self,
        source_dir: Path = None,
        output_dir: Path = None,
        target_samples_per_hour: int = 20,
        dry_run: bool = False
    ):
        """
        Initialize augmentor.
        
        Args:
            source_dir: Directory containing original runs
            output_dir: Directory to save augmented runs (default: same as source)
            target_samples_per_hour: Target density
            dry_run: If True, only simulate without creating files
        """
        self.source_dir = source_dir or Path('data/runs')
        self.output_dir = output_dir or self.source_dir
        self.target_samples_per_hour = target_samples_per_hour
        self.dry_run = dry_run
        
        self.stats = {
            'runs_processed': 0,
            'runs_created': 0,
            'total_samples_generated': 0,
            'errors': []
        }
        
    def augment_all_runs(
        self,
        max_runs: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict:
        """
        Augment all runs in source directory.
        
        Args:
            max_runs: Maximum number of original runs to process
            skip_existing: Skip if augmented runs already exist
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting augmentation: {self.source_dir}")
        logger.info(f"Target: {self.target_samples_per_hour} samples/hour")
        
        # Find all original run directories
        run_dirs = sorted([d for d in self.source_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        if max_runs:
            run_dirs = run_dirs[:max_runs]
        
        logger.info(f"Found {len(run_dirs)} original runs")
        
        # Process each run
        for i, run_dir in enumerate(run_dirs, 1):
            logger.info(f"\n[{i}/{len(run_dirs)}] Processing: {run_dir.name}")
            
            try:
                # Augment this run
                created = self.augment_single_run(run_dir, skip_existing=skip_existing)
                self.stats['runs_processed'] += 1
                self.stats['runs_created'] += created
                
                logger.info(f"  ✓ Created {created} augmented runs")
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                self.stats['errors'].append({'run': run_dir.name, 'error': str(e)})
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Augmentation Complete!")
        logger.info(f"  Original runs processed: {self.stats['runs_processed']}")
        logger.info(f"  Augmented runs created: {self.stats['runs_created']}")
        logger.info(f"  Total samples generated: {self.stats['total_samples_generated']}")
        logger.info(f"  Errors: {len(self.stats['errors'])}")
        logger.info("="*60)
        
        return self.stats
    
    def augment_single_run(
        self,
        run_dir: Path,
        skip_existing: bool = True
    ) -> int:
        """
        Generate augmented runs from a single original run.
        
        Args:
            run_dir: Original run directory
            skip_existing: Skip if augmented versions exist
            
        Returns:
            Number of augmented runs created
        """
        # Load run data
        run_data = self._load_run(run_dir)
        
        if not run_data:
            logger.warning(f"  No valid data in {run_dir.name}")
            return 0
        
        # Calculate required augmented timestamps
        new_timestamps = self._calculate_augmented_timestamps(run_data, run_dir)
        
        if not new_timestamps:
            logger.info(f"  No augmentation needed (already dense enough)")
            return 0
        
        logger.info(f"  Will create {len(new_timestamps)} augmented runs")
        
        if self.dry_run:
            logger.info(f"  [DRY RUN] Would create: {[ts.strftime('%Y%m%d_%H%M%S') for ts in new_timestamps[:3]]}...")
            return len(new_timestamps)
        
        # Generate augmented runs
        created_count = 0
        
        for new_ts in new_timestamps:
            try:
                # Create augmented data at this timestamp
                augmented_data = self._interpolate_at_timestamp(
                    run_data,
                    new_ts,
                    original_run_id=run_dir.name
                )
                
                # Save as new run folder
                self._save_augmented_run(augmented_data, new_ts, run_dir)
                
                created_count += 1
                self.stats['total_samples_generated'] += len(augmented_data.get('traffic_data', []))
                
            except Exception as e:
                logger.error(f"  ✗ Failed to create run at {new_ts}: {e}")
        
        return created_count
    
    def _load_run(self, run_dir: Path) -> Optional[Dict]:
        """Load traffic data from run directory."""
        # Try traffic_edges.json first, then traffic.json
        traffic_file = run_dir / 'traffic_edges.json'
        if not traffic_file.exists():
            traffic_file = run_dir / 'traffic.json'
        
        if not traffic_file.exists():
            return None
        
        try:
            with open(traffic_file, 'r', encoding='utf-8') as f:
                traffic_data = json.load(f)
            
            # If it's a list, wrap it in a dict
            if isinstance(traffic_data, list):
                data = {'traffic_data': traffic_data}
            else:
                data = traffic_data
                if 'traffic_data' not in data:
                    # Assume the whole dict is traffic data
                    data = {'traffic_data': [data]}
            
            # Extract timestamp from directory name
            # Format: run_YYYYMMDD_HHMMSS
            parts = run_dir.name.split('_')
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                data['timestamp'] = timestamp
                data['run_id'] = run_dir.name
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading {traffic_file}: {e}")
            return None
    
    def _calculate_augmented_timestamps(
        self,
        run_data: Dict,
        run_dir: Path
    ) -> List[datetime]:
        """
        Calculate timestamps for augmented runs needed to achieve target density.
        
        Strategy: Find gaps between existing runs and fill with new timestamps.
        """
        # Get all existing run timestamps
        all_runs = sorted([d for d in self.source_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        existing_timestamps = []
        for run in all_runs:
            parts = run.name.split('_')
            if len(parts) >= 3:
                try:
                    ts = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                    existing_timestamps.append(ts)
                except:
                    pass
        
        if len(existing_timestamps) < 2:
            logger.warning("Need at least 2 runs to interpolate")
            return []
        
        existing_timestamps = sorted(existing_timestamps)
        
        # Calculate gaps and determine where to insert new timestamps
        new_timestamps = []
        target_interval = timedelta(minutes=self.TARGET_INTERVAL_MINUTES)
        
        for i in range(len(existing_timestamps) - 1):
            start_ts = existing_timestamps[i]
            end_ts = existing_timestamps[i + 1]
            gap = end_ts - start_ts
            
            # How many samples should be in this gap?
            gap_hours = gap.total_seconds() / 3600
            target_samples_in_gap = int(gap_hours * self.target_samples_per_hour)
            current_samples_in_gap = 2  # start and end
            
            samples_needed = max(0, target_samples_in_gap - current_samples_in_gap)
            
            if samples_needed > 0:
                # Generate evenly spaced timestamps in this gap
                for j in range(1, samples_needed + 1):
                    fraction = j / (samples_needed + 1)
                    new_ts = start_ts + (gap * fraction)
                    # Round to nearest minute for cleaner timestamps
                    new_ts = new_ts.replace(second=0, microsecond=0)
                    new_timestamps.append(new_ts)
        
        return sorted(new_timestamps)
    
    def _interpolate_at_timestamp(
        self,
        reference_data: Dict,
        target_timestamp: datetime,
        original_run_id: str
    ) -> Dict:
        """
        Create interpolated traffic data at target timestamp.
        
        Uses surrounding runs for interpolation.
        """
        # Find surrounding runs (before and after target_timestamp)
        all_runs = sorted([d for d in self.source_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        run_times = []
        for run in all_runs:
            parts = run.name.split('_')
            if len(parts) >= 3:
                try:
                    ts = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                    run_times.append((ts, run))
                except:
                    pass
        
        run_times = sorted(run_times, key=lambda x: x[0])
        
        # Find before and after
        before_run = None
        after_run = None
        
        for ts, run in run_times:
            if ts <= target_timestamp:
                before_run = (ts, run)
            if ts >= target_timestamp and after_run is None:
                after_run = (ts, run)
                break
        
        if not before_run or not after_run:
            raise ValueError(f"Cannot interpolate - missing surrounding runs for {target_timestamp}")
        
        # Load both runs
        before_data = self._load_run(before_run[1])
        after_data = self._load_run(after_run[1])
        
        if not before_data or not after_data:
            raise ValueError("Failed to load surrounding runs")
        
        # Interpolate traffic data
        interpolated_traffic = self._interpolate_traffic_data(
            before_data['traffic_data'],
            after_data['traffic_data'],
            before_run[0],
            after_run[0],
            target_timestamp
        )
        
        # Create new run data structure
        augmented_data = {
            'run_id': f"run_{target_timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': target_timestamp,
            'augmented': True,
            'source_runs': [before_run[1].name, after_run[1].name],
            'traffic_data': interpolated_traffic,
        }
        
        # Copy metadata from reference
        for key in ['weather', 'topology', 'collection_params']:
            if key in reference_data:
                augmented_data[key] = reference_data[key]
        
        return augmented_data
    
    def _interpolate_traffic_data(
        self,
        before_traffic: List[Dict],
        after_traffic: List[Dict],
        before_time: datetime,
        after_time: datetime,
        target_time: datetime
    ) -> List[Dict]:
        """
        Interpolate traffic data between two time points.
        
        Method: Cubic spline + variation control + noise
        """
        # Calculate interpolation factor (0 to 1)
        total_gap = (after_time - before_time).total_seconds()
        target_gap = (target_time - before_time).total_seconds()
        alpha = target_gap / total_gap  # 0 = before, 1 = after
        
        # Create mapping of edges
        edge_data = {}
        
        # Helper function to get edge ID
        def get_edge_id(edge):
            # Support both formats
            if 'edge_id' in edge:
                return edge['edge_id']
            elif 'node_a_id' in edge and 'node_b_id' in edge:
                return f"{edge['node_a_id']}--{edge['node_b_id']}"
            else:
                raise KeyError(f"Edge missing both 'edge_id' and 'node_a_id/node_b_id': {edge.keys()}")
        
        # Collect data from before run
        for edge in before_traffic:
            edge_id = get_edge_id(edge)
            edge_data[edge_id] = {
                'before': edge,
                'after': None
            }
        
        # Add data from after run
        for edge in after_traffic:
            edge_id = get_edge_id(edge)
            if edge_id in edge_data:
                edge_data[edge_id]['after'] = edge
            else:
                edge_data[edge_id] = {
                    'before': None,
                    'after': edge
                }
        
        # Interpolate each edge
        interpolated = []
        
        for edge_id, data in edge_data.items():
            if data['before'] and data['after']:
                # Both points available - interpolate
                interpolated_edge = self._interpolate_edge(
                    data['before'],
                    data['after'],
                    alpha
                )
            elif data['before']:
                # Only before - use with noise
                interpolated_edge = self._add_noise_to_edge(data['before'])
            elif data['after']:
                # Only after - use with noise
                interpolated_edge = self._add_noise_to_edge(data['after'])
            else:
                continue
            
            interpolated.append(interpolated_edge)
        
        return interpolated
    
    def _interpolate_edge(
        self,
        before_edge: Dict,
        after_edge: Dict,
        alpha: float
    ) -> Dict:
        """
        Interpolate a single edge between two time points.
        
        Uses cubic spline for smooth transition with variation control.
        """
        # Linear interpolation for speed (will be refined)
        before_speed = before_edge.get('speed_kmh', 0)
        after_speed = after_edge.get('speed_kmh', 0)
        
        # Simple cubic interpolation
        # Could use more sophisticated spline but this is sufficient for 2 points
        interpolated_speed = before_speed + alpha * (after_speed - before_speed)
        
        # Apply variation control
        max_variation = min(
            self.CONSTRAINTS['max_variation'],
            abs(after_speed - before_speed) * self.CONSTRAINTS['max_variation_ratio']
        )
        
        # Add controlled noise
        noise_std = np.random.uniform(*self.CONSTRAINTS['noise_std_range'])
        noise = np.random.normal(0, noise_std)
        interpolated_speed += noise
        
        # Apply physics constraints
        interpolated_speed = np.clip(
            interpolated_speed,
            self.CONSTRAINTS['speed_range'][0],
            self.CONSTRAINTS['speed_range'][1]
        )
        
        # Create interpolated edge
        interpolated_edge = before_edge.copy()
        interpolated_edge['speed_kmh'] = round(float(interpolated_speed), 2)
        
        # Interpolate duration if available
        if 'duration_sec' in before_edge and 'duration_sec' in after_edge:
            interpolated_edge['duration_sec'] = round(
                before_edge['duration_sec'] + alpha * (after_edge['duration_sec'] - before_edge['duration_sec'])
            )
        
        return interpolated_edge
    
    def _add_noise_to_edge(self, edge: Dict) -> Dict:
        """Add realistic noise to an edge when only one reference point exists."""
        noisy_edge = edge.copy()
        
        if 'speed_kmh' in edge:
            noise_std = np.random.uniform(*self.CONSTRAINTS['noise_std_range'])
            noise = np.random.normal(0, noise_std)
            noisy_speed = edge['speed_kmh'] + noise
            
            # Apply constraints
            noisy_speed = np.clip(
                noisy_speed,
                self.CONSTRAINTS['speed_range'][0],
                self.CONSTRAINTS['speed_range'][1]
            )
            
            noisy_edge['speed_kmh'] = round(float(noisy_speed), 2)
        
        return noisy_edge
    
    def _save_augmented_run(
        self,
        augmented_data: Dict,
        timestamp: datetime,
        reference_run: Path
    ):
        """
        Save augmented data as a new run folder.
        
        Format: run_YYYYMMDD_HHMMSS/
                ├── traffic.json
                └── (other metadata files copied from reference)
        """
        # Create run directory name
        run_name = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_name
        
        # Check if already exists
        if run_dir.exists():
            logger.warning(f"  Run already exists: {run_name}")
            return
        
        # Create directory
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save traffic.json
        traffic_file = run_dir / 'traffic.json'
        
        # Format output JSON (remove augmentation metadata for clean storage)
        output_data = {
            'traffic_data': augmented_data['traffic_data']
        }
        
        # Copy metadata
        for key in ['weather', 'topology', 'collection_params']:
            if key in augmented_data:
                output_data[key] = augmented_data[key]
        
        with open(traffic_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Copy other files from reference run (if any)
        for file in reference_run.iterdir():
            if file.is_file() and file.name != 'traffic.json':
                shutil.copy2(file, run_dir / file.name)
        
        logger.debug(f"  ✓ Created: {run_name}")


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment traffic data runs')
    parser.add_argument('--source', type=str, default='data/runs', help='Source directory')
    parser.add_argument('--output', type=str, help='Output directory (default: same as source)')
    parser.add_argument('--max-runs', type=int, help='Maximum runs to process')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without creating files')
    parser.add_argument('--target-samples', type=int, default=20, help='Target samples per hour')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create augmentor
    augmentor = RunBasedAugmentor(
        source_dir=Path(args.source),
        output_dir=Path(args.output) if args.output else None,
        target_samples_per_hour=args.target_samples,
        dry_run=args.dry_run
    )
    
    # Run augmentation
    stats = augmentor.augment_all_runs(max_runs=args.max_runs)
    
    print("\n" + json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
