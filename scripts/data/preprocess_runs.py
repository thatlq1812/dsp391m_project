#!/usr/bin/env python3
"""
Quick Data Preprocessing Script
Convert downloaded JSON runs to optimized Parquet format for fast analysis

This script:
1. Loads all JSON runs from data/runs/
2. Converts to Pandas DataFrames
3. Adds derived features (time-based, speed categories, etc.)
4. Saves to Parquet format (10x faster loading)
5. Creates a cache for quick reloading

Usage:
    # Process all runs
    python scripts/data/preprocess_runs.py
    
    # Process specific runs
    python scripts/data/preprocess_runs.py --runs run_20251030_032457 run_20251030_032440
    
    # Force refresh (ignore cache)
    python scripts/data/preprocess_runs.py --force
    
    # Output to custom directory
    python scripts/data/preprocess_runs.py --output data/processed_custom
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class RunPreprocessor:
    """Fast preprocessing for traffic data runs"""
    
    def __init__(self, data_dir='data/runs', output_dir='data/processed', force=False):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.force = force
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for quick checking
        self.cache_file = self.output_dir / '_cache_info.json'
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load preprocessing cache"""
        if self.cache_file.exists() and not self.force:
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save preprocessing cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _needs_processing(self, run_name):
        """Check if run needs preprocessing"""
        if self.force:
            return True
        
        if run_name not in self.cache:
            return True
        
        # Check if output files exist
        parquet_file = self.output_dir / f"{run_name}.parquet"
        if not parquet_file.exists():
            return True
        
        return False
    
    def process_run(self, run_dir):
        """Process a single run directory"""
        run_name = run_dir.name
        
        if not self._needs_processing(run_name):
            print(f"✓ {run_name} (cached)")
            return None
        
        try:
            start_time = time.time()
            
            # Load JSON files
            with open(run_dir / 'nodes.json', 'r') as f:
                nodes_data = json.load(f)
            
            with open(run_dir / 'traffic_edges.json', 'r') as f:
                traffic_data = json.load(f)
            
            with open(run_dir / 'weather_snapshot.json', 'r') as f:
                weather_data = json.load(f)
            
            # Convert to DataFrames
            df_traffic = pd.DataFrame(traffic_data)
            df_weather = pd.DataFrame(weather_data)
            df_nodes = pd.DataFrame(nodes_data)
            
            # Parse timestamps
            df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
            df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
            
            # Add time-based features
            df_traffic['hour'] = df_traffic['timestamp'].dt.hour
            df_traffic['minute'] = df_traffic['timestamp'].dt.minute
            df_traffic['day_of_week'] = df_traffic['timestamp'].dt.dayofweek
            df_traffic['day_name'] = df_traffic['timestamp'].dt.day_name()
            df_traffic['is_weekend'] = df_traffic['day_of_week'].isin([5, 6])
            
            # Add congestion categories
            df_traffic['congestion_level'] = pd.cut(
                df_traffic['speed_kmh'],
                bins=[0, 15, 25, 40, 150],
                labels=['heavy', 'moderate', 'light', 'free_flow']
            )
            
            # Add speed categories
            df_traffic['speed_category'] = pd.cut(
                df_traffic['speed_kmh'],
                bins=5,
                labels=['very_slow', 'slow', 'moderate', 'fast', 'very_fast']
            )
            
            # Merge with weather (using node_a_id)
            df_merged = df_traffic.merge(
                df_weather[['node_id', 'temperature_c', 'wind_speed_kmh', 'precipitation_mm']],
                left_on='node_a_id',
                right_on='node_id',
                how='left',
                suffixes=('', '_weather')
            )
            
            # Add node importance (from nodes data)
            node_importance = df_nodes.set_index('node_id')['importance_score'].to_dict()
            df_merged['node_importance'] = df_merged['node_a_id'].map(node_importance)
            
            # Add run metadata
            df_merged['run_id'] = run_name
            df_merged['collection_time'] = df_traffic['timestamp'].iloc[0] if len(df_traffic) > 0 else None
            
            # Save to Parquet (columnar format, much faster than JSON)
            output_file = self.output_dir / f"{run_name}.parquet"
            df_merged.to_parquet(output_file, index=False, compression='snappy')
            
            # Also save individual components for flexibility
            df_nodes.to_parquet(self.output_dir / f"{run_name}_nodes.parquet", index=False)
            df_weather.to_parquet(self.output_dir / f"{run_name}_weather.parquet", index=False)
            
            # Update cache
            elapsed = time.time() - start_time
            self.cache[run_name] = {
                'processed_at': datetime.now().isoformat(),
                'num_records': len(df_merged),
                'file_size_mb': output_file.stat().st_size / 1024 / 1024,
                'processing_time_sec': elapsed
            }
            
            print(f"✓ {run_name} ({len(df_merged)} records, {elapsed:.2f}s)")
            
            return df_merged
            
        except Exception as e:
            print(f"✗ {run_name}: {e}")
            return None
    
    def process_all(self, run_names=None):
        """Process all runs or specific runs"""
        if run_names:
            run_dirs = [self.data_dir / name for name in run_names]
        else:
            run_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        print(f"\n{'='*70}")
        print(f"🔄 PREPROCESSING TRAFFIC DATA RUNS")
        print(f"{'='*70}")
        print(f"Input:  {self.data_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Runs:   {len(run_dirs)}")
        print(f"Mode:   {'Force refresh' if self.force else 'Incremental (cached)'}")
        print()
        
        results = []
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                print(f"⚠️  Skipping {run_dir.name} (not a directory)")
                continue
            
            df = self.process_run(run_dir)
            if df is not None:
                results.append(df)
        
        # Save cache
        self._save_cache()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {len([r for r in run_dirs if self.cache.get(r.name)])} runs")
        print(f"Total records: {sum(c['num_records'] for c in self.cache.values()):,}")
        print(f"Total size: {sum(c['file_size_mb'] for c in self.cache.values()):.2f} MB")
        print()
        print(f"📁 Output files:")
        print(f"   • {self.output_dir}/*.parquet (main data)")
        print(f"   • {self.output_dir}/*_nodes.parquet (topology)")
        print(f"   • {self.output_dir}/*_weather.parquet (weather)")
        print(f"   • {self.cache_file} (cache info)")
        print()
        print(f"💡 Quick load in Python:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_parquet('{self.output_dir}/run_name.parquet')")
        print()
        
        return results
    
    def create_combined_dataset(self):
        """Create a single combined dataset from all processed runs"""
        print(f"\n🔗 Creating combined dataset...")
        
        parquet_files = sorted(self.output_dir.glob("run_*.parquet"))
        
        if not parquet_files:
            print("⚠️  No processed runs found. Run preprocessing first.")
            return None
        
        dfs = []
        for pf in parquet_files:
            if '_nodes' in pf.name or '_weather' in pf.name:
                continue  # Skip component files
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
        
        # Save combined dataset
        combined_file = self.output_dir / 'all_runs_combined.parquet'
        df_combined.to_parquet(combined_file, index=False, compression='snappy')
        
        print(f"✓ Combined dataset created: {combined_file}")
        print(f"  • Records: {len(df_combined):,}")
        print(f"  • Size: {combined_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  • Time range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
        
        return df_combined


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess traffic data runs to Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        default='data/runs',
        help='Input directory containing run folders (default: data/runs)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/processed',
        help='Output directory for Parquet files (default: data/processed)'
    )
    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        help='Specific run names to process (default: all)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reprocessing (ignore cache)'
    )
    parser.add_argument(
        '--combine', '-c',
        action='store_true',
        help='Create combined dataset from all runs'
    )
    
    args = parser.parse_args()
    
    preprocessor = RunPreprocessor(
        data_dir=args.input,
        output_dir=args.output,
        force=args.force
    )
    
    # Process runs
    preprocessor.process_all(run_names=args.runs)
    
    # Create combined dataset if requested
    if args.combine:
        preprocessor.create_combined_dataset()


if __name__ == '__main__':
    main()
