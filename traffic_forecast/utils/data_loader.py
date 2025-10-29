"""
Quick Data Loader Utilities
Fast and convenient data loading for EDA and modeling

This module provides optimized data loading functions that:
- Use Parquet format for 10x faster loading
- Cache preprocessed data
- Provide convenient filtering and sampling
- Handle large datasets efficiently
"""

from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class QuickDataLoader:
    """Fast data loader with caching and convenience methods"""
    
    def __init__(self, processed_dir='data/processed', runs_dir='data/runs'):
        self.processed_dir = Path(processed_dir)
        self.runs_dir = Path(runs_dir)
        self._cache = {}
    
    def load_run(self, run_name: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Load a single run quickly
        
        Args:
            run_name: Name of the run (e.g., 'run_20251030_032457')
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with traffic, weather, and node data merged
        
        Example:
            >>> loader = QuickDataLoader()
            >>> df = loader.load_run('run_20251030_032457')
            >>> print(df.shape)
            (144, 25)
        """
        cache_key = f"run_{run_name}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try Parquet first (fast)
        parquet_file = self.processed_dir / f"{run_name}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            if use_cache:
                self._cache[cache_key] = df
            return df
        
        # Fallback to JSON (slower)
        run_dir = self.runs_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_name}")
        
        print(f"⚠️  Loading from JSON (slower). Run preprocessing first:")
        print(f"   python scripts/data/preprocess_runs.py")
        
        import json
        
        with open(run_dir / 'traffic_edges.json', 'r') as f:
            traffic_data = json.load(f)
        
        df = pd.DataFrame(traffic_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if use_cache:
            self._cache[cache_key] = df
        
        return df
    
    def load_all_runs(self, use_cache: bool = True, max_runs: Optional[int] = None) -> pd.DataFrame:
        """
        Load all runs as a combined dataset
        
        Args:
            use_cache: Whether to use cached data
            max_runs: Maximum number of runs to load (None = all)
        
        Returns:
            Combined DataFrame from all runs
        
        Example:
            >>> loader = QuickDataLoader()
            >>> df = loader.load_all_runs()
            >>> print(f"Loaded {len(df):,} records")
        """
        cache_key = "all_runs"
        
        if use_cache and cache_key in self._cache:
            df = self._cache[cache_key]
            if max_runs:
                return df.head(max_runs * 144)  # ~144 records per run
            return df
        
        # Try combined Parquet first
        combined_file = self.processed_dir / 'all_runs_combined.parquet'
        if combined_file.exists():
            df = pd.read_parquet(combined_file)
            if use_cache:
                self._cache[cache_key] = df
            if max_runs:
                return df.head(max_runs * 144)
            return df
        
        # Load individual runs
        parquet_files = sorted(self.processed_dir.glob("run_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                "No processed runs found. Run preprocessing first:\n"
                "  python scripts/data/preprocess_runs.py --combine"
            )
        
        dfs = []
        for i, pf in enumerate(parquet_files):
            if '_nodes' in pf.name or '_weather' in pf.name:
                continue
            if max_runs and i >= max_runs:
                break
            dfs.append(pd.read_parquet(pf))
        
        df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
        
        if use_cache:
            self._cache[cache_key] = df
        
        return df
    
    def load_latest(self, n: int = 1) -> pd.DataFrame:
        """
        Load the latest N runs
        
        Args:
            n: Number of latest runs to load
        
        Returns:
            DataFrame from latest runs
        """
        run_dirs = sorted(self.runs_dir.glob("run_*"), reverse=True)
        latest_runs = [d.name for d in run_dirs[:n]]
        
        dfs = []
        for run_name in latest_runs:
            dfs.append(self.load_run(run_name))
        
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    
    def load_by_date_range(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Load data within a date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            Filtered DataFrame
        """
        df = self.load_all_runs()
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        return df[mask].reset_index(drop=True)
    
    def load_by_hours(self, hours: List[int]) -> pd.DataFrame:
        """
        Load data from specific hours of day
        
        Args:
            hours: List of hours (0-23)
        
        Returns:
            Filtered DataFrame
        
        Example:
            >>> # Load only peak hours
            >>> df = loader.load_by_hours([7, 8, 17, 18])
        """
        df = self.load_all_runs()
        return df[df['hour'].isin(hours)].reset_index(drop=True)
    
    def sample_data(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        stratify_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Sample data for quick analysis
        
        Args:
            n: Number of samples
            frac: Fraction of data to sample (0.0-1.0)
            stratify_by: Column to stratify sampling
        
        Returns:
            Sampled DataFrame
        
        Example:
            >>> # Get 10% sample stratified by hour
            >>> df_sample = loader.sample_data(frac=0.1, stratify_by='hour')
        """
        df = self.load_all_runs()
        
        if stratify_by and stratify_by in df.columns:
            return df.groupby(stratify_by, group_keys=False).apply(
                lambda x: x.sample(n=n, frac=frac) if len(x) > 0 else x
            ).reset_index(drop=True)
        
        if n:
            return df.sample(n=min(n, len(df))).reset_index(drop=True)
        elif frac:
            return df.sample(frac=frac).reset_index(drop=True)
        
        return df
    
    def get_summary_stats(self) -> dict:
        """Get quick summary statistics"""
        try:
            df = self.load_all_runs()
            
            return {
                'total_records': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'num_runs': df['run_id'].nunique() if 'run_id' in df.columns else 'N/A',
                'avg_speed_kmh': df['speed_kmh'].mean(),
                'avg_duration_sec': df['duration_sec'].mean(),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache = {}
        print("✓ Cache cleared")


# Convenience functions for quick usage
def quick_load(run_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Quick load function - loads latest run if no name specified
    
    Example:
        >>> df = quick_load()  # Load latest
        >>> df = quick_load('run_20251030_032457')  # Load specific
    """
    loader = QuickDataLoader()
    
    if run_name:
        return loader.load_run(run_name, **kwargs)
    else:
        return loader.load_latest(1)


def load_for_modeling(
    test_size: float = 0.2,
    time_based_split: bool = True,
    **kwargs
) -> tuple:
    """
    Load data ready for ML modeling with train/test split
    
    Args:
        test_size: Fraction for test set
        time_based_split: If True, use temporal split (latest data as test)
        **kwargs: Additional arguments for load_all_runs
    
    Returns:
        (X_train, X_test, y_train, y_test)
    
    Example:
        >>> X_train, X_test, y_train, y_test = load_for_modeling()
    """
    loader = QuickDataLoader()
    df = loader.load_all_runs(**kwargs)
    
    # Define features and target
    feature_cols = [
        'distance_km', 'hour', 'day_of_week', 'is_weekend',
        'temperature_c', 'wind_speed_kmh', 'precipitation_mm',
        'node_importance'
    ]
    
    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['speed_kmh']
    
    if time_based_split:
        # Use latest data as test set
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    return X_train, X_test, y_train, y_test


# Print helpful info when imported
def _show_usage_hint():
    """Show usage hint on import"""
    print("💡 Quick Data Loader imported!")
    print("   from traffic_forecast.utils.data_loader import quick_load, QuickDataLoader")
    print("   df = quick_load()  # Load latest run")
    print("   df_all = QuickDataLoader().load_all_runs()  # Load all runs")


if __name__ == '__main__':
    # Demo usage
    print("Quick Data Loader - Demo")
    print("=" * 60)
    
    loader = QuickDataLoader()
    stats = loader.get_summary_stats()
    
    print("\n📊 Dataset Summary:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n💡 Usage Examples:")
    print("   # Load latest run")
    print("   df = quick_load()")
    print("")
    print("   # Load all runs")
    print("   loader = QuickDataLoader()")
    print("   df = loader.load_all_runs()")
    print("")
    print("   # Load specific time range")
    print("   df = loader.load_by_date_range('2025-10-29', '2025-10-30')")
    print("")
    print("   # Sample 10% for quick analysis")
    print("   df_sample = loader.sample_data(frac=0.1)")
