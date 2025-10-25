"""
Enhanced data collection with historical storage.

This script wraps the existing collection pipeline and adds:
- Automatic storage of collected data to history database
- Lag feature computation using historical data
- Efficient data retrieval without re-collection

Usage:
    # Collect once with history storage
    python scripts/collect_with_history.py --once
    
    # Collect with interval (stores each run)
    python scripts/collect_with_history.py --interval 900  # 15 minutes
    
    # View storage stats
    python scripts/collect_with_history.py --stats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime, timedelta
import pandas as pd

from traffic_forecast.storage import TrafficHistoryStore
from traffic_forecast.features.pipeline import FeatureEngineeringPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_latest_collection() -> tuple:
    """
    Load latest collected data from node directories.
    
    Returns:
        (timestamp, traffic_data, weather_data, nodes_data)
    """
    data_dir = Path('data/node')
    
    # Find latest collection directory
    collections = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
    if not collections:
        raise ValueError("No collection data found in data/node/")
    
    latest = collections[0]
    timestamp = datetime.strptime(latest.name, '%Y%m%d%H%M%S')
    
    logger.info(f"Loading collection from {latest.name}")
    
    # Load traffic snapshot
    traffic_file = latest / 'collectors' / 'mock' / 'traffic_snapshot_normalized.json'
    if not traffic_file.exists():
        # Try google traffic
        traffic_file = latest / 'collectors' / 'google' / 'traffic_edges.json'
    
    if traffic_file.exists():
        with open(traffic_file, 'r', encoding='utf-8') as f:
            traffic_data = json.load(f)
    else:
        logger.warning("No traffic data found")
        traffic_data = []
    
    # Load weather
    weather_file = latest / 'collectors' / 'open_meteo' / 'weather_snapshot.json'
    if weather_file.exists():
        with open(weather_file, 'r', encoding='utf-8') as f:
            weather_data = json.load(f)
    else:
        logger.warning("No weather data found")
        weather_data = []
    
    # Load nodes
    nodes_file = latest / 'collectors' / 'overpass' / 'nodes.json'
    if nodes_file.exists():
        with open(nodes_file, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
    else:
        logger.warning("No nodes data found")
        nodes_data = []
    
    return timestamp, traffic_data, weather_data, nodes_data


def merge_traffic_weather(
    traffic_data: list,
    weather_data: list,
    timestamp: datetime
) -> pd.DataFrame:
    """
    Merge traffic and weather data.
    
    Returns:
        DataFrame with combined data and timestamp
    """
    # Convert to DataFrames
    if isinstance(traffic_data, list):
        df_traffic = pd.DataFrame(traffic_data)
    else:
        df_traffic = pd.DataFrame([traffic_data])
    
    # Weather is per-node
    weather_map = {w['node_id']: w for w in weather_data}
    
    # Merge
    records = []
    for _, row in df_traffic.iterrows():
        node_id = row.get('node_id')
        
        record = {
            'ts': timestamp,
            'node_id': node_id,
            'avg_speed_kmh': row.get('avg_speed_kmh', row.get('speed_kmh')),
            'congestion_level': row.get('congestion_level', 0)
        }
        
        # Add weather if available
        if node_id in weather_map:
            weather = weather_map[node_id]
            record.update({
                'temperature_c': weather.get('temperature_c'),
                'rain_mm': weather.get('rain_mm', 0),
                'wind_speed_kmh': weather.get('wind_speed_kmh'),
                'cloud_cover_pct': weather.get('cloud_cover_pct'),
                'visibility_km': weather.get('visibility_km'),
                'pressure_mb': weather.get('pressure_mb'),
                # Forecasts
                'forecast_temp_t5_c': weather.get('forecast_temp_t5_c'),
                'forecast_temp_t15_c': weather.get('forecast_temp_t15_c'),
                'forecast_temp_t30_c': weather.get('forecast_temp_t30_c'),
                'forecast_temp_t60_c': weather.get('forecast_temp_t60_c'),
                'forecast_rain_t5_mm': weather.get('forecast_rain_t5_mm', 0),
                'forecast_rain_t15_mm': weather.get('forecast_rain_t15_mm', 0),
                'forecast_rain_t30_mm': weather.get('forecast_rain_t30_mm', 0),
                'forecast_rain_t60_mm': weather.get('forecast_rain_t60_mm', 0),
                'forecast_wind_t5_kmh': weather.get('forecast_wind_t5_kmh'),
                'forecast_wind_t15_kmh': weather.get('forecast_wind_t15_kmh'),
                'forecast_wind_t30_kmh': weather.get('forecast_wind_t30_kmh'),
                'forecast_wind_t60_kmh': weather.get('forecast_wind_t60_kmh'),
            })
        
        records.append(record)
    
    df = pd.DataFrame(records)
    logger.info(f"Merged {len(df)} traffic records with weather")
    
    return df


def compute_lag_features_from_history(
    current_data: pd.DataFrame,
    store: TrafficHistoryStore,
    current_time: datetime
) -> pd.DataFrame:
    """
    Add lag features using historical data from store.
    
    Args:
        current_data: Current traffic snapshot
        store: History store
        current_time: Current timestamp
    
    Returns:
        DataFrame with lag features added
    """
    logger.info("Computing lag features from historical data...")
    
    # Get historical data
    lags = store.get_lag_data(current_time, lag_minutes=[5, 15, 30, 60])
    
    if not lags:
        logger.warning("No historical data available for lag features")
        return current_data
    
    # Merge lag data
    df = current_data.copy()
    
    for lag_min, lag_df in lags.items():
        if lag_df is None or len(lag_df) == 0:
            logger.warning(f"No data for {lag_min}min lag")
            continue
        
        # Create lookup: node_id -> speed
        lag_lookup = dict(zip(lag_df['node_id'], lag_df['avg_speed_kmh']))
        congestion_lookup = dict(zip(lag_df['node_id'], lag_df.get('congestion_level', [0]*len(lag_df))))
        
        # Add lag features
        df[f'speed_lag_{lag_min}min'] = df['node_id'].map(lag_lookup)
        df[f'congestion_lag_{lag_min}min'] = df['node_id'].map(congestion_lookup)
        
        logger.info(f"Added lag features for {lag_min} minutes")
    
    # Compute rolling features (need more history)
    history_df = store.get_recent_history(current_time, hours=2)
    
    if history_df is not None and len(history_df) > 0:
        # Group by node and compute rolling stats
        for node_id in df['node_id'].unique():
            node_history = history_df[history_df['node_id'] == node_id].copy()
            node_history = node_history.sort_values('ts')
            
            if len(node_history) >= 3:  # Need at least 3 points
                # 15min rolling (3 points @ 5min intervals)
                df.loc[df['node_id'] == node_id, 'speed_rolling_15min_mean'] = \
                    node_history['avg_speed_kmh'].tail(3).mean()
                df.loc[df['node_id'] == node_id, 'speed_rolling_15min_std'] = \
                    node_history['avg_speed_kmh'].tail(3).std()
                
                # 30min rolling (6 points)
                if len(node_history) >= 6:
                    df.loc[df['node_id'] == node_id, 'speed_rolling_30min_mean'] = \
                        node_history['avg_speed_kmh'].tail(6).mean()
                    df.loc[df['node_id'] == node_id, 'speed_rolling_30min_std'] = \
                        node_history['avg_speed_kmh'].tail(6).std()
        
        logger.info("Added rolling features from recent history")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Collect traffic data with historical storage')
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run collection once and exit'
    )
    parser.add_argument(
        '--interval',
        type=int,
        help='Collection interval in seconds'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show storage statistics and exit'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Cleanup old data (>7 days) and exit'
    )
    parser.add_argument(
        '--db-path',
        default='data/traffic_history.db',
        help='Path to history database'
    )
    
    args = parser.parse_args()
    
    # Initialize storage
    store = TrafficHistoryStore(args.db_path)
    
    # Handle stats/cleanup commands
    if args.stats:
        stats = store.get_stats()
        print("\n" + "=" * 60)
        print("TRAFFIC HISTORY STORAGE STATS")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key:25s}: {value}")
        print("=" * 60)
        return
    
    if args.cleanup:
        deleted = store.cleanup_old_data()
        print(f"Cleaned up {deleted} old records")
        return
    
    # Run collection
    logger.info("=" * 60)
    logger.info("ENHANCED DATA COLLECTION WITH HISTORY STORAGE")
    logger.info("=" * 60)
    
    # Load latest collection
    timestamp, traffic_data, weather_data, nodes_data = load_latest_collection()
    
    # Merge data
    df_current = merge_traffic_weather(traffic_data, weather_data, timestamp)
    
    # Convert timestamp to string for JSON serialization
    df_current['ts'] = df_current['ts'].astype(str)
    
    # Save to history store
    saved = store.save_snapshot(timestamp, df_current.to_dict('records'))
    logger.info(f"Saved {saved} records to history store")
    
    # Compute lag features from history
    df_enhanced = compute_lag_features_from_history(df_current, store, timestamp)
    
    # Add temporal features
    pipeline = FeatureEngineeringPipeline()
    df_enhanced = pipeline.create_temporal_features(df_enhanced)
    
    # Save enhanced data
    output_path = Path('data/features_with_lags.csv')
    df_enhanced.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced data to {output_path}")
    
    # Show summary
    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Timestamp:      {timestamp}")
    logger.info(f"Records saved:  {saved}")
    logger.info(f"Total features: {len(df_enhanced.columns)}")
    
    # Show which lag features were added
    lag_features = [c for c in df_enhanced.columns if 'lag' in c or 'rolling' in c]
    logger.info(f"Lag features:   {len(lag_features)}")
    if lag_features:
        logger.info(f"  - {', '.join(lag_features[:5])}...")
    
    # Show storage stats
    stats = store.get_stats()
    logger.info(f"Storage size:   {stats['database_size_mb']} MB")
    logger.info(f"Total records:  {stats['total_records']}")
    logger.info(f"Time range:     {stats['earliest_timestamp']} to {stats['latest_timestamp']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
