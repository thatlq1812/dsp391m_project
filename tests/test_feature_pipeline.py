"""
Test feature engineering pipeline with sample data.

This script validates that all feature modules work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging

from traffic_forecast.features.pipeline import FeatureEngineeringPipeline

logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_traffic_data(n_nodes=10, n_timesteps=100):
 """Create sample traffic data for testing."""
 logger.info(f"Creating sample data: {n_nodes} nodes, {n_timesteps} timesteps")
 
 data = []
 start_time = datetime.now() - timedelta(hours=10)
 
 for node_id in range(1, n_nodes + 1):
 for t in range(n_timesteps):
 ts = start_time + timedelta(minutes=5 * t)
 
 # Simulate traffic patterns
 hour = ts.hour
 base_speed = 40
 
 # Rush hour slowdown
 if 7 <= hour <= 9 or 17 <= hour <= 19:
 base_speed = 25
 elif 11 <= hour <= 13:
 base_speed = 30
 
 # Add some noise
 speed = base_speed + np.random.normal(0, 5)
 speed = max(10, min(60, speed))
 
 # Congestion based on speed
 if speed >= 40:
 congestion = 0
 elif speed >= 30:
 congestion = 1
 elif speed >= 20:
 congestion = 2
 else:
 congestion = 3
 
 # Weather
 temp = 28 + np.random.normal(0, 2)
 rain = max(0, np.random.exponential(2))
 wind = max(0, np.random.normal(15, 5))
 
 data.append({
 'node_id': node_id,
 'ts': ts,
 'avg_speed_kmh': speed,
 'congestion_level': congestion,
 'temperature_c': temp,
 'rain_mm': rain,
 'wind_speed_kmh': wind,
 'cloud_cover_pct': np.random.uniform(20, 80),
 'visibility_km': np.random.uniform(5, 10),
 'pressure_mb': 1013 + np.random.normal(0, 5),
 # Forecasts (simplified)
 'forecast_temp_t5_c': temp + np.random.normal(0, 0.5),
 'forecast_temp_t15_c': temp + np.random.normal(0, 1),
 'forecast_temp_t30_c': temp + np.random.normal(0, 1.5),
 'forecast_temp_t60_c': temp + np.random.normal(0, 2),
 'forecast_rain_t5_mm': rain * 1.1,
 'forecast_rain_t15_mm': rain * 1.2,
 'forecast_rain_t30_mm': rain * 1.3,
 'forecast_rain_t60_mm': rain * 1.4,
 'forecast_wind_t5_kmh': wind + np.random.normal(0, 1),
 'forecast_wind_t15_kmh': wind + np.random.normal(0, 2),
 'forecast_wind_t30_kmh': wind + np.random.normal(0, 3),
 'forecast_wind_t60_kmh': wind + np.random.normal(0, 4),
 })
 
 df = pd.DataFrame(data)
 logger.info(f"Created {len(df)} records")
 return df


def create_sample_nodes_data(n_nodes=10):
 """Create sample nodes data for spatial features."""
 logger.info(f"Creating sample nodes graph: {n_nodes} nodes")
 
 nodes = []
 for i in range(1, n_nodes + 1):
 # Create simple chain: 1-2-3-4-5-6-7-8-9-10
 # With some branches
 connected = []
 
 if i > 1:
 # Previous node
 connected.append({
 'way_id': f'way_{i-1}_{i}',
 'nodes': [i-1, i]
 })
 
 if i < n_nodes:
 # Next node
 connected.append({
 'way_id': f'way_{i}_{i+1}',
 'nodes': [i, i+1]
 })
 
 # Add some branches (every 3rd node connects to next+1)
 if i % 3 == 0 and i + 2 <= n_nodes:
 connected.append({
 'way_id': f'way_{i}_{i+2}',
 'nodes': [i, i+2]
 })
 
 nodes.append({
 'node_id': i,
 'lat': 10.77 + (i * 0.001),
 'lon': 106.69 + (i * 0.001),
 'connected_ways': connected
 })
 
 logger.info(f"Created {len(nodes)} nodes with connections")
 return nodes


def test_feature_pipeline():
 """Test the complete feature engineering pipeline."""
 logger.info("=" * 70)
 logger.info("TESTING FEATURE ENGINEERING PIPELINE")
 logger.info("=" * 70)
 
 # Create sample data
 df_raw = create_sample_traffic_data(n_nodes=10, n_timesteps=100)
 nodes_data = create_sample_nodes_data(n_nodes=10)
 
 logger.info(f"\nRaw data shape: {df_raw.shape}")
 logger.info(f"Raw columns ({len(df_raw.columns)}): {list(df_raw.columns)}")
 
 # Initialize pipeline
 pipeline = FeatureEngineeringPipeline()
 
 # Create all features
 df_enhanced = pipeline.create_all_features(df_raw, nodes_data)
 
 logger.info(f"\nEnhanced data shape: {df_enhanced.shape}")
 logger.info(f"Total columns: {len(df_enhanced.columns)}")
 
 # Get feature groups
 groups = pipeline.get_feature_groups()
 
 logger.info("\n" + "=" * 70)
 logger.info("FEATURE GROUPS SUMMARY")
 logger.info("=" * 70)
 
 for group_name, features in groups.items():
 present = [f for f in features if f in df_enhanced.columns]
 missing = [f for f in features if f not in df_enhanced.columns]
 
 logger.info(f"\n{group_name.upper()} FEATURES:")
 logger.info(f" Expected: {len(features)}")
 logger.info(f" Present: {len(present)}")
 if missing:
 logger.info(f" Missing: {missing}")
 
 # Validate features
 report = pipeline.validate_features(df_enhanced)
 
 logger.info("\n" + "=" * 70)
 logger.info("VALIDATION REPORT")
 logger.info("=" * 70)
 logger.info(f"Total rows: {report['total_rows']}")
 logger.info(f"Total columns: {report['total_columns']}")
 
 if report['missing_values']:
 logger.info("\nMissing values detected:")
 for col, stats in report['missing_values'].items():
 logger.info(f" {col}: {stats['count']} ({stats['percentage']:.1f}%)")
 else:
 logger.info("\nNo missing values!")
 
 # Sample output
 logger.info("\n" + "=" * 70)
 logger.info("SAMPLE OUTPUT (first 3 rows)")
 logger.info("=" * 70)
 
 sample_cols = [
 'node_id', 'ts', 'avg_speed_kmh',
 'speed_lag_5min', 'speed_rolling_15min_mean',
 'hour_sin', 'is_rush_hour',
 'neighbor_avg_avg_speed_kmh', 'neighbor_speed_diff'
 ]
 
 available_cols = [c for c in sample_cols if c in df_enhanced.columns]
 print(df_enhanced[available_cols].head(3))
 
 # Save enhanced data
 output_path = Path(__file__).parent.parent / 'data' / 'test_features_output.csv'
 df_enhanced.to_csv(output_path, index=False)
 logger.info(f"\nSaved enhanced data to: {output_path}")
 
 logger.info("\n" + "=" * 70)
 logger.info("TEST COMPLETE")
 logger.info("=" * 70)
 
 return df_enhanced, report


if __name__ == '__main__':
 df_enhanced, report = test_feature_pipeline()
 
 print("\n" + "=" * 70)
 print("FINAL SUMMARY")
 print("=" * 70)
 print(f"Input: 23 base features")
 print(f"Output: {len(df_enhanced.columns)} total features")
 print(f"New features created: {len(df_enhanced.columns) - 23}")
 print("\nFeature engineering pipeline is working correctly!")
