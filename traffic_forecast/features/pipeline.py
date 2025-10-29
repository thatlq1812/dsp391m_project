"""
Feature engineering pipeline orchestrator.

Combines all feature modules into unified pipeline:
1. Lag features (traffic history)
2. Temporal features (time patterns)
3. Spatial features (network effects)

Usage:
 from traffic_forecast.features.pipeline import FeatureEngineeringPipeline

 pipeline = FeatureEngineeringPipeline(config)
 df_enhanced = pipeline.create_all_features(df, nodes_data)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
import yaml

from .lag_features import create_all_lag_features
from .temporal_features import add_temporal_features
from .spatial_features import add_spatial_features

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates feature engineering process.

    Creates ~60 features from raw traffic data:
    - 30 lag features (history patterns)
    - 18 temporal features (time patterns)
    - 9 spatial features (network effects)
    - 23 base features (current state + weather)
    """

    def __init__(self, config: Optional[Dict] = None):
    """
 Initialize pipeline with config.

 Args:
 config: Configuration dict from project_config.yaml
 If None, loads from default location
 """
    if config is None:
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'project_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

    self.config = config
    self.pipelines_config = config.get('pipelines', {})
    self.lag_config = self.pipelines_config.get('lag_config', {})
    self.temporal_config = self.pipelines_config.get('temporal_config', {})
    self.spatial_config = self.pipelines_config.get('spatial_config', {})

    logger.info("Feature engineering pipeline initialized")

    def create_lag_features(
        self,
        df: pd.DataFrame,
        group_by: str = 'node_id'
    ) -> pd.DataFrame:
    """
 Create lag and rolling features.

 Args:
 df: DataFrame with traffic data
 group_by: Column to group by (default: 'node_id')

 Returns:
 DataFrame with lag features
 """
    logger.info("Step 1/3: Creating lag features...")

    df = create_all_lag_features(
        df,
        config=self.lag_config,
        group_by=group_by
    )

    logger.info(f"Lag features created: {len(df.columns)} total columns")
    return df

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        ts_column: str = 'ts'
    ) -> pd.DataFrame:
    """
 Create temporal encoding features.

 Args:
 df: DataFrame with traffic data
 ts_column: Timestamp column name

 Returns:
 DataFrame with temporal features
 """
    logger.info("Step 2/3: Creating temporal features...")

    df = add_temporal_features(
        df,
        config=self.temporal_config,
        ts_column=ts_column
    )

    logger.info(f"Temporal features created: {len(df.columns)} total columns")
    return df

    def create_spatial_features(
        self,
        df: pd.DataFrame,
        nodes_data: List[dict]
    ) -> pd.DataFrame:
    """
 Create spatial network features.

 Args:
 df: DataFrame with traffic data
 nodes_data: List of node dicts (for graph construction)

 Returns:
 DataFrame with spatial features
 """
    logger.info("Step 3/3: Creating spatial features...")

    df = add_spatial_features(
        df,
        nodes_data=nodes_data,
        config=self.spatial_config
    )

    logger.info(f"Spatial features created: {len(df.columns)} total columns")
    return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        nodes_data: Optional[List[dict]] = None,
        group_by: str = 'node_id',
        ts_column: str = 'ts'
    ) -> pd.DataFrame:
    """
 Create all features at once.

 Args:
 df: DataFrame with raw traffic data
 Required columns: node_id, ts, avg_speed_kmh
 nodes_data: List of node dicts (for spatial features)
 If None, spatial features will be skipped
 group_by: Column to group by for lag features
 ts_column: Timestamp column name

 Returns:
 DataFrame with ~60 engineered features

 Example:
 pipeline = FeatureEngineeringPipeline()
 df_enhanced = pipeline.create_all_features(df_raw, nodes_data)

 print(df_enhanced.columns)
 # ['node_id', 'ts', 'avg_speed_kmh', 'congestion_level',
 # 'temperature_c', 'rain_mm', 'wind_speed_kmh',
 # 'forecast_temp_t5_c', 'forecast_temp_t15_c', ...
 # 'speed_lag_5min', 'speed_lag_15min', ...
 # 'speed_rolling_15min_mean', 'speed_rolling_30min_std', ...
 # 'hour_sin', 'hour_cos', 'is_rush_hour', ...
 # 'neighbor_avg_speed', 'neighbor_speed_diff', ...]
 """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE STARTED")
    logger.info("=" * 60)

    initial_columns = len(df.columns)
    initial_rows = len(df)

    logger.info(f"Input: {initial_rows} rows, {initial_columns} columns")

    # Validate required columns
    required = ['node_id', 'ts', 'avg_speed_kmh']
    missing = [c for c in required if c not in df.columns]
    if missing:
    raise ValueError(f"Missing required columns: {missing}")

    # Step 1: Lag features
    df = self.create_lag_features(df, group_by=group_by)
    after_lag = len(df.columns)

    # Step 2: Temporal features
    df = self.create_temporal_features(df, ts_column=ts_column)
    after_temporal = len(df.columns)

    # Step 3: Spatial features (optional)
    if nodes_data is not None and self.spatial_config.get('enabled', True):
    df = self.create_spatial_features(df, nodes_data)
    after_spatial = len(df.columns)
    else:
    logger.info("Step 3/3: Skipping spatial features (no nodes_data or disabled)")
    after_spatial = after_temporal

    # Summary
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input columns: {initial_columns}")
    logger.info(f"After lag: {after_lag} (+{after_lag - initial_columns})")
    logger.info(f"After temporal: {after_temporal} (+{after_temporal - after_lag})")
    logger.info(f"After spatial: {after_spatial} (+{after_spatial - after_temporal})")
    logger.info(f"Total features: {after_spatial} (+{after_spatial - initial_columns} new)")
    logger.info(f"Final rows: {len(df)} (lost {initial_rows - len(df)} to lag creation)")
    logger.info("=" * 60)

    return df

    def get_feature_groups(self) -> Dict[str, List[str]]:
    """
 Get feature names organized by group.

 Returns:
 Dict with feature groups:
 - base: Original features
 - lag: Lag and rolling features
 - temporal: Time encoding features
 - spatial: Network features
 """
    expected_features = self.pipelines_config.get('feature_columns', [])

    groups = {
        'base': [
            'avg_speed_kmh', 'congestion_level',
            'temperature_c', 'rain_mm', 'wind_speed_kmh',
            'cloud_cover_pct', 'visibility_km', 'pressure_mb',
            'forecast_temp_t5_c', 'forecast_temp_t15_c', 'forecast_temp_t30_c', 'forecast_temp_t60_c',
            'forecast_rain_t5_mm', 'forecast_rain_t15_mm', 'forecast_rain_t30_mm', 'forecast_rain_t60_mm',
            'forecast_wind_t5_kmh', 'forecast_wind_t15_kmh', 'forecast_wind_t30_kmh', 'forecast_wind_t60_kmh'
        ],
        'lag': [
            'speed_lag_5min', 'speed_lag_15min', 'speed_lag_30min', 'speed_lag_60min',
            'congestion_lag_5min', 'congestion_lag_15min', 'congestion_lag_30min', 'congestion_lag_60min',
            'speed_rolling_15min_mean', 'speed_rolling_15min_std', 'speed_rolling_15min_min', 'speed_rolling_15min_max',
            'speed_rolling_30min_mean', 'speed_rolling_30min_std', 'speed_rolling_30min_min', 'speed_rolling_30min_max',
            'speed_rolling_60min_mean', 'speed_rolling_60min_std', 'speed_rolling_60min_min', 'speed_rolling_60min_max',
            'speed_change_5min', 'speed_pct_change_5min',
            'speed_acceleration_5min', 'congestion_change_5min'
        ],
        'temporal': [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
            'is_lunch_time', 'is_off_peak',
            'is_weekend', 'is_weekday', 'is_friday', 'is_monday',
            'is_holiday', 'is_pre_holiday', 'days_to_next_holiday'
        ],
        'spatial': [
            'neighbor_avg_avg_speed_kmh', 'neighbor_min_avg_speed_kmh',
            'neighbor_max_avg_speed_kmh', 'neighbor_std_avg_speed_kmh',
            'neighbor_count', 'neighbor_speed_diff', 'neighbor_is_bottleneck',
            'neighbor_congested_count', 'neighbor_congested_fraction'
        ]
    }

    return groups

    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
    """
 Validate created features.

 Args:
 df: DataFrame with engineered features

 Returns:
 Validation report dict
 """
    groups = self.get_feature_groups()

    report = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'missing_values': {},
        'feature_coverage': {}
    }

    for group_name, expected_features in groups.items():
    present = [f for f in expected_features if f in df.columns]
    missing = [f for f in expected_features if f not in df.columns]

    report['feature_coverage'][group_name] = {
        'expected': len(expected_features),
        'present': len(present),
        'missing': missing
    }

    # Check for missing values
    for col in df.columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
    report['missing_values'][col] = {
        'count': int(null_count),
        'percentage': float(null_count / len(df) * 100)
    }

    return report


def create_all_features(
    df: pd.DataFrame,
    nodes_data: Optional[List[dict]] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Convenience function to create all features.

    Args:
    df: DataFrame with raw traffic data
    nodes_data: List of node dicts (for spatial features)
    config: Configuration dict (loads from file if None)

    Returns:
    DataFrame with engineered features

    Example:
    from traffic_forecast.features.pipeline import create_all_features

    df_enhanced = create_all_features(df_raw, nodes_data)
    """
    pipeline = FeatureEngineeringPipeline(config)
    return pipeline.create_all_features(df, nodes_data)
