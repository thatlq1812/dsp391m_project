"""
Feature engineering module for ML pipeline.
Creates temporal, spatial, and weather-derived features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime
import warnings


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Add temporal features from timestamp column.

    Features added:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - day_of_month: Day of month (1-31)
    - month: Month (1-12)
    - is_weekend: Boolean indicator for weekend
    - is_rush_hour: Boolean indicator for rush hours (7-9am, 5-7pm)
    - time_of_day: Categorical (morning, afternoon, evening, night)

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added temporal features
    """
    result = df.copy()

    if timestamp_col not in result.columns:
        warnings.warn(f"Column '{timestamp_col}' not found. Skipping temporal features.")
        return result

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

    # Extract datetime components
    result['hour'] = result[timestamp_col].dt.hour
    result['day_of_week'] = result[timestamp_col].dt.dayofweek
    result['day_of_month'] = result[timestamp_col].dt.day
    result['month'] = result[timestamp_col].dt.month
    result['year'] = result[timestamp_col].dt.year

    # Weekend indicator
    result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)

    # Rush hour indicator (7-9am, 5-7pm)
    result['is_rush_hour'] = result['hour'].isin([7, 8, 17, 18]).astype(int)

    # Time of day categories
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 0  # morning
        elif 12 <= hour < 17:
            return 1  # afternoon
        elif 17 <= hour < 21:
            return 2  # evening
        else:
            return 3  # night

    result['time_of_day'] = result['hour'].apply(categorize_time)

    # Cyclical encoding for hour (preserves circular nature)
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

    # Cyclical encoding for day of week
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

    return result


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial features from coordinate data.

    Features added:
    - distance_km: Already exists in traffic data
    - lat_diff: Difference in latitude between nodes
    - lon_diff: Difference in longitude between nodes
    - bearing: Direction angle from node A to node B

    Args:
        df: DataFrame with lat, lon columns

    Returns:
        DataFrame with added spatial features
    """
    result = df.copy()

    # Check if we have necessary columns
    has_coords = 'lat' in result.columns and 'lon' in result.columns

    if not has_coords:
        warnings.warn("Coordinate columns not found. Skipping spatial features.")
        return result

    # Add coordinate difference features
    if 'lat_node' in result.columns and 'lon_node' in result.columns:
        result['lat_diff'] = result['lat'] - result['lat_node']
        result['lon_diff'] = result['lon'] - result['lon_node']

    return result


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-derived features.

    Features added:
    - is_raining: Boolean indicator for precipitation > 0
    - precipitation_category: Categorical (none, light, moderate, heavy)
    - temp_category: Categorical (cool, comfortable, hot)
    - wind_category: Categorical (calm, moderate, strong)
    - weather_severity: Combined weather impact score

    Args:
        df: DataFrame with weather columns

    Returns:
        DataFrame with added weather features
    """
    result = df.copy()

    # Check for weather columns
    has_weather = all(col in result.columns for col in
                      ['temperature_c', 'precipitation_mm', 'wind_speed_kmh'])

    if not has_weather:
        warnings.warn("Weather columns not found. Skipping weather features.")
        return result

    # Rain indicator
    result['is_raining'] = (result['precipitation_mm'] > 0).astype(int)

    # Precipitation categories
    def categorize_precipitation(precip):
        if precip == 0:
            return 0  # none
        elif precip < 2.5:
            return 1  # light
        elif precip < 10:
            return 2  # moderate
        else:
            return 3  # heavy

    result['precipitation_category'] = result['precipitation_mm'].apply(categorize_precipitation)

    # Temperature categories (Celsius)
    def categorize_temperature(temp):
        if temp < 20:
            return 0  # cool
        elif temp < 30:
            return 1  # comfortable
        else:
            return 2  # hot

    result['temp_category'] = result['temperature_c'].apply(categorize_temperature)

    # Wind categories (km/h)
    def categorize_wind(wind):
        if wind < 20:
            return 0  # calm
        elif wind < 40:
            return 1  # moderate
        else:
            return 2  # strong

    result['wind_category'] = result['wind_speed_kmh'].apply(categorize_wind)

    # Weather severity score (0-1, higher = worse conditions)
    # Normalize and combine weather factors
    result['weather_severity'] = (
        (result['precipitation_mm'] / 50).clip(0, 1) * 0.5 +  # Rain impact
        (result['wind_speed_kmh'] / 60).clip(0, 1) * 0.3 +    # Wind impact
        (abs(result['temperature_c'] - 25) / 15).clip(0, 1) * 0.2  # Temp deviation
    )

    return result


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add traffic-derived features.

    Features added:
    - speed_category: Categorical speed bin
    - is_congested: Boolean indicator for low speed
    - speed_to_distance_ratio: Speed normalized by distance

    Args:
        df: DataFrame with traffic columns

    Returns:
        DataFrame with added traffic features
    """
    result = df.copy()

    if 'speed_kmh' not in result.columns:
        warnings.warn("Speed column not found. Skipping traffic features.")
        return result

    # Speed categories
    def categorize_speed(speed):
        if speed < 15:
            return 0  # very slow / stopped
        elif speed < 30:
            return 1  # slow
        elif speed < 50:
            return 2  # moderate
        else:
            return 3  # fast

    result['speed_category'] = result['speed_kmh'].apply(categorize_speed)

    # Congestion indicator (speed < 20 km/h)
    result['is_congested'] = (result['speed_kmh'] < 20).astype(int)

    # Speed to distance ratio (can indicate road type)
    if 'distance_km' in result.columns:
        result['speed_to_distance_ratio'] = result['speed_kmh'] / (result['distance_km'] + 0.001)

    return result


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = 'speed_kmh',
    group_cols: Optional[List[str]] = None,
    lags: List[int] = [1, 2, 3]
) -> pd.DataFrame:
    """
    Add lag features (previous time period values).

    Args:
        df: DataFrame sorted by timestamp
        value_col: Column to create lags for
        group_cols: Columns to group by (e.g., ['node_a_id'])
        lags: List of lag periods to create

    Returns:
        DataFrame with lag features
    """
    result = df.copy()

    if value_col not in result.columns:
        warnings.warn(f"Column '{value_col}' not found. Skipping lag features.")
        return result

    if group_cols:
        for lag in lags:
            result[f'{value_col}_lag_{lag}'] = result.groupby(group_cols)[value_col].shift(lag)
    else:
        for lag in lags:
            result[f'{value_col}_lag_{lag}'] = result[value_col].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = 'speed_kmh',
    group_cols: Optional[List[str]] = None,
    windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Add rolling window features (moving averages, std).

    Args:
        df: DataFrame sorted by timestamp
        value_col: Column to calculate rolling stats for
        group_cols: Columns to group by
        windows: List of window sizes

    Returns:
        DataFrame with rolling features
    """
    result = df.copy()

    if value_col not in result.columns:
        warnings.warn(f"Column '{value_col}' not found. Skipping rolling features.")
        return result

    if group_cols:
        for window in windows:
            result[f'{value_col}_rolling_mean_{window}'] = \
                result.groupby(group_cols)[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
            )
            result[f'{value_col}_rolling_std_{window}'] = \
                result.groupby(group_cols)[value_col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
            )
    else:
        for window in windows:
            result[f'{value_col}_rolling_mean_{window}'] = \
                result[value_col].rolling(window, min_periods=1).mean()
            result[f'{value_col}_rolling_std_{window}'] = \
                result[value_col].rolling(window, min_periods=1).std()

    return result


def build_features(
    df: pd.DataFrame,
    include_temporal: bool = True,
    include_spatial: bool = True,
    include_weather: bool = True,
    include_traffic: bool = True,
    include_lags: bool = False,
    include_rolling: bool = False,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Build all features for the dataset.

    Args:
        df: Input DataFrame
        include_temporal: Add temporal features
        include_spatial: Add spatial features
        include_weather: Add weather features
        include_traffic: Add traffic features
        include_lags: Add lag features
        include_rolling: Add rolling window features
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with all selected features
    """
    result = df.copy()

    if include_temporal:
        result = add_temporal_features(result, timestamp_col)

    if include_spatial:
        result = add_spatial_features(result)

    if include_weather:
        result = add_weather_features(result)

    if include_traffic:
        result = add_traffic_features(result)

    if include_lags:
        result = add_lag_features(result)

    if include_rolling:
        result = add_rolling_features(result)

    return result


def get_feature_importance_names() -> List[str]:
    """
    Get list of all possible feature names created by feature engineering.

    Returns:
        List of feature names
    """
    temporal = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'year',
        'is_weekend', 'is_rush_hour', 'time_of_day',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]

    spatial = ['lat_diff', 'lon_diff']

    weather = [
        'is_raining', 'precipitation_category', 'temp_category',
        'wind_category', 'weather_severity'
    ]

    traffic = ['speed_category', 'is_congested', 'speed_to_distance_ratio']

    return temporal + spatial + weather + traffic
