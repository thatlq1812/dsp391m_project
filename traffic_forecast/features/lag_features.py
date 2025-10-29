"""
Traffic lag features: historical speed and congestion patterns.

These features are CRITICAL for traffic prediction as traffic exhibits
strong temporal autocorrelation (current speed depends heavily on past speed).
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_lag_features(
    df: pd.DataFrame,
    lag_minutes: List[int] = [5, 15, 30, 60],
    interval_minutes: int = 5,
    target_column: str = 'avg_speed_kmh',
    congestion_column: Optional[str] = 'congestion_level',
    node_id_column: str = 'node_id'
) -> pd.DataFrame:
    """
    Create lag features for traffic speed and congestion.

    Traffic exhibits strong temporal autocorrelation:
    - Current speed highly correlated with speed 5-15 min ago
    - Rush hour "builds up" gradually over time
    - Congestion propagates with some delay

    Args:
    df: DataFrame with traffic data (must have ts, node_id, speed columns)
    lag_minutes: List of lag periods in minutes
    interval_minutes: Data collection interval (default 5 min)
    target_column: Speed column name
    congestion_column: Congestion level column (optional)
    node_id_column: Node identifier column

    Returns:
    DataFrame with lag features added

    Example:
    >>> df = create_lag_features(df, lag_minutes=[5, 15, 30])
    >>> # Adds: speed_lag_5min, speed_lag_15min, speed_lag_30min, etc.
    """
    df = df.copy()

    # Ensure sorted by node and time
    df = df.sort_values([node_id_column, 'ts'])

    logger.info(f"Creating lag features for {lag_minutes} minutes...")

    for lag_min in lag_minutes:
        # Calculate number of periods to shift
    periods = lag_min // interval_minutes

    # Speed lags
    lag_col = f'speed_lag_{lag_min}min'
    df[lag_col] = (
        df.groupby(node_id_column)[target_column]
        .shift(periods)
    )

    # Congestion lags (if column exists)
    if congestion_column and congestion_column in df.columns:
    cong_lag_col = f'congestion_lag_{lag_min}min'
    df[cong_lag_col] = (
        df.groupby(node_id_column)[congestion_column]
        .shift(periods)
    )

    logger.info(f"Created {len(lag_minutes) * 2} lag features")

    return df


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 6, 12],  # 15min, 30min, 60min for 5min intervals
    interval_minutes: int = 5,
    target_column: str = 'avg_speed_kmh',
    node_id_column: str = 'node_id'
) -> pd.DataFrame:
    """
    Create rolling window statistics for speed.

    Rolling statistics help:
    - Smooth out noise in measurements
    - Capture trends and patterns
    - Detect sudden changes (via std)

    Args:
    df: DataFrame with traffic data
    windows: List of window sizes in periods
    interval_minutes: Data interval (for naming)
    target_column: Speed column name
    node_id_column: Node identifier column

    Returns:
    DataFrame with rolling features added

    Example:
    >>> df = create_rolling_features(df, windows=[3, 6, 12])
    >>> # For window=3 (15 min): mean, std, min, max over last 15 min
    """
    df = df.copy()
    df = df.sort_values([node_id_column, 'ts'])

    logger.info(f"Creating rolling features for windows {windows}...")

    for window in windows:
    window_min = window * interval_minutes
    prefix = f'speed_rolling_{window_min}min'

    # Rolling mean
    df[f'{prefix}_mean'] = (
        df.groupby(node_id_column)[target_column]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    # Rolling std (variability indicator)
    df[f'{prefix}_std'] = (
        df.groupby(node_id_column)[target_column]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
    )

    # Rolling min/max (speed range)
    df[f'{prefix}_min'] = (
        df.groupby(node_id_column)[target_column]
        .transform(lambda x: x.rolling(window, min_periods=1).min())
    )

    df[f'{prefix}_max'] = (
        df.groupby(node_id_column)[target_column]
        .transform(lambda x: x.rolling(window, min_periods=1).max())
    )

    # Use first window for standard rolling features
    if windows:
    first_window = windows[0]
    window_min = first_window * interval_minutes

    # Create standard names (backward compatibility)
    df['speed_rolling_mean_15min'] = df[f'speed_rolling_{window_min}min_mean']
    df['speed_rolling_std_15min'] = df[f'speed_rolling_{window_min}min_std']
    df['speed_rolling_min_15min'] = df[f'speed_rolling_{window_min}min_min']
    df['speed_rolling_max_15min'] = df[f'speed_rolling_{window_min}min_max']

    logger.info(f"Created {len(windows) * 4} rolling features")

    return df


def create_speed_change_features(
    df: pd.DataFrame,
    target_column: str = 'avg_speed_kmh',
    node_id_column: str = 'node_id'
) -> pd.DataFrame:
    """
    Create speed change features (absolute and percentage).

    Speed changes help detect:
    - Sudden slowdowns (accident, incident)
    - Rush hour onset
    - Traffic clearing

    Args:
    df: DataFrame with traffic data
    target_column: Speed column name
    node_id_column: Node identifier column

    Returns:
    DataFrame with change features added
    """
    df = df.copy()
    df = df.sort_values([node_id_column, 'ts'])

    logger.info("Creating speed change features...")

    # Absolute change
    df['speed_change_5min'] = (
        df.groupby(node_id_column)[target_column].diff()
    )

    # Percentage change
    df['speed_pct_change_5min'] = (
        df.groupby(node_id_column)[target_column].pct_change()
    )

    # Replace inf with NaN (happens when dividing by 0)
    df['speed_pct_change_5min'].replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("Created 2 speed change features")

    return df


def create_all_lag_features(
    df: pd.DataFrame,
    config: dict = None
) -> pd.DataFrame:
    """
    Create all lag-based features at once.

    Args:
    df: DataFrame with traffic data
    config: Configuration dict from project_config.yaml

    Returns:
    DataFrame with all lag features
    """
    if config is None:
    config = {
        'lag_minutes': [5, 15, 30, 60],
        'rolling_windows': [3, 6, 12],
        'interval_minutes': 5
    }

    lag_minutes = config.get('lag_minutes', [5, 15, 30, 60])
    rolling_windows = config.get('rolling_windows', [3, 6, 12])
    interval_minutes = config.get('interval_minutes', 5)

    logger.info("Creating all lag features...")

    # Add lag features
    df = create_lag_features(df, lag_minutes=lag_minutes,
                             interval_minutes=interval_minutes)

    # Add rolling features
    df = create_rolling_features(df, windows=rolling_windows,
                                 interval_minutes=interval_minutes)

    # Add change features
    df = create_speed_change_features(df)

    total_features = len(lag_minutes) * 2 + len(rolling_windows) * 4 + 2
    logger.info(f"Created total {total_features} lag-based features")

    return df
