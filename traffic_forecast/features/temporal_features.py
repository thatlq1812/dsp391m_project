"""
Temporal features: hour, day, weekend, rush hour encoding.

These features capture temporal patterns in traffic:
- Rush hour peaks (7-9h, 17-19h)
- Weekend vs weekday patterns
- Lunch time traffic
- Holiday effects
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def add_cyclical_encoding(
    df: pd.DataFrame,
    ts_column: str = 'ts'
) -> pd.DataFrame:
    """
    Add cyclical encoding for hour and day of week.

    Cyclical encoding preserves the circular nature of time:
    - Hour 23 and Hour 0 are adjacent (not 23 units apart)
    - Sunday and Monday are adjacent

    Uses sin/cos transformation:
    hour_sin = sin(2π × hour / 24)
    hour_cos = cos(2π × hour / 24)

    Args:
    df: DataFrame with timestamp column
    ts_column: Timestamp column name

    Returns:
    DataFrame with cyclical features

    Example:
    Hour 0 → sin=0, cos=1
    Hour 6 → sin=1, cos=0
    Hour 12 → sin=0, cos=-1
    Hour 18 → sin=-1, cos=0
    Hour 23 → sin≈0, cos≈1 (close to Hour 0!)
    """
    df = df.copy()
    df[ts_column] = pd.to_datetime(df[ts_column])

    logger.info("Adding cyclical encoding for hour and day...")

    # Extract hour
    df['hour'] = df[ts_column].dt.hour

    # Hour encoding (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[ts_column].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month (for seasonality)
    df['month'] = df[ts_column].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    logger.info("Created 9 cyclical encoding features")

    return df


def add_rush_hour_flags(
    df: pd.DataFrame,
    morning_rush: Tuple[int, int] = (7, 9),
    evening_rush: Tuple[int, int] = (17, 19),
    lunch_time: Tuple[int, int] = (11, 13),
    ts_column: str = 'ts'
) -> pd.DataFrame:
    """
    Add boolean flags for rush hours and special periods.

    Rush hours in HCMC:
    - Morning: 7-9 AM (people going to work)
    - Evening: 5-7 PM (people going home)
    - Lunch: 11 AM - 1 PM (moderate traffic)

    Args:
    df: DataFrame with timestamp column
    morning_rush: (start_hour, end_hour) for morning rush
    evening_rush: (start_hour, end_hour) for evening rush
    lunch_time: (start_hour, end_hour) for lunch period
    ts_column: Timestamp column name

    Returns:
    DataFrame with rush hour flags
    """
    df = df.copy()

    if 'hour' not in df.columns:
    df[ts_column] = pd.to_datetime(df[ts_column])
    df['hour'] = df[ts_column].dt.hour

    logger.info("Adding rush hour flags...")

    # Morning rush hour
    df['is_morning_rush'] = df['hour'].between(
        morning_rush[0], morning_rush[1], inclusive='left'
    )

    # Evening rush hour
    df['is_evening_rush'] = df['hour'].between(
        evening_rush[0], evening_rush[1], inclusive='left'
    )

    # Combined rush hour
    df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']

    # Lunch time
    df['is_lunch_time'] = df['hour'].between(
        lunch_time[0], lunch_time[1], inclusive='left'
    )

    # Off-peak (not rush hour, not lunch)
    df['is_off_peak'] = ~(df['is_rush_hour'] | df['is_lunch_time'])

    logger.info("Created 5 rush hour flag features")

    return df


def add_weekend_features(
    df: pd.DataFrame,
    ts_column: str = 'ts'
) -> pd.DataFrame:
    """
    Add weekend and weekday features.

    Args:
    df: DataFrame with timestamp column
    ts_column: Timestamp column name

    Returns:
    DataFrame with weekend features
    """
    df = df.copy()

    if 'day_of_week' not in df.columns:
    df[ts_column] = pd.to_datetime(df[ts_column])
    df['day_of_week'] = df[ts_column].dt.dayofweek

    logger.info("Adding weekend features...")

    # Weekend flag (Saturday=5, Sunday=6)
    df['is_weekend'] = df['day_of_week'] >= 5

    # Weekday flag
    df['is_weekday'] = ~df['is_weekend']

    # Friday (often has different traffic pattern)
    df['is_friday'] = df['day_of_week'] == 4

    # Monday (post-weekend, often heavier traffic)
    df['is_monday'] = df['day_of_week'] == 0

    logger.info("Created 4 weekend features")

    return df


def add_holiday_features(
    df: pd.DataFrame,
    holidays: Optional[List[str]] = None,
    ts_column: str = 'ts'
) -> pd.DataFrame:
    """
    Add holiday features.

    Major Vietnamese holidays:
    - Tết (Lunar New Year) - varies, usually late Jan/Feb
    - Reunification Day - April 30
    - International Workers' Day - May 1
    - National Day - September 2

    Args:
    df: DataFrame with timestamp column
    holidays: List of holiday dates in 'YYYY-MM-DD' format
    ts_column: Timestamp column name

    Returns:
    DataFrame with holiday features
    """
    df = df.copy()
    df[ts_column] = pd.to_datetime(df[ts_column])

    logger.info("Adding holiday features...")

    if holidays is None:
        # Default Vietnamese public holidays for 2025
    holidays = [
        '2025-01-01',  # New Year
        '2025-01-28',  # Tết (example - varies by lunar calendar)
        '2025-01-29',
        '2025-01-30',
        '2025-01-31',
        '2025-02-01',
        '2025-04-30',  # Reunification Day
        '2025-05-01',  # Labor Day
        '2025-09-02',  # National Day
    ]

    # Convert to datetime
    holiday_dates = pd.to_datetime(holidays)

    # Holiday flag
    df['is_holiday'] = df[ts_column].dt.date.isin(
        [d.date() for d in holiday_dates]
    )

    # Days to next holiday (useful for pre-holiday traffic patterns)
    df['days_to_next_holiday'] = 0
    for idx, row in df.iterrows():
    future_holidays = [h for h in holiday_dates if h > row[ts_column]]
    if future_holidays:
    next_holiday = min(future_holidays)
    df.loc[idx, 'days_to_next_holiday'] = (
        next_holiday - row[ts_column]
    ).days
    else:
    df.loc[idx, 'days_to_next_holiday'] = 365  # No upcoming holiday

    # Day before holiday (often has increased traffic)
    df['is_pre_holiday'] = df['days_to_next_holiday'] == 1

    logger.info("Created 3 holiday features")

    return df


def add_temporal_features(
    df: pd.DataFrame,
    config: dict = None,
    ts_column: str = 'ts'
) -> pd.DataFrame:
    """
    Add all temporal features at once.

    Args:
    df: DataFrame with traffic data
    config: Configuration dict from project_config.yaml
    ts_column: Timestamp column name

    Returns:
    DataFrame with all temporal features
    """
    if config is None:
    config = {
        'cyclical_encoding': True,
        'rush_hours': {
            'morning': [7, 9],
            'evening': [17, 19],
            'lunch': [11, 13]
        }
    }

    logger.info("Creating all temporal features...")

    # Cyclical encoding
    if config.get('cyclical_encoding', True):
    df = add_cyclical_encoding(df, ts_column)

    # Rush hour flags
    rush_config = config.get('rush_hours', {})
    df = add_rush_hour_flags(
        df,
        morning_rush=tuple(rush_config.get('morning', [7, 9])),
        evening_rush=tuple(rush_config.get('evening', [17, 19])),
        lunch_time=tuple(rush_config.get('lunch', [11, 13])),
        ts_column=ts_column
    )

    # Weekend features
    df = add_weekend_features(df, ts_column)

    # Holiday features (optional, requires holiday list)
    # df = add_holiday_features(df, ts_column=ts_column)

    total_features = 9 + 5 + 4  # cyclical + rush + weekend
    logger.info(f"Created total {total_features} temporal features")

    return df
