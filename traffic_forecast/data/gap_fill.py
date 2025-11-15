"""Gap-filling utilities for building weekly 15-minute datasets.

This module keeps augmentation grounded in the observed distribution by
re-sampling each edge on a uniform timeline, interpolating short gaps,
and bootstrapping longer horizons with hour-of-week statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GapFillSummary:
    """Lightweight report describing a gap-filled dataset."""

    freq: str
    target_days: int
    total_rows: int
    real_rows: int
    augmented_rows: int
    timeline_start: pd.Timestamp
    timeline_end: pd.Timestamp

    @property
    def augmented_ratio(self) -> float:
        return 0.0 if self.total_rows == 0 else self.augmented_rows / self.total_rows

    @property
    def coverage_days(self) -> float:
        delta = self.timeline_end - self.timeline_start
        return delta.total_seconds() / 86400.0


class GapFillAugmentor:
    """Fill temporal gaps on a fixed cadence with restrained augmentation.

    The resulting dataset keeps the real distribution intact while ensuring
    every 15-minute slot inside the target horizon has a full graph snapshot.
    """

    def __init__(
        self,
        freq: str = "15min",
        target_days: int = 7,
        max_interp_minutes: int = 90,
        speed_noise: float = 0.05,
        weather_noise: float = 0.02,
        min_speed: float = 3.0,
        max_speed: float = 80.0,
        random_seed: int = 42,
    ) -> None:
        self.freq = freq
        self.target_days = target_days
        self.offset = pd.tseries.frequencies.to_offset(freq)
        # pandas offsets expose ``delta`` for fixed-frequency offsets
        freq_delta = getattr(self.offset, "delta", pd.Timedelta(minutes=15))
        self.freq_minutes = max(int(freq_delta / pd.Timedelta(minutes=1)), 1)
        self.max_interp_steps = max(int(max_interp_minutes / self.freq_minutes), 1)
        self.speed_noise = speed_noise
        self.weather_noise = weather_noise
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.rng = np.random.default_rng(random_seed)

    def build(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, GapFillSummary]:
        """Return a weekly, gap-filled DataFrame and a summary report."""

        if df.empty:
            raise ValueError("Input dataframe is empty; cannot build gap-filled data")

        base = df.copy()
        base['timestamp'] = pd.to_datetime(base['timestamp'])
        base.sort_values('timestamp', inplace=True)
        base['edge_key'] = base['node_a_id'].astype(str) + '--' + base['node_b_id'].astype(str)
        base['timestamp_aligned'] = base['timestamp'].dt.floor(self.offset)
        base = base.drop_duplicates(subset=['edge_key', 'timestamp_aligned'], keep='first')

        if 'humidity_percent' in base.columns:
            base['humidity_percent'] = pd.to_numeric(base['humidity_percent'], errors='coerce')

        start = base['timestamp_aligned'].min()
        target_end = start + pd.Timedelta(days=self.target_days)
        # Ensure timeline end is exclusive by subtracting one frequency step
        inclusive_end = target_end - getattr(self.offset, 'delta', pd.Timedelta(minutes=15))
        timeline = pd.date_range(start=start, end=inclusive_end, freq=self.freq)

        profiles = self._build_speed_profiles(base)
        numeric_cols = base.select_dtypes(include=['number']).columns.tolist()
        constant_cols = [
            col for col in ['node_a_id', 'node_b_id', 'distance_km', 'duration_min',
                            'lat_a', 'lon_a', 'lat_b', 'lon_b'] if col in base.columns
        ]
        weather_cols = [col for col in ['temperature_c', 'wind_speed_kmh', 'precipitation_mm', 'humidity_percent']
                        if col in base.columns]

        filled_frames: List[pd.DataFrame] = []
        for edge_key, edge_df in base.groupby('edge_key'):
            filled_frames.append(
                self._fill_edge(edge_key, edge_df, timeline, numeric_cols,
                                 constant_cols, weather_cols, profiles)
            )

        result = pd.concat(filled_frames, ignore_index=True)
        result.sort_values('timestamp', inplace=True)
        summary = self._summarize(result, timeline)
        return result, summary

    def _fill_edge(
        self,
        edge_key: str,
        edge_df: pd.DataFrame,
        timeline: pd.DatetimeIndex,
        numeric_cols: List[str],
        constant_cols: List[str],
        weather_cols: List[str],
        profiles: Dict[str, Dict],
    ) -> pd.DataFrame:
        edge_df = edge_df.sort_values('timestamp_aligned')
        edge_df = edge_df.set_index('timestamp_aligned')
        reindexed = edge_df.reindex(timeline)
        if 'timestamp' in reindexed.columns:
            reindexed = reindexed.rename(columns={'timestamp': 'observed_timestamp'})

        reindexed['source_run_id'] = reindexed['run_id']
        reindexed['edge_key'] = edge_key

        def _median(series: pd.Series) -> float:
            cleaned = series.dropna()
            return cleaned.median() if not cleaned.empty else 0.0

        # Interpolate numeric columns for short gaps
        for col in numeric_cols:
            if col not in reindexed.columns:
                continue
            reindexed[col] = reindexed[col].interpolate(
                method='time', limit=self.max_interp_steps, limit_direction='both'
            )

        # Fill residual numeric NaNs with per-edge medians
        for col in numeric_cols:
            if col not in reindexed.columns:
                continue
            median_value = _median(edge_df[col]) if col in edge_df.columns else 0.0
            if reindexed[col].isna().all():
                reindexed[col] = median_value
            reindexed[col] = reindexed[col].fillna(median_value)

        # Handle speed specifically with profile fallback
        missing_speed = reindexed['speed_kmh'].isna()
        if missing_speed.any():
            fallback = self._sample_speed(edge_key, timeline[missing_speed], profiles)
            reindexed.loc[missing_speed, 'speed_kmh'] = fallback
        reindexed['speed_kmh'] = reindexed['speed_kmh'].clip(self.min_speed, self.max_speed)

        # Static columns per edge
        for col in constant_cols:
            if col not in edge_df.columns:
                continue
            value = edge_df[col].iloc[0]
            reindexed[col] = value

        # Weather tweaks with light noise on augmented rows
        augmented_mask = reindexed['source_run_id'].isna()
        for col in weather_cols:
            if col not in edge_df.columns:
                continue
            median_value = _median(edge_df[col])
            reindexed[col] = reindexed[col].fillna(median_value)
            if self.weather_noise > 0 and augmented_mask.any():
                noise = self.rng.normal(0.0, self.weather_noise, augmented_mask.sum())
                base_values = reindexed.loc[augmented_mask, col].values
                reindexed.loc[augmented_mask, col] = base_values * (1.0 + noise)

        if 'weather_description' in edge_df.columns:
            mode = edge_df['weather_description'].dropna().mode()
            default_weather = mode.iloc[0] if not mode.empty else 'synthetic-clear'
            reindexed['weather_description'] = reindexed['weather_description'].fillna(default_weather)

        reindexed['augmented'] = reindexed['source_run_id'].isna()
        reindexed.index.name = 'timestamp'
        timestamp_series = reindexed.index.to_series()
        reindexed['run_id'] = timestamp_series.dt.strftime('run_gap_%Y%m%d_%H%M')
        reindexed = reindexed.reset_index()
        reindexed.drop(columns=['timestamp_aligned'], inplace=True, errors='ignore')
        reindexed.drop(columns=['edge_key'], inplace=True, errors='ignore')

        return reindexed

    def _build_speed_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        working = df.copy()
        working['hour'] = working['timestamp'].dt.hour
        working['dow'] = working['timestamp'].dt.dayofweek

        edge_hour = working.groupby(['edge_key', 'hour'])['speed_kmh'].median()
        dow_hour = working.groupby(['dow', 'hour'])['speed_kmh'].median()
        hourly = working.groupby('hour')['speed_kmh'].median()
        global_mean = working['speed_kmh'].mean()

        return {
            'edge_hour': edge_hour.to_dict(),
            'dow_hour': dow_hour.to_dict(),
            'hourly': hourly.to_dict(),
            'global_mean': global_mean,
        }

    def _sample_speed(
        self,
        edge_key: str,
        timestamps: Iterable[pd.Timestamp],
        profiles: Dict[str, Dict],
    ) -> np.ndarray:
        samples: List[float] = []
        for ts in timestamps:
            hour = ts.hour
            dow = ts.dayofweek
            key_edge = (edge_key, hour)
            key_dow = (dow, hour)

            if key_edge in profiles['edge_hour']:
                value = profiles['edge_hour'][key_edge]
            elif key_dow in profiles['dow_hour']:
                value = profiles['dow_hour'][key_dow]
            else:
                value = profiles['hourly'].get(hour, profiles['global_mean'])

            jitter = self.rng.normal(0.0, self.speed_noise)
            value = np.clip(value * (1.0 + jitter), self.min_speed, self.max_speed)
            samples.append(value)

        return np.asarray(samples)

    def _summarize(self, df: pd.DataFrame, timeline: pd.DatetimeIndex) -> GapFillSummary:
        total = len(df)
        augmented = int(df['augmented'].sum()) if 'augmented' in df.columns else 0
        real = total - augmented
        return GapFillSummary(
            freq=self.freq,
            target_days=self.target_days,
            total_rows=total,
            real_rows=real,
            augmented_rows=augmented,
            timeline_start=timeline.min(),
            timeline_end=timeline.max(),
        )
