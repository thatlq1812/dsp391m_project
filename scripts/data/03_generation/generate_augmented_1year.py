#!/usr/bin/env python3
"""
Augmented 1-Year Dataset Generator

Generates challenging augmented traffic dataset with:
- Realistic base patterns (rush hour, weekends)
- Seasonal variations (school cycles, holidays)
- Random incidents (accidents, breakdowns)
- Construction zones (long-term disruptions)
- Weather events (rain, fog)
- Special events (concerts, sports, festivals)

Usage:
    python scripts/data/03_generation/generate_augmented_1year.py \
        --config configs/super_dataset_config.yaml \
        --output data/processed/augmented_1year.parquet \
        --visualize
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class SuperDatasetGenerator:
    """Generate 1-year challenging traffic dataset."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rng = np.random.default_rng(self.config['random_seed'])
        self.metadata = {
            'incidents': [],
            'construction_zones': [],
            'weather_events': [],
            'special_events': [],
            'holidays': []
        }
        
        print(f"[SuperDataset] Initialized with config: {config_path}")
        print(f"  Duration: {self.config['temporal']['duration_days']} days")
        print(f"  Interval: {self.config['temporal']['interval_minutes']} minutes")
        print(f"  Total timestamps: {self.config['temporal']['total_timestamps']:,}")
    
    def smooth_temporal(self, speeds: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply temporal smoothing to reduce unrealistic jumps."""
        from scipy.ndimage import uniform_filter1d
        # Smooth along time axis (axis=0)
        return uniform_filter1d(speeds, size=window, axis=0, mode='nearest')
    
    def load_topology(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load existing edge topology."""
        topo_file = self.config['spatial']['topology_file']
        df = pd.read_parquet(topo_file)
        
        # Get unique edges
        edges = df[['node_a_id', 'node_b_id']].drop_duplicates()
        edge_ids = [f"{row['node_a_id']}_{row['node_b_id']}" 
                    for _, row in edges.iterrows()]
        
        print(f"[Topology] Loaded {len(edge_ids)} edges from {topo_file}")
        return edges, edge_ids
    
    def generate_timestamps(self) -> pd.DatetimeIndex:
        """Generate timestamp sequence."""
        start = pd.to_datetime(self.config['temporal']['start_date'])
        periods = self.config['temporal']['total_timestamps']
        freq = f"{self.config['temporal']['interval_minutes']}min"
        
        timestamps = pd.date_range(start=start, periods=periods, freq=freq)
        print(f"[Timestamps] Generated {len(timestamps):,} from {timestamps[0]} to {timestamps[-1]}")
        return timestamps
    
    def generate_base_pattern(
        self, 
        timestamp: pd.Timestamp, 
        edge_id: str
    ) -> float:
        """
        Generate base speed for given timestamp and edge.
        
        Args:
            timestamp: Current timestamp
            edge_id: Edge identifier
            
        Returns:
            Base speed in km/h
        """
        cfg = self.config['base_patterns']
        hour = timestamp.hour
        is_weekday = timestamp.weekday() < 5  # Mon-Fri
        
        # Determine base speed range by edge (hash-based for consistency)
        edge_hash = hash(edge_id) % 3
        if edge_hash == 0:  # Highway
            base_range = cfg['highway_speed_range']
        elif edge_hash == 1:  # Main road
            base_range = cfg['main_road_range']
        else:  # Local road
            base_range = cfg['local_road_range']
        
        base_speed = np.mean(base_range)
        
        # Apply time-of-day pattern
        if is_weekday:
            # Weekday pattern
            morning_rush = cfg['weekday_morning_rush']
            evening_rush = cfg['weekday_evening_rush']
            
            if morning_rush[0] <= hour < morning_rush[1]:
                # Morning rush hour - reduce speed
                base_speed *= cfg['rush_hour_reduction']
            elif evening_rush[0] <= hour < evening_rush[1]:
                # Evening rush hour - reduce speed
                base_speed *= cfg['rush_hour_reduction']
            elif 23 <= hour or hour < 6:
                # Night time - increase speed
                base_speed *= cfg['night_speed_multiplier']
        else:
            # Weekend pattern
            leisure = cfg['weekend_leisure']
            if leisure[0] <= hour < leisure[1]:
                # Leisure hours - moderate reduction
                base_speed *= 0.85
            elif 23 <= hour or hour < 6:
                # Night time - increase speed
                base_speed *= cfg['night_speed_multiplier']
        
        # Add realistic noise
        noise = self.rng.normal(0, cfg['base_noise_std'])
        speed = np.clip(base_speed + noise, base_range[0], base_range[1])
        
        # Global clip to valid range
        return np.clip(speed, 3.0, 52.0)
    
    def apply_seasonal_overlay(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """Apply seasonal patterns to base speeds."""
        if not self.config['seasonal']['enabled']:
            return speeds
        
        cfg_seasonal = self.config['seasonal']
        speeds_modified = speeds.copy()
        
        for t_idx, ts in enumerate(timestamps):
            month = ts.month
            day = ts.day
            hour = ts.hour
            is_weekday = ts.weekday() < 5
            
            multiplier = 1.0
            
            # School calendar impact
            school_cal = cfg_seasonal['school_calendar']
            school_start = school_cal['school_year'][0]
            school_end = school_cal['school_year'][1]
            
            is_school_year = (
                (month > school_start[0]) or 
                (month == school_start[0] and day >= school_start[1]) or
                (month < school_end[0]) or
                (month == school_end[0] and day <= school_end[1])
            )
            
            if is_school_year and is_weekday:
                # Morning rush more intense
                if 6 <= hour < 9:
                    multiplier *= (1 - school_cal['morning_rush_increase'])
                # Afternoon pickup time
                elif 15 <= hour < 16:
                    multiplier *= 0.85
            else:
                # Summer break - less traffic
                if not is_school_year:
                    multiplier *= (1 + school_cal['summer_reduction'])
            
            # Economic cycles
            econ = cfg_seasonal['economic_cycles']
            # Month-end increase
            if day >= 25:
                multiplier *= (1 - econ['month_end_increase'])
            
            # Quarter-end increase (Mar, Jun, Sep, Dec last week)
            if month in [3, 6, 9, 12] and day >= 24:
                multiplier *= (1 - econ['quarter_end_increase'])
            
            speeds_modified[t_idx, :] *= multiplier
        
        return speeds_modified
    
    def inject_incidents(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex,
        edge_ids: List[str]
    ) -> np.ndarray:
        """Inject random traffic incidents."""
        if not self.config['incidents']['enabled']:
            return speeds
        
        cfg = self.config['incidents']
        speeds_modified = speeds.copy()
        
        # Calculate number of incidents
        num_weeks = len(timestamps) / (7 * 24 * 6)  # 10-min intervals
        num_incidents = int(self.rng.poisson(cfg['rate_per_week'] * num_weeks))
        
        print(f"  Generating {num_incidents} incidents...")
        
        for _ in range(num_incidents):
            # Select incident type
            incident_types = cfg['types']
            type_probs = [t['probability'] for t in incident_types]
            incident = self.rng.choice(incident_types, p=type_probs)
            
            # Random timing
            incident_start = self.rng.integers(0, len(timestamps) - 100)
            duration_min = self.rng.integers(*incident['duration_range'])
            duration_steps = duration_min // self.config['temporal']['interval_minutes']
            
            # Random edge
            affected_edge_idx = self.rng.integers(0, len(edge_ids))
            
            # Severity
            severity = self.rng.uniform(*incident['severity_range'])
            
            # Apply incident impact
            incident_end = min(incident_start + duration_steps, len(timestamps))
            for t_idx in range(incident_start, incident_end):
                # Direct impact on affected edge
                speeds_modified[t_idx, affected_edge_idx] *= (1 - severity)
                
                # Spatial propagation (simplified - affect nearby edges)
                prop_cfg = self.config['spatial']['incident_propagation']
                for e_idx in range(len(edge_ids)):
                    if e_idx == affected_edge_idx:
                        continue
                    
                    # Random neighbor effect (simplified without graph)
                    if abs(e_idx - affected_edge_idx) <= 3:
                        hop_impact = self.rng.uniform(*prop_cfg['hop_1_impact'])
                        speeds_modified[t_idx, e_idx] *= (1 - severity * hop_impact)
                    elif abs(e_idx - affected_edge_idx) <= 6:
                        hop_impact = self.rng.uniform(*prop_cfg['hop_2_impact'])
                        speeds_modified[t_idx, e_idx] *= (1 - severity * hop_impact)
            
            # Recovery period
            recovery_steps = cfg['recovery_time'] // self.config['temporal']['interval_minutes']
            for t_idx in range(incident_end, min(incident_end + recovery_steps, len(timestamps))):
                progress = (t_idx - incident_end) / recovery_steps
                recovery_factor = 1 - (1 - progress) * severity
                speeds_modified[t_idx, affected_edge_idx] *= recovery_factor
            
            # Store metadata
            self.metadata['incidents'].append({
                'timestamp': str(timestamps[incident_start]),
                'edge_idx': int(affected_edge_idx),
                'type': incident['name'],
                'severity': float(severity),
                'duration_min': int(duration_min)
            })
        
        return speeds_modified
    
    def add_construction_zones(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex,
        edge_ids: List[str]
    ) -> np.ndarray:
        """Add construction zone disruptions."""
        if not self.config['construction']['enabled']:
            return speeds
        
        cfg = self.config['construction']
        speeds_modified = speeds.copy()
        
        num_zones = cfg['num_zones_per_year']
        print(f"  Placing {num_zones} construction zones...")
        
        # Distribute zones across the year
        for zone_idx in range(num_zones):
            # Random start date
            max_start = len(timestamps) - (56 * 24 * 6)  # Reserve space for max duration
            zone_start = self.rng.integers(0, max(1, max_start))
            
            # Duration in days
            duration_days = self.rng.integers(*cfg['duration_range'])
            duration_steps = duration_days * 24 * 6  # 10-min intervals
            zone_end = min(zone_start + duration_steps, len(timestamps))
            
            # Select affected edge
            affected_edge_idx = self.rng.integers(0, len(edge_ids))
            
            # Speed reduction
            reduction = self.rng.uniform(*cfg['speed_reduction'])
            
            # Apply construction impact (only during active hours on weekdays)
            active_hours = cfg['active_hours']
            for t_idx in range(zone_start, zone_end):
                ts = timestamps[t_idx]
                hour = ts.hour
                is_weekday = ts.weekday() < 5
                
                if is_weekday and active_hours[0] <= hour < active_hours[1]:
                    # Apply to main edge
                    speeds_modified[t_idx, affected_edge_idx] *= (1 - reduction)
                    
                    # Spillover to adjacent edges
                    spillover = self.config['spatial']['construction_spillover']
                    for e_idx in range(len(edge_ids)):
                        if e_idx == affected_edge_idx:
                            continue
                        if abs(e_idx - affected_edge_idx) <= 2:
                            impact = self.rng.uniform(*spillover['adjacent_impact'])
                            speeds_modified[t_idx, e_idx] *= (1 - reduction * impact)
            
            # Store metadata
            self.metadata['construction_zones'].append({
                'start': str(timestamps[zone_start]),
                'end': str(timestamps[zone_end - 1]),
                'edge_idx': int(affected_edge_idx),
                'duration_days': int(duration_days),
                'reduction': float(reduction)
            })
        
        return speeds_modified
    
    def apply_weather_effects(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply weather-based speed reductions."""
        if not self.config['weather']['enabled']:
            weather_conditions = ['clear'] * len(timestamps)
            return speeds, weather_conditions
        
        cfg = self.config['weather']
        speeds_modified = speeds.copy()
        weather_conditions = ['clear'] * len(timestamps)
        
        # Process day by day
        current_date = timestamps[0].date()
        day_start_idx = 0
        
        for t_idx, ts in enumerate(timestamps):
            if ts.date() != current_date:
                # New day - decide weather
                current_date = ts.date()
                day_start_idx = t_idx
                
                month = ts.month
                
                # Rain probability based on month
                rain_prob = cfg['rain']['probability_by_month'].get(month, 0.1)
                if self.rng.random() < rain_prob:
                    # Rainy day
                    is_heavy = self.rng.random() < cfg['rain']['heavy_rain']['probability']
                    
                    if is_heavy:
                        rain_cfg = cfg['rain']['heavy_rain']
                        rain_type = 'heavy_rain'
                    else:
                        rain_cfg = cfg['rain']['light_rain']
                        rain_type = 'light_rain'
                    
                    # Duration
                    duration_min = self.rng.integers(*rain_cfg['duration_range'])
                    duration_steps = duration_min // self.config['temporal']['interval_minutes']
                    
                    # Start time (afternoon bias)
                    rain_start_hour = self.rng.integers(12, 20)
                    rain_start_idx = t_idx + (rain_start_hour - ts.hour) * 6
                    rain_end_idx = min(rain_start_idx + duration_steps, len(timestamps))
                    
                    # Speed reduction
                    reduction = self.rng.uniform(*rain_cfg['speed_reduction'])
                    
                    # Apply
                    for r_idx in range(rain_start_idx, rain_end_idx):
                        if r_idx < len(timestamps):
                            speeds_modified[r_idx, :] *= (1 - reduction)
                            weather_conditions[r_idx] = rain_type
                    
                    # Store metadata
                    self.metadata['weather_events'].append({
                        'start': str(timestamps[rain_start_idx]),
                        'end': str(timestamps[min(rain_end_idx - 1, len(timestamps) - 1)]),
                        'type': rain_type,
                        'reduction': float(reduction)
                    })
                
                # Fog (independent of rain)
                fog_prob = cfg['fog']['probability']
                if self.rng.random() < fog_prob:
                    fog_hours = cfg['fog']['hours']
                    reduction = self.rng.uniform(*cfg['fog']['speed_reduction'])
                    
                    # Apply fog in morning hours
                    for h in range(fog_hours[0], fog_hours[1]):
                        fog_idx = day_start_idx + (h - timestamps[day_start_idx].hour) * 6
                        if 0 <= fog_idx < len(timestamps):
                            for step in range(6):  # 1 hour = 6 steps
                                if fog_idx + step < len(timestamps):
                                    speeds_modified[fog_idx + step, :] *= (1 - reduction)
                                    if weather_conditions[fog_idx + step] == 'clear':
                                        weather_conditions[fog_idx + step] = 'fog'
                    
                    self.metadata['weather_events'].append({
                        'date': str(current_date),
                        'type': 'fog',
                        'reduction': float(reduction)
                    })
        
        return speeds_modified, weather_conditions
    
    def inject_special_events(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex,
        edge_ids: List[str]
    ) -> np.ndarray:
        """Inject special events (concerts, sports, etc)."""
        if not self.config['special_events']['enabled']:
            return speeds
        
        cfg = self.config['special_events']
        speeds_modified = speeds.copy()
        
        # Calculate number of events
        num_months = len(timestamps) / (30 * 24 * 6)
        num_events = int(cfg['per_month'] * num_months)
        
        print(f"  Scheduling {num_events} special events...")
        
        for _ in range(num_events):
            # Select event type
            event_types = cfg['types']
            type_probs = [t['probability'] for t in event_types]
            event = self.rng.choice(event_types, p=type_probs)
            
            # Select venue
            venue = self.rng.choice(cfg['venues'])
            venue_edges = venue['edges']
            
            # Random date (avoid first/last week)
            event_start = self.rng.integers(7 * 24 * 6, len(timestamps) - 7 * 24 * 6)
            
            # Align to event hours
            event_hour = self.rng.integers(*event['hours'])
            current_hour = timestamps[event_start].hour
            event_start += (event_hour - current_hour) * 6
            
            # Duration
            duration_steps = event['duration'] // self.config['temporal']['interval_minutes']
            event_end = min(event_start + duration_steps, len(timestamps))
            
            # Congestion multiplier
            congestion = event['congestion_multiplier']
            
            # Apply event impact
            for t_idx in range(event_start, event_end):
                for venue_edge in venue_edges:
                    if venue_edge < len(edge_ids):
                        # Direct venue impact (increased congestion = reduced speed)
                        speeds_modified[t_idx, venue_edge] /= congestion
                        
                        # Radius-based impact
                        radius = event['affected_radius']
                        for e_idx in range(len(edge_ids)):
                            if abs(e_idx - venue_edge) <= radius * 2:
                                impact_factor = 1 + (congestion - 1) * 0.3  # 30% of main impact
                                speeds_modified[t_idx, e_idx] /= impact_factor
            
            # Store metadata
            self.metadata['special_events'].append({
                'timestamp': str(timestamps[event_start]),
                'type': event['name'],
                'venue_edges': venue_edges,
                'duration_min': int(event['duration']),
                'congestion_factor': float(congestion)
            })
        
        return speeds_modified
    
    def apply_holidays(
        self,
        speeds: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> np.ndarray:
        """Apply holiday traffic reductions."""
        if not self.config['holidays']['enabled']:
            return speeds
        
        cfg = self.config['holidays']
        speeds_modified = speeds.copy()
        
        # Process fixed holidays
        for holiday in cfg['fixed']:
            month, day = map(int, holiday['date'].split('-'))
            duration = holiday['duration']
            reduction = holiday['traffic_reduction']
            
            # Find matching dates
            for t_idx, ts in enumerate(timestamps):
                if ts.month == month and ts.day == day:
                    # Apply for duration days
                    for d in range(duration):
                        day_start = t_idx + d * 24 * 6
                        day_end = min(day_start + 24 * 6, len(timestamps))
                        for step in range(day_start, day_end):
                            if step < len(timestamps):
                                speeds_modified[step, :] *= (1 + reduction)  # Reduction means less traffic
                    
                    self.metadata['holidays'].append({
                        'name': holiday['name'],
                        'date': f"{ts.year}-{month:02d}-{day:02d}",
                        'type': 'fixed',
                        'reduction': float(reduction)
                    })
                    break
        
        # Process lunar holidays (approximate)
        for holiday in cfg['lunar']:
            month, day = map(int, holiday['approximate_date'].split('-'))
            duration = holiday['duration']
            reduction = holiday['traffic_reduction']
            
            # Find matching date (approximate)
            for t_idx, ts in enumerate(timestamps):
                if ts.month == month and abs(ts.day - day) <= 3:
                    # Main holiday
                    for d in range(duration):
                        day_start = t_idx + d * 24 * 6
                        day_end = min(day_start + 24 * 6, len(timestamps))
                        for step in range(day_start, day_end):
                            if step < len(timestamps):
                                speeds_modified[step, :] *= (1 + reduction)
                    
                    # Pre-holiday increase (shopping traffic)
                    if 'pre_holiday_days' in holiday:
                        pre_days = holiday['pre_holiday_days']
                        pre_increase = holiday['pre_holiday_increase']
                        for d in range(pre_days):
                            day_start = max(0, t_idx - (pre_days - d) * 24 * 6)
                            day_end = max(0, t_idx - (pre_days - d - 1) * 24 * 6)
                            for step in range(day_start, day_end):
                                if step < len(timestamps):
                                    speeds_modified[step, :] *= (1 - pre_increase)
                    
                    # Post-holiday increase (return travel)
                    if 'post_holiday_days' in holiday:
                        post_days = holiday['post_holiday_days']
                        post_increase = holiday['post_holiday_increase']
                        holiday_end = t_idx + duration * 24 * 6
                        for d in range(post_days):
                            day_start = holiday_end + d * 24 * 6
                            day_end = min(day_start + 24 * 6, len(timestamps))
                            for step in range(day_start, day_end):
                                if step < len(timestamps):
                                    speeds_modified[step, :] *= (1 - post_increase)
                    
                    self.metadata['holidays'].append({
                        'name': holiday['name'],
                        'date': f"{ts.year}-{month:02d}-{day:02d}",
                        'type': 'lunar',
                        'reduction': float(reduction)
                    })
                    break
        
        # Long weekend patterns
        long_weekend = cfg['long_weekends']
        for t_idx, ts in enumerate(timestamps):
            # Friday evening spike
            if ts.weekday() == 4 and 17 <= ts.hour < 22:
                increase = long_weekend['friday_evening_increase']
                speeds_modified[t_idx, :] *= (1 - increase)
            
            # Sunday evening spike
            if ts.weekday() == 6 and 15 <= ts.hour < 22:
                increase = long_weekend['sunday_evening_increase']
                speeds_modified[t_idx, :] *= (1 - increase)
            
            # Saturday CBD reduction
            if ts.weekday() == 5 and 8 <= ts.hour < 18:
                reduction = long_weekend['saturday_reduction']
                # Apply only to CBD edges (first 30% of edges)
                cbd_edges = int(len(speeds_modified[t_idx]) * 0.3)
                speeds_modified[t_idx, :cbd_edges] *= (1 + reduction)
        
        return speeds_modified
    
    def propagate_spatial_impact(
        self,
        speeds: np.ndarray,
        affected_edge_idx: int,
        impact_severity: float,
        edge_ids: List[str]
    ) -> np.ndarray:
        """
        Propagate impact to neighboring edges.
        
        Note: Simplified implementation without full graph topology.
        Uses edge index distance as proxy for spatial distance.
        """
        speeds_modified = speeds.copy()
        
        cfg = self.config['spatial']['incident_propagation']
        
        for e_idx in range(len(edge_ids)):
            if e_idx == affected_edge_idx:
                continue
            
            # Calculate "distance" (simplified)
            distance = abs(e_idx - affected_edge_idx)
            
            if distance <= 3:
                # 1-hop neighbors
                impact = self.rng.uniform(*cfg['hop_1_impact'])
                speeds_modified[e_idx] *= (1 - impact_severity * impact)
            elif distance <= 6:
                # 2-hop neighbors
                impact = self.rng.uniform(*cfg['hop_2_impact'])
                speeds_modified[e_idx] *= (1 - impact_severity * impact)
            elif distance <= 9:
                # 3-hop neighbors
                impact = self.rng.uniform(*cfg['hop_3_impact'])
                speeds_modified[e_idx] *= (1 - impact_severity * impact)
        
        return speeds_modified
    
    def validate_dataset(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """Validate generated dataset quality."""
        cfg = self.config['validation']
        stats = {}
        
        print("\n[Validation Checks]")
        
        # 1. Speed range
        min_speed = df['speed_kmh'].min()
        max_speed = df['speed_kmh'].max()
        stats['speed_range'] = [float(min_speed), float(max_speed)]
        
        valid_range = cfg['min_speed'] <= min_speed and max_speed <= cfg['max_speed']
        print(f"  Speed range: [{min_speed:.2f}, {max_speed:.2f}] km/h - {'✓' if valid_range else '✗'}")
        
        # 2. No invalid values
        num_invalid = df['speed_kmh'].isna().sum() + (df['speed_kmh'] <= 0).sum()
        stats['num_invalid'] = int(num_invalid)
        print(f"  Invalid values: {num_invalid} - {'✓' if num_invalid == 0 else '✗'}")
        
        # 3. Temporal smoothness (sample one edge)
        sample_edge = df.groupby(['node_a_id', 'node_b_id']).size().idxmax()
        edge_data = df[
            (df['node_a_id'] == sample_edge[0]) & 
            (df['node_b_id'] == sample_edge[1])
        ].sort_values('timestamp')
        
        speed_diffs = np.abs(np.diff(edge_data['speed_kmh'].values))
        max_jump = speed_diffs.max()
        stats['max_temporal_jump'] = float(max_jump)
        
        valid_smoothness = max_jump <= cfg['max_jump_per_timestep']
        print(f"  Max temporal jump: {max_jump:.2f} km/h - {'✓' if valid_smoothness else '⚠'}")
        
        # 4. Autocorrelation (lag-12 check)
        from pandas import Series
        speeds = edge_data['speed_kmh'].values[:1000]  # Sample for speed
        if len(speeds) > 24:
            autocorr_lag12 = Series(speeds).autocorr(lag=12)
            stats['autocorr_lag12'] = float(autocorr_lag12)
            
            valid_autocorr = (
                cfg['min_autocorr_lag12'] <= autocorr_lag12 <= cfg['max_autocorr_lag12']
            )
            print(f"  Autocorr lag-12: {autocorr_lag12:.4f} - {'✓' if valid_autocorr else '⚠'}")
        
        # 5. Event frequency
        if 'is_incident' in df.columns:
            incident_rate = df['is_incident'].sum() / len(df) * 100
            stats['incident_rate_percent'] = float(incident_rate)
            print(f"  Incident rate: {incident_rate:.3f}%")
        
        if 'is_construction' in df.columns:
            construction_rate = df['is_construction'].sum() / len(df) * 100
            stats['construction_rate_percent'] = float(construction_rate)
            print(f"  Construction rate: {construction_rate:.3f}%")
        
        # 6. Overall statistics
        stats['mean_speed'] = float(df['speed_kmh'].mean())
        stats['std_speed'] = float(df['speed_kmh'].std())
        stats['total_rows'] = len(df)
        stats['unique_timestamps'] = int(df['timestamp'].nunique())
        stats['unique_edges'] = int(df.groupby(['node_a_id', 'node_b_id']).ngroups)
        
        print(f"  Mean speed: {stats['mean_speed']:.2f} km/h")
        print(f"  Std speed: {stats['std_speed']:.2f} km/h")
        print(f"  Total rows: {stats['total_rows']:,}")
        
        return stats
    
    def generate(self) -> pd.DataFrame:
        """Main generation pipeline."""
        print("\n" + "=" * 80)
        print("SUPER DATASET GENERATION")
        print("=" * 80)
        
        # Step 1: Load topology
        edges_df, edge_ids = self.load_topology()
        num_edges = len(edge_ids)
        
        # Step 2: Generate timestamps
        timestamps = self.generate_timestamps()
        num_timestamps = len(timestamps)
        
        # Step 3: Initialize speed matrix
        print(f"\n[Step 1/8] Initializing {num_timestamps:,} × {num_edges} speed matrix...")
        speeds = np.zeros((num_timestamps, num_edges))
        
        # Apply temporal smoothing after each modification step
        
        # Step 4: Generate base patterns
        print("[Step 2/8] Generating base traffic patterns...")
        for t_idx, ts in enumerate(tqdm(timestamps, desc="Base patterns")):
            for e_idx, edge_id in enumerate(edge_ids):
                speeds[t_idx, e_idx] = self.generate_base_pattern(ts, edge_id)
        
        # Step 5: Apply seasonal overlay
        print("[Step 3/8] Applying seasonal patterns...")
        speeds = self.apply_seasonal_overlay(speeds, timestamps)
        
        # Step 6: Inject disruptions
        print("[Step 4/8] Injecting traffic incidents...")
        speeds = self.inject_incidents(speeds, timestamps, edge_ids)
        
        print("[Step 5/8] Adding construction zones...")
        speeds = self.add_construction_zones(speeds, timestamps, edge_ids)
        
        # Step 7: Apply environmental effects
        print("[Step 6/8] Applying weather effects...")
        speeds, weather_conditions = self.apply_weather_effects(speeds, timestamps)
        
        print("[Step 7/8] Injecting special events...")
        speeds = self.inject_special_events(speeds, timestamps, edge_ids)
        
        print("[Step 8/8] Applying holiday patterns...")
        speeds = self.apply_holidays(speeds, timestamps)
        
        # Step 9: Final smoothing and clipping
        print("\n[Post-processing] Applying temporal smoothing...")
        smoothing_window = self.config['base_patterns']['temporal_smoothing']
        speeds = self.smooth_temporal(speeds, window=smoothing_window)
        
        # Final clip to valid range
        speeds = np.clip(speeds, 3.0, 52.0)
        print(f"  Speed range after processing: [{speeds.min():.2f}, {speeds.max():.2f}] km/h")
        
        # Step 10: Create DataFrame
        print("\n[Assembly] Building final DataFrame...")
        
        # Create incident/construction flags
        incident_flags = np.zeros((num_timestamps, num_edges), dtype=bool)
        construction_flags = np.zeros((num_timestamps, num_edges), dtype=bool)
        holiday_flags = np.zeros(num_timestamps, dtype=bool)
        
        for inc in self.metadata['incidents']:
            inc_ts = pd.to_datetime(inc['timestamp'])
            t_idx = timestamps.get_loc(inc_ts) if inc_ts in timestamps else -1
            if t_idx >= 0:
                incident_flags[t_idx, inc['edge_idx']] = True
        
        for zone in self.metadata['construction_zones']:
            zone_start = pd.to_datetime(zone['start'])
            zone_end = pd.to_datetime(zone['end'])
            mask = (timestamps >= zone_start) & (timestamps <= zone_end)
            incident_flags[mask, zone['edge_idx']] = True
        
        for holiday in self.metadata['holidays']:
            if 'date' in holiday:
                h_date = pd.to_datetime(holiday['date']).date()
                mask = timestamps.date == h_date
                holiday_flags[mask] = True
        
        # Build DataFrame more efficiently
        rows = []
        for t_idx, ts in enumerate(tqdm(timestamps, desc="Assembly")):
            for e_idx, edge_id in enumerate(edge_ids):
                node_a, node_b = edge_id.split('_')
                rows.append({
                    'timestamp': ts,
                    'node_a_id': node_a,
                    'node_b_id': node_b,
                    'speed_kmh': speeds[t_idx, e_idx],
                    'is_incident': incident_flags[t_idx, e_idx],
                    'is_construction': construction_flags[t_idx, e_idx],
                    'weather_condition': weather_conditions[t_idx],
                    'is_holiday': holiday_flags[t_idx],
                    'temperature_c': self.rng.uniform(25, 35),  # Tropical climate
                    'precipitation_mm': 10.0 if 'rain' in weather_conditions[t_idx] else 0.0,
                    'wind_speed_kmh': self.rng.uniform(5, 20),
                    'humidity_percent': self.rng.uniform(60, 90)
                })
        
        df = pd.DataFrame(rows)
        print(f"[OK] Created DataFrame: {len(df):,} rows")
        
        # Step 9: Validate
        print("\n[Validation] Checking dataset quality...")
        stats = self.validate_dataset(df)
        
        return df, stats
    
    def save(
        self,
        df: pd.DataFrame,
        stats: Dict,
        output_path: str
    ):
        """Save dataset and metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        print(f"\n[Save] Writing to {output_path}...")
        df.to_parquet(output_path, compression='snappy')
        print(f"[OK] Dataset saved: {output_path.stat().st_size / 1e6:.1f} MB")
        
        # Save metadata
        meta_path = output_path.parent / "augmented_1year_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"[OK] Metadata saved: {meta_path}")
        
        # Save statistics
        stats_path = output_path.parent / "augmented_1year_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[OK] Statistics saved: {stats_path}")
        
        # Generate splits
        splits = self.create_splits(df)
        splits_path = output_path.parent / "augmented_1year_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2, default=str)
        print(f"[OK] Splits saved: {splits_path}")
    
    def create_splits(self, df: pd.DataFrame) -> Dict:
        """Create train/val/test split boundaries."""
        cfg = self.config['splits']
        
        unique_timestamps = sorted(df['timestamp'].unique())
        total_timestamps = len(unique_timestamps)
        
        # Calculate boundaries
        train_months = cfg['train']['end_month'] - cfg['train']['start_month'] + 1
        val_months = cfg['val']['end_month'] - cfg['val']['start_month'] + 1
        test_months = cfg['test']['end_month'] - cfg['test']['start_month'] + 1
        
        # Approximate timestamps per month
        timestamps_per_month = total_timestamps / 12
        
        train_end_idx = int(train_months * timestamps_per_month)
        gap_duration = cfg['gap']['duration_weeks'] * 7 * 24 * 6
        val_start_idx = train_end_idx + gap_duration
        val_end_idx = val_start_idx + int(val_months * timestamps_per_month)
        
        splits = {
            'train': {
                'start': str(unique_timestamps[0]),
                'end': str(unique_timestamps[train_end_idx - 1]),
                'num_timestamps': train_end_idx,
                'percentage': cfg['train']['percentage']
            },
            'gap': {
                'start': str(unique_timestamps[train_end_idx]),
                'end': str(unique_timestamps[val_start_idx - 1]),
                'num_timestamps': gap_duration
            },
            'val': {
                'start': str(unique_timestamps[val_start_idx]),
                'end': str(unique_timestamps[val_end_idx - 1]),
                'num_timestamps': val_end_idx - val_start_idx,
                'percentage': cfg['val']['percentage']
            },
            'test': {
                'start': str(unique_timestamps[val_end_idx]),
                'end': str(unique_timestamps[-1]),
                'num_timestamps': total_timestamps - val_end_idx,
                'percentage': cfg['test']['percentage']
            }
        }
        
        return splits
    
    def visualize(self, df: pd.DataFrame, output_dir: Path):
        """Create visualization plots."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        print("\n[Creating Visualizations]")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Sample week time series
        ax1 = plt.subplot(3, 2, 1)
        sample_week = df[df['timestamp'] < df['timestamp'].min() + pd.Timedelta(days=7)]
        sample_edge = sample_week.groupby(['node_a_id', 'node_b_id']).size().idxmax()
        edge_data = sample_week[
            (sample_week['node_a_id'] == sample_edge[0]) & 
            (sample_week['node_b_id'] == sample_edge[1])
        ].sort_values('timestamp')
        
        ax1.plot(edge_data['timestamp'], edge_data['speed_kmh'], linewidth=1)
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Sample Week - Traffic Speed Pattern')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Speed distribution
        ax2 = plt.subplot(3, 2, 2)
        ax2.hist(df['speed_kmh'], bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(df['speed_kmh'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["speed_kmh"].mean():.2f} km/h')
        ax2.set_xlabel('Speed (km/h)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Speed Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Event timeline
        ax3 = plt.subplot(3, 2, 3)
        if self.metadata['incidents']:
            incident_times = [pd.to_datetime(inc['timestamp']) for inc in self.metadata['incidents']]
            incident_severities = [inc['severity'] for inc in self.metadata['incidents']]
            ax3.scatter(incident_times, incident_severities, alpha=0.5, label='Incidents')
        
        if self.metadata['special_events']:
            event_times = [pd.to_datetime(evt['timestamp']) for evt in self.metadata['special_events']]
            event_congestion = [evt['congestion_factor'] for evt in self.metadata['special_events']]
            ax3.scatter(event_times, event_congestion, alpha=0.5, marker='^', label='Events')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Impact Severity / Congestion')
        ax3.set_title('Event Timeline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Hourly pattern
        ax4 = plt.subplot(3, 2, 4)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_mean = df.groupby('hour')['speed_kmh'].mean()
        ax4.plot(hourly_mean.index, hourly_mean.values, marker='o')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Mean Speed (km/h)')
        ax4.set_title('Daily Pattern (Average by Hour)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 2))
        
        # Plot 5: Weekly pattern
        ax5 = plt.subplot(3, 2, 5)
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        daily_mean = df.groupby('dayofweek')['speed_kmh'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax5.bar(range(7), daily_mean.values)
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Mean Speed (km/h)')
        ax5.set_title('Weekly Pattern')
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(days)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Statistics summary
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = f"""
DATASET STATISTICS

Size:
  Total rows: {len(df):,}
  Timestamps: {df['timestamp'].nunique():,}
  Edges: {df.groupby(['node_a_id', 'node_b_id']).ngroups}
  
Speed:
  Mean: {df['speed_kmh'].mean():.2f} km/h
  Std: {df['speed_kmh'].std():.2f} km/h
  Min: {df['speed_kmh'].min():.2f} km/h
  Max: {df['speed_kmh'].max():.2f} km/h
  
Events:
  Incidents: {len(self.metadata['incidents'])}
  Construction zones: {len(self.metadata['construction_zones'])}
  Weather events: {len(self.metadata['weather_events'])}
  Special events: {len(self.metadata['special_events'])}
  Holidays: {len(self.metadata['holidays'])}
  
Date Range:
  Start: {df['timestamp'].min()}
  End: {df['timestamp'].max()}
  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days
        """
        
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Super Dataset Analysis - 1 Year Traffic Simulation', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        viz_path = output_dir / 'augmented_1year_analysis.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {viz_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Super Traffic Dataset")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data/augmented_1year.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/augmented_1year.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test configuration without full generation'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SuperDatasetGenerator(args.config)
    
    if args.dry_run:
        print("\n[DRY RUN] Configuration loaded successfully!")
        print("Remove --dry-run to start generation.")
        return
    
    # Generate dataset
    df, stats = generator.generate()
    
    # Save
    generator.save(df, stats, args.output)
    
    # Visualize
    if args.visualize:
        print("\n[Visualization] Creating plots...")
        output_dir = Path(args.output).parent
        generator.visualize(df, output_dir)
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Dataset: {args.output}")
    print(f"Rows: {len(df):,}")
    print(f"Size: {Path(args.output).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
