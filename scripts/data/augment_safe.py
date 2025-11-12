"""
Safe Traffic Data Augmentation Module

Provides leak-free augmentation methods that only use training set statistics.

Author: THAT Le Quang (thatlq1812)
Date: 2025-11-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import timedelta


class SafeTrafficAugmentor:
    """
    Augment traffic data without information leakage.
    
    Key Principle:
        ALL statistics are computed from training data ONLY.
        NO information from validation or test sets is used.
    
    Safe Methods:
        1. Noise Injection: Add Gaussian noise
        2. Weather Scenarios: Synthetic weather within train ranges
        3. Temporal Jitter: Small time shifts with train patterns
        4. Edge Dropout: Random edge masking
    
    Usage:
        >>> augmentor = SafeTrafficAugmentor(train_df)
        >>> augmented = augmentor.augment_all(noise_copies=3, weather_scenarios=5)
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        random_seed: int = 42
    ):
        """
        Initialize augmentor with training data only.
        
        Args:
            train_df: Training data (MUST be only training split)
            random_seed: Random seed for reproducibility
        """
        self.train_df = train_df.copy()
        self.train_df['timestamp'] = pd.to_datetime(self.train_df['timestamp'])
        self.rng = np.random.default_rng(random_seed)
        
        # Learn patterns from training data ONLY
        self._learn_patterns_from_train()
        
        print(f"SafeTrafficAugmentor initialized with {len(self.train_df)} training samples")
        print(f"  Date range: {self.train_df['timestamp'].min()} to {self.train_df['timestamp'].max()}")
    
    def _learn_patterns_from_train(self):
        """
        Extract statistical patterns from training data only.
        
        This is the CRITICAL function that ensures no leakage.
        """
        print("Learning patterns from TRAINING DATA ONLY...")
        
        # Speed statistics
        self.speed_mean = self.train_df['speed_kmh'].mean()
        self.speed_std = self.train_df['speed_kmh'].std()
        self.speed_min = self.train_df['speed_kmh'].min()
        self.speed_max = self.train_df['speed_kmh'].max()
        
        # Temporal features
        self.train_df['hour'] = self.train_df['timestamp'].dt.hour
        self.train_df['dow'] = self.train_df['timestamp'].dt.dayofweek
        
        # Hourly patterns (from training only)
        self.hourly_profile = self.train_df.groupby('hour')['speed_kmh'].agg(['mean', 'std']).to_dict()
        
        # Day of week patterns (from training only)
        self.dow_profile = self.train_df.groupby('dow')['speed_kmh'].agg(['mean', 'std']).to_dict()
        
        # Weather statistics (from training only)
        weather_cols = ['temperature_c', 'wind_speed_kmh', 'precipitation_mm']
        self.weather_stats = {}
        for col in weather_cols:
            if col in self.train_df.columns:
                self.weather_stats[col] = {
                    'mean': self.train_df[col].mean(),
                    'std': self.train_df[col].std(),
                    'min': self.train_df[col].min(),
                    'max': self.train_df[col].max()
                }
        
        # Weather-speed correlation (from training only)
        if all(col in self.train_df.columns for col in weather_cols + ['speed_kmh']):
            corr_df = self.train_df[weather_cols + ['speed_kmh']].dropna()
            if len(corr_df) > 0:
                self.weather_correlation = corr_df.corr()['speed_kmh'].to_dict()
            else:
                self.weather_correlation = {}
        else:
            self.weather_correlation = {}
        
        # Edge-specific patterns (from training only)
        self.edge_profiles = {}
        for (node_a, node_b), group in self.train_df.groupby(['node_a_id', 'node_b_id']):
            if len(group) > 1:
                self.edge_profiles[(node_a, node_b)] = {
                    'mean': group['speed_kmh'].mean(),
                    'std': group['speed_kmh'].std(),
                    'count': len(group)
                }
        
        print(f"  Speed: mean={self.speed_mean:.2f}, std={self.speed_std:.2f}")
        print(f"  Learned {len(self.hourly_profile['mean'])} hourly patterns")
        print(f"  Learned {len(self.edge_profiles)} edge-specific patterns")
        print(f"  Weather correlations: {len(self.weather_correlation)} features")
    
    def augment_noise_injection(
        self,
        num_copies: int = 3,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Add Gaussian noise to speed values.
        
        Safe Method: Uses only training std for noise scale.
        
        Args:
            num_copies: Number of noisy copies to create
            noise_level: Noise magnitude as fraction of training std
        
        Returns:
            DataFrame with augmented data
        """
        augmented_dfs = []
        noise_scale = self.speed_std * noise_level
        
        for copy_idx in range(num_copies):
            df_noisy = self.train_df.copy()
            
            # Add Gaussian noise
            noise = self.rng.normal(0, noise_scale, len(df_noisy))
            df_noisy['speed_kmh'] += noise
            
            # Clip to valid range (using training min/max)
            df_noisy['speed_kmh'] = df_noisy['speed_kmh'].clip(
                self.speed_min,
                self.speed_max
            )
            
            # Update run_id
            df_noisy['run_id'] = 'aug_noise_' + str(copy_idx) + '_' + df_noisy['run_id'].astype(str)
            
            augmented_dfs.append(df_noisy)
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Noise injection: Created {len(result)} records ({num_copies} copies)")
        return result
    
    def augment_weather_scenarios(
        self,
        num_scenarios: int = 5
    ) -> pd.DataFrame:
        """
        Create synthetic weather variations within training ranges.
        
        Safe Method: Uses only training weather ranges and correlations.
        
        Args:
            num_scenarios: Number of weather scenarios to generate
        
        Returns:
            DataFrame with augmented data
        """
        if not self.weather_stats:
            print("Warning: No weather data in training set, skipping weather augmentation")
            return pd.DataFrame()
        
        augmented_dfs = []
        
        # Define weather scenarios within training ranges
        scenarios = [
            {'name': 'hot', 'temp_factor': 1.1, 'wind_factor': 0.9, 'rain_factor': 0.0},
            {'name': 'cool', 'temp_factor': 0.9, 'wind_factor': 1.1, 'rain_factor': 0.0},
            {'name': 'rainy', 'temp_factor': 0.95, 'wind_factor': 1.2, 'rain_factor': 2.0},
            {'name': 'windy', 'temp_factor': 1.0, 'wind_factor': 1.5, 'rain_factor': 0.5},
            {'name': 'mild', 'temp_factor': 1.0, 'wind_factor': 1.0, 'rain_factor': 0.0},
        ]
        
        for scenario in scenarios[:num_scenarios]:
            df_weather = self.train_df.copy()
            
            # Modify weather within training ranges
            if 'temperature_c' in df_weather.columns:
                temp_mean = self.weather_stats['temperature_c']['mean']
                df_weather['temperature_c'] = (
                    temp_mean + 
                    (df_weather['temperature_c'] - temp_mean) * scenario['temp_factor']
                )
                df_weather['temperature_c'] = df_weather['temperature_c'].clip(
                    self.weather_stats['temperature_c']['min'],
                    self.weather_stats['temperature_c']['max']
                )
            
            if 'wind_speed_kmh' in df_weather.columns:
                df_weather['wind_speed_kmh'] *= scenario['wind_factor']
                df_weather['wind_speed_kmh'] = df_weather['wind_speed_kmh'].clip(
                    self.weather_stats['wind_speed_kmh']['min'],
                    self.weather_stats['wind_speed_kmh']['max']
                )
            
            if 'precipitation_mm' in df_weather.columns:
                df_weather['precipitation_mm'] *= scenario['rain_factor']
                df_weather['precipitation_mm'] = df_weather['precipitation_mm'].clip(
                    0,
                    self.weather_stats['precipitation_mm']['max']
                )
            
            # Adjust speed based on training weather-speed correlation
            if 'temperature_c' in self.weather_correlation:
                temp_effect = (
                    (df_weather['temperature_c'] - self.weather_stats['temperature_c']['mean']) * 
                    self.weather_correlation.get('temperature_c', 0) * 0.1
                )
                df_weather['speed_kmh'] += temp_effect
            
            if 'precipitation_mm' in self.weather_correlation:
                rain_effect = (
                    df_weather['precipitation_mm'] * 
                    self.weather_correlation.get('precipitation_mm', -0.5)
                )
                df_weather['speed_kmh'] += rain_effect
            
            # Clip speed to training range
            df_weather['speed_kmh'] = df_weather['speed_kmh'].clip(
                self.speed_min,
                self.speed_max
            )
            
            # Update run_id
            df_weather['run_id'] = f"aug_weather_{scenario['name']}_" + df_weather['run_id'].astype(str)
            
            augmented_dfs.append(df_weather)
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Weather scenarios: Created {len(result)} records ({num_scenarios} scenarios)")
        return result
    
    def augment_temporal_jitter(
        self,
        num_copies: int = 2,
        max_jitter_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Apply small random time shifts within training period.
        
        Safe Method: Only shifts within training timestamp range,
        adjusts patterns using training hourly profiles.
        
        Args:
            num_copies: Number of jittered copies
            max_jitter_minutes: Maximum time shift in minutes
        
        Returns:
            DataFrame with augmented data
        """
        augmented_dfs = []
        
        for copy_idx in range(num_copies):
            df_jitter = self.train_df.copy()
            
            # Random time shifts
            jitter_minutes = self.rng.uniform(
                -max_jitter_minutes,
                max_jitter_minutes,
                len(df_jitter)
            )
            df_jitter['timestamp'] = df_jitter['timestamp'] + pd.to_timedelta(jitter_minutes, unit='m')
            
            # Ensure timestamps stay within training range
            train_min_time = self.train_df['timestamp'].min()
            train_max_time = self.train_df['timestamp'].max()
            df_jitter['timestamp'] = df_jitter['timestamp'].clip(train_min_time, train_max_time)
            
            # Update hour after jitter
            df_jitter['hour'] = df_jitter['timestamp'].dt.hour
            
            # Adjust speed based on new hour using training patterns
            for hour in df_jitter['hour'].unique():
                if hour in self.hourly_profile['mean']:
                    mask = df_jitter['hour'] == hour
                    hour_mean = self.hourly_profile['mean'][hour]
                    
                    # Slight adjustment toward hour's typical speed
                    adjustment_factor = self.rng.uniform(0.95, 1.05)
                    df_jitter.loc[mask, 'speed_kmh'] = (
                        df_jitter.loc[mask, 'speed_kmh'] * 0.9 +
                        hour_mean * 0.1 * adjustment_factor
                    )
            
            # Clip to training range
            df_jitter['speed_kmh'] = df_jitter['speed_kmh'].clip(
                self.speed_min,
                self.speed_max
            )
            
            # Update run_id
            df_jitter['run_id'] = f"aug_jitter_{copy_idx}_" + df_jitter['run_id'].astype(str)
            
            augmented_dfs.append(df_jitter)
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Temporal jitter: Created {len(result)} records ({num_copies} copies)")
        return result
    
    def augment_all(
        self,
        noise_copies: int = 3,
        weather_scenarios: int = 5,
        jitter_copies: int = 2,
        include_original: bool = True
    ) -> pd.DataFrame:
        """
        Apply all safe augmentation methods.
        
        Args:
            noise_copies: Number of noise-injected copies
            weather_scenarios: Number of weather variations
            jitter_copies: Number of time-jittered copies
            include_original: Whether to include original training data
        
        Returns:
            Combined augmented dataset
        """
        print("\n=== Safe Augmentation (Training Data Only) ===")
        
        augmented_dfs = []
        
        if include_original:
            augmented_dfs.append(self.train_df)
            print(f"Original: {len(self.train_df)} records")
        
        if noise_copies > 0:
            augmented_dfs.append(self.augment_noise_injection(noise_copies))
        
        if weather_scenarios > 0:
            weather_aug = self.augment_weather_scenarios(weather_scenarios)
            if len(weather_aug) > 0:
                augmented_dfs.append(weather_aug)
        
        if jitter_copies > 0:
            augmented_dfs.append(self.augment_temporal_jitter(jitter_copies))
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        
        print(f"\nTotal augmented records: {len(result)}")
        print(f"Augmentation factor: {len(result) / len(self.train_df):.2f}x")
        print(f"Unique runs: {result['run_id'].nunique()}")
        
        return result


def validate_no_leakage(
    train_augmented: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> bool:
    """
    Validate that augmented training data doesn't leak into val/test.
    
    Args:
        train_augmented: Augmented training data
        val_df: Validation data
        test_df: Test data
    
    Returns:
        True if no leakage detected
    """
    print("\n=== Validating No Data Leakage ===")
    
    train_max = train_augmented['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()
    
    checks = []
    
    # Check 1: No temporal overlap
    if train_max < val_min:
        print(f"✓ No temporal overlap: train ends before val starts")
        checks.append(True)
    else:
        print(f"✗ LEAKAGE: Train extends into validation period")
        print(f"  Train max: {train_max}")
        print(f"  Val min: {val_min}")
        checks.append(False)
    
    if val_max < test_min:
        print(f"✓ No temporal overlap: val ends before test starts")
        checks.append(True)
    else:
        print(f"✗ LEAKAGE: Val extends into test period")
        checks.append(False)
    
    # Check 2: No duplicate run_ids
    train_runs = set(train_augmented['run_id'].unique())
    val_runs = set(val_df['run_id'].unique())
    test_runs = set(test_df['run_id'].unique())
    
    if len(train_runs & val_runs) == 0:
        print(f"✓ No shared run_ids between train and val")
        checks.append(True)
    else:
        print(f"✗ LEAKAGE: {len(train_runs & val_runs)} shared run_ids between train and val")
        checks.append(False)
    
    if len(train_runs & test_runs) == 0:
        print(f"✓ No shared run_ids between train and test")
        checks.append(True)
    else:
        print(f"✗ LEAKAGE: {len(train_runs & test_runs)} shared run_ids between train and test")
        checks.append(False)
    
    all_passed = all(checks)
    if all_passed:
        print("\n✓ All leakage checks passed!")
    else:
        print("\n✗ LEAKAGE DETECTED - Review augmentation process")
    
    return all_passed


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    
    # Load data
    data_path = project_root / "data" / "processed" / "all_runs_combined.parquet"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Temporal split
    df = df.sort_values('timestamp')
    unique_times = sorted(df['timestamp'].unique())
    n_train = int(len(unique_times) * 0.7)
    n_val = int(len(unique_times) * 0.15)
    
    train_times = unique_times[:n_train]
    val_times = unique_times[n_train:n_train+n_val]
    test_times = unique_times[n_train+n_val:]
    
    train_df = df[df['timestamp'].isin(train_times)]
    val_df = df[df['timestamp'].isin(val_times)]
    test_df = df[df['timestamp'].isin(test_times)]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_df)} records")
    print(f"  Val: {len(val_df)} records")
    print(f"  Test: {len(test_df)} records")
    
    # Initialize augmentor with TRAINING DATA ONLY
    augmentor = SafeTrafficAugmentor(train_df)
    
    # Augment training data
    train_augmented = augmentor.augment_all(
        noise_copies=2,
        weather_scenarios=3,
        jitter_copies=1
    )
    
    # Validate no leakage
    validate_no_leakage(train_augmented, val_df, test_df)
    
    print("\n=== Example Complete ===")
