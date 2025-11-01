"""
Smart Data Augmentation for Traffic Forecasting

Generates realistic synthetic traffic data from real observations using:
1. Temporal interpolation (smooth transitions between hours)
2. Noise injection (realistic variance)
3. Pattern-based generation (daily/weekly patterns)
4. Physics-based constraints (speed limits, congestion patterns)

Author: thatlq1812
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TrafficDataAugmentor:
    """
    Intelligent traffic data augmentation with realistic constraints.
    
    Strategy:
    1. Temporal Interpolation - Fill gaps between runs
    2. Noise Injection - Add realistic variance
    3. Pattern Learning - Extract and replicate daily patterns
    4. Physics Constraints - Maintain realistic speed bounds
    """
    
    def __init__(
        self,
        min_speed: float = 5.0,
        max_speed: float = 80.0,
        noise_std_ratio: float = 0.1
    ):
        """
        Initialize augmentor.
        
        Args:
            min_speed: Minimum realistic speed (km/h)
            max_speed: Maximum realistic speed (km/h)
            noise_std_ratio: Noise std as ratio of speed range
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.noise_std_ratio = noise_std_ratio
        
        self.scaler = StandardScaler()
        
    def temporal_interpolation(
        self,
        df: pd.DataFrame,
        target_frequency: str = '5min',
        method: str = 'cubic'
    ) -> pd.DataFrame:
        """
        Interpolate traffic data to higher frequency.
        
        Args:
            df: DataFrame with traffic data (must have 'timestamp', 'edge_id', 'speed_kmh')
            target_frequency: Target sampling frequency ('1min', '5min', '15min', etc.)
            method: Interpolation method ('linear', 'cubic', 'quadratic')
            
        Returns:
            Augmented DataFrame with interpolated samples
        """
        logger.info(f"Starting temporal interpolation to {target_frequency}...")
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        augmented_dfs = []
        
        # Process each edge separately
        for edge_id in df['edge_id'].unique():
            edge_df = df[df['edge_id'] == edge_id].sort_values('timestamp')
            
            if len(edge_df) < 2:
                # Not enough points to interpolate
                augmented_dfs.append(edge_df)
                continue
            
            # Create time range
            time_min = edge_df['timestamp'].min()
            time_max = edge_df['timestamp'].max()
            new_times = pd.date_range(time_min, time_max, freq=target_frequency)
            
            # Convert to numeric for interpolation
            times_numeric = (edge_df['timestamp'] - time_min).dt.total_seconds().values
            new_times_numeric = (new_times - time_min).total_seconds().values
            
            # Interpolate speed
            if method == 'cubic' and len(edge_df) >= 4:
                # Cubic spline needs at least 4 points
                interp_func = CubicSpline(times_numeric, edge_df['speed_kmh'].values)
            else:
                # Fallback to linear
                interp_func = interp1d(
                    times_numeric, 
                    edge_df['speed_kmh'].values,
                    kind='linear',
                    fill_value='extrapolate'
                )
            
            new_speeds = interp_func(new_times_numeric)
            
            # Apply constraints
            new_speeds = np.clip(new_speeds, self.min_speed, self.max_speed)
            
            # Create new DataFrame
            new_df = pd.DataFrame({
                'timestamp': new_times,
                'edge_id': edge_id,
                'speed_kmh': new_speeds,
                'augmented': True,
                'augmentation_method': 'temporal_interpolation'
            })
            
            # Copy other columns from nearest real sample
            for col in edge_df.columns:
                if col not in new_df.columns and col != 'timestamp':
                    # Use forward fill then backward fill
                    merged = pd.merge_asof(
                        new_df[['timestamp']].sort_values('timestamp'),
                        edge_df[['timestamp', col]].sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                    new_df[col] = merged[col].values
            
            augmented_dfs.append(new_df)
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        
        logger.info(f"✓ Interpolated {len(df)} → {len(result)} samples "
                   f"({len(result)/len(df):.1f}x increase)")
        
        return result
    
    def add_realistic_noise(
        self,
        df: pd.DataFrame,
        noise_columns: List[str] = None,
        preserve_real: bool = True
    ) -> pd.DataFrame:
        """
        Add realistic Gaussian noise to augmented samples.
        
        Args:
            df: DataFrame with traffic data
            noise_columns: Columns to add noise to (default: ['speed_kmh'])
            preserve_real: Don't add noise to original real samples
            
        Returns:
            DataFrame with noise added
        """
        if noise_columns is None:
            noise_columns = ['speed_kmh']
        
        df = df.copy()
        
        # Mark original data if not already marked
        if 'augmented' not in df.columns:
            df['augmented'] = False
        
        for col in noise_columns:
            if col not in df.columns:
                continue
            
            # Calculate noise std based on data variance
            data_std = df[col].std()
            noise_std = data_std * self.noise_std_ratio
            
            # Generate noise
            noise = np.random.normal(0, noise_std, len(df))
            
            # Apply only to augmented samples
            if preserve_real:
                mask = df['augmented'] == True
                df.loc[mask, col] = df.loc[mask, col] + noise[mask]
            else:
                df[col] = df[col] + noise
            
            # Apply constraints
            if col == 'speed_kmh':
                df[col] = np.clip(df[col], self.min_speed, self.max_speed)
        
        logger.info(f"✓ Added realistic noise (std={noise_std:.2f}) to {len(df[df['augmented']])} samples")
        
        return df
    
    def pattern_based_augmentation(
        self,
        df: pd.DataFrame,
        pattern_type: str = 'hourly',
        n_synthetic_days: int = 5
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on learned patterns.
        
        Args:
            df: DataFrame with traffic data
            pattern_type: 'hourly', 'daily', or 'weekly'
            n_synthetic_days: Number of synthetic days to generate
            
        Returns:
            DataFrame with pattern-based synthetic data
        """
        logger.info(f"Generating pattern-based synthetic data ({pattern_type})...")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract hour of day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Learn hourly patterns for each edge
        synthetic_samples = []
        
        for edge_id in df['edge_id'].unique():
            edge_df = df[df['edge_id'] == edge_id]
            
            # Calculate mean speed per hour
            hourly_pattern = edge_df.groupby('hour')['speed_kmh'].agg(['mean', 'std']).reset_index()
            
            # Fill missing hours with interpolation
            all_hours = pd.DataFrame({'hour': range(24)})
            hourly_pattern = all_hours.merge(hourly_pattern, on='hour', how='left')
            hourly_pattern['mean'] = hourly_pattern['mean'].interpolate(method='linear')
            hourly_pattern['std'] = hourly_pattern['std'].fillna(hourly_pattern['std'].mean())
            
            # Generate synthetic days
            base_date = df['timestamp'].max() + timedelta(days=1)
            
            for day in range(n_synthetic_days):
                for hour in range(24):
                    # Generate samples for this hour (e.g., every 5 minutes)
                    for minute in range(0, 60, 5):
                        timestamp = base_date + timedelta(days=day, hours=hour, minutes=minute)
                        
                        # Get pattern mean and std
                        pattern = hourly_pattern[hourly_pattern['hour'] == hour].iloc[0]
                        mean_speed = pattern['mean']
                        std_speed = pattern['std']
                        
                        # Generate with noise
                        speed = np.random.normal(mean_speed, std_speed * self.noise_std_ratio)
                        speed = np.clip(speed, self.min_speed, self.max_speed)
                        
                        synthetic_samples.append({
                            'timestamp': timestamp,
                            'edge_id': edge_id,
                            'speed_kmh': speed,
                            'augmented': True,
                            'augmentation_method': f'pattern_{pattern_type}'
                        })
        
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        # Copy other columns from template
        template = df.iloc[0:1].copy()
        for col in template.columns:
            if col not in synthetic_df.columns:
                synthetic_df[col] = template[col].values[0]
        
        logger.info(f"✓ Generated {len(synthetic_df)} pattern-based samples")
        
        return pd.concat([df, synthetic_df], ignore_index=True)
    
    def augment_dataset(
        self,
        df: pd.DataFrame,
        strategy: str = 'interpolation',
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply data augmentation strategy.
        
        Args:
            df: Original DataFrame
            strategy: Augmentation strategy:
                - 'interpolation': Temporal interpolation only
                - 'interpolation+noise': Interpolation + noise
                - 'pattern': Pattern-based generation
                - 'full': All methods combined
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Augmented DataFrame
        """
        logger.info(f"Starting data augmentation: {strategy}")
        
        original_size = len(df)
        
        if strategy == 'interpolation':
            df = self.temporal_interpolation(
                df, 
                target_frequency=kwargs.get('target_frequency', '5min'),
                method=kwargs.get('method', 'cubic')
            )
        
        elif strategy == 'interpolation+noise':
            df = self.temporal_interpolation(
                df,
                target_frequency=kwargs.get('target_frequency', '5min'),
                method=kwargs.get('method', 'cubic')
            )
            df = self.add_realistic_noise(df)
        
        elif strategy == 'pattern':
            df = self.pattern_based_augmentation(
                df,
                pattern_type=kwargs.get('pattern_type', 'hourly'),
                n_synthetic_days=kwargs.get('n_synthetic_days', 5)
            )
        
        elif strategy == 'full':
            # Interpolation
            df = self.temporal_interpolation(df, target_frequency='5min')
            # Add noise
            df = self.add_realistic_noise(df)
            # Generate pattern-based data
            df = self.pattern_based_augmentation(df, n_synthetic_days=3)
        
        augmented_size = len(df)
        
        logger.info(f"✓ Augmentation complete: {original_size} → {augmented_size} samples "
                   f"({augmented_size/original_size:.1f}x)")
        
        return df
    
    def validate_augmented_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate augmented data quality.
        
        Returns:
            Validation metrics
        """
        metrics = {
            'total_samples': len(df),
            'real_samples': len(df[~df.get('augmented', False)]),
            'augmented_samples': len(df[df.get('augmented', False)]),
            'speed_mean': df['speed_kmh'].mean(),
            'speed_std': df['speed_kmh'].std(),
            'speed_min': df['speed_kmh'].min(),
            'speed_max': df['speed_kmh'].max(),
            'out_of_bounds': len(df[(df['speed_kmh'] < self.min_speed) | 
                                    (df['speed_kmh'] > self.max_speed)])
        }
        
        return metrics


def quick_augment(
    input_path: Path,
    output_path: Path,
    target_frequency: str = '5min',
    add_noise: bool = True
) -> pd.DataFrame:
    """
    Quick data augmentation utility.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save augmented data
        target_frequency: Target sampling frequency
        add_noise: Whether to add realistic noise
        
    Returns:
        Augmented DataFrame
    """
    # Load data
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} samples from {input_path}")
    
    # Augment
    augmentor = TrafficDataAugmentor()
    
    strategy = 'interpolation+noise' if add_noise else 'interpolation'
    df_augmented = augmentor.augment_dataset(
        df,
        strategy=strategy,
        target_frequency=target_frequency
    )
    
    # Validate
    metrics = augmentor.validate_augmented_data(df_augmented)
    
    logger.info("Validation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_parquet(output_path)
    
    logger.info(f"✓ Saved augmented data to {output_path}")
    
    return df_augmented


if __name__ == '__main__':
    """Test data augmentation"""
    from traffic_forecast import PROJECT_ROOT
    
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    input_file = PROJECT_ROOT / 'data/processed/all_runs_combined.parquet'
    output_file = PROJECT_ROOT / 'data/processed/augmented_5min.parquet'
    
    # Test interpolation
    df_aug = quick_augment(
        input_file,
        output_file,
        target_frequency='5min',
        add_noise=True
    )
    
    print(f"\nAugmentation complete!")
    print(f"Original: {len(pd.read_parquet(input_file))} samples")
    print(f"Augmented: {len(df_aug)} samples")
    print(f"Increase: {len(df_aug) / len(pd.read_parquet(input_file)):.1f}x")
