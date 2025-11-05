"""Advanced Data Augmentation for STMGT datasets."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from traffic_forecast.data.dataset_validation import (
    DEFAULT_REQUIRED_COLUMNS,
    validate_processed_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented STMGT training datasets.")
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=Path("data/processed/all_runs_combined.parquet"),
        help="Baseline processed parquet to augment (default: data/processed/all_runs_combined.parquet)",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("data/processed/all_runs_augmented.parquet"),
        help="Destination parquet for augmented data (default: data/processed/all_runs_augmented.parquet)",
    )
    parser.add_argument(
        "--temporal-start",
        default="2025-10-01",
        help="Start date for temporal extrapolation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--temporal-end",
        default="2025-10-29",
        help="End date for temporal extrapolation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--pattern-variations",
        type=int,
        default=5,
        help="Number of pattern variation copies to generate",
    )
    parser.add_argument(
        "--noise-copies",
        type=int,
        default=3,
        help="Number of noise-injected dataset copies",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.15,
        help="Noise level multiplier for speed standard deviation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the augmented parquet after writing",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Extra column names required during validation",
    )
    return parser.parse_args(argv)

class TrafficDataAugmentor:
    """
    Advanced augmentation với statistical guarantees
    """
    
    def __init__(self, df_original):
        self.df_orig = df_original.copy()
        self.df_orig['timestamp'] = pd.to_datetime(self.df_orig['timestamp'])
        
        # Learn patterns from original data
        self._learn_patterns()
    
    def _learn_patterns(self):
        """Extract statistical patterns"""
        print("Learning patterns from original data...")
        
        # Hourly patterns
        self.df_orig['hour'] = self.df_orig['timestamp'].dt.hour
        self.hourly_profile = self.df_orig.groupby('hour')['speed_kmh'].agg(['mean', 'std']).to_dict()
        
        # Day of week patterns  
        self.df_orig['dow'] = self.df_orig['timestamp'].dt.dayofweek
        self.dow_profile = self.df_orig.groupby('dow')['speed_kmh'].agg(['mean', 'std']).to_dict()
        
        # Edge-specific patterns
        self.edge_profiles = {}
        for (node_a, node_b), group in self.df_orig.groupby(['node_a_id', 'node_b_id']):
            self.edge_profiles[(node_a, node_b)] = {
                'mean': group['speed_kmh'].mean(),
                'std': group['speed_kmh'].std(),
                'min': group['speed_kmh'].min(),
                'max': group['speed_kmh'].max()
            }
        
        # Weather-speed correlation
        corr_data = self.df_orig[['speed_kmh', 'temperature_c', 'wind_speed_kmh', 'precipitation_mm']].dropna()
        if len(corr_data) > 0:
            self.weather_corr = corr_data.corr()['speed_kmh'].to_dict()
        else:
            self.weather_corr = {}
        
        # Overall statistics
        self.global_stats = {
            'speed_mean': self.df_orig['speed_kmh'].mean(),
            'speed_std': self.df_orig['speed_kmh'].std(),
            'speed_min': self.df_orig['speed_kmh'].min(),
            'speed_max': self.df_orig['speed_kmh'].max(),
            'temp_mean': self.df_orig['temperature_c'].mean(),
            'temp_std': self.df_orig['temperature_c'].std(),
            'wind_mean': self.df_orig['wind_speed_kmh'].mean(),
            'wind_std': self.df_orig['wind_speed_kmh'].std()
        }
        
        print(f"  Learned {len(self.hourly_profile['mean'])} hourly patterns")
        print(f"  Learned {len(self.edge_profiles)} edge profiles")
        print(f"  Weather-Speed correlation: {self.weather_corr.get('temperature_c', 0):.3f}")
    
    def augment_temporal_extrapolation(self, start_date='2025-10-01', end_date='2025-10-29'):
        """
        Method 1: Extrapolate backwards in time
        """
        print(f"\nAugmenting: Temporal Extrapolation ({start_date} to {end_date})")
        
        # Get original date range
        orig_start = self.df_orig['timestamp'].min()
        orig_end = self.df_orig['timestamp'].max()
        orig_duration = (orig_end - orig_start).total_seconds()
        
        # Target date range
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)
        
        # Calculate how many "cycles" to create
        target_duration = (target_end - target_start).total_seconds()
        num_cycles = int(target_duration / orig_duration) + 1
        
        augmented_dfs = []
        
        for cycle in range(num_cycles):
            # Shift timestamp
            time_shift = timedelta(seconds=-cycle * orig_duration)
            df_cycle = self.df_orig.copy()
            df_cycle['timestamp'] = df_cycle['timestamp'] + time_shift
            
            # Filter to target range
            df_cycle = df_cycle[
                (df_cycle['timestamp'] >= target_start) &
                (df_cycle['timestamp'] <= target_end)
            ]
            
            if len(df_cycle) == 0:
                continue
            
            # Add variation based on day of week
            df_cycle['dow'] = df_cycle['timestamp'].dt.dayofweek
            
            # Scale speed based on day patterns
            for dow in df_cycle['dow'].unique():
                if dow in self.dow_profile['mean']:
                    mask = df_cycle['dow'] == dow
                    base_dow_mean = self.dow_profile['mean'][dow]
                    
                    # Add realistic variation (±10%)
                    scale_factor = np.random.uniform(0.9, 1.1)
                    df_cycle.loc[mask, 'speed_kmh'] *= scale_factor
            
            # Add noise to preserve variance
            noise = np.random.normal(0, self.global_stats['speed_std'] * 0.1, len(df_cycle))
            df_cycle['speed_kmh'] += noise
            df_cycle['speed_kmh'] = df_cycle['speed_kmh'].clip(
                self.global_stats['speed_min'],
                self.global_stats['speed_max']
            )
            
            # Generate new run_id
            df_cycle['run_id'] = f"aug_temporal_{start_date}_{cycle}_" + df_cycle['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_cycle)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} augmented runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_pattern_variations(self, num_variations=5):
        """
        Method 2: Create variations based on learned patterns
        """
        print(f"\nAugmenting: Pattern Variations (x{num_variations})")
        
        augmented_dfs = []
        
        for var_id in range(num_variations):
            df_var = self.df_orig.copy()
            
            # Variation type
            variation_type = var_id % 5
            
            if variation_type == 0:
                # Rush hour intensity variation
                df_var['hour'] = df_var['timestamp'].dt.hour
                rush_hours = [7, 8, 9, 17, 18, 19]
                mask = df_var['hour'].isin(rush_hours)
                intensity = np.random.uniform(0.7, 0.95)  # More congestion
                df_var.loc[mask, 'speed_kmh'] *= intensity
                suffix = f"rush_var_{var_id}"
                
            elif variation_type == 1:
                # Weather impact variation
                # Stronger correlation with weather
                temp_effect = (df_var['temperature_c'] - self.global_stats['temp_mean']) * np.random.uniform(-0.3, -0.1)
                wind_effect = (df_var['wind_speed_kmh'] - self.global_stats['wind_mean']) * np.random.uniform(-0.2, 0)
                df_var['speed_kmh'] += temp_effect + wind_effect
                suffix = f"weather_var_{var_id}"
                
            elif variation_type == 2:
                # Weekend pattern
                df_var['dow'] = df_var['timestamp'].dt.dayofweek
                weekend_mask = df_var['dow'] >= 5
                df_var.loc[weekend_mask, 'speed_kmh'] *= np.random.uniform(1.05, 1.15)  # Faster on weekends
                suffix = f"weekend_var_{var_id}"
                
            elif variation_type == 3:
                # Random events (accidents, roadworks)
                # Randomly select 10% of edges to have "events"
                edge_list = list(self.edge_profiles.keys())
                num_event_edges = max(1, len(edge_list) // 10)
                event_edge_indices = np.random.choice(len(edge_list), size=num_event_edges, replace=False)
                event_edges = [edge_list[i] for i in event_edge_indices]
                
                for edge in event_edges:
                    mask = (df_var['node_a_id'] == edge[0]) & (df_var['node_b_id'] == edge[1])
                    df_var.loc[mask, 'speed_kmh'] *= np.random.uniform(0.5, 0.8)  # Slower due to event
                suffix = f"event_var_{var_id}"
                
            else:
                # Seasonal variation (early Oct vs late Oct)
                # Gradual change in base speed
                seasonal_factor = np.random.uniform(0.95, 1.05)
                df_var['speed_kmh'] *= seasonal_factor
                suffix = f"seasonal_var_{var_id}"
            
            # Clip to valid range
            df_var['speed_kmh'] = df_var['speed_kmh'].clip(
                self.global_stats['speed_min'],
                self.global_stats['speed_max']
            )
            
            # Add small noise
            noise = np.random.normal(0, self.global_stats['speed_std'] * 0.05, len(df_var))
            df_var['speed_kmh'] += noise
            df_var['speed_kmh'] = df_var['speed_kmh'].clip(
                self.global_stats['speed_min'],
                self.global_stats['speed_max']
            )
            
            # New run IDs
            df_var['run_id'] = f"aug_{suffix}_" + df_var['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_var)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} variation runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_noise_injection(self, num_copies=3, noise_level=0.15):
        """
        Method 3: Simple noise injection
        """
        print(f"\nAugmenting: Noise Injection (x{num_copies}, σ={noise_level})")
        
        augmented_dfs = []
        
        for copy_id in range(num_copies):
            df_noise = self.df_orig.copy()
            
            # Add Gaussian noise
            noise = np.random.normal(0, self.global_stats['speed_std'] * noise_level, len(df_noise))
            df_noise['speed_kmh'] += noise
            df_noise['speed_kmh'] = df_noise['speed_kmh'].clip(
                self.global_stats['speed_min'],
                self.global_stats['speed_max']
            )
            
            # Weather noise
            temp_noise = np.random.normal(0, self.global_stats['temp_std'] * 0.1, len(df_noise))
            df_noise['temperature_c'] += temp_noise
            
            wind_noise = np.random.normal(0, self.global_stats['wind_std'] * 0.1, len(df_noise))
            df_noise['wind_speed_kmh'] += wind_noise
            df_noise['wind_speed_kmh'] = df_noise['wind_speed_kmh'].clip(0, None)
            
            # New run IDs
            df_noise['run_id'] = f"aug_noise_{copy_id}_" + df_noise['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_noise)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} noise-augmented runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def validate_augmented_data(self, df_augmented):
        """Check if augmented data preserves statistical properties"""
        print("\nValidating augmented data...")
        
        # Speed distribution
        orig_mean = self.df_orig['speed_kmh'].mean()
        orig_std = self.df_orig['speed_kmh'].std()
        aug_mean = df_augmented['speed_kmh'].mean()
        aug_std = df_augmented['speed_kmh'].std()
        
        print(f"  Speed mean: {orig_mean:.2f} → {aug_mean:.2f} (Δ {abs(aug_mean-orig_mean)/orig_mean*100:.1f}%)")
        print(f"  Speed std: {orig_std:.2f} → {aug_std:.2f} (Δ {abs(aug_std-orig_std)/orig_std*100:.1f}%)")
        
        # Weather correlations
        aug_corr_data = df_augmented[['speed_kmh', 'temperature_c', 'wind_speed_kmh']].dropna()
        if len(aug_corr_data) > 0:
            aug_corr = aug_corr_data.corr()['speed_kmh']
            orig_temp_corr = self.weather_corr.get('temperature_c', 0)
            aug_temp_corr = aug_corr.get('temperature_c', 0)
            print(f"  Speed-Temp corr: {orig_temp_corr:.3f} → {aug_temp_corr:.3f}")
        
        # Range preservation
        print(f"  Speed range: [{self.global_stats['speed_min']:.1f}, {self.global_stats['speed_max']:.1f}] → [{df_augmented['speed_kmh'].min():.1f}, {df_augmented['speed_kmh'].max():.1f}]")
        
        # Statistical test
        # Sample to same size for fair comparison
        sample_size = min(len(self.df_orig), len(df_augmented), 1000)
        orig_sample = self.df_orig['speed_kmh'].sample(sample_size, random_state=42)
        aug_sample = df_augmented['speed_kmh'].sample(sample_size, random_state=42)
        
        ks_stat, ks_p = stats.ks_2samp(orig_sample, aug_sample)
        print(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
        if ks_p > 0.05:
            print("  ✓ Distributions are similar (p > 0.05)")
        else:
            print("  ⚠ Distributions differ (p < 0.05) - expected with augmentation")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)

    print("=" * 70)
    print("ADVANCED DATA AUGMENTATION")
    print("=" * 70)

    input_path = _resolve_path(args.input_dataset)
    output_path = _resolve_path(args.output_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_parquet(input_path)
    print(f"\nOriginal data: {len(df_orig)} records, {df_orig['run_id'].nunique()} runs")

    augmentor = TrafficDataAugmentor(df_orig)

    augmented_parts = []

    df_temporal = augmentor.augment_temporal_extrapolation(args.temporal_start, args.temporal_end)
    if not df_temporal.empty:
        augmented_parts.append(df_temporal)

    df_patterns = augmentor.augment_pattern_variations(num_variations=args.pattern_variations)
    if not df_patterns.empty:
        augmented_parts.append(df_patterns)

    df_noise = augmentor.augment_noise_injection(num_copies=args.noise_copies, noise_level=args.noise_level)
    if not df_noise.empty:
        augmented_parts.append(df_noise)

    if augmented_parts:
        df_all_augmented = pd.concat(augmented_parts, ignore_index=True)
        df_combined = pd.concat([df_orig, df_all_augmented], ignore_index=True)

        print("\n" + "=" * 70)
        print("AUGMENTATION SUMMARY")
        print("=" * 70)
        print(f"Original runs: {df_orig['run_id'].nunique()}")
        print(f"Augmented runs: {df_all_augmented['run_id'].nunique()}")
        print(f"Total runs: {df_combined['run_id'].nunique()}")
        print(f"Multiplication factor: {df_combined['run_id'].nunique() / df_orig['run_id'].nunique():.1f}x")
        print(f"\nOriginal records: {len(df_orig)}")
        print(f"Augmented records: {len(df_all_augmented)}")
        print(f"Total records: {len(df_combined)}")

        augmentor.validate_augmented_data(df_all_augmented)

        df_combined.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to {output_path}")

        if not args.no_validate:
            required = list(DEFAULT_REQUIRED_COLUMNS) + ["temperature_c", "wind_speed_kmh", "precipitation_mm"]
            if args.require:
                required.extend(args.require)
            validation = validate_processed_dataset(output_path, required)
            print("\n== Dataset Validation Report ==")
            print(f"Path: {validation.path}")
            print(f"Rows: {validation.rows}")
            print(f"Missing Columns: {validation.missing_columns or '-'}")
            print(f"Columns With Nulls: {validation.null_columns or '-'}")
            print(f"Duplicate Rows: {validation.duplicate_rows}")
            if validation.errors:
                print("Errors:")
                for error in validation.errors:
                    print(f"  - {error}")
            if not validation.is_valid:
                raise SystemExit("Augmented dataset validation failed; please review output.")

        print("\n" + "=" * 70)
        print("EXPECTED TRAINING SAMPLES")
        print("=" * 70)

        total_runs = df_combined['run_id'].nunique()
        configs = [
            (12, 12, "Original (High quality)"),
            (6, 3, "Moderate (Balanced)"),
            (4, 2, "Conservative (More samples)"),
        ]

        for seq_len, pred_len, desc in configs:
            window_size = seq_len + pred_len
            samples = max(0, total_runs - window_size + 1)
            print(f"{desc}:")
            print(f"  seq_len={seq_len}, pred_len={pred_len}")
            print(f"  Window size: {window_size} runs")
            print(f"  Training samples: ~{samples}")
            print()

        print("=" * 70)

    else:
        print("\n⚠ No augmented data generated")


if __name__ == '__main__':
    main()
