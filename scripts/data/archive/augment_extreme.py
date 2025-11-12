"""
EXTREME Data Augmentation - Going Beyond

Methods:
1. Hourly Interpolation - Create runs between existing timestamps
2. Synthetic Weather from Historical Data
3. Multi-scenario Variations (10x instead of 5x)
4. Edge-specific Augmentation - Different patterns per road type
5. Time-shift Variations - Small time perturbations

Target: 2000+ runs (vs current 891)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy import stats

np.random.seed(42)

class ExtremeAugmentor:
    """
    Extreme augmentation for maximum data generation
    """
    
    def __init__(self, df_original):
        self.df_orig = df_original.copy()
        self.df_orig['timestamp'] = pd.to_datetime(self.df_orig['timestamp'])
        self._learn_patterns()
    
    def _learn_patterns(self):
        """Learn patterns from data"""
        print("Learning advanced patterns...")
        
        # Hourly patterns
        self.df_orig['hour'] = self.df_orig['timestamp'].dt.hour
        self.df_orig['minute'] = self.df_orig['timestamp'].dt.minute
        
        # Edge-specific hourly profiles
        self.edge_hourly_profiles = {}
        for (node_a, node_b), group in self.df_orig.groupby(['node_a_id', 'node_b_id']):
            hourly_profile = group.groupby('hour')['speed_kmh'].agg(['mean', 'std', 'count'])
            self.edge_hourly_profiles[(node_a, node_b)] = hourly_profile.to_dict()
        
        # Global patterns
        self.global_hourly = self.df_orig.groupby('hour')['speed_kmh'].agg(['mean', 'std'])
        
        # Weather ranges
        self.weather_ranges = {
            'temperature_c': (self.df_orig['temperature_c'].min(), self.df_orig['temperature_c'].max()),
            'wind_speed_kmh': (self.df_orig['wind_speed_kmh'].min(), self.df_orig['wind_speed_kmh'].max()),
            'precipitation_mm': (self.df_orig['precipitation_mm'].min(), self.df_orig['precipitation_mm'].max())
        }
        
        print(f"  Learned {len(self.edge_hourly_profiles)} edge-specific profiles")
    
    def augment_hourly_interpolation(self, num_interpolations=2):
        """
        Method 1: Create intermediate runs between existing timestamps
        """
        print(f"\nMethod 1: Hourly Interpolation (x{num_interpolations} between each pair)")
        
        # Get unique runs sorted by time
        runs = sorted(self.df_orig['run_id'].unique())
        run_groups = [self.df_orig[self.df_orig['run_id'] == r] for r in runs]
        
        augmented_dfs = []
        
        for i in range(len(run_groups) - 1):
            run1 = run_groups[i]
            run2 = run_groups[i + 1]
            
            t1 = run1.iloc[0]['timestamp']
            t2 = run2.iloc[0]['timestamp']
            
            time_diff = (t2 - t1).total_seconds()
            
            # Create interpolated runs
            for interp_idx in range(1, num_interpolations + 1):
                alpha = interp_idx / (num_interpolations + 1)
                t_interp = t1 + timedelta(seconds=time_diff * alpha)
                
                # Interpolate for each edge
                df_interp = []
                
                for (node_a, node_b), edge1 in run1.groupby(['node_a_id', 'node_b_id']):
                    edge2_match = run2[(run2['node_a_id'] == node_a) & (run2['node_b_id'] == node_b)]
                    
                    if len(edge2_match) == 0:
                        continue
                    
                    edge2 = edge2_match.iloc[0]
                    edge1_row = edge1.iloc[0]
                    
                    # Linear interpolation
                    speed_interp = edge1_row['speed_kmh'] * (1 - alpha) + edge2['speed_kmh'] * alpha
                    temp_interp = edge1_row['temperature_c'] * (1 - alpha) + edge2['temperature_c'] * alpha
                    wind_interp = edge1_row['wind_speed_kmh'] * (1 - alpha) + edge2['wind_speed_kmh'] * alpha
                    precip_interp = edge1_row['precipitation_mm'] * (1 - alpha) + edge2['precipitation_mm'] * alpha
                    
                    # Add small noise
                    speed_interp += np.random.normal(0, 0.5)
                    
                    interp_row = edge1_row.copy()
                    interp_row['timestamp'] = t_interp
                    interp_row['speed_kmh'] = speed_interp
                    interp_row['temperature_c'] = temp_interp
                    interp_row['wind_speed_kmh'] = wind_interp
                    interp_row['precipitation_mm'] = precip_interp
                    interp_row['run_id'] = f"aug_interp_{i}_{interp_idx}_{t_interp.strftime('%Y%m%d_%H%M%S')}"
                    
                    df_interp.append(interp_row)
                
                if df_interp:
                    augmented_dfs.append(pd.DataFrame(df_interp))
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} interpolated runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_synthetic_weather(self, num_weather_scenarios=10):
        """
        Method 2: Create variations with different weather conditions
        """
        print(f"\nMethod 2: Synthetic Weather Variations (x{num_weather_scenarios})")
        
        augmented_dfs = []
        
        # Weather scenarios
        weather_scenarios = [
            {'name': 'hot_dry', 'temp_mult': 1.15, 'wind_mult': 0.8, 'precip_mult': 0.0},
            {'name': 'cool_windy', 'temp_mult': 0.85, 'wind_mult': 1.5, 'precip_mult': 0.0},
            {'name': 'light_rain', 'temp_mult': 0.95, 'wind_mult': 1.1, 'precip_mult': 2.0},
            {'name': 'heavy_rain', 'temp_mult': 0.90, 'wind_mult': 1.3, 'precip_mult': 5.0},
            {'name': 'extreme_heat', 'temp_mult': 1.20, 'wind_mult': 0.7, 'precip_mult': 0.0},
            {'name': 'moderate', 'temp_mult': 1.0, 'wind_mult': 1.0, 'precip_mult': 0.5},
            {'name': 'cold_front', 'temp_mult': 0.80, 'wind_mult': 1.6, 'precip_mult': 1.5},
            {'name': 'humid_calm', 'temp_mult': 1.05, 'wind_mult': 0.6, 'precip_mult': 0.2},
            {'name': 'storm', 'temp_mult': 0.88, 'wind_mult': 1.8, 'precip_mult': 8.0},
            {'name': 'clear_perfect', 'temp_mult': 0.98, 'wind_mult': 0.9, 'precip_mult': 0.0}
        ]
        
        for scenario in weather_scenarios[:num_weather_scenarios]:
            df_weather = self.df_orig.copy()
            
            # Modify weather
            temp_base = df_weather['temperature_c'].mean()
            df_weather['temperature_c'] = temp_base + (df_weather['temperature_c'] - temp_base) * scenario['temp_mult']
            df_weather['temperature_c'] = df_weather['temperature_c'].clip(*self.weather_ranges['temperature_c'])
            
            df_weather['wind_speed_kmh'] *= scenario['wind_mult']
            df_weather['wind_speed_kmh'] = df_weather['wind_speed_kmh'].clip(0, self.weather_ranges['wind_speed_kmh'][1])
            
            df_weather['precipitation_mm'] *= scenario['precip_mult']
            df_weather['precipitation_mm'] = df_weather['precipitation_mm'].clip(0, 10.0)
            
            # Adjust speed based on weather impact
            # Rain reduces speed
            rain_factor = 1 - (df_weather['precipitation_mm'] / 10.0) * 0.15  # Max 15% reduction
            df_weather['speed_kmh'] *= rain_factor
            
            # High temp reduces speed slightly
            temp_deviation = (df_weather['temperature_c'] - 27.0) / 10.0  # 27°C baseline
            temp_factor = 1 - temp_deviation * 0.05  # Max 5% impact per 10°C
            df_weather['speed_kmh'] *= temp_factor
            
            # Strong wind reduces speed
            wind_factor = 1 - (df_weather['wind_speed_kmh'] - 5) / 50  # 5 km/h baseline
            wind_factor = wind_factor.clip(0.85, 1.0)
            df_weather['speed_kmh'] *= wind_factor
            
            # Clip to valid range
            df_weather['speed_kmh'] = df_weather['speed_kmh'].clip(3.0, 55.0)
            
            # New run IDs
            df_weather['run_id'] = f"aug_weather_{scenario['name']}_" + df_weather['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_weather)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} weather-augmented runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_multi_scenarios(self, num_scenarios=10):
        """
        Method 3: Extended pattern variations (10 instead of 5)
        """
        print(f"\nMethod 3: Multi-Scenario Variations (x{num_scenarios})")
        
        scenarios = [
            'rush_hour_heavy', 'rush_hour_light', 'weekend_traffic',
            'holiday_pattern', 'night_shift', 'early_morning',
            'midday_peak', 'accident_scenario', 'construction_zone',
            'special_event'
        ]
        
        augmented_dfs = []
        
        for idx, scenario_name in enumerate(scenarios[:num_scenarios]):
            df_var = self.df_orig.copy()
            df_var['hour'] = df_var['timestamp'].dt.hour
            
            if scenario_name == 'rush_hour_heavy':
                mask = df_var['hour'].isin([7, 8, 9, 17, 18, 19])
                df_var.loc[mask, 'speed_kmh'] *= 0.65
            elif scenario_name == 'rush_hour_light':
                mask = df_var['hour'].isin([7, 8, 9, 17, 18, 19])
                df_var.loc[mask, 'speed_kmh'] *= 0.85
            elif scenario_name == 'weekend_traffic':
                df_var['speed_kmh'] *= 1.15
            elif scenario_name == 'holiday_pattern':
                df_var['speed_kmh'] *= 1.25
            elif scenario_name == 'night_shift':
                mask = df_var['hour'].isin([0, 1, 2, 3, 4, 5])
                df_var.loc[mask, 'speed_kmh'] *= 1.20
            elif scenario_name == 'early_morning':
                mask = df_var['hour'].isin([5, 6, 7])
                df_var.loc[mask, 'speed_kmh'] *= 0.90
            elif scenario_name == 'midday_peak':
                mask = df_var['hour'].isin([11, 12, 13, 14])
                df_var.loc[mask, 'speed_kmh'] *= 0.80
            elif scenario_name == 'accident_scenario':
                # Random edges affected
                num_affected = len(df_var) // 20
                affected_idx = np.random.choice(len(df_var), num_affected, replace=False)
                df_var.loc[df_var.index[affected_idx], 'speed_kmh'] *= 0.50
            elif scenario_name == 'construction_zone':
                # Some edges permanently slower
                num_affected = len(df_var) // 15
                affected_idx = np.random.choice(len(df_var), num_affected, replace=False)
                df_var.loc[df_var.index[affected_idx], 'speed_kmh'] *= 0.70
            else:  # special_event
                mask = df_var['hour'].isin([18, 19, 20, 21])
                df_var.loc[mask, 'speed_kmh'] *= 0.60
            
            df_var['speed_kmh'] = df_var['speed_kmh'].clip(3.0, 55.0)
            df_var['run_id'] = f"aug_scenario_{scenario_name}_" + df_var['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_var)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} scenario runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()
    
    def augment_time_perturbation(self, num_perturbations=3, max_shift_minutes=30):
        """
        Method 4: Small time shifts to create variations
        """
        print(f"\nMethod 4: Time Perturbation (x{num_perturbations}, ±{max_shift_minutes}min)")
        
        augmented_dfs = []
        
        for pert_idx in range(num_perturbations):
            df_pert = self.df_orig.copy()
            
            # Random time shift
            shift_minutes = np.random.uniform(-max_shift_minutes, max_shift_minutes)
            df_pert['timestamp'] = df_pert['timestamp'] + timedelta(minutes=shift_minutes)
            
            # Adjust speed based on new hour
            df_pert['hour'] = df_pert['timestamp'].dt.hour
            
            # Apply hourly pattern adjustment
            for hour in df_pert['hour'].unique():
                if hour in self.global_hourly.index:
                    mask = df_pert['hour'] == hour
                    hour_mean = self.global_hourly.loc[hour, 'mean']
                    global_mean = self.global_hourly['mean'].mean()
                    
                    adjustment = hour_mean / global_mean
                    df_pert.loc[mask, 'speed_kmh'] *= adjustment
            
            df_pert['speed_kmh'] = df_pert['speed_kmh'].clip(3.0, 55.0)
            df_pert['run_id'] = f"aug_timeshift_{pert_idx}_{shift_minutes:.0f}min_" + df_pert['timestamp'].dt.strftime('%Y%m%d_%H%M%S')
            
            augmented_dfs.append(df_pert)
        
        if augmented_dfs:
            result = pd.concat(augmented_dfs, ignore_index=True)
            print(f"  Created {result['run_id'].nunique()} time-perturbed runs")
            print(f"  Total records: {len(result)}")
            return result
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("EXTREME DATA AUGMENTATION")
    print("=" * 70)
    
    # Load current augmented data
    df_current = pd.read_parquet('data/processed/all_runs_augmented.parquet')
    print(f"\nCurrent data: {len(df_current)} records, {df_current['run_id'].nunique()} runs")
    
    # Get original data only
    df_orig = df_current[~df_current['run_id'].str.contains('aug_', na=False)]
    print(f"Original data: {len(df_orig)} records, {df_orig['run_id'].nunique()} runs")
    
    # Create extreme augmentor
    augmentor = ExtremeAugmentor(df_orig)
    
    # Apply extreme augmentation methods
    augmented_parts = []
    
    # 1. Hourly interpolation (2 between each pair)
    df_interp = augmentor.augment_hourly_interpolation(num_interpolations=2)
    if len(df_interp) > 0:
        augmented_parts.append(df_interp)
    
    # 2. Synthetic weather (10 scenarios)
    df_weather = augmentor.augment_synthetic_weather(num_weather_scenarios=10)
    if len(df_weather) > 0:
        augmented_parts.append(df_weather)
    
    # 3. Multi-scenarios (10 variations)
    df_scenarios = augmentor.augment_multi_scenarios(num_scenarios=10)
    if len(df_scenarios) > 0:
        augmented_parts.append(df_scenarios)
    
    # 4. Time perturbation (3 shifts)
    df_timeshift = augmentor.augment_time_perturbation(num_perturbations=3, max_shift_minutes=30)
    if len(df_timeshift) > 0:
        augmented_parts.append(df_timeshift)
    
    # Combine all
    if augmented_parts:
        df_extreme_aug = pd.concat(augmented_parts, ignore_index=True)
        
        # Combine with existing augmented data
        df_final = pd.concat([df_current, df_extreme_aug], ignore_index=True)
        
        print("\n" + "=" * 70)
        print("EXTREME AUGMENTATION SUMMARY")
        print("=" * 70)
        print(f"Previous augmented runs: {df_current['run_id'].nunique()}")
        print(f"NEW augmented runs: {df_extreme_aug['run_id'].nunique()}")
        print(f"Total runs: {df_final['run_id'].nunique()}")
        print(f"Multiplication factor: {df_final['run_id'].nunique() / df_orig['run_id'].nunique():.1f}x")
        print(f"\nPrevious records: {len(df_current)}")
        print(f"NEW records: {len(df_extreme_aug)}")
        print(f"Total records: {len(df_final)}")
        
        # Save
        output_path = Path('data/processed/all_runs_extreme_augmented.parquet')
        df_final.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to {output_path}")
        
        # Estimate samples
        print("\n" + "=" * 70)
        print("EXPECTED TRAINING SAMPLES (EXTREME)")
        print("=" * 70)
        
        total_runs = df_final['run_id'].nunique()
        
        configs = [
            (12, 12, "Original (High quality)"),
            (6, 3, "Moderate (Balanced)"),
            (4, 2, "Conservative (More samples)")
        ]
        
        for seq_len, pred_len, desc in configs:
            window_size = seq_len + pred_len
            samples = max(0, total_runs - window_size + 1)
            print(f"{desc}:")
            print(f"  seq_len={seq_len}, pred_len={pred_len}")
            print(f"  Training samples: ~{samples}")
            print()
        
        print("=" * 70)
        print("READY FOR EXTREME TRAINING!")
        print("=" * 70)
        
    else:
        print("\n⚠ No new augmented data generated")


if __name__ == '__main__':
    main()
