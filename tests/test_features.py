"""
Unit tests for feature engineering modules.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from traffic_forecast.features.temporal_features import create_temporal_features
from traffic_forecast.features.spatial_features import calculate_spatial_features


class TestTemporalFeatures(unittest.TestCase):
    """Test temporal feature extraction."""
    
    def test_create_temporal_features(self):
        """Test temporal features are created correctly."""
        # Create sample data with timestamps
        timestamps = pd.date_range('2025-10-27 08:00', periods=10, freq='15min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'speed_kmh': np.random.uniform(20, 60, 10)
        })
        
        result = create_temporal_features(df)
        
        # Check temporal features exist
        self.assertIn('hour', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertIn('is_weekend', result.columns)
        self.assertIn('is_peak_hour', result.columns)
        
        # Check values are reasonable
        self.assertTrue(all(result['hour'] >= 0))
        self.assertTrue(all(result['hour'] <= 23))
        self.assertTrue(all(result['day_of_week'] >= 0))
        self.assertTrue(all(result['day_of_week'] <= 6))
    
    def test_peak_hour_detection(self):
        """Test peak hour detection."""
        # Morning peak: 7-9 AM
        morning_peak = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-10-27 08:00')]
        })
        result = create_temporal_features(morning_peak)
        self.assertTrue(result['is_peak_hour'].iloc[0])
        
        # Off-peak: 2 AM
        off_peak = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-10-27 02:00')]
        })
        result = create_temporal_features(off_peak)
        self.assertFalse(result['is_peak_hour'].iloc[0])
    
    def test_weekend_detection(self):
        """Test weekend detection."""
        # Saturday
        weekend = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-11-01 12:00')]  # Saturday
        })
        result = create_temporal_features(weekend)
        self.assertTrue(result['is_weekend'].iloc[0])
        
        # Monday
        weekday = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-10-27 12:00')]  # Monday
        })
        result = create_temporal_features(weekday)
        self.assertFalse(result['is_weekend'].iloc[0])


class TestSpatialFeatures(unittest.TestCase):
    """Test spatial feature calculation."""
    
    def test_calculate_spatial_features(self):
        """Test spatial features calculation."""
        # Create sample node data
        nodes = pd.DataFrame({
            'node_id': ['A', 'B', 'C'],
            'lat': [10.762622, 10.772622, 10.782622],
            'lon': [106.660172, 106.670172, 106.680172],
            'speed_kmh': [30, 40, 50]
        })
        
        result = calculate_spatial_features(nodes)
        
        # Should have neighbor features
        self.assertIn('neighbor_avg_speed', result.columns)
        self.assertIn('neighbor_count', result.columns)
        
        # Values should be reasonable
        self.assertTrue(all(result['neighbor_avg_speed'] > 0))
        self.assertTrue(all(result['neighbor_count'] >= 0))
    
    def test_distance_based_neighbors(self):
        """Test neighbor selection by distance."""
        nodes = pd.DataFrame({
            'node_id': ['A', 'B'],
            'lat': [10.762622, 10.762622],
            'lon': [106.660172, 106.670172],  # ~1km apart
            'speed_kmh': [30, 40]
        })
        
        result = calculate_spatial_features(nodes, radius_km=2.0)
        
        # Nodes should find each other as neighbors
        self.assertTrue(all(result['neighbor_count'] > 0))


class TestLagFeatures(unittest.TestCase):
    """Test lag feature creation."""
    
    def test_lag_feature_creation(self):
        """Test lag features are created correctly."""
        from traffic_forecast.features.lag_features import create_lag_features
        
        # Create time series data
        timestamps = pd.date_range('2025-10-27 08:00', periods=20, freq='15min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'node_id': ['A'] * 20,
            'speed_kmh': np.random.uniform(20, 60, 20)
        })
        
        result = create_lag_features(df, lag_periods=[1, 2, 4])
        
        # Check lag features exist
        self.assertIn('speed_lag_1', result.columns)
        self.assertIn('speed_lag_2', result.columns)
        self.assertIn('speed_lag_4', result.columns)
        
        # First few rows should have NaN for lags
        self.assertTrue(pd.isna(result['speed_lag_1'].iloc[0]))
        self.assertTrue(pd.notna(result['speed_lag_1'].iloc[-1]))


if __name__ == '__main__':
    unittest.main()
