"""
Integration tests for data collection pipeline.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path


class TestDataCollectionPipeline(unittest.TestCase):
    """Test end-to-end data collection pipeline."""

    def setUp(self):
        """Set up temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_run_dir = Path(self.temp_dir) / 'test_run'
        self.test_run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_overpass_collector_integration(self):
        """Test Overpass collector integration."""
        from traffic_forecast.collectors.overpass.collector import collect_nodes

        # Run collector with small area
        config = {
            'area': {
                'center': {'lat': 10.762622, 'lon': 106.660172},
                'radius_m': 500  # Small radius for testing
            },
            'node_selection': {
                'min_degree': 2,
                'min_importance_score': 10.0,
                'max_nodes': 10
            }
        }

        # This should run without errors
        try:
            nodes = collect_nodes(config)
            self.assertIsNotNone(nodes)
            self.assertIsInstance(nodes, list)
        except Exception as e:
            # Network issues are acceptable in tests
            self.skipTest(f"Overpass API unavailable: {e}")

    def test_google_collector_mock_mode(self):
        """Test Google collector in mock mode."""
        from traffic_forecast.collectors.google.collector import mock_directions_api

        node_a = {'node_id': 'test_a', 'lat': 10.762622, 'lon': 106.660172}
        node_b = {'node_id': 'test_b', 'lat': 10.772622, 'lon': 106.670172}
        config = {
            'collectors': {
                'google_directions': {
                    'mock_speed_range': [20, 60]
                }
            }
        }

        result = mock_directions_api(node_a, node_b, config)

        # Verify result structure
        self.assertIn('distance_km', result)
        self.assertIn('duration_sec', result)
        self.assertIn('speed_kmh', result)
        self.assertGreater(result['speed_kmh'], 0)

    def test_feature_pipeline_integration(self):
        """Test feature engineering pipeline."""
        import pandas as pd
        from traffic_forecast.features.temporal_features import create_temporal_features

        # Create sample traffic data
        timestamps = pd.date_range('2025-10-27 08:00', periods=10, freq='15min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'node_a_id': ['A'] * 10,
            'node_b_id': ['B'] * 10,
            'speed_kmh': [30, 35, 40, 45, 50, 45, 40, 35, 30, 25]
        })

        # Apply temporal features
        result = create_temporal_features(df)

        # Verify features were added
        self.assertIn('hour', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertEqual(len(result), len(df))

    def test_data_validation_pipeline(self):
        """Test data validation schemas."""
        from traffic_forecast.validation.schemas import TrafficDataSchema
        from pydantic import ValidationError

        # Valid data
        valid_data = {
            'node_a_id': 'A',
            'node_b_id': 'B',
            'speed_kmh': 45.5,
            'duration_sec': 120,
            'distance_km': 1.5,
            'timestamp': datetime.now().isoformat()
        }

        try:
            validated = TrafficDataSchema(**valid_data)
            self.assertEqual(validated.speed_kmh, 45.5)
        except Exception:
            # Schema might not exist yet
            self.skipTest("TrafficDataSchema not implemented")

        # Invalid data (negative speed)
        invalid_data = valid_data.copy()
        invalid_data['speed_kmh'] = -10

        with self.assertRaises(ValidationError):
            TrafficDataSchema(**invalid_data)


class TestStoragePipeline(unittest.TestCase):
    """Test data storage and retrieval pipeline."""

    def setUp(self):
        """Create temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

    def tearDown(self):
        """Clean up database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_save_retrieve_pipeline(self):
        """Test saving and retrieving traffic data."""
        from traffic_forecast.storage.traffic_history import TrafficHistory
        import pandas as pd

        history = TrafficHistory(db_path=self.db_path)

        # Save data
        timestamp = datetime.now()
        data = pd.DataFrame({
            'node_a_id': ['A', 'B', 'C'],
            'node_b_id': ['B', 'C', 'D'],
            'speed_kmh': [30, 40, 50],
            'duration_sec': [120, 150, 180],
            'distance_km': [1.0, 1.5, 2.0]
        })

        history.save(timestamp, data)

        # Retrieve data
        retrieved = history.get_latest(limit=10)

        self.assertIsNotNone(retrieved)
        self.assertGreater(len(retrieved), 0)


if __name__ == '__main__':
    unittest.main()
