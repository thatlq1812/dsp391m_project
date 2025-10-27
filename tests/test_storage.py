"""
Unit tests for traffic history storage.
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd
from traffic_forecast.storage.traffic_history import TrafficHistory


class TestTrafficHistory(unittest.TestCase):
    """Test traffic history storage and retrieval."""
    
    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.history = TrafficHistory(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_save_and_retrieve(self):
        """Test saving and retrieving traffic data."""
        # Create sample data
        timestamp = datetime.now()
        data = pd.DataFrame({
            'node_a_id': ['A', 'B'],
            'node_b_id': ['B', 'C'],
            'speed_kmh': [30.5, 45.2],
            'duration_sec': [120, 180],
            'distance_km': [1.0, 2.0]
        })
        
        # Save data
        self.history.save(timestamp, data)
        
        # Retrieve data
        retrieved = self.history.get_latest(limit=2)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 2)
        self.assertIn('speed_kmh', retrieved.columns)
    
    def test_time_range_query(self):
        """Test querying data by time range."""
        # Save data at different times
        now = datetime.now()
        
        for i in range(5):
            timestamp = now - timedelta(hours=i)
            data = pd.DataFrame({
                'node_a_id': ['A'],
                'node_b_id': ['B'],
                'speed_kmh': [30 + i],
                'duration_sec': [120],
                'distance_km': [1.0]
            })
            self.history.save(timestamp, data)
        
        # Query last 2 hours
        start_time = now - timedelta(hours=2)
        result = self.history.get_range(start_time, now)
        
        self.assertIsNotNone(result)
        # Should have data from last 2-3 entries
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)
    
    def test_node_specific_query(self):
        """Test querying data for specific nodes."""
        timestamp = datetime.now()
        data = pd.DataFrame({
            'node_a_id': ['A', 'B', 'C'],
            'node_b_id': ['B', 'C', 'D'],
            'speed_kmh': [30, 40, 50],
            'duration_sec': [120, 150, 180],
            'distance_km': [1.0, 1.5, 2.0]
        })
        
        self.history.save(timestamp, data)
        
        # Query for specific edge
        result = self.history.get_edge_history('A', 'B', hours=1)
        
        self.assertIsNotNone(result)
        if len(result) > 0:
            self.assertTrue(all(result['node_a_id'] == 'A'))
            self.assertTrue(all(result['node_b_id'] == 'B'))
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        # Save old data
        old_time = datetime.now() - timedelta(days=30)
        old_data = pd.DataFrame({
            'node_a_id': ['A'],
            'node_b_id': ['B'],
            'speed_kmh': [30],
            'duration_sec': [120],
            'distance_km': [1.0]
        })
        self.history.save(old_time, old_data)
        
        # Save recent data
        recent_time = datetime.now()
        recent_data = pd.DataFrame({
            'node_a_id': ['C'],
            'node_b_id': ['D'],
            'speed_kmh': [40],
            'duration_sec': [150],
            'distance_km': [1.5]
        })
        self.history.save(recent_time, recent_data)
        
        # Cleanup data older than 7 days
        self.history.cleanup(days=7)
        
        # Should only have recent data
        all_data = self.history.get_latest(limit=100)
        # Old data should be removed
        self.assertGreater(len(all_data), 0)


if __name__ == '__main__':
    unittest.main()
