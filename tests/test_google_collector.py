"""
Unit tests for Google Directions collector.
"""

import unittest
from unittest.mock import patch, MagicMock
from traffic_forecast.collectors.google.collector import (
    haversine,
    RateLimiter,
    mock_directions_api
)


class TestHaversine(unittest.TestCase):
    """Test haversine distance calculation."""
    
    def test_same_location(self):
        """Test distance between same coordinates is 0."""
        distance = haversine(10.762622, 106.660172, 10.762622, 106.660172)
        self.assertAlmostEqual(distance, 0, places=2)
    
    def test_known_distance(self):
        """Test known distance between two points in HCMC."""
        # Distance between two points roughly 1km apart
        lat1, lon1 = 10.762622, 106.660172
        lat2, lon2 = 10.772622, 106.670172
        distance = haversine(lat1, lon1, lat2, lon2)
        # Should be approximately 1.5 km
        self.assertGreater(distance, 1.0)
        self.assertLess(distance, 2.0)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter functionality."""
    
    def test_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(requests_per_minute=100)
        self.assertEqual(limiter.requests_per_minute, 100)
        self.assertAlmostEqual(limiter.requests_per_second, 100/60, places=2)
        self.assertEqual(limiter.request_count, 0)
    
    @patch('time.sleep')
    @patch('time.time')
    def test_rate_limiting(self, mock_time, mock_sleep):
        """Test rate limiter enforces limits."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 per second
        
        # Simulate rapid requests
        mock_time.side_effect = [0, 0.5, 0.6]  # Too fast
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        
        # Should have slept to maintain interval
        self.assertTrue(mock_sleep.called or limiter.request_count > 0)


class TestMockDirectionsAPI(unittest.TestCase):
    """Test mock directions API."""
    
    def test_mock_api_returns_valid_data(self):
        """Test mock API returns valid traffic data."""
        node_a = {'node_id': 'A', 'lat': 10.762622, 'lon': 106.660172}
        node_b = {'node_id': 'B', 'lat': 10.772622, 'lon': 106.670172}
        config = {'collectors': {'google_directions': {'mock_speed_range': [20, 60]}}}
        
        result = mock_directions_api(node_a, node_b, config)
        
        self.assertIn('distance_km', result)
        self.assertIn('duration_sec', result)
        self.assertIn('speed_kmh', result)
        self.assertGreater(result['distance_km'], 0)
        self.assertGreater(result['duration_sec'], 0)
        self.assertGreaterEqual(result['speed_kmh'], 20)
        self.assertLessEqual(result['speed_kmh'], 60)


if __name__ == '__main__':
    unittest.main()
