"""
Unit tests for Pydantic validation schemas
"""

import unittest
from datetime import datetime
from pydantic import ValidationError
from traffic_forecast.validation.schemas import (
 TrafficNode,
 TrafficEdge,
 WeatherSnapshot,
 TrafficSnapshot,
 NodeFeatures,
 validate_nodes,
 validate_edges,
 generate_quality_report
)


class TestValidationSchemas(unittest.TestCase):
 """Test cases for Pydantic validation schemas."""
 
 def test_traffic_node_valid(self):
 """Test valid traffic node."""
 node_data = {
 'node_id': 'node-10.772-106.698',
 'lat': 10.772,
 'lon': 106.698,
 'degree': 4,
 'importance_score': 25.5,
 'road_type': 'motorway',
 'connected_road_types': ['motorway', 'primary'],
 'is_major_intersection': True
 }
 
 node = TrafficNode(**node_data)
 self.assertEqual(node.node_id, 'node-10.772-106.698')
 self.assertEqual(node.degree, 4)
 
 def test_traffic_node_invalid_lat(self):
 """Test traffic node with invalid latitude."""
 node_data = {
 'node_id': 'node-100.000-106.698',
 'lat': 100.0, # Invalid: > 90
 'lon': 106.698,
 'degree': 4,
 'importance_score': 25.5,
 'road_type': 'motorway',
 'connected_road_types': ['motorway'],
 }
 
 with self.assertRaises(ValidationError):
 TrafficNode(**node_data)
 
 def test_traffic_node_invalid_id(self):
 """Test traffic node with invalid node_id format."""
 node_data = {
 'node_id': 'invalid-id', # Should start with 'node-'
 'lat': 10.772,
 'lon': 106.698,
 'degree': 4,
 'importance_score': 25.5,
 'road_type': 'motorway',
 'connected_road_types': ['motorway'],
 }
 
 with self.assertRaises(ValidationError):
 TrafficNode(**node_data)
 
 def test_traffic_edge_valid(self):
 """Test valid traffic edge."""
 edge_data = {
 'u': 'node-10.772-106.698',
 'v': 'node-10.773-106.699',
 'distance_m': 150.5,
 'way_id': 12345,
 'road_type': 'primary',
 'lanes': '4',
 'maxspeed': '60',
 'name': 'Main Street'
 }
 
 edge = TrafficEdge(**edge_data)
 self.assertEqual(edge.distance_m, 150.5)
 
 def test_traffic_edge_self_loop(self):
 """Test that self-loops are rejected."""
 edge_data = {
 'u': 'node-10.772-106.698',
 'v': 'node-10.772-106.698', # Same as u
 'distance_m': 150.5,
 'way_id': 12345,
 'road_type': 'primary'
 }
 
 with self.assertRaises(ValidationError):
 TrafficEdge(**edge_data)
 
 def test_weather_snapshot_valid(self):
 """Test valid weather snapshot."""
 weather_data = {
 'timestamp': datetime.now(),
 'temperature_c': 28.5,
 'precipitation_mm': 0.0,
 'wind_speed_kmh': 12.3,
 'forecast_temp_t5_c': 29.0,
 'forecast_temp_t15_c': 29.5
 }
 
 weather = WeatherSnapshot(**weather_data)
 self.assertEqual(weather.temperature_c, 28.5)
 
 def test_weather_snapshot_invalid_temp(self):
 """Test weather snapshot with invalid temperature."""
 weather_data = {
 'timestamp': datetime.now(),
 'temperature_c': 100.0, # Too high
 'precipitation_mm': 0.0,
 'wind_speed_kmh': 12.3
 }
 
 with self.assertRaises(ValidationError):
 WeatherSnapshot(**weather_data)
 
 def test_traffic_snapshot_valid(self):
 """Test valid traffic snapshot."""
 snapshot_data = {
 'node_id': 'node-10.772-106.698',
 'timestamp': datetime.now(),
 'avg_speed_kmh': 45.5,
 'sample_count': 10,
 'congestion_level': 2,
 'reliability': 0.95
 }
 
 snapshot = TrafficSnapshot(**snapshot_data)
 self.assertEqual(snapshot.avg_speed_kmh, 45.5)
 
 def test_traffic_snapshot_invalid_speed(self):
 """Test traffic snapshot with invalid speed."""
 snapshot_data = {
 'node_id': 'node-10.772-106.698',
 'timestamp': datetime.now(),
 'avg_speed_kmh': -10.0, # Negative speed
 'sample_count': 10
 }
 
 with self.assertRaises(ValidationError):
 TrafficSnapshot(**snapshot_data)
 
 def test_validate_nodes_function(self):
 """Test validate_nodes utility function."""
 nodes = [
 {
 'node_id': 'node-10.772-106.698',
 'lat': 10.772,
 'lon': 106.698,
 'degree': 4,
 'importance_score': 25.5,
 'road_type': 'motorway',
 'connected_road_types': ['motorway'],
 },
 {
 'node_id': 'invalid', # Invalid node
 'lat': 10.773,
 'lon': 106.699,
 'degree': 3,
 'importance_score': 20.0,
 'road_type': 'primary',
 'connected_road_types': ['primary'],
 }
 ]
 
 valid_nodes, errors = validate_nodes(nodes)
 
 self.assertEqual(len(valid_nodes), 1) # Only first node is valid
 self.assertEqual(len(errors), 1) # One error
 self.assertIn('node_id', errors[0])
 
 def test_validate_edges_function(self):
 """Test validate_edges utility function."""
 edges = [
 {
 'u': 'node-10.772-106.698',
 'v': 'node-10.773-106.699',
 'distance_m': 150.5,
 'way_id': 12345,
 'road_type': 'primary'
 },
 {
 'u': 'node-10.772-106.698',
 'v': 'node-10.772-106.698', # Self-loop (invalid)
 'distance_m': 0.0,
 'way_id': 67890,
 'road_type': 'secondary'
 }
 ]
 
 valid_edges, errors = validate_edges(edges)
 
 self.assertEqual(len(valid_edges), 1)
 self.assertEqual(len(errors), 1)
 
 def test_generate_quality_report(self):
 """Test quality report generation."""
 report = generate_quality_report(
 dataset_name='test_nodes',
 total_records=100,
 valid_records=95,
 validation_errors=['Error 1', 'Error 2']
 )
 
 self.assertEqual(report.dataset_name, 'test_nodes')
 self.assertEqual(report.total_records, 100)
 self.assertEqual(report.valid_records, 95)
 self.assertEqual(report.invalid_records, 5)
 self.assertEqual(report.validity_pct, 95.0)
 self.assertEqual(len(report.validation_errors), 2)
 
 def test_node_features_schema(self):
 """Test node features schema for ML."""
 features_data = {
 'node_id': 'node-10.772-106.698',
 'timestamp': datetime.now(),
 'avg_speed_kmh': 45.5,
 'temperature_c': 28.5,
 'rain_mm': 0.0,
 'wind_speed_kmh': 12.3,
 'forecast_temp_t5_c': 29.0
 }
 
 features = NodeFeatures(**features_data)
 self.assertEqual(features.avg_speed_kmh, 45.5)
 
 def test_node_features_extra_fields(self):
 """Test that NodeFeatures allows extra fields."""
 features_data = {
 'node_id': 'node-10.772-106.698',
 'timestamp': datetime.now(),
 'avg_speed_kmh': 45.5,
 'temperature_c': 28.5,
 'rain_mm': 0.0,
 'wind_speed_kmh': 12.3,
 'extra_field': 'allowed' # Extra field should be allowed
 }
 
 features = NodeFeatures(**features_data)
 self.assertTrue(hasattr(features, 'extra_field'))


if __name__ == '__main__':
 unittest.main()
