"""
Unit tests for NodeSelector - Major Intersection Selection Algorithm
"""

import unittest
from traffic_forecast.collectors.overpass.node_selector import NodeSelector


class TestNodeSelector(unittest.TestCase):
 """Test cases for NodeSelector class."""
 
 def setUp(self):
 """Set up test fixtures."""
 self.selector = NodeSelector(min_degree=3, min_importance_score=15.0)
 
 # Sample OSM data
 self.sample_osm_data = {
 'elements': [
 {
 'type': 'way',
 'id': 1,
 'tags': {'highway': 'motorway', 'name': 'Highway 1'},
 'geometry': [
 {'lat': 10.772, 'lon': 106.698},
 {'lat': 10.773, 'lon': 106.699},
 {'lat': 10.774, 'lon': 106.700}
 ]
 },
 {
 'type': 'way',
 'id': 2,
 'tags': {'highway': 'primary', 'name': 'Road 2'},
 'geometry': [
 {'lat': 10.774, 'lon': 106.700},
 {'lat': 10.775, 'lon': 106.701}
 ]
 },
 {
 'type': 'way',
 'id': 3,
 'tags': {'highway': 'secondary', 'name': 'Road 3'},
 'geometry': [
 {'lat': 10.774, 'lon': 106.700},
 {'lat': 10.773, 'lon': 106.701}
 ]
 },
 {
 'type': 'way',
 'id': 4,
 'tags': {'highway': 'tertiary', 'name': 'Road 4'},
 'geometry': [
 {'lat': 10.774, 'lon': 106.700},
 {'lat': 10.775, 'lon': 106.699}
 ]
 }
 ]
 }
 
 def test_road_weights(self):
 """Test road importance weights."""
 self.assertEqual(NodeSelector.ROAD_WEIGHTS['motorway'], 10)
 self.assertEqual(NodeSelector.ROAD_WEIGHTS['trunk'], 9)
 self.assertEqual(NodeSelector.ROAD_WEIGHTS['primary'], 8)
 self.assertEqual(NodeSelector.ROAD_WEIGHTS['secondary'], 7)
 
 def test_extract_major_intersections(self):
 """Test major intersection extraction."""
 nodes, edges = self.selector.extract_major_intersections(self.sample_osm_data)
 
 # Should find at least one major intersection
 self.assertGreater(len(nodes), 0)
 
 # All nodes should have required fields
 for node in nodes:
 self.assertIn('node_id', node)
 self.assertIn('lat', node)
 self.assertIn('lon', node)
 self.assertIn('degree', node)
 self.assertIn('importance_score', node)
 self.assertIn('road_type', node)
 
 def test_intersection_degree(self):
 """Test that intersections have minimum degree."""
 nodes, _ = self.selector.extract_major_intersections(self.sample_osm_data)
 
 for node in nodes:
 self.assertGreaterEqual(node['degree'], self.selector.min_degree)
 
 def test_importance_score_calculation(self):
 """Test importance score calculation."""
 # Motorway + Primary connection
 connections = [
 ('way1', 'motorway'),
 ('way2', 'primary')
 ]
 score = self.selector._calculate_importance_score(connections)
 
 # Score = motorway(10) + primary(8) + diversity_bonus(2*2) = 22
 expected_score = 10 + 8 + (2 * 2)
 self.assertEqual(score, expected_score)
 
 def test_get_primary_road_type(self):
 """Test primary road type selection."""
 road_types = ['motorway', 'primary', 'secondary']
 primary = self.selector._get_primary_road_type(road_types)
 self.assertEqual(primary, 'motorway') # Highest weight
 
 road_types = ['tertiary', 'residential']
 primary = self.selector._get_primary_road_type(road_types)
 self.assertEqual(primary, 'tertiary') # Higher than residential
 
 def test_haversine_distance(self):
 """Test haversine distance calculation."""
 # Distance between two points ~1km apart
 lat1, lon1 = 10.772, 106.698
 lat2, lon2 = 10.782, 106.698
 
 distance = NodeSelector._haversine_km(lat1, lon1, lat2, lon2)
 
 # Should be approximately 1.1 km
 self.assertGreater(distance, 1.0)
 self.assertLess(distance, 1.5)
 
 def test_statistics_generation(self):
 """Test statistics generation."""
 nodes, edges = self.selector.extract_major_intersections(self.sample_osm_data)
 stats = self.selector.get_statistics(nodes, edges)
 
 # Check required stat fields
 self.assertIn('total_nodes', stats)
 self.assertIn('total_edges', stats)
 self.assertIn('avg_degree', stats)
 self.assertIn('avg_importance', stats)
 self.assertIn('road_type_distribution', stats)
 
 # Values should be non-negative
 self.assertGreaterEqual(stats['total_nodes'], 0)
 self.assertGreaterEqual(stats['total_edges'], 0)
 
 def test_empty_data(self):
 """Test handling of empty OSM data."""
 empty_data = {'elements': []}
 nodes, edges = self.selector.extract_major_intersections(empty_data)
 
 self.assertEqual(len(nodes), 0)
 self.assertEqual(len(edges), 0)
 
 def test_min_importance_threshold(self):
 """Test minimum importance score filtering."""
 # Use high threshold
 strict_selector = NodeSelector(min_degree=3, min_importance_score=50.0)
 nodes, _ = strict_selector.extract_major_intersections(self.sample_osm_data)
 
 # All nodes should meet threshold
 for node in nodes:
 self.assertGreaterEqual(node['importance_score'], 50.0)


if __name__ == '__main__':
 unittest.main()
