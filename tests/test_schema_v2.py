"""
Test schema v2 validation.
"""

import json
import unittest
import os

class TestSchemaV2(unittest.TestCase):
    def test_schema_fields(self):
        with open('configs/nodes_schema_v2.json', 'r') as f:
            schema = json.load(f)
        
        required_fields = [f['name'] for f in schema['fields'] if f.get('required')]
        forecast_fields = [f['name'] for f in schema['fields'] if 'forecast_' in f['name']]
        
        self.assertIn('node_id', required_fields)
        self.assertEqual(len(forecast_fields), 12)  # 12 forecast fields
        
        # Check data
        if os.path.exists('data/features_nodes_v2.json'):
            with open('data/features_nodes_v2.json', 'r') as f:
                data = json.load(f)
            
            if data:
                sample = data[0]
                for field in forecast_fields:
                    self.assertIn(field, sample)
                    # Check type (basic)
                    if sample[field] is not None:
                        self.assertIsInstance(sample[field], (int, float))

if __name__ == '__main__':
    unittest.main()