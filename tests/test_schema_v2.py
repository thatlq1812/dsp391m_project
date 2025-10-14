"""
Test schema v2 validation.
"""

import json
import unittest
from traffic_forecast import PROJECT_ROOT

class TestSchemaV2(unittest.TestCase):
    def test_schema_fields(self):
        schema_path = PROJECT_ROOT / 'configs' / 'nodes_schema_v2.json'
        with schema_path.open('r', encoding='utf-8') as f:
            schema = json.load(f)
        
        required_fields = [f['name'] for f in schema['fields'] if f.get('required')]
        forecast_fields = [f['name'] for f in schema['fields'] if 'forecast_' in f['name']]
        
        self.assertIn('node_id', required_fields)
        self.assertEqual(len(forecast_fields), 12)  # 12 forecast fields
        
        # Check data
        features_path = PROJECT_ROOT / 'data' / 'features_nodes_v2.json'
        if features_path.exists():
            with features_path.open('r', encoding='utf-8') as f:
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
