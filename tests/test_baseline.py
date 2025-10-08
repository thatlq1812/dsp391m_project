"""
Unit tests for the platform.
"""

import unittest
from models.baseline import predict_speed_persistence

class TestBaseline(unittest.TestCase):
    def test_persistence(self):
        historical = [30, 35, 40]
        predictions = predict_speed_persistence(historical, [5, 15, 30])
        self.assertEqual(predictions, [40, 40, 40])

if __name__ == '__main__':
    unittest.main()