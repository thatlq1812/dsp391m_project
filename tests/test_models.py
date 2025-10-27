"""
Unit tests for model training and prediction.
"""

import unittest
import numpy as np
import pandas as pd
from traffic_forecast.models.baseline import predict_speed_persistence
from traffic_forecast.models.ensemble import TrafficEnsemble


class TestBaselineModel(unittest.TestCase):
    """Test baseline persistence model."""
    
    def test_persistence_prediction(self):
        """Test persistence model predicts last value."""
        historical = [30, 35, 40, 45, 50]
        horizons = [5, 15, 30]
        
        predictions = predict_speed_persistence(historical, horizons)
        
        # All predictions should be the last value
        self.assertEqual(len(predictions), len(horizons))
        for pred in predictions:
            self.assertEqual(pred, 50)
    
    def test_empty_historical(self):
        """Test handling of empty historical data."""
        historical = []
        horizons = [5, 15, 30]
        
        predictions = predict_speed_persistence(historical, horizons)
        
        # Should return default or None
        self.assertIsNotNone(predictions)
    
    def test_single_value(self):
        """Test with single historical value."""
        historical = [42]
        horizons = [5, 15, 30]
        
        predictions = predict_speed_persistence(historical, horizons)
        
        for pred in predictions:
            self.assertEqual(pred, 42)


class TestEnsembleModel(unittest.TestCase):
    """Test ensemble model."""
    
    def setUp(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_peak_hour': np.random.randint(0, 2, n_samples),
            'speed_lag_1': np.random.uniform(20, 60, n_samples),
            'speed_lag_2': np.random.uniform(20, 60, n_samples),
            'neighbor_avg_speed': np.random.uniform(20, 60, n_samples)
        })
        
        # Target is roughly correlated with lag features
        self.y_train = (
            0.5 * self.X_train['speed_lag_1'] + 
            0.3 * self.X_train['speed_lag_2'] + 
            np.random.normal(0, 5, n_samples)
        )
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = TrafficEnsemble()
        self.assertIsNotNone(model)
    
    def test_model_training(self):
        """Test model can be trained."""
        model = TrafficEnsemble()
        model.fit(self.X_train, self.y_train)
        
        # Model should be fitted
        self.assertTrue(hasattr(model, 'models') or hasattr(model, 'model'))
    
    def test_model_prediction(self):
        """Test model can make predictions."""
        model = TrafficEnsemble()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions on training data
        predictions = model.predict(self.X_train)
        
        self.assertEqual(len(predictions), len(self.X_train))
        # Predictions should be reasonable
        self.assertTrue(all(predictions > 0))
        self.assertTrue(all(predictions < 100))
    
    def test_model_feature_importance(self):
        """Test feature importance extraction."""
        model = TrafficEnsemble()
        model.fit(self.X_train, self.y_train)
        
        # Should be able to get feature importance
        if hasattr(model, 'feature_importance'):
            importance = model.feature_importance()
            self.assertIsNotNone(importance)
            self.assertEqual(len(importance), len(self.X_train.columns))


class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""
    
    def test_model_registration(self):
        """Test registering and retrieving models."""
        from traffic_forecast.models.registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Register a model
        model = TrafficEnsemble()
        registry.register('test_model', model)
        
        # Retrieve the model
        retrieved = registry.get('test_model')
        self.assertIsNotNone(retrieved)
    
    def test_list_models(self):
        """Test listing registered models."""
        from traffic_forecast.models.registry import ModelRegistry
        
        registry = ModelRegistry()
        registry.register('model1', TrafficEnsemble())
        registry.register('model2', TrafficEnsemble())
        
        models = registry.list_models()
        self.assertGreaterEqual(len(models), 2)


if __name__ == '__main__':
    unittest.main()
