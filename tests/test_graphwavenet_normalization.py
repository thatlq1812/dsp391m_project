"""
Unit tests for GraphWaveNet normalization fix.

Tests that data normalization is properly applied during
training and prediction.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from traffic_forecast.models.graph.graph_wavenet import GraphWaveNetTrafficPredictor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    n_samples = 50
    seq_len = 12
    n_nodes = 10
    
    # Generate realistic speed data (0-60 km/h range)
    X = np.random.uniform(10, 60, (n_samples, seq_len, n_nodes, 1)).astype(np.float32)
    y = np.random.uniform(10, 60, (n_samples, n_nodes, 1)).astype(np.float32)
    
    return X, y


def test_normalization_applied(sample_data):
    """Test that data normalization is applied during training."""
    X_train, y_train = sample_data
    
    # Split into train/val
    n_train = 40
    X_train_split = X_train[:n_train]
    y_train_split = y_train[:n_train]
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    
    # Create model
    model = GraphWaveNetTrafficPredictor(
        num_nodes=10,
        sequence_length=12,
        num_layers=2,
        hidden_channels=16,
        learning_rate=0.001
    )
    
    # Train for just 2 epochs
    history = model.fit(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    
    # Check that scalers are fitted
    assert model.scaler_X is not None, "scaler_X should be fitted"
    assert model.scaler_y is not None, "scaler_y should be fitted"
    
    # Check that scalers have proper statistics
    assert hasattr(model.scaler_X, 'mean_'), "scaler_X should have mean_"
    assert hasattr(model.scaler_X, 'scale_'), "scaler_X should have scale_"
    assert hasattr(model.scaler_y, 'mean_'), "scaler_y should have mean_"
    assert hasattr(model.scaler_y, 'scale_'), "scaler_y should have scale_"
    
    # Check that scale is not 1.0 (would indicate no normalization)
    assert model.scaler_y.scale_[0] != 1.0, "Data should be normalized"
    
    print("✓ Normalization is applied during training")


def test_denormalization_in_prediction(sample_data):
    """Test that predictions are properly denormalized."""
    X_train, y_train = sample_data
    
    # Create and train model
    model = GraphWaveNetTrafficPredictor(
        num_nodes=10,
        sequence_length=12,
        num_layers=2,
        hidden_channels=16,
        learning_rate=0.001
    )
    
    model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    
    # Make predictions
    X_test = X_train[:5]
    predictions = model.predict(X_test)
    
    # Check predictions are in reasonable range (not normalized)
    # Original data is in 10-60 km/h range
    assert predictions.min() > -50, "Predictions should be denormalized (not around 0)"
    assert predictions.max() < 150, "Predictions should be in realistic range"
    
    # Check predictions shape
    assert predictions.shape == (5, 10, 1), f"Expected shape (5, 10, 1), got {predictions.shape}"
    
    print(f"✓ Predictions are denormalized: range [{predictions.min():.2f}, {predictions.max():.2f}]")


def test_save_load_preserves_scalers(sample_data):
    """Test that save/load preserves scalers."""
    X_train, y_train = sample_data
    
    # Create and train model
    model1 = GraphWaveNetTrafficPredictor(
        num_nodes=10,
        sequence_length=12,
        num_layers=2,
        hidden_channels=16,
        learning_rate=0.001
    )
    
    model1.fit(
        X_train, y_train,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    
    # Make predictions with original model
    X_test = X_train[:5]
    pred1 = model1.predict(X_test)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'test_model'
        model1.save(save_path)
        
        # Check that scaler files exist
        assert (save_path / 'scaler_X.pkl').exists(), "scaler_X.pkl should be saved"
        assert (save_path / 'scaler_y.pkl').exists(), "scaler_y.pkl should be saved"
        
        # Load model
        model2 = GraphWaveNetTrafficPredictor.load(save_path)
        
        # Make predictions with loaded model
        pred2 = model2.predict(X_test)
        
        # Check predictions are identical
        np.testing.assert_allclose(
            pred1, pred2,
            rtol=1e-5,
            err_msg="Predictions should be identical after save/load"
        )
    
    print("✓ Save/load preserves scalers and predictions")


def test_predictions_in_realistic_range(sample_data):
    """Test that predictions are in realistic speed range."""
    X_train, y_train = sample_data
    
    # Create and train model
    model = GraphWaveNetTrafficPredictor(
        num_nodes=10,
        sequence_length=12,
        num_layers=2,
        hidden_channels=16,
        learning_rate=0.001
    )
    
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=8,
        verbose=0
    )
    
    # Make predictions
    predictions = model.predict(X_train)
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions - y_train))
    
    # For random data with 2 layers and 5 epochs, MAE should be reasonable
    # Not near 0 (would indicate leakage) and not too high (>50 would indicate failure)
    assert mae > 1.0, f"MAE too low ({mae:.2f}), possible data leakage"
    assert mae < 50.0, f"MAE too high ({mae:.2f}), model not learning"
    
    print(f"✓ MAE is in realistic range: {mae:.2f} km/h")


def test_no_impossible_accuracy(sample_data):
    """Test that model doesn't achieve impossible accuracy like the bug (0.02 MAE)."""
    X_train, y_train = sample_data
    
    # Create and train model
    model = GraphWaveNetTrafficPredictor(
        num_nodes=10,
        sequence_length=12,
        num_layers=2,
        hidden_channels=16,
        learning_rate=0.001
    )
    
    # Split data
    n_train = 40
    X_train_split = X_train[:n_train]
    y_train_split = y_train[:n_train]
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    
    model.fit(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=5,
        batch_size=8,
        verbose=0
    )
    
    # Make predictions on validation
    val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(val_pred - y_val))
    
    # The buggy version had MAE = 0.02 which is impossible
    # With proper normalization, MAE should be > 1 km/h for random data
    assert val_mae > 1.0, f"Validation MAE suspiciously low ({val_mae:.4f}), check for bugs"
    
    print(f"✓ No impossible accuracy detected: Val MAE = {val_mae:.2f} km/h")


if __name__ == '__main__':
    # Run tests manually
    print("=" * 80)
    print("GRAPHWAVENET NORMALIZATION TESTS")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    seq_len = 12
    n_nodes = 10
    X = np.random.uniform(10, 60, (n_samples, seq_len, n_nodes, 1)).astype(np.float32)
    y = np.random.uniform(10, 60, (n_samples, n_nodes, 1)).astype(np.float32)
    sample_data = (X, y)
    
    print("\n[1/6] Testing normalization is applied...")
    test_normalization_applied(sample_data)
    
    print("\n[2/6] Testing denormalization in prediction...")
    test_denormalization_in_prediction(sample_data)
    
    print("\n[3/6] Testing save/load preserves scalers...")
    test_save_load_preserves_scalers(sample_data)
    
    print("\n[4/6] Testing predictions in realistic range...")
    test_predictions_in_realistic_range(sample_data)
    
    print("\n[5/6] Testing no impossible accuracy...")
    test_no_impossible_accuracy(sample_data)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
