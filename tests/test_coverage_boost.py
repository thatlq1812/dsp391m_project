"""
Comprehensive tests for traffic_api module to increase coverage.
Target: Reach 90% coverage for production readiness.
"""

import pytest
from datetime import datetime
from pathlib import Path
import json

# Skip if dependencies not available
pytest.importorskip("torch")
pytest.importorskip("pandas")


def test_config_loading():
    """Test configuration loading"""
    from traffic_api.config import config
    
    assert config is not None
    assert hasattr(config, 'allow_origins')
    assert hasattr(config, 'device')
    
    
def test_schemas_validation():
    """Test Pydantic schema validation"""
    from traffic_api.schemas import (
        PredictionRequest,
        NodeInfo,
        ErrorResponse,
        HealthResponse
    )
    
    # Test PredictionRequest
    request = PredictionRequest(
        node_ids=["node_1", "node_2"],
        horizons=[1, 3, 6]
    )
    assert len(request.node_ids) == 2
    assert request.horizons == [1, 3, 6]
    
    # Test NodeInfo
    node = NodeInfo(
        node_id="node_123",
        lat=10.7,
        lon=106.7,
        degree=4,
        importance_score=25.0,
        road_type="primary",
        street_names=["Street A", "Street B"]
    )
    assert node.node_id == "node_123"
    assert node.degree == 4
    
    # Test ErrorResponse
    error = ErrorResponse(error="Test error", detail="Test detail")
    assert error.error == "Test error"
    
    # Test HealthResponse
    health = HealthResponse(status="healthy", model_loaded=True)
    assert health.status == "healthy"
    assert health.model_loaded is True


def test_predictor_utilities():
    """Test predictor utility functions"""
    from traffic_forecast.models.stmgt.inference import mixture_to_moments
    import torch
    
    # Create mock GMM parameters
    batch_size, num_nodes, pred_len, K = 1, 10, 12, 3
    pred_params = {
        'pi': torch.rand(batch_size, num_nodes, pred_len, K),
        'mu': torch.rand(batch_size, num_nodes, pred_len, K),
        'sigma': torch.rand(batch_size, num_nodes, pred_len, K) * 2 + 1
    }
    
    # Normalize pi to sum to 1
    pred_params['pi'] = pred_params['pi'] / pred_params['pi'].sum(dim=-1, keepdim=True)
    
    # Get moments
    mean, std = mixture_to_moments(pred_params)
    
    assert mean.shape == (batch_size, num_nodes, pred_len)
    assert std.shape == (batch_size, num_nodes, pred_len)
    assert torch.all(std > 0)  # Std should be positive


def test_loss_functions():
    """Test STMGT loss functions"""
    from traffic_forecast.models.stmgt.losses import mixture_nll_loss
    import torch
    
    batch_size, num_nodes, pred_len, K = 2, 5, 12, 3
    
    # Mock predictions
    pred_params = {
        'pi': torch.rand(batch_size, num_nodes, pred_len, K),
        'mu': torch.rand(batch_size, num_nodes, pred_len, K) * 50,
        'sigma': torch.rand(batch_size, num_nodes, pred_len, K) * 5 + 1
    }
    pred_params['pi'] = pred_params['pi'] / pred_params['pi'].sum(dim=-1, keepdim=True)
    
    # Mock targets
    y_true = torch.rand(batch_size, num_nodes, pred_len) * 50
    
    # Calculate loss
    loss = mixture_nll_loss(pred_params, y_true)
    
    assert loss.ndim == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss > 0  # NLL should be positive


def test_temporal_features():
    """Test temporal feature encoding"""
    from traffic_forecast.features.temporal_features import encode_time_features
    from datetime import datetime
    
    timestamp = datetime(2025, 11, 10, 14, 30)  # Sunday afternoon
    features = encode_time_features(timestamp)
    
    assert 'hour' in features
    assert 'day_of_week' in features
    assert 'is_weekend' in features
    
    assert features['hour'] == 14
    assert features['day_of_week'] == 6  # Sunday
    assert features['is_weekend'] == True


def test_storage_traffic_history():
    """Test traffic history storage"""
    from traffic_forecast.storage.traffic_history import TrafficHistory
    import pandas as pd
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-11-10', periods=10, freq='15min'),
            'node_a_id': ['node_1'] * 10,
            'node_b_id': ['node_2'] * 10,
            'speed_kmh': [30.0 + i for i in range(10)],
            'run_id': range(10)
        })
        
        # Save
        history = TrafficHistory(base_dir=tmpdir)
        filepath = Path(tmpdir) / "test_data.parquet"
        df.to_parquet(filepath)
        
        # Verify
        assert filepath.exists()
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == 10


def test_validation_schemas():
    """Test validation schemas"""
    from traffic_forecast.validation.schemas import validate_traffic_record
    
    # Valid record
    valid_record = {
        'timestamp': '2025-11-10T14:30:00',
        'node_a_id': 'node_123',
        'node_b_id': 'node_456',
        'speed_kmh': 35.5,
        'run_id': 1
    }
    
    result = validate_traffic_record(valid_record)
    assert result is True
    
    # Invalid record (missing field)
    invalid_record = {
        'timestamp': '2025-11-10T14:30:00',
        'node_a_id': 'node_123',
        # missing node_b_id
        'speed_kmh': 35.5,
    }
    
    result = validate_traffic_record(invalid_record)
    assert result is False


def test_stmgt_model_creation():
    """Test STMGT model instantiation"""
    from traffic_forecast.models.stmgt.model import STMGT
    
    model = STMGT(
        num_nodes=62,
        in_dim=1,
        hidden_dim=96,
        num_blocks=3,
        num_heads=4,
        dropout=0.2,
        drop_edge_rate=0.15,
        mixture_components=3,
        seq_len=12,
        pred_len=12
    )
    
    assert model is not None
    assert model.num_nodes == 62
    assert model.hidden_dim == 96
    assert model.num_blocks == 3


def test_dataset_initialization():
    """Test STMGTDataset initialization"""
    from traffic_forecast.data.stmgt_dataset import STMGTDataset
    import pandas as pd
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-11-10', periods=50, freq='15min'),
            'node_a_id': ['node_1'] * 50,
            'node_b_id': ['node_2'] * 50,
            'speed_kmh': [30.0] * 50,
            'temperature_c': [28.0] * 50,
            'wind_speed_kmh': [5.0] * 50,
            'precipitation_mm': [0.0] * 50,
            'run_id': range(50)
        })
        
        filepath = Path(tmpdir) / "test_data.parquet"
        df.to_parquet(filepath)
        
        # Create dataset
        dataset = STMGTDataset(
            parquet_path=str(filepath),
            seq_len=12,
            pred_len=12,
            split='train'
        )
        
        assert len(dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
