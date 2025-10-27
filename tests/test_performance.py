"""
Performance benchmarks for the traffic forecast system.
"""

import time
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestPerformance:
    """Performance benchmarks for key operations."""
    
    @pytest.mark.benchmark
    def test_feature_engineering_performance(self, benchmark):
        """Benchmark feature engineering pipeline."""
        from traffic_forecast.features.temporal_features import create_temporal_features
        
        # Create large dataset
        timestamps = pd.date_range('2025-01-01', periods=10000, freq='15min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'node_id': ['N' + str(i % 100) for i in range(10000)],
            'speed_kmh': np.random.uniform(20, 80, 10000)
        })
        
        # Benchmark
        result = benchmark(create_temporal_features, df)
        assert len(result) == len(df)
    
    @pytest.mark.benchmark
    def test_haversine_distance_performance(self, benchmark):
        """Benchmark haversine distance calculation."""
        from traffic_forecast.collectors.google.collector import haversine
        
        # Benchmark
        result = benchmark(
            haversine,
            10.762622, 106.660172,
            10.772622, 106.670172
        )
        assert result > 0
    
    @pytest.mark.benchmark
    def test_data_validation_performance(self, benchmark):
        """Benchmark data quality validation."""
        from traffic_forecast.validation.data_quality_validator import DataQualityValidator
        
        validator = DataQualityValidator()
        
        # Create large dataset
        df = pd.DataFrame({
            'node_a_id': ['A' + str(i) for i in range(1000)],
            'node_b_id': ['B' + str(i) for i in range(1000)],
            'speed_kmh': np.random.uniform(20, 80, 1000),
            'duration_sec': np.random.uniform(60, 600, 1000),
            'distance_km': np.random.uniform(1, 10, 1000)
        })
        
        # Benchmark
        result = benchmark(validator.validate_traffic_data, df)
        assert result['total_records'] == 1000
    
    @pytest.mark.benchmark
    def test_storage_write_performance(self, benchmark):
        """Benchmark traffic history storage writes."""
        import tempfile
        from pathlib import Path
        from traffic_forecast.storage.traffic_history import TrafficHistory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'
            history = TrafficHistory(db_path=str(db_path))
            
            timestamp = datetime.now()
            data = pd.DataFrame({
                'node_a_id': ['A'] * 100,
                'node_b_id': ['B'] * 100,
                'speed_kmh': np.random.uniform(20, 80, 100),
                'duration_sec': np.random.uniform(60, 600, 100),
                'distance_km': np.random.uniform(1, 10, 100)
            })
            
            # Benchmark
            benchmark(history.save, timestamp, data)
    
    @pytest.mark.benchmark
    def test_model_prediction_performance(self, benchmark):
        """Benchmark model prediction speed."""
        from traffic_forecast.models.baseline import predict_speed_persistence
        
        historical = list(np.random.uniform(20, 80, 100))
        horizons = [5, 15, 30, 60]
        
        # Benchmark
        result = benchmark(predict_speed_persistence, historical, horizons)
        assert len(result) == len(horizons)


class TestScalability:
    """Test system scalability with increasing data volumes."""
    
    def test_linear_scaling_features(self):
        """Test that feature engineering scales linearly."""
        from traffic_forecast.features.temporal_features import create_temporal_features
        
        sizes = [1000, 5000, 10000]
        times = []
        
        for size in sizes:
            timestamps = pd.date_range('2025-01-01', periods=size, freq='15min')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'speed_kmh': np.random.uniform(20, 80, size)
            })
            
            start = time.time()
            create_temporal_features(df)
            times.append(time.time() - start)
        
        # Check linear scaling (within tolerance)
        # time[1] should be ~5x time[0]
        # time[2] should be ~10x time[0]
        assert times[1] / times[0] < 7  # Should be ~5x, allow 7x
        assert times[2] / times[0] < 15  # Should be ~10x, allow 15x
    
    def test_memory_efficiency(self):
        """Test memory usage stays reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        timestamps = pd.date_range('2025-01-01', periods=50000, freq='15min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'node_id': ['N' + str(i % 1000) for i in range(50000)],
            'speed_kmh': np.random.uniform(20, 80, 50000)
        })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500 MB for 50k rows)
        assert memory_increase < 500


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-only'])
