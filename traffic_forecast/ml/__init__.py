"""
Machine Learning Pipeline for Traffic Forecast System.

This module provides end-to-end ML pipeline:
- Data loading from collected runs
- Preprocessing and feature engineering
- Deep Learning model training and evaluation (LSTM, ATSCGN)
- Prediction and inference
"""

from traffic_forecast.ml.data_loader import (
    DataLoader,
    load_latest_data,
    load_all_data
)
from traffic_forecast.ml.preprocess import (
    DataPreprocessor,
    split_data,
    prepare_features_target,
    get_preprocessing_pipeline
)
from traffic_forecast.ml.features import (
    build_features,
    add_temporal_features,
    add_spatial_features,
    add_weather_features,
    add_traffic_features
)

# Deep Learning models (requires TensorFlow)
try:
    from traffic_forecast.ml.dl_trainer import DLModelTrainer
    HAS_DL = True
except ImportError:
    HAS_DL = False
    DLModelTrainer = None

__all__ = [
    # Data loading
    'DataLoader',
    'load_latest_data',
    'load_all_data',
    # Preprocessing
    'DataPreprocessor',
    'split_data',
    'prepare_features_target',
    'get_preprocessing_pipeline',
    # Feature engineering
    'build_features',
    'add_temporal_features',
    'add_spatial_features',
    'add_weather_features',
    'add_traffic_features',
    # Deep Learning Training
    'compare_models',
    # Training (Deep Learning)
    'DLModelTrainer',
    'compare_dl_models',
    'HAS_DL',
]
