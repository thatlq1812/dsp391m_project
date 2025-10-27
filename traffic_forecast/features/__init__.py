"""
Feature engineering modules for traffic forecasting.
Includes lag features, temporal encoding, and spatial aggregation.
"""

from traffic_forecast.features.lag_features import (
 create_lag_features,
 create_rolling_features,
 create_speed_change_features,
 create_all_lag_features
)

from traffic_forecast.features.temporal_features import (
 add_temporal_features,
 add_cyclical_encoding,
 add_rush_hour_flags,
 add_holiday_features,
 add_weekend_features
)

from traffic_forecast.features.spatial_features import (
 add_neighbor_features,
 add_spatial_features,
 add_congestion_propagation,
 build_adjacency_graph
)

from traffic_forecast.features.pipeline import (
 FeatureEngineeringPipeline,
 create_all_features
)

__all__ = [
 # Lag features
 'create_lag_features',
 'create_rolling_features',
 'create_speed_change_features',
 'create_all_lag_features',
 
 # Temporal features
 'add_temporal_features',
 'add_cyclical_encoding',
 'add_rush_hour_flags',
 'add_holiday_features',
 'add_weekend_features',
 
 # Spatial features
 'add_neighbor_features',
 'add_spatial_features',
 'add_congestion_propagation',
 'build_adjacency_graph',
 
 # Pipeline
 'FeatureEngineeringPipeline',
 'create_all_features'
]
