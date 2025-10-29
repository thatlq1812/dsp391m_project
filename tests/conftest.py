"""
Pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from traffic_forecast import PROJECT_ROOT


@pytest.fixture
def sample_node_data():
    """Sample node data for testing."""
    return {
        'node_id': 'node-10.772-106.698',
        'lat': 10.772,
        'lon': 106.698,
        'degree': 4,
        'importance_score': 25.5,
        'road_type': 'motorway',
        'connected_road_types': ['motorway', 'primary'],
        'is_major_intersection': True
    }


@pytest.fixture
def sample_edge_data():
    """Sample edge data for testing."""
    return {
        'u': 'node-10.772-106.698',
        'v': 'node-10.773-106.699',
        'distance_m': 150.5,
        'way_id': 12345,
        'road_type': 'primary',
        'lanes': '4',
        'maxspeed': '60',
        'name': 'Main Street'
    }


@pytest.fixture
def sample_osm_data():
    """Sample OSM data for testing."""
    return {
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
            }
        ]
    }


@pytest.fixture
def sample_features_df():
    """Sample features DataFrame for ML testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        'node_id': [f'node-{i}' for i in range(n_samples)],
        'avg_speed_kmh': np.random.uniform(20, 60, n_samples),
        'temperature_c': np.random.uniform(25, 35, n_samples),
        'rain_mm': np.random.uniform(0, 10, n_samples),
        'wind_speed_kmh': np.random.uniform(0, 30, n_samples),
        'forecast_temp_t5_c': np.random.uniform(25, 35, n_samples),
        'forecast_temp_t15_c': np.random.uniform(25, 35, n_samples),
        'forecast_temp_t30_c': np.random.uniform(25, 35, n_samples),
        'forecast_temp_t60_c': np.random.uniform(25, 35, n_samples),
    })


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return PROJECT_ROOT
