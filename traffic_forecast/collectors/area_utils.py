"""
Utility functions for area configuration.
"""

import os
import math
import yaml
from traffic_forecast import PROJECT_ROOT


def bbox_from_point_radius(lon, lat, radius_m):
    """
    Given (lon, lat) center and radius in meters, return bounding box.
    Returns (min_lat, min_lon, max_lat, max_lon).
    """
    R = 6371000  # Earth radius in meters
    lat_rad = math.radians(lat)
    
    # Latitude offset
    delta_lat = (radius_m / R) * (180 / math.pi)
    
    # Longitude offset (adjusted for latitude)
    delta_lon = (radius_m / (R * math.cos(lat_rad))) * (180 / math.pi)
    
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    
    return [min_lat, min_lon, max_lat, max_lon]


def load_area_config(collector_name, cli_area=None):
    """
    Load area configuration for a given collector.
    Priority: CLI args > Env vars > collector config > global config.
    
    Returns a dict with keys: mode and bbox (min_lat,min_lon,max_lat,max_lon).
    """
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    
    c = cfg.get('collectors', {}).get(collector_name, {})
    area = c.get('area', {}) or {}

    # fallback to global area if not provided
    if not area:
        glob_area = cfg.get('globals', {}).get('area')
        if glob_area:
            area = dict(glob_area)

    # Apply CLI overrides first (highest priority)
    if cli_area:
        area.update({k: v for k, v in cli_area.items() if v is not None})

    # Env overrides
    env_mode = os.getenv(f"{collector_name.upper()}_MODE")
    env_bbox = os.getenv(f"{collector_name.upper()}_BBOX")
    env_center = os.getenv(f"{collector_name.upper()}_CENTER")
    env_radius = os.getenv(f"{collector_name.upper()}_RADIUS")

    if env_mode and 'mode' not in area:
        area['mode'] = env_mode
    if env_bbox and 'bbox' not in area:
        # expected as min_lat,min_lon,max_lat,max_lon
        area['mode'] = 'bbox'
        area['bbox'] = list(map(float, env_bbox.split(',')))
    if env_center and 'center' not in area:
        area['mode'] = 'point_radius'
        area['center'] = list(map(float, env_center.split(',')))
    if env_radius and 'radius_m' not in area:
        area['mode'] = 'point_radius'
        area['radius_m'] = float(env_radius)

    mode = area.get('mode', 'bbox')

    if mode == 'bbox':
        bbox = area.get('bbox')
        if not bbox:
            raise ValueError(f"Collector {collector_name}: bbox mode but no bbox provided")
        return {'mode': 'bbox', 'bbox': bbox}
    elif mode in ('point_radius', 'circle'):
        center = area.get('center')
        radius = area.get('radius_m')
        if not center or not radius:
            raise ValueError(f"Collector {collector_name}: point_radius mode requires center and radius_m")
        lon, lat = center
        bbox = bbox_from_point_radius(lon, lat, radius)
        return {'mode': 'point_radius', 'bbox': bbox, 'center': center, 'radius_m': radius}
    else:
        raise ValueError(f"Unknown area mode: {mode}")


def get_run_output_base():
    """Return base output directory for runs. Priority: RUN_DIR env > globals.output_base > data_runs"""
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = os.getenv('RUN_DIR') or cfg.get('globals', {}).get('output_base') or 'data_runs'
    return base
