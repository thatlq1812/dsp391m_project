import math
import os
import yaml
from traffic_forecast import PROJECT_ROOT

EARTH_RADIUS_M = 6371000.0


def bbox_from_point_radius(lon, lat, radius_m):
 """Return [min_lat, min_lon, max_lat, max_lon] for easier Overpass/Open-Meteo use.
 lon, lat in degrees, radius_m in meters.
 """
 # delta latitude in degrees
 delta_lat = (radius_m / EARTH_RADIUS_M) * (180.0 / math.pi)
 # delta longitude depends on latitude
 lat_rad = math.radians(lat)
 cos_lat = max(1e-6, math.cos(lat_rad))
 delta_lon = (radius_m / (EARTH_RADIUS_M * cos_lat)) * (180.0 / math.pi)

 min_lat = lat - delta_lat
 max_lat = lat + delta_lat
 min_lon = lon - delta_lon
 max_lon = lon + delta_lon

 # clamp
 min_lat = max(-90.0, min_lat)
 max_lat = min(90.0, max_lat)
 min_lon = (min_lon + 180) % 360 - 180
 max_lon = (max_lon + 180) % 360 - 180

 return [min_lat, min_lon, max_lat, max_lon]


def load_area_config(collector_name, cli_area=None):
 """Load area config for a given collector from configs/project_config.yaml.
 Priority: CLI overrides (cli_area) > environment variables > config file.

 cli_area: optional dict with keys mode, bbox, center, radius_m.

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
