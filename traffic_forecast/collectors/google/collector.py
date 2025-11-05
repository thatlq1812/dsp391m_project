"""Mock Google Directions API interactions used in the tests.

The real project talks to Google Directions; here we generate lightweight
synthetic values so the integration pipeline can stay offline.
"""

from __future__ import annotations

import random
from typing import Dict


def mock_directions_api(
    node_a: Dict[str, float], node_b: Dict[str, float], config: Dict[str, object]
) -> Dict[str, float]:
    """Return a synthetic route summary between two nodes.

    Parameters
    ----------
    node_a, node_b
        Dictionaries containing at least ``lat`` and ``lon``.
    config
        Collector configuration with an optional ``mock_speed_range``.
    """

    collector_cfg = config.get("collectors", {}).get("google_directions", {})
    speed_range = collector_cfg.get("mock_speed_range", [20.0, 60.0])
    speed_min, speed_max = sorted([float(speed_range[0]), float(speed_range[-1])])
    speed_kmh = random.uniform(speed_min, speed_max)

    # Simple straight-line distance approximation in kilometres.
    delta_lat = float(node_b["lat"]) - float(node_a["lat"])
    delta_lon = float(node_b["lon"]) - float(node_a["lon"])
    distance_km = (delta_lat**2 + delta_lon**2) ** 0.5 * 111.0

    duration_hours = distance_km / max(speed_kmh, 1e-3)
    duration_sec = duration_hours * 3600.0

    return {
        "distance_km": distance_km,
        "duration_sec": duration_sec,
        "speed_kmh": speed_kmh,
    }
