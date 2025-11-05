"""Lightweight Overpass collector fakes for integration testing.

The real project performs HTTP requests against the Overpass API to fetch
road network nodes. For the test suite we keep things deterministic and avoid
network access by synthesising a repeatable list of nodes that roughly match
what the pipeline expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class SyntheticNode:
    """Simple representation of an Overpass node."""

    node_id: str
    lat: float
    lon: float
    importance: float

    def as_dict(self) -> Dict[str, float]:
        """Return a serialisable dictionary form for downstream code."""

        return {
            "node_id": self.node_id,
            "lat": self.lat,
            "lon": self.lon,
            "importance": self.importance,
        }


def _iterate_nodes(config: Dict[str, object]) -> Iterable[SyntheticNode]:
    center = config.get("area", {}).get("center", {})
    lat = float(center.get("lat", 0.0))
    lon = float(center.get("lon", 0.0))
    radius = float(config.get("area", {}).get("radius_m", 500.0))
    selection = config.get("node_selection", {})
    max_nodes = int(selection.get("max_nodes", 10))
    min_importance = float(selection.get("min_importance_score", 0.0))

    step = max(radius / max(max_nodes, 1), 1.0)
    for idx in range(max_nodes):
        yield SyntheticNode(
            node_id=f"node_{idx:03d}",
            lat=lat + 0.0001 * idx,
            lon=lon + 0.0001 * idx,
            importance=min_importance + 0.1 * idx,
        )


def collect_nodes(config: Dict[str, object]) -> List[Dict[str, float]]:
    """Return a deterministic list of nodes for the requested configuration."""

    return [node.as_dict() for node in _iterate_nodes(config)]
