"""Pydantic schemas for API request/response."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NodeInfo(BaseModel):
    """Information about a traffic node (intersection)."""

    node_id: str
    lat: float
    lon: float
    degree: int
    importance_score: float
    road_type: str
    street_names: list[str]
    intersection_name: str
    is_major_intersection: bool


class PredictionRequest(BaseModel):
    """Request for traffic predictions."""

    timestamp: Optional[datetime] = Field(default=None, description="Forecast timestamp (defaults to now)")
    node_ids: Optional[list[str]] = Field(default=None, description="Specific nodes (defaults to all)")
    horizons: list[int] = Field(default=[1, 2, 3, 6, 9, 12], description="Forecast horizons (timesteps)")


class NodePrediction(BaseModel):
    """Prediction for a single node."""

    node_id: str
    lat: float
    lon: float
    forecasts: list[dict[str, float]]  # List of {horizon, mean, std, lower, upper}
    current_speed: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response containing predictions for all nodes."""
    
    model_config = {"protected_namespaces": ()}

    timestamp: datetime
    forecast_time: datetime
    nodes: list[NodePrediction]
    model_version: str = "stmgt_v2"
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_checkpoint: Optional[str]
    device: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime


class EdgeTraffic(BaseModel):
    """Current traffic status for an edge."""

    edge_id: str
    node_a_id: str
    node_b_id: str
    speed_kmh: float
    color: str  # Hex color code
    color_category: str  # 'blue', 'green', 'yellow', 'orange', 'red'
    timestamp: datetime
    lat_a: float
    lon_a: float
    lat_b: float
    lon_b: float


class TrafficCurrentResponse(BaseModel):
    """Response for current traffic on all edges."""

    edges: list[EdgeTraffic]
    timestamp: datetime
    total_edges: int


class RouteRequest(BaseModel):
    """Request for route planning."""

    start_node_id: str = Field(..., description="Starting node ID")
    end_node_id: str = Field(..., description="Destination node ID")
    departure_time: Optional[datetime] = Field(default=None, description="Departure time (defaults to now)")


class RouteSegment(BaseModel):
    """One segment of a route."""

    edge_id: str
    node_a_id: str
    node_b_id: str
    distance_km: float
    predicted_speed_kmh: float
    predicted_travel_time_min: float
    uncertainty_std: float


class Route(BaseModel):
    """One possible route from A to B."""

    route_type: str  # 'fastest', 'shortest', 'balanced'
    segments: list[RouteSegment]
    total_distance_km: float
    expected_travel_time_min: float
    travel_time_uncertainty_min: float
    confidence_level: float  # 0-1


class RoutePlanResponse(BaseModel):
    """Response for route planning."""

    start_node_id: str
    end_node_id: str
    departure_time: datetime
    routes: list[Route]  # 3 routes: fastest, shortest, balanced
    timestamp: datetime
