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
