"""Pydantic schemas for validating traffic data records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class TrafficDataSchema(BaseModel):
    node_a_id: str = Field(..., min_length=1)
    node_b_id: str = Field(..., min_length=1)
    speed_kmh: float = Field(..., gt=0.0)
    duration_sec: float = Field(..., gt=0.0)
    distance_km: float = Field(..., gt=0.0)
    timestamp: datetime

    @validator("timestamp", pre=True)
    def _parse_timestamp(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("timestamp must be ISO-8601 parsable") from exc

    class Config:
        allow_mutation = False
        anystr_strip_whitespace = True
