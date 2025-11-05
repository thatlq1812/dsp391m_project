"""Schema validation utilities for the model registry configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class HyperparameterSchema(BaseModel):
    """Validate dashboard hyperparameter descriptors."""

    key: str
    type: str = Field(..., description="Widget type used by the dashboard")
    label: str
    group: str = Field(default="Parameters")
    default: Optional[Any] = None
    options: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    @model_validator(mode='after')
    def _check_numeric_bounds(self) -> 'HyperparameterSchema':
        if self.type in {"int_slider", "float_slider"}:
            if self.min is None or self.max is None:
                raise ValueError(f"Slider parameter '{self.key}' requires min and max values")
        if self.type == "select" and not self.options:
            raise ValueError(f"Select parameter '{self.key}' requires options")
        if self.type == "dataset_select" and self.options:
            raise ValueError("dataset_select parameters derive options internally and should omit them")
        return self


class TrainSectionSchema(BaseModel):
    """Structure for the training section inside a model registry entry."""

    script: str
    config: Dict[str, Any]


class ModelEntrySchema(BaseModel):
    """Single model entry inside the registry."""

    key: str
    display_name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    train: TrainSectionSchema
    hyperparameters: List[HyperparameterSchema] = Field(default_factory=list)


class ModelRegistrySchema(BaseModel):
    """Top level registry container."""

    models: List[ModelEntrySchema] = Field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return json.loads(self.json())


def load_model_registry(path: Path) -> ModelRegistrySchema:
    """Load and validate the model registry JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelRegistrySchema(**payload)


__all__ = ["ModelRegistrySchema", "ModelEntrySchema", "HyperparameterSchema", "load_model_registry"]
