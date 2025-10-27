"""
Data validation schemas using Pydantic for traffic forecast pipeline.
Ensures data quality and consistency across all pipeline stages.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RoadType(str, Enum):
 """Valid road types from OSM."""
 MOTORWAY = "motorway"
 TRUNK = "trunk"
 PRIMARY = "primary"
 SECONDARY = "secondary"
 TERTIARY = "tertiary"
 RESIDENTIAL = "residential"
 UNCLASSIFIED = "unclassified"
 UNKNOWN = "unknown"


class TrafficNode(BaseModel):
 """Schema for a traffic node (intersection)."""
 node_id: str = Field(..., description="Unique node identifier")
 lat: float = Field(..., ge=-90, le=90, description="Latitude")
 lon: float = Field(..., ge=-180, le=180, description="Longitude")
 degree: int = Field(..., ge=0, description="Number of connecting roads")
 importance_score: float = Field(..., ge=0, description="Intersection importance score")
 road_type: str = Field(..., description="Primary road type at intersection")
 connected_road_types: List[str] = Field(default_factory=list)
 street_names: List[str] = Field(default_factory=list, description="Names of streets at intersection")
 intersection_name: Optional[str] = Field(None, description="Human-readable intersection name")
 way_ids: List[int] = Field(default_factory=list, description="OSM way IDs")
 is_major_intersection: bool = Field(default=True)
 
 @validator('node_id')
 def validate_node_id(cls, v):
 if not v.startswith('node-'):
 raise ValueError('node_id must start with "node-"')
 return v
 
 @validator('connected_road_types')
 def validate_road_types(cls, v):
 if not v:
 raise ValueError('connected_road_types cannot be empty')
 return v
 
 class Config:
 use_enum_values = True


class TrafficEdge(BaseModel):
 """Schema for a traffic edge (road segment between intersections)."""
 u: str = Field(..., description="Source node ID")
 v: str = Field(..., description="Target node ID")
 distance_m: float = Field(..., gt=0, description="Distance in meters")
 way_id: int = Field(..., description="OSM way ID")
 road_type: str = Field(..., description="Road type")
 lanes: Optional[str] = None
 maxspeed: Optional[str] = None
 name: Optional[str] = None
 
 @validator('u', 'v')
 def validate_node_ids(cls, v):
 if not v.startswith('node-'):
 raise ValueError('Node IDs must start with "node-"')
 return v
 
 @root_validator(skip_on_failure=True)
 def validate_edge(cls, values):
 if values.get('u') == values.get('v'):
 raise ValueError('Edge cannot connect node to itself')
 return values


class WeatherSnapshot(BaseModel):
 """Schema for weather data snapshot."""
 timestamp: datetime
 temperature_c: float = Field(..., ge=-50, le=60)
 precipitation_mm: float = Field(..., ge=0)
 wind_speed_kmh: float = Field(..., ge=0)
 
 # Forecast horizons (optional)
 forecast_temp_t5_c: Optional[float] = None
 forecast_temp_t15_c: Optional[float] = None
 forecast_temp_t30_c: Optional[float] = None
 forecast_temp_t60_c: Optional[float] = None
 
 forecast_rain_t5_mm: Optional[float] = None
 forecast_rain_t15_mm: Optional[float] = None
 forecast_rain_t30_mm: Optional[float] = None
 forecast_rain_t60_mm: Optional[float] = None
 
 forecast_wind_t5_kmh: Optional[float] = None
 forecast_wind_t15_kmh: Optional[float] = None
 forecast_wind_t30_kmh: Optional[float] = None
 forecast_wind_t60_kmh: Optional[float] = None


class TrafficSnapshot(BaseModel):
 """Schema for traffic data snapshot at a node."""
 node_id: str
 timestamp: datetime
 avg_speed_kmh: float = Field(..., ge=0, le=200, description="Average speed in km/h")
 sample_count: int = Field(..., ge=0, description="Number of samples aggregated")
 
 # Optional metadata
 congestion_level: Optional[int] = Field(None, ge=0, le=4)
 reliability: Optional[float] = Field(None, ge=0, le=1)
 
 @validator('node_id')
 def validate_node_id(cls, v):
 if not v.startswith('node-'):
 raise ValueError('node_id must start with "node-"')
 return v


class NodeFeatures(BaseModel):
 """Schema for ML features at a node."""
 node_id: str
 timestamp: datetime
 
 # Traffic features
 avg_speed_kmh: float = Field(..., ge=0, le=200)
 
 # Weather features
 temperature_c: float
 rain_mm: float = Field(..., ge=0)
 wind_speed_kmh: float = Field(..., ge=0)
 
 # Weather forecast features
 forecast_temp_t5_c: Optional[float] = None
 forecast_temp_t15_c: Optional[float] = None
 forecast_temp_t30_c: Optional[float] = None
 forecast_temp_t60_c: Optional[float] = None
 
 forecast_rain_t5_mm: Optional[float] = None
 forecast_rain_t15_mm: Optional[float] = None
 forecast_rain_t30_mm: Optional[float] = None
 forecast_rain_t60_mm: Optional[float] = None
 
 forecast_wind_t5_kmh: Optional[float] = None
 forecast_wind_t15_kmh: Optional[float] = None
 forecast_wind_t30_kmh: Optional[float] = None
 forecast_wind_t60_kmh: Optional[float] = None
 
 @validator('node_id')
 def validate_node_id(cls, v):
 if not v.startswith('node-'):
 raise ValueError('node_id must start with "node-"')
 return v
 
 class Config:
 # Allow extra fields for backward compatibility
 extra = 'allow'


class PipelineRunMetadata(BaseModel):
 """Metadata for a pipeline run."""
 run_id: str
 timestamp: datetime
 pipeline_name: str
 status: str = Field(..., pattern='^(success|failed|running)$')
 
 # Metrics
 nodes_processed: int = Field(..., ge=0)
 edges_processed: int = Field(default=0, ge=0)
 processing_time_seconds: float = Field(..., ge=0)
 
 # Error tracking
 errors: List[str] = Field(default_factory=list)
 warnings: List[str] = Field(default_factory=list)
 
 # Additional metadata
 config: Dict[str, Any] = Field(default_factory=dict)


class ModelPrediction(BaseModel):
 """Schema for model predictions."""
 node_id: str
 timestamp: datetime
 horizon_min: int = Field(..., gt=0, description="Forecast horizon in minutes")
 predicted_speed_kmh: float = Field(..., ge=0, le=200)
 confidence: Optional[float] = Field(None, ge=0, le=1)
 model_name: str
 model_version: str
 
 @validator('node_id')
 def validate_node_id(cls, v):
 if not v.startswith('node-'):
 raise ValueError('node_id must start with "node-"')
 return v


class DataQualityReport(BaseModel):
 """Data quality report for a dataset."""
 timestamp: datetime
 dataset_name: str
 total_records: int = Field(..., ge=0)
 valid_records: int = Field(..., ge=0)
 invalid_records: int = Field(default=0, ge=0)
 
 # Quality metrics
 completeness_pct: float = Field(..., ge=0, le=100)
 validity_pct: float = Field(..., ge=0, le=100)
 
 # Issues found
 missing_values: Dict[str, int] = Field(default_factory=dict)
 outliers: Dict[str, int] = Field(default_factory=dict)
 duplicates: int = Field(default=0, ge=0)
 
 # Validation errors
 validation_errors: List[str] = Field(default_factory=list)
 
 @root_validator(skip_on_failure=True)
 def validate_counts(cls, values):
 total = values.get('total_records', 0)
 valid = values.get('valid_records', 0)
 invalid = values.get('invalid_records', 0)
 
 if valid + invalid != total:
 raise ValueError('valid_records + invalid_records must equal total_records')
 
 return values


# Utility functions for validation

def validate_nodes(nodes: List[dict]) -> tuple[List[TrafficNode], List[str]]:
 """
 Validate a list of node dictionaries.
 
 Returns:
 Tuple of (valid_nodes, error_messages)
 """
 valid_nodes = []
 errors = []
 
 for i, node_data in enumerate(nodes):
 try:
 validated_node = TrafficNode(**node_data)
 valid_nodes.append(validated_node)
 except Exception as e:
 errors.append(f"Node {i}: {str(e)}")
 
 return valid_nodes, errors


def validate_edges(edges: List[dict]) -> tuple[List[TrafficEdge], List[str]]:
 """
 Validate a list of edge dictionaries.
 
 Returns:
 Tuple of (valid_edges, error_messages)
 """
 valid_edges = []
 errors = []
 
 for i, edge_data in enumerate(edges):
 try:
 validated_edge = TrafficEdge(**edge_data)
 valid_edges.append(validated_edge)
 except Exception as e:
 errors.append(f"Edge {i}: {str(e)}")
 
 return valid_edges, errors


def generate_quality_report(
 dataset_name: str,
 total_records: int,
 valid_records: int,
 validation_errors: List[str],
 missing_values: Dict[str, int] = None,
 outliers: Dict[str, int] = None,
 duplicates: int = 0
) -> DataQualityReport:
 """Generate a data quality report."""
 invalid_records = total_records - valid_records
 completeness_pct = (valid_records / total_records * 100) if total_records > 0 else 0
 validity_pct = (valid_records / total_records * 100) if total_records > 0 else 0
 
 return DataQualityReport(
 timestamp=datetime.now(),
 dataset_name=dataset_name,
 total_records=total_records,
 valid_records=valid_records,
 invalid_records=invalid_records,
 completeness_pct=round(completeness_pct, 2),
 validity_pct=round(validity_pct, 2),
 missing_values=missing_values or {},
 outliers=outliers or {},
 duplicates=duplicates,
 validation_errors=validation_errors
 )
