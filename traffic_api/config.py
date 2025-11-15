"""API Configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class APIConfig(BaseModel):
    """API configuration settings."""
    
    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}

    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    model_checkpoint: Optional[Path] = None
    data_path: Optional[Path] = None  # Training data path
    topology_cache: Path = project_root / "cache" / "overpass_topology.json"
    
    # Model settings
    device: str = "cuda"  # "cuda" or "cpu"
    batch_size: int = 16
    num_workers: int = 4
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = True
    
    # CORS
    allow_origins: list[str] = ["*"]  # For development, restrict in production
    
    # Cache TTL (seconds)
    prediction_cache_ttl: int = 900  # 15 minutes
    node_cache_ttl: int = 86400  # 24 hours


# Global config instance
config = APIConfig()

# Auto-detect best model checkpoint and data path
outputs_dir = config.project_root / "outputs"
if outputs_dir.exists():
    model_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if model_dirs:
        best_model = model_dirs[0] / "best_model.pt"
        if best_model.exists():
            config.model_checkpoint = best_model
            print(f"Auto-detected model: {best_model}")
            
            # Try to detect data path from config
            config_json = model_dirs[0] / "config.json"
            if config_json.exists():
                import json
                with open(config_json) as f:
                    model_config = json.load(f)
                    if 'metadata' in model_config and 'data_path' in model_config['metadata']:
                        config.data_path = Path(model_config['metadata']['data_path'])
                        print(f"Auto-detected data: {config.data_path}")
            
            # Fallback data paths (prefer gap-filled weekly dataset)
            if config.data_path is None or not config.data_path.exists():
                for data_file in ['all_runs_gapfilled_week.parquet', 'all_runs_extreme_augmented.parquet', 'all_runs_augmented.parquet', 'all_runs_combined.parquet']:
                    test_path = config.project_root / "data" / "processed" / data_file
                    if test_path.exists():
                        config.data_path = test_path
                        print(f"Using fallback data: {config.data_path}")
                        break
