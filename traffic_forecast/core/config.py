"""
Configuration management for the traffic forecasting pipeline.

Author: thatlq1812
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json


@dataclass
class ModelConfig:
    """Configuration for a deep learning model."""
    
    name: str  # 'lstm' or 'astgcn'
    enabled: bool = True
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    sequence_length: int = 12  # Number of past time steps
    
    # Model-specific parameters
    hidden_units: List[int] = field(default_factory=lambda: [64, 64])
    dropout_rate: float = 0.2
    
    # ASTGCN-specific
    num_of_vertices: Optional[int] = None  # Number of nodes in graph
    num_of_features: int = 1  # Traffic speed only, or multivariate
    
    # Advanced
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'num_of_vertices': self.num_of_vertices,
            'num_of_features': self.num_of_features,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
        }


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path('data/runs'))
    processed_dir: Path = field(default_factory=lambda: Path('data/processed'))
    models_dir: Path = field(default_factory=lambda: Path('traffic_forecast/models/saved'))
    cache_dir: Path = field(default_factory=lambda: Path('cache'))
    
    # Pipeline steps control
    enable_data_exploration: bool = True
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_model_training: bool = True
    enable_evaluation: bool = True
    enable_save_models: bool = True
    
    # Data options
    use_existing_data: bool = True
    use_preprocessed: bool = False
    max_runs: Optional[int] = None  # Limit number of runs to load
    
    # Feature engineering
    target_column: str = 'speed_kmh'
    feature_columns: List[str] = field(default_factory=lambda: [
        'hour', 'day_of_week', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'distance_km', 'duration_sec',
        'temperature_c', 'wind_speed_kmh', 'precipitation_mm',
        'speed_lag_1', 'speed_lag_2', 'speed_lag_3',
        'speed_rolling_mean_3', 'speed_rolling_std_3',
        'is_morning_rush', 'is_evening_rush', 'is_rush_hour'
    ])
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 42
    
    # Models to train
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        'lstm': ModelConfig(
            name='lstm',
            enabled=True,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            hidden_units=[64, 64],
            dropout_rate=0.2
        ),
        'astgcn': ModelConfig(
            name='astgcn',
            enabled=True,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            hidden_units=[64, 64, 64],
            dropout_rate=0.2
        )
    })
    
    # Logging
    verbose: bool = True
    plot_training_history: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert paths
        for key in ['data_dir', 'processed_dir', 'models_dir', 'cache_dir']:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        
        # Convert models
        if 'models' in config_dict:
            models = {}
            for model_name, model_cfg in config_dict['models'].items():
                models[model_name] = ModelConfig(name=model_name, **model_cfg)
            config_dict['models'] = models
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert paths
        for key in ['data_dir', 'processed_dir', 'models_dir', 'cache_dir']:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        
        # Convert models
        if 'models' in config_dict:
            models = {}
            for model_name, model_cfg in config_dict['models'].items():
                models[model_name] = ModelConfig(name=model_name, **model_cfg)
            config_dict['models'] = models
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, json_path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_dir': str(self.data_dir),
            'processed_dir': str(self.processed_dir),
            'models_dir': str(self.models_dir),
            'cache_dir': str(self.cache_dir),
            'enable_data_exploration': self.enable_data_exploration,
            'enable_preprocessing': self.enable_preprocessing,
            'enable_feature_engineering': self.enable_feature_engineering,
            'enable_model_training': self.enable_model_training,
            'enable_evaluation': self.enable_evaluation,
            'enable_save_models': self.enable_save_models,
            'use_existing_data': self.use_existing_data,
            'use_preprocessed': self.use_preprocessed,
            'max_runs': self.max_runs,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'models': {name: cfg.to_dict() for name, cfg in self.models.items()},
            'verbose': self.verbose,
            'plot_training_history': self.plot_training_history,
        }


# Default configuration
DEFAULT_CONFIG = PipelineConfig()
