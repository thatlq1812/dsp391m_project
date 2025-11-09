"""
Model wrapper interface for unified evaluation.

Each model implements this interface for consistent evaluation across
different architectures (STMGT, LSTM, ASTGCN, etc.).
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Optional
import pandas as pd


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    Each model (STMGT, LSTM, ASTGCN, etc.) implements this interface
    to enable unified evaluation through consistent API.
    
    Example:
        >>> wrapper = STMGTWrapper(model_config)
        >>> wrapper.load_checkpoint("best_model.pt")
        >>> predictions, stds = wrapper.predict(test_data)
    """
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on data.
        
        Args:
            data: DataFrame with required features (speed, weather, etc.)
            device: Device to run inference on ('cuda' or 'cpu')
            
        Returns:
            Tuple of (predictions, std) where:
                - predictions: (n_samples,) array of predicted speeds
                - std: (n_samples,) array of prediction uncertainty 
                       or None for deterministic models
        """
        pass
    
    @abstractmethod
    def parameters(self):
        """
        Return model parameters for counting.
        
        Returns:
            Iterator over model parameters
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model name."""
        pass
