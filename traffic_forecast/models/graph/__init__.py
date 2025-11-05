"""Graph neural network models for traffic forecasting."""

from __future__ import annotations

# Lazy import to avoid TensorFlow dependency issues
# ASTGCN requires TensorFlow which can cause import errors
# Only import when explicitly needed
def get_astgcn_models():
    """Lazy load ASTGCN models to avoid TensorFlow import issues."""
    try:
        from traffic_forecast.models.graph.astgcn_traffic import (
            ASTGCNConfig,
            ASTGCNComponentConfig,
            ASTGCNTrafficModel,
        )
        return ASTGCNConfig, ASTGCNComponentConfig, ASTGCNTrafficModel
    except (ImportError, IndentationError, SyntaxError):
        return None, None, None

__all__ = ['get_astgcn_models']
