"""Graph neural network models for traffic forecasting."""

from __future__ import annotations

try:
    from traffic_forecast.models.graph.astgcn_traffic import (
        ASTGCNConfig,
        ASTGCNComponentConfig,
        ASTGCNTrafficModel,
    )

    __all__ = ['ASTGCNConfig', 'ASTGCNComponentConfig', 'ASTGCNTrafficModel']
except (ImportError, IndentationError, SyntaxError):
    __all__ = []
