"""
Evaluation utilities for traffic forecasting models.

This module provides unified evaluation framework for comparing
different models (STMGT, LSTM, ASTGCN, etc.) on consistent metrics
and data splits.
"""

from .unified_evaluator import (
    UnifiedEvaluator,
    EvaluationMetrics,
    ModelComparison
)
from .model_wrapper import ModelWrapper

__all__ = [
    'UnifiedEvaluator',
    'EvaluationMetrics',
    'ModelComparison',
    'ModelWrapper'
]
