"""Model utilities and trained artifacts."""

from importlib import import_module

__all__ = []


def _register_graph_models() -> None:
    """Register graph neural network models - lazy loading to avoid TensorFlow."""
    # Skip auto-registration to avoid TensorFlow import issues
    # Models can be loaded on-demand via get_astgcn_models()
    pass


def _register_lstm_models() -> None:
    """Register LSTM models."""
    try:
        from traffic_forecast.models.lstm_traffic import LSTMTrafficPredictor
        globals()["LSTMTrafficPredictor"] = LSTMTrafficPredictor
        __all__.append("LSTMTrafficPredictor")
    except (ImportError, IndentationError, SyntaxError):
        pass


def _register_astgcn() -> None:
    """Expose the ASTGCN runner - lazy loading to avoid TensorFlow."""
    # Skip auto-registration to avoid TensorFlow import issues
    # ASTGCN can be imported directly when needed
    pass


_register_graph_models()
_register_lstm_models()
_register_astgcn()

