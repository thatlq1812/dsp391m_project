"""Model utilities and trained artifacts."""

from importlib import import_module

__all__ = []


def _register_graph_models() -> None:
    """Register graph neural network models."""
    try:
        module = import_module("traffic_forecast.models.graph")
        for symbol in ["ASTGCNConfig", "ASTGCNComponentConfig", "ASTGCNTrafficModel"]:
            if hasattr(module, symbol):
                globals()[symbol] = getattr(module, symbol)
                __all__.append(symbol)
    except (ImportError, IndentationError, SyntaxError):
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
    """Expose the ASTGCN runner."""
    try:
        from traffic_forecast.models.astgcn import (
            NotebookBaselineConfig,
            NotebookBaselineRunner,
            run_astgcn,
        )

        globals()["NotebookBaselineConfig"] = NotebookBaselineConfig
        globals()["NotebookBaselineRunner"] = NotebookBaselineRunner
        globals()["run_astgcn"] = run_astgcn
        __all__.extend(
            [
                "NotebookBaselineConfig",
                "NotebookBaselineRunner",
                "run_astgcn",
            ]
        )
    except (ImportError, IndentationError, SyntaxError):
        pass


_register_graph_models()
_register_lstm_models()
_register_astgcn()
