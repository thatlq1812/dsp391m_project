"""Model utilities and trained artifacts."""

from importlib import import_module

__all__ = []


def _maybe_register_astgcn() -> None:
	try:
		module = import_module("traffic_forecast.models.research.astgcn")
	except ImportError:
		return

	for symbol in (
		"ASTGCNComponentConfig",
		"ASTGCNConfig",
		"ASTGCNTrafficModel",
	):
		if hasattr(module, symbol):
			globals()[symbol] = getattr(module, symbol)
			__all__.append(symbol)


_maybe_register_astgcn()
