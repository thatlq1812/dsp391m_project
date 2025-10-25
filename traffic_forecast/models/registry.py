"""Model registry providing configurable regressors for traffic forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class ModelFactory:
    name: str
    creator: Callable[..., object]
    supported_params: Iterable[str]

    def build(self, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in self.supported_params}
        return self.creator(**filtered)


REGISTRY: Dict[str, ModelFactory] = {
    "linear_regression": ModelFactory(
        name="linear_regression",
        creator=LinearRegression,
        supported_params=["fit_intercept", "positive"],
    ),
    "random_forest": ModelFactory(
        name="random_forest",
        creator=RandomForestRegressor,
        supported_params=[
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "n_jobs",
            "random_state",
        ],
    ),
    "gradient_boosting": ModelFactory(
        name="gradient_boosting",
        creator=GradientBoostingRegressor,
        supported_params=[
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "random_state",
        ],
    ),
}


def build_model(model_type: str, **kwargs):
    factory = REGISTRY.get(model_type)
    if not factory:
        raise ValueError(f"Unsupported model type: {model_type}")
    return factory.build(**kwargs)


def list_available_models() -> Iterable[str]:
    return REGISTRY.keys()
