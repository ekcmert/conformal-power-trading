from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _validate_quantile_pair(lower_quantile: float, upper_quantile: float) -> None:
    if not 0.0 < lower_quantile < 1.0:
        raise ValueError(f"lower_quantile must be between 0 and 1, got {lower_quantile}.")
    if not 0.0 < upper_quantile < 1.0:
        raise ValueError(f"upper_quantile must be between 0 and 1, got {upper_quantile}.")
    if lower_quantile >= upper_quantile:
        raise ValueError(
            f"lower_quantile must be smaller than upper_quantile, got {lower_quantile} >= {upper_quantile}."
        )


def _normalize_bounds(
    lower_values: np.ndarray,
    upper_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_values = np.asarray(lower_values, dtype=float).reshape(-1)
    upper_values = np.asarray(upper_values, dtype=float).reshape(-1)
    return np.minimum(lower_values, upper_values), np.maximum(lower_values, upper_values)


class SeparateQuantileEstimator:
    def __init__(
        self,
        *,
        estimator_builder: Callable[[float], object],
        lower_quantile: float,
        upper_quantile: float,
    ) -> None:
        _validate_quantile_pair(lower_quantile, upper_quantile)
        self.estimator_builder = estimator_builder
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y) -> "SeparateQuantileEstimator":
        self.lower_model_ = self.estimator_builder(self.lower_quantile)
        self.upper_model_ = self.estimator_builder(self.upper_quantile)
        self.lower_model_.fit(X, y)
        self.upper_model_.fit(X, y)
        self.interval_models_ = [self.lower_model_, self.upper_model_]
        self.n_features_in_ = getattr(self.lower_model_, "n_features_in_", None)
        return self

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        lower_values = self.lower_model_.predict(X)
        upper_values = self.upper_model_.predict(X)
        return _normalize_bounds(lower_values, upper_values)

    def predict(self, X) -> np.ndarray:
        lower_values, upper_values = self.predict_interval(X)
        return (lower_values + upper_values) / 2.0


class MultiQuantileEstimator:
    def __init__(
        self,
        *,
        estimator_builder: Callable[[Sequence[float]], object],
        quantiles: Sequence[float],
    ) -> None:
        if len(quantiles) < 2:
            raise ValueError(f"Expected at least two quantiles, got {quantiles}.")
        _validate_quantile_pair(float(quantiles[0]), float(quantiles[-1]))
        self.estimator_builder = estimator_builder
        self.quantiles = tuple(float(quantile) for quantile in quantiles)

    def fit(self, X, y) -> "MultiQuantileEstimator":
        self.model_ = self.estimator_builder(self.quantiles)
        self.model_.fit(X, y)
        self.n_features_in_ = getattr(self.model_, "n_features_in_", None)
        return self

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.asarray(self.model_.predict(X), dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] < 2:
            raise ValueError(
                f"Expected a 2D multi-quantile prediction array, got shape {predictions.shape}."
            )
        lower_values = predictions[:, 0]
        upper_values = predictions[:, -1]
        return _normalize_bounds(lower_values, upper_values)

    def predict(self, X) -> np.ndarray:
        lower_values, upper_values = self.predict_interval(X)
        return (lower_values + upper_values) / 2.0


class TreeEnsembleQuantileIntervalEstimator:
    def __init__(
        self,
        *,
        ensemble_builder: Callable[[], object],
        lower_quantile: float,
        upper_quantile: float,
    ) -> None:
        _validate_quantile_pair(lower_quantile, upper_quantile)
        self.ensemble_builder = ensemble_builder
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y) -> "TreeEnsembleQuantileIntervalEstimator":
        self.model_ = self.ensemble_builder()
        self.model_.fit(X, y)
        self.n_features_in_ = getattr(self.model_, "n_features_in_", None)
        return self

    def _estimator_predictions(self, X) -> np.ndarray:
        if not hasattr(self.model_, "estimators_"):
            raise AttributeError(f"{type(self.model_).__name__} does not expose estimators_.")
        estimator_input = X.to_numpy() if hasattr(X, "to_numpy") else X
        per_tree_predictions = [
            np.asarray(estimator.predict(estimator_input), dtype=float).reshape(-1)
            for estimator in self.model_.estimators_
        ]
        return np.column_stack(per_tree_predictions)

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        prediction_matrix = self._estimator_predictions(X)
        lower_values = np.quantile(prediction_matrix, self.lower_quantile, axis=1)
        upper_values = np.quantile(prediction_matrix, self.upper_quantile, axis=1)
        return _normalize_bounds(lower_values, upper_values)

    def predict(self, X) -> np.ndarray:
        lower_values, upper_values = self.predict_interval(X)
        return (lower_values + upper_values) / 2.0
