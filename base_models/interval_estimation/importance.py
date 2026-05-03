from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _series_from_values(values: np.ndarray, feature_names: list[str]) -> pd.Series:
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    if flat_values.shape[0] != len(feature_names):
        return pd.Series(dtype=float, name="importance")
    return pd.Series(flat_values, index=feature_names, name="importance").sort_values(ascending=False)


def _combine_importances(importances: Iterable[pd.Series]) -> pd.Series:
    valid_importances = [importance for importance in importances if not importance.empty]
    if not valid_importances:
        return pd.Series(dtype=float, name="importance")
    return (
        pd.concat(valid_importances, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
    )


def _compute_importance_from_model(model: object, feature_names: list[str]) -> pd.Series:
    if model is None:
        return pd.Series(dtype=float, name="importance")

    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
        return _series_from_values(np.asarray(values, dtype=float), feature_names)

    if hasattr(model, "get_feature_importance"):
        try:
            values = model.get_feature_importance()
        except TypeError:
            values = None
        if values is not None:
            return _series_from_values(np.asarray(values, dtype=float), feature_names)

    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(getattr(model, "coef_"), dtype=float)).reshape(-1)
        return _series_from_values(values, feature_names)

    nested_models: list[object] = []
    for attr_name in ("interval_models_", "model_", "lower_model_", "upper_model_"):
        if hasattr(model, attr_name):
            attr_value = getattr(model, attr_name)
            if isinstance(attr_value, (list, tuple)):
                nested_models.extend(attr_value)
            else:
                nested_models.append(attr_value)
    if nested_models:
        return _combine_importances(
            _compute_importance_from_model(nested_model, feature_names)
            for nested_model in nested_models
        )

    return pd.Series(dtype=float, name="importance")


def compute_importance(
    model: object,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
) -> pd.Series:
    del y_reference
    return _compute_importance_from_model(model, X_reference.columns.tolist())
