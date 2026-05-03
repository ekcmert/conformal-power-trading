from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_residual_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_prediction_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
    return alpha


def _conformal_quantile(residuals: FloatArray, *, alpha: float) -> float:
    if residuals.size == 0:
        return 0.0

    sorted_residuals = np.sort(residuals)
    quantile_rank = math.ceil((sorted_residuals.size + 1) * (1.0 - alpha))
    quantile_rank = min(max(quantile_rank, 1), sorted_residuals.size)
    return float(sorted_residuals[quantile_rank - 1])


@dataclass(frozen=True)
class ConformalResult:
    margin_upper: FloatArray
    margin_lower: FloatArray
    calibrated_upper: FloatArray
    calibrated_lower: FloatArray


class ConformalBase:
    """Compute conformal margins for upper and lower interval bounds."""

    def __init__(
        self,
        *,
        residuals_upper: ArrayLike,
        residuals_lower: ArrayLike,
        predictions_upper: ArrayLike,
        predictions_lower: ArrayLike,
        alpha: float,
    ) -> None:
        self.residuals_upper = _as_residual_array(
            residuals_upper,
            name="residuals_upper",
        )
        self.residuals_lower = _as_residual_array(
            residuals_lower,
            name="residuals_lower",
        )
        self.predictions_upper = _as_prediction_array(
            predictions_upper,
            name="predictions_upper",
        )
        self.predictions_lower = _as_prediction_array(
            predictions_lower,
            name="predictions_lower",
        )
        self.alpha = _validate_alpha(alpha)

        if self.predictions_upper.shape != self.predictions_lower.shape:
            raise ValueError(
                "predictions_upper and predictions_lower must have matching shapes, "
                f"got {self.predictions_upper.shape} and {self.predictions_lower.shape}."
            )

    @property
    def margin_upper_scalar(self) -> float:
        return _conformal_quantile(self.residuals_upper, alpha=self.alpha)

    @property
    def margin_lower_scalar(self) -> float:
        return _conformal_quantile(self.residuals_lower, alpha=self.alpha)

    def predict_margins(self) -> tuple[FloatArray, FloatArray]:
        margin_upper = np.full(
            self.predictions_upper.shape,
            self.margin_upper_scalar,
            dtype=float,
        )
        margin_lower = np.full(
            self.predictions_lower.shape,
            self.margin_lower_scalar,
            dtype=float,
        )
        return margin_upper, margin_lower

    def predict_interval(self) -> tuple[FloatArray, FloatArray]:
        margin_upper, margin_lower = self.predict_margins()
        calibrated_lower = self.predictions_lower - margin_lower
        calibrated_upper = self.predictions_upper + margin_upper
        return calibrated_lower, calibrated_upper

    def get_result(self) -> ConformalResult:
        margin_upper, margin_lower = self.predict_margins()
        calibrated_lower, calibrated_upper = self.predict_interval()
        return ConformalResult(
            margin_upper=margin_upper,
            margin_lower=margin_lower,
            calibrated_upper=calibrated_upper,
            calibrated_lower=calibrated_lower,
        )


class BaseConformalMethod:
    """Shared interface for methods that wrap an internal ConformalBase."""

    base_: ConformalBase

    @property
    def margin_upper_scalar(self) -> float:
        return self.base_.margin_upper_scalar

    @property
    def margin_lower_scalar(self) -> float:
        return self.base_.margin_lower_scalar

    def predict_margins(self) -> tuple[FloatArray, FloatArray]:
        return self.base_.predict_margins()

    def predict_interval(self) -> tuple[FloatArray, FloatArray]:
        return self.base_.predict_interval()

    def get_result(self) -> ConformalResult:
        return self.base_.get_result()
