from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .conformal_base import (
    BaseConformalMethod,
    ConformalBase,
    _as_prediction_array,
    _as_residual_array,
    _validate_alpha,
)


class ConformalizedQuantileRegression(BaseConformalMethod):
    """Conformalize precomputed lower and upper quantile predictions."""

    def __init__(
        self,
        *,
        residuals_upper: ArrayLike,
        residuals_lower: ArrayLike,
        predictions_upper: ArrayLike,
        predictions_lower: ArrayLike,
        alpha: float,
    ) -> None:
        self.alpha = _validate_alpha(alpha)
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

        if self.residuals_upper.shape != self.residuals_lower.shape:
            raise ValueError(
                "ConformalizedQuantileRegression expects residuals_upper and "
                "residuals_lower to be aligned elementwise."
            )
        if self.predictions_upper.shape != self.predictions_lower.shape:
            raise ValueError(
                "predictions_upper and predictions_lower must have matching shapes, "
                f"got {self.predictions_upper.shape} and {self.predictions_lower.shape}."
            )

        self.shared_residuals = np.maximum(-self.residuals_lower, self.residuals_upper)
        self.base_ = ConformalBase(
            residuals_upper=self.shared_residuals,
            residuals_lower=self.shared_residuals,
            predictions_upper=self.predictions_upper,
            predictions_lower=self.predictions_lower,
            alpha=self.alpha,
        )
